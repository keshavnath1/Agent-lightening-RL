from __future__ import annotations

import os
import time
import uuid
from threading import Lock
from typing import Any

import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = os.getenv("LOCAL_LLM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
ADAPTER_PATH = os.getenv("LOCAL_LLM_ADAPTER_PATH")
MAX_NEW_TOKENS = int(os.getenv("LOCAL_LLM_MAX_NEW_TOKENS", "384"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" and torch.cuda.is_bf16_supported() else torch.float16 if DEVICE == "cuda" else torch.float32

app = FastAPI(title="Local OpenAI-Compatible Transformers LLM", version="0.1.0")
_gen_lock = Lock()
_tokenizer: AutoTokenizer | None = None
_model: AutoModelForCausalLM | None = None
_loaded_at: float | None = None
_last_adapter_path: str | None = None


class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[Message]
    temperature: float = 0.2
    max_tokens: int | None = Field(default=None, alias="max_tokens")
    top_p: float = 0.95
    stream: bool = False


class LoadLoraRequest(BaseModel):
    lora_name: str | None = None
    lora_path: str | None = None
    adapter_name: str | None = None
    adapter_path: str | None = None


def _load() -> None:
    global _tokenizer, _model, _loaded_at, _last_adapter_path
    if _model is not None and _tokenizer is not None:
        return
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=DTYPE,
        device_map="auto" if DEVICE == "cuda" else None,
        trust_remote_code=True,
    )
    if DEVICE == "cpu":
        _model.to(DEVICE)
    if ADAPTER_PATH:
        from peft import PeftModel
        _model = PeftModel.from_pretrained(_model, ADAPTER_PATH)
        _last_adapter_path = ADAPTER_PATH
    _model.eval()
    _loaded_at = time.time()


def _prompt_from_messages(messages: list[Message]) -> str:
    assert _tokenizer is not None
    rows = [m.model_dump() for m in messages]
    if hasattr(_tokenizer, "apply_chat_template"):
        return _tokenizer.apply_chat_template(rows, tokenize=False, add_generation_prompt=True)
    return "\n".join(f"{m.role}: {m.content}" for m in messages) + "\nassistant:"


@app.on_event("startup")
def startup() -> None:
    _load()


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": _model is not None,
        "model": MODEL_NAME,
        "device": DEVICE,
        "dtype": str(DTYPE),
        "loaded_at": _loaded_at,
        "last_adapter_path": _last_adapter_path,
    }


@app.get("/v1/models")
def models() -> dict[str, Any]:
    return {"object": "list", "data": [{"id": MODEL_NAME, "object": "model", "created": int(_loaded_at or time.time()), "owned_by": "local-runpod"}]}


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest) -> dict[str, Any]:
    if req.stream:
        raise ValueError("Streaming is not implemented in this lightweight validation server.")
    _load()
    assert _tokenizer is not None and _model is not None
    prompt = _prompt_from_messages(req.messages)
    max_new = min(int(req.max_tokens or MAX_NEW_TOKENS), MAX_NEW_TOKENS)
    inputs = _tokenizer(prompt, return_tensors="pt").to(_model.device)
    with _gen_lock, torch.no_grad():
        output_ids = _model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=req.temperature > 0,
            temperature=max(req.temperature, 1e-5),
            top_p=req.top_p,
            pad_token_id=_tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0, inputs["input_ids"].shape[-1]:]
    text = _tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    prompt_tokens = int(inputs["input_ids"].numel())
    completion_tokens = int(new_tokens.numel())
    return {
        "id": "chatcmpl-" + uuid.uuid4().hex,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model or MODEL_NAME,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": prompt_tokens + completion_tokens},
    }


@app.post("/v1/load_lora_adapter")
def load_lora_adapter(req: LoadLoraRequest) -> dict[str, Any]:
    global _last_adapter_path
    _last_adapter_path = req.lora_path or req.adapter_path
    return {
        "ok": True,
        "status": "accepted_noop",
        "adapter_name": req.lora_name or req.adapter_name,
        "adapter_path": _last_adapter_path,
        "note": "The lightweight Transformers validation server records the adapter reload request. Use vLLM for true LoRA hot-swap in production.",
    }
