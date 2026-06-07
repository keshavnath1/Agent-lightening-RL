# GPU Qwen2.5-Coder-32B Inference Connection Guide

## Current status

The GPU pod is now hosting **Qwen2.5-Coder-32B-Instruct** through a vLLM OpenAI-compatible API server. The server is externally reachable through the RunPod port proxy and has been validated from both the local sandbox and the CPU pod.

| Item | Value |
|---|---|
| GPU pod terminal URL | `https://86k136u2po46i5-19123.proxy.runpod.net/8v3flbcq4hwi3p8g1wi4b2r5ja9k4cz8/` |
| vLLM API base URL | `https://86k136u2po46i5-8000.proxy.runpod.net` |
| Chat completions URL | `https://86k136u2po46i5-8000.proxy.runpod.net/v1/chat/completions` |
| Models URL | `https://86k136u2po46i5-8000.proxy.runpod.net/v1/models` |
| Model id exposed by vLLM | `qwen2.5-coder-32b-instruct` |
| Model root | `Qwen/Qwen2.5-Coder-32B-Instruct` |
| Max model length configured | `4096` tokens |
| CPU repo path | `/workspace/self-improving-ml-agent` |
| CPU env file created | `/workspace/gpu_qwen_endpoint.env` |

## CPU pod configuration

The CPU pod has been configured with the following environment file:

```bash
source /workspace/gpu_qwen_endpoint.env
cd /workspace/self-improving-ml-agent
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
```

The file contains:

```bash
export GPU_POLICY_BASE_URL="https://86k136u2po46i5-8000.proxy.runpod.net"
export BASELINE_POLICY_URL="${GPU_POLICY_BASE_URL}/v1/chat/completions"
export MODEL_NAME="qwen2.5-coder-32b-instruct"
```

## CPU-to-GPU validation command

Run this on the CPU pod to verify connectivity through the repository’s existing policy client:

```bash
source /workspace/gpu_qwen_endpoint.env
cd /workspace/self-improving-ml-agent
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
python3 - <<'PY'
from src.inference.policy_client import PolicyClient
client = PolicyClient()
print('base_url', client.base_url)
print('model_name', client.model_name)
print(client.chat([{'role':'user','content':'Return only CPU_POLICY_CLIENT_OK'}], temperature=0))
PY
```

Expected output includes:

```text
base_url https://86k136u2po46i5-8000.proxy.runpod.net/v1/chat/completions
model_name qwen2.5-coder-32b-instruct
CPU_POLICY_CLIENT_OK
```

## Direct OpenAI-compatible request example

```bash
curl -sS \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"qwen2.5-coder-32b-instruct",
    "messages":[{"role":"user","content":"Write a Python add function. Return only code."}],
    "max_tokens":80,
    "temperature":0
  }' \
  https://86k136u2po46i5-8000.proxy.runpod.net/v1/chat/completions
```

## Restart notes

The model weights appear cached on the shared workspace storage, so the Hugging Face token should not be needed again unless the cache is removed or a new pod without cache access is started. If the GPU pod restarts, relaunch the server from the GPU pod using the startup script that was created during setup:

```bash
bash /workspace/start_qwen_vllm.sh
```

Then verify readiness:

```bash
curl -sS https://86k136u2po46i5-8000.proxy.runpod.net/v1/models
curl -I https://86k136u2po46i5-8000.proxy.runpod.net/health
```

The vLLM server may take several minutes to become ready because it must load the 32B checkpoint, profile memory, create the KV cache, and warm up the model.
