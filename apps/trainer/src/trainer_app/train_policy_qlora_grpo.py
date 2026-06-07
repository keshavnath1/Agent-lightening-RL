from __future__ import annotations

import argparse
import inspect
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Iterable


SYSTEM_PROMPT = (
    'You are the policy model inside a self-improving multi-agent ML workflow. '
    'Given a compact task state and previous agent actions, emit the next high-quality '
    'agent action as JSON. Do not expose raw dataset rows in the response.'
)


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    rows: list[dict[str, Any]] = []
    for line in p.read_text(encoding='utf-8').splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _compact_tool_calls(step: dict[str, Any]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for call in step.get('tool_calls', []) or []:
        compact.append({
            'tool_name': call.get('tool_name'),
            'status': call.get('status'),
            'has_error': bool(call.get('error')),
            'output_ref': call.get('output_ref'),
        })
    return compact


def _build_prompt(task_id: str, trajectory: dict[str, Any], step_index: int) -> str:
    prior_steps = []
    for prior in trajectory.get('steps', [])[:step_index]:
        prior_steps.append({
            'agent_name': prior.get('agent_name'),
            'action': prior.get('action'),
            'tool_statuses': [c.get('status') for c in prior.get('tool_calls', []) or []],
        })
    state = {
        'task_id': task_id,
        'policy_version': trajectory.get('policy_version'),
        'trajectory_id': trajectory.get('trajectory_id'),
        'step_index': step_index,
        'previous_agent_actions': prior_steps,
        'final_reward_available_for_training_only': trajectory.get('reward', 0.0),
        'instruction': 'Return the next agent action JSON with agent_name, action, reasoning_summary, and expected_tool_calls.',
    }
    return SYSTEM_PROMPT + '\n\nSTATE_JSON:\n' + json.dumps(state, sort_keys=True)


def _build_completion(step: dict[str, Any]) -> str:
    target = {
        'agent_name': step.get('agent_name'),
        'action': step.get('action'),
        'reasoning_summary': step.get('reasoning_summary'),
        'expected_tool_calls': _compact_tool_calls(step),
    }
    return json.dumps(target, sort_keys=True)


def build_training_examples(
    grouped_rollouts: Iterable[dict[str, Any]],
    max_trajectories_per_task: int = 1,
    min_reward: float | None = None,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for group in grouped_rollouts:
        task_id = group.get('task_id')
        ranked = group.get('ranked_trajectories', []) or []
        kept = 0
        for trajectory in ranked:
            reward = float(trajectory.get('reward') or 0.0)
            if min_reward is not None and reward < min_reward:
                continue
            for step_index, step in enumerate(trajectory.get('steps', []) or []):
                if not step.get('agent_name') or not step.get('action'):
                    continue
                examples.append({
                    'task_id': task_id,
                    'trajectory_id': trajectory.get('trajectory_id'),
                    'reward': reward,
                    'prompt': _build_prompt(str(task_id), trajectory, step_index),
                    'completion': _build_completion(step),
                    'text': _build_prompt(str(task_id), trajectory, step_index) + '\n\nACTION_JSON:\n' + _build_completion(step),
                })
            kept += 1
            if kept >= max_trajectories_per_task:
                break
    if not examples:
        raise ValueError('No training examples were produced from the grouped rollout dataset.')
    return examples


def write_examples(examples: list[dict[str, Any]], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    examples_path = output_dir / 'policy_training_examples.jsonl'
    with examples_path.open('w', encoding='utf-8') as f:
        for row in examples:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    return examples_path


def _common_lora_config(r: int, alpha: int, dropout: float):
    from peft import LoraConfig

    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias='none',
        task_type='CAUSAL_LM',
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
    )


def train_qlora_sft(
    examples: list[dict[str, Any]],
    output_dir: Path,
    model_name: str,
    epochs: float,
    batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    max_seq_length: int,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
) -> dict[str, Any]:
    import torch
    from datasets import Dataset
    from peft import get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map='auto',
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, _common_lora_config(lora_r, lora_alpha, lora_dropout))

    dataset = Dataset.from_list([{'text': row['text']} for row in examples])

    def tokenize(batch: dict[str, list[str]]) -> dict[str, Any]:
        tokenized = tokenizer(
            batch['text'],
            truncation=True,
            max_length=max_seq_length,
            padding=False,
        )
        tokenized['labels'] = [ids.copy() for ids in tokenized['input_ids']]
        return tokenized

    tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=['text'])
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        logging_steps=5,
        save_strategy='epoch',
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        report_to=['none'],
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    return {
        'trainer': 'qlora_sft',
        'adapter_status': 'trained',
        'num_examples': len(examples),
        'model_name': model_name,
        'output_dir': str(output_dir),
    }


def _json_validity_reward(completions: list[str], **_: Any) -> list[float]:
    rewards: list[float] = []
    for completion in completions:
        try:
            parsed = json.loads(completion)
            score = 0.2
            for key in ['agent_name', 'action', 'reasoning_summary']:
                if parsed.get(key):
                    score += 0.2
            if parsed.get('expected_tool_calls') is not None:
                score += 0.2
            rewards.append(min(score, 1.0))
        except Exception:
            rewards.append(0.0)
    return rewards


def train_trl_grpo(
    examples: list[dict[str, Any]],
    output_dir: Path,
    model_name: str,
    epochs: float,
    batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    max_seq_length: int,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
) -> dict[str, Any]:
    import torch
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    train_dataset = Dataset.from_list([{'prompt': row['prompt'], 'reference_completion': row['completion']} for row in examples])
    peft_config = _common_lora_config(lora_r, lora_alpha, lora_dropout)

    # 4-bit QLoRA quantization config (mirrors train_qlora_sft).
    # model_init_kwargs is accepted by GRPOTrainer when `model` is a string.
    try:
        from transformers import BitsAndBytesConfig as _BnB
        _bnb_config = _BnB(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
        _model_init_kwargs: dict[str, Any] = {
            'quantization_config': _bnb_config,
            'device_map': 'auto',
            'trust_remote_code': True,
        }
    except ImportError:
        # bitsandbytes not installed (CPU-only environment) – run full precision.
        _model_init_kwargs = {'device_map': 'auto', 'trust_remote_code': True}

    config_kwargs = {
        'output_dir': str(output_dir),
        'num_train_epochs': epochs,
        'per_device_train_batch_size': batch_size,
        'gradient_accumulation_steps': gradient_accumulation_steps,
        'learning_rate': learning_rate,
        'logging_steps': 5,
        'save_strategy': 'epoch',
        'max_prompt_length': max_seq_length,
        'report_to': ['none'],
    }
    config_sig = inspect.signature(GRPOConfig)
    config_kwargs = {k: v for k, v in config_kwargs.items() if k in config_sig.parameters}
    args = GRPOConfig(**config_kwargs)
    trainer_kwargs = {
        'model': model_name,
        'reward_funcs': _json_validity_reward,
        'args': args,
        'train_dataset': train_dataset,
        'peft_config': peft_config,
        'model_init_kwargs': _model_init_kwargs,
    }
    trainer_sig = inspect.signature(GRPOTrainer)
    trainer_kwargs = {k: v for k, v in trainer_kwargs.items() if k in trainer_sig.parameters}
    trainer = GRPOTrainer(**trainer_kwargs)
    trainer.train()
    trainer.save_model(str(output_dir))
    return {
        'trainer': 'trl_grpo',
        'adapter_status': 'trained',
        'num_examples': len(examples),
        'model_name': model_name,
        'output_dir': str(output_dir),
    }


def run_official_agent_lightning_training(
    tasks_path: str,
    output_dir: Path,
    policy_version: str,
    limit: int | None,
) -> dict[str, Any]:
    from src.training.agent_lightning_official_runner import run_official_agent_lightning

    return run_official_agent_lightning(
        tasks_path=tasks_path,
        output_dir=output_dir,
        policy_version=policy_version,
        limit=limit,
    )


def run_verl_training_handoff(
    grouped_dataset_path: str,
    examples_path: Path,
    output_dir: Path,
    model_name: str,
    command: str | None = None,
) -> dict[str, Any]:
    """Execute an operator-supplied official veRL command with strict failure semantics.

    The repository prepares trajectory-derived training examples and grouped rollouts,
    then hands them to the GPU pod command declared in VERL_TRAIN_CMD or --verl-command.
    This intentionally avoids pretending to run a veRL cluster when the pod has not
    installed/configured veRL; a missing command is a hard configuration error.
    """
    effective_command = command or os.getenv('VERL_TRAIN_CMD')
    if not effective_command:
        raise RuntimeError(
            'veRL mode requires VERL_TRAIN_CMD or --verl-command. Install/configure official veRL '
            'on the GPU pod and provide the command that launches the veRL trainer/cluster.'
        )
    env = os.environ.copy()
    env.update({
        'GRPO_DATASET_PATH': str(grouped_dataset_path),
        'POLICY_TRAINING_EXAMPLES_PATH': str(examples_path),
        'POLICY_OUTPUT_DIR': str(output_dir),
        'MODEL_NAME': str(model_name),
    })
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / 'verl_train_command.log'
    with log_path.open('w', encoding='utf-8') as log_file:
        completed = subprocess.run(
            effective_command,
            shell=True,
            cwd=Path(__file__).resolve().parents[2],
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f'veRL command failed with exit code {completed.returncode}. See {log_path}.'
        )
    return {
        'trainer': 'official_verl_handoff',
        'adapter_status': 'trained_or_exported_by_verl_command',
        'model_name': model_name,
        'output_dir': str(output_dir),
        'verl_command_log': str(log_path),
        'verl_command': effective_command,
    }


def write_metadata(output_dir: Path, metadata: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'adapter_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser(description='Train a LoRA policy adapter from scored agent rollouts.')
    parser.add_argument('--dataset', default='data/grpo/grouped_rollouts.jsonl')
    parser.add_argument('--output-dir', default='checkpoints/qwen25-3b-agent-lora')
    parser.add_argument('--model-name', default='Qwen/Qwen2.5-3B-Instruct')
    parser.add_argument('--trainer', choices=['qlora_sft', 'trl_grpo', 'agent_lightning_official', 'verl'], default='trl_grpo')
    parser.add_argument('--epochs', type=float, default=1.0)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--gradient-accumulation-steps', type=int, default=8)
    parser.add_argument('--learning-rate', type=float, default=2e-4)
    parser.add_argument('--max-seq-length', type=int, default=2048)
    parser.add_argument('--max-trajectories-per-task', type=int, default=1)
    parser.add_argument('--min-reward', type=float, default=None)
    parser.add_argument('--lora-r', type=int, default=16)
    parser.add_argument('--lora-alpha', type=int, default=32)
    parser.add_argument('--lora-dropout', type=float, default=0.05)
    parser.add_argument('--agent-lightning-tasks', default='data/synthetic/tasks.jsonl')
    parser.add_argument('--agent-lightning-policy-version', default='agent_lightning_policy')
    parser.add_argument('--agent-lightning-limit', type=int, default=None)
    parser.add_argument('--verl-command', default=None)
    args = parser.parse_args()

    out = Path(args.output_dir)
    grouped: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []
    examples_path: Path | None = None

    try:
        if args.trainer == 'agent_lightning_official':
            metadata = run_official_agent_lightning_training(
                tasks_path=args.agent_lightning_tasks,
                output_dir=out,
                policy_version=args.agent_lightning_policy_version,
                limit=args.agent_lightning_limit,
            )
        else:
            grouped = _read_jsonl(args.dataset)
            examples = build_training_examples(grouped, args.max_trajectories_per_task, args.min_reward)
            examples_path = write_examples(examples, out)
            common_kwargs = dict(
                examples=examples,
                output_dir=out,
                model_name=args.model_name,
                epochs=args.epochs,
                batch_size=args.batch_size,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                learning_rate=args.learning_rate,
                max_seq_length=args.max_seq_length,
                lora_r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
            )
            if args.trainer == 'trl_grpo':
                metadata = train_trl_grpo(**common_kwargs)
            elif args.trainer == 'qlora_sft':
                metadata = train_qlora_sft(**common_kwargs)
            elif args.trainer == 'verl':
                metadata = run_verl_training_handoff(
                    grouped_dataset_path=args.dataset,
                    examples_path=examples_path,
                    output_dir=out,
                    model_name=args.model_name,
                    command=args.verl_command,
                )
            else:  # pragma: no cover - argparse enforces choices
                raise ValueError(f'Unsupported trainer: {args.trainer}')
    except Exception as exc:
        failure_metadata = {
            'trainer': args.trainer,
            'adapter_status': 'training_failed',
            'model_name': args.model_name,
            'dataset_path': args.dataset,
            'examples_path': str(examples_path) if examples_path else None,
            'num_examples': len(examples),
            'error': f'{exc.__class__.__name__}: {exc}',
            'note': 'No fake adapter was created and no silent fallback was used. Fix the requested trainer/runtime and rerun on the CPU/GPU pod.',
        }
        write_metadata(out, failure_metadata)
        raise

    metadata.update({
        'dataset_path': args.dataset,
        'examples_path': str(examples_path) if examples_path else None,
        'training_method': metadata.get('trainer') or args.trainer,
        'grouped_rollout_groups': len(grouped),
        'fallback_used': False,
    })
    write_metadata(out, metadata)
    print(f'Trained policy adapter metadata written to {out / "adapter_metadata.json"}')


if __name__ == '__main__':
    main()
