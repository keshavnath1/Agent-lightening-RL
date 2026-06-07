# Full-Design CPU/GPU Orchestration Runbook

This repository now separates **strict execution modes** from local development utilities. The CPU pod owns orchestration, hosted PostgreSQL/MCP access, telemetry, scoring, and benchmarking. The GPU pod owns OpenAI-compatible model serving and adapter/RL training. The code no longer silently falls back from one trainer to another when a strict full-design mode is selected.

## Runtime roles

| Plane | Responsibility | Required runtime state |
|---|---|---|
| CPU pod | MCP/FastAPI wrapper, DataEngineer extraction from hosted PostgreSQL, agent rollouts, Agent Lightning orchestration, reward scoring, benchmark comparison | `DATABASE_URL`, `POSTGRES_SCHEMA=agentic_ml`, Python CPU requirements, optional `agentlightning` package for official Lightning mode |
| GPU pod | vLLM/OpenAI-compatible policy endpoints, TRL GRPO LoRA training, optional veRL cluster command | GPU requirements, model cache, Hugging Face token when needed, `BASELINE_POLICY_URL`, `V2_POLICY_URL`, `TUNED_POLICY_URL`, and matching model names |

## Strict trainer modes

| `POLICY_TRAINER` | What it does | Failure behavior |
|---|---|---|
| `trl_grpo` | Runs the repository-native TRL GRPO trainer against grouped rollout data. | Fails if TRL/GPU dependencies are missing or incompatible. |
| `verl` | Prepares trajectory examples and hands control to the operator-provided official veRL command in `VERL_TRAIN_CMD`. | Fails if `VERL_TRAIN_CMD` is unset or exits non-zero. |
| `agent_lightning_official` | Runs the CPU workflow through an official `agentlightning.LitAgent`/`Trainer` integration. | Fails if the official Agent Lightning package/API is not installed. |

## Suggested restart sequence

Start the GPU pod first and expose one or more OpenAI-compatible `/v1/chat/completions` endpoints. A single GPU can be reused by changing the loaded model/adapter between benchmark phases, or separate URLs can be used for baseline, V2, and tuned policies.

```bash
cd /workspace/self-improving-ml-agent
source .env
# Start vLLM according to your selected GPU image/model.
# Example shape only; tune for the selected model and GPU memory.
python -m vllm.entrypoints.openai.api_server \
  --host 0.0.0.0 \
  --port 8000 \
  --model "$VLLM_MODEL_ID"
```

Then start the CPU pod services and validation.

```bash
cd /workspace/self-improving-ml-agent
source .env
export DATABASE_URL='<hosted postgres url>'
export POSTGRES_SCHEMA=agentic_ml
bash scripts/cpu/start_mcp_server.sh
bash scripts/cpu/start_mlflow.sh
bash scripts/cpu/validate_architecture.sh
```

## Official Agent Lightning run

Install the official runtime in the CPU pod and run the strict launcher. If the package is absent or incompatible, this command fails instead of pretending that local-only training was official Lightning.

```bash
cd /workspace/self-improving-ml-agent
source .env
pip install agentlightning
bash scripts/cpu/run_05c_agent_lightning_official.sh
```

## veRL handoff run

Install/configure veRL in the GPU pod using the official project instructions for your image, Ray topology, and accelerator layout. Then provide the exact veRL launch command through `VERL_TRAIN_CMD`. The repository exports these variables to the command: `GRPO_DATASET_PATH`, `POLICY_TRAINING_EXAMPLES_PATH`, `POLICY_OUTPUT_DIR`, and `MODEL_NAME`.

```bash
cd /workspace/self-improving-ml-agent
source .env
export POLICY_TRAINER=verl
export VERL_TRAIN_CMD='python -m verl.trainer.main_ppo trainer.project_name=self_improving_ml_agent'
bash scripts/gpu/run_02_train_policy_qlora_grpo.sh
```

## Baseline, V2, and tuned endpoint benchmark

The benchmark script requires live endpoint profiles. It does not create localhost defaults. `V2_POLICY_URL` can point to a second GPU pod, a second model server, or the same GPU pod after loading a different model or adapter.

```bash
cd /workspace/self-improving-ml-agent
source .env
export BASELINE_POLICY_URL='https://<baseline-gpu>-8000.proxy.runpod.net/v1/chat/completions'
export BASELINE_POLICY_MODEL='qwen2.5-coder-32b-instruct'
export V2_POLICY_URL='https://<v2-gpu>-8000.proxy.runpod.net/v1/chat/completions'
export V2_POLICY_MODEL='qwen2.5-coder-32b-instruct-v2'
export TUNED_POLICY_URL='https://<tuned-gpu>-8000.proxy.runpod.net/v1/chat/completions'
export TUNED_POLICY_MODEL='qwen2.5-coder-32b-instruct-lora'
export BENCHMARK_POLICIES='baseline v2 tuned'
export BENCHMARK_LIMIT=4
bash scripts/cpu/run_08_benchmark_policy_endpoints.sh
```

The output report is written to `reports/baseline_v2_tuned_benchmark.md`. The report includes a `live_policy_endpoint_rate` column so a benchmark cannot accidentally look like a live GPU evaluation when the policy endpoint was not used.

## GRPO Track A → Track B run

**Date:** June 07, 2026

The current branch is set up for a gated end-to-end workflow. **Track A** first executes against the baseline policy endpoint and produces grouped, scored trajectories. **Track B** trains a real PEFT LoRA adapter with the repository's TRL GRPO path, redeploys the OpenAI-compatible endpoint with `LOCAL_LLM_ADAPTER_PATH` set, and then runs the same benchmark again.

| Checkpoint | Evidence artifact | Result |
|---|---|---:|
| Track A baseline, four OpenML tasks | `reports/e2e_tracka_baseline_latest.md` | Written by `scripts/runpod_bootstrap_e2e.sh` |
| Track B adapter, same four tasks | `reports/e2e_tracka_vs_trackb_summary_latest.json` | Written by the post-redeploy benchmark |
| Track B adapter endpoint run | `reports/e2e_trackb_adapter_latest.md` | Written when `RUN_LIVE_POLICY=1` |
| Grouped GRPO evidence | `data/grpo/grouped_rollouts.jsonl` | Used by TRL GRPO training |

The lightweight validation endpoint now loads the adapter during process start through `peft.PeftModel.from_pretrained(...)` when `LOCAL_LLM_ADAPTER_PATH` is provided. The `/v1/load_lora_adapter` route remains a compatibility endpoint for reload requests, while production hot-swap should still use a vLLM server with LoRA support. The dashboard Results tab has been updated to discover the new E2E reports first, so the Streamlit app surfaces the latest Track A/Track B benchmark evidence before older baseline reports.
