# Fresh RunPod E2E Bootstrap Prompt

You are on a fresh RunPod with an empty network volume. The user has provided this GitHub repository URL and will provide runtime credentials out of band.

Your objective is to clone the repository, create the runtime environment, ingest/register the benchmark task in PostgreSQL, run Track A, run Track B, optionally redeploy the tuned adapter, rerun Track A, and write reports/logs that are visible in Streamlit.

## Required Inputs

The only hard required secret is:

```bash
export DATABASE_URL='postgres://USER:PASSWORD@HOST:PORT/DB?sslmode=require'
```

Use the same value for the three database roles unless the user explicitly gives separate role URLs:

```bash
export MCP_DATABASE_URL="${MCP_DATABASE_URL:-$DATABASE_URL}"
export EXECUTION_DATABASE_URL="${EXECUTION_DATABASE_URL:-$DATABASE_URL}"
```

## Optional Inputs

For model downloads, ask for one of these when missing and model download fails, when a gated/private model is requested, or when Hugging Face rate limits are likely:

```bash
export HF_TOKEN='hf_...'
# or
export HUGGINGFACE_HUB_TOKEN='hf_...'
```

Do not commit or print real tokens. It is fine to export them in the pod shell or put them in an untracked `.env` copied from `.env.example`.

## Clone

```bash
cd /workspace
git clone https://github.com/keshavnath1/Agent-lightening-RL.git
cd Agent-lightening-RL
git checkout E2E_feature
```

If SSH auth is configured and preferred:

```bash
git clone git@github.com:keshavnath1/Agent-lightening-RL.git
```

## Fast Path

After `DATABASE_URL` is exported, run the full bootstrap:

```bash
bash scripts/runpod_bootstrap_e2e.sh
```

For dashboard on RunPod's exposed HTTP port:

```bash
START_DASHBOARD=1 DASHBOARD_PORT=8888 bash scripts/runpod_bootstrap_e2e.sh
```

For endpoint redeploy comparison after Track B:

```bash
RUN_LIVE_POLICY=1 START_DASHBOARD=1 DASHBOARD_PORT=8888 bash scripts/runpod_bootstrap_e2e.sh
```

The live endpoint mode starts two lightweight OpenAI-compatible local model servers:

```text
baseline: http://127.0.0.1:18080/v1/chat/completions
tuned:    http://127.0.0.1:18081/v1/chat/completions
```

## What The Script Does

1. Creates `.venv`.
2. Installs `requirements-cpu.txt`.
3. Installs GPU dependencies when `nvidia-smi` is present or `INSTALL_GPU_DEPS=1`.
4. Exports `MCP_DATABASE_URL` and `EXECUTION_DATABASE_URL` from `DATABASE_URL`.
5. Runs source validations.
6. Ingests OpenML 31 German Credit into:
   - `ml_data.openml_31_german_credit`
   - `ml_registry.real_benchmark_tasks`
   - `ml_registry.real_dataset_summaries`
   - `ml_execution.dataset_sources`
7. Starts the Lightning server on port `19124`.
8. Runs Track A strict PostgreSQL/MCP rollouts.
9. Scores trajectories and writes `data/grpo/grouped_rollouts.jsonl`.
10. Runs Track B QLoRA SFT into `checkpoints/trackb_qlora_sft_runpod`.
11. Writes benchmark reports under `reports/`.
12. Optionally starts Streamlit.

## Main Outputs

```text
reports/run_logs/
reports/service_logs/
reports/tracka_initial_benchmark.md
reports/tracka_initial_vs_baseline_vs_trackb_redeploy.md
data/grpo/grouped_rollouts.jsonl
checkpoints/trackb_qlora_sft_runpod/
```

## Default Dataset

The default E2E task is OpenML 31 German Credit:

```bash
OPENML_ID=31
DATASET_KEY=openml_31_german_credit
TRACKA_TASK_ID=mltask_openml_31_german_credit_baseline
TARGET_COLUMN=class
PRIMARY_METRIC=roc_auc
```

Override these env vars only when the user asks for a different task.

## Streamlit

On RunPod, direct raw IP ports may time out. Prefer the RunPod HTTP proxy URL for the exposed port. If `8888` is the exposed HTTP service, run:

```bash
START_DASHBOARD=1 DASHBOARD_PORT=8888 bash scripts/runpod_bootstrap_e2e.sh
```

Then open the RunPod proxy URL for port `8888`.

## Safety Requirements

- Never commit `.env`, tokens, model weights, checkpoints, trajectories, `reports/run_logs`, database dumps, or raw dataset rows.
- Raw rows may be read only by the execution boundary using `EXECUTION_DATABASE_URL`.
- MCP and agent prompts receive metadata/profile summaries only.
- If Docker is unavailable in the pod, `CODE_INTERPRETER_MODE=host_subprocess` is acceptable for RunPod validation and must be reported.
- If a command fails due to a missing dependency, install the smallest missing package and rerun.
- If Hugging Face download fails with auth/rate-limit/gated-model errors, ask the user for `HF_TOKEN`.

## Manual Debug Commands

```bash
curl http://127.0.0.1:19124/health
python -m src.agents.supervisor --task-source postgres_mcp --task-limit 1 --lightning-server-url http://127.0.0.1:19124
python -m src.rewards.scorer --input-dir trajectories/tracka_initial --output-dir trajectories/tracka_initial_scored
python -m src.training.prepare_grpo_dataset --input-dir trajectories/tracka_initial_scored --output data/grpo/grouped_rollouts.jsonl
TRAINER=qlora_sft POLICY_OUTPUT_DIR=checkpoints/trackb_qlora_sft_runpod bash scripts/gpu/run_02_train_policy_qlora_grpo.sh
```
