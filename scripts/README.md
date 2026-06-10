# Scripts Overview

This directory is the operator layer for the project. Most files are thin wrappers around importable Python modules in `src/`, `apps/`, `packages/`, and `services/`.

For a fresh RunPod, the primary command is:

```bash
bash scripts/runpod_bootstrap_e2e.sh
```

That script runs the current end-to-end flow: validate code wiring, ingest four OpenML tasks into PostgreSQL, run Track A rollouts, score trajectories, build grouped GRPO data, train Track B with TRL GRPO, optionally rerun baseline and tuned policy endpoints, then optionally start Streamlit.

## Default E2E Flow

```text
scripts/runpod_bootstrap_e2e.sh
  -> scripts/ingest_openml_to_postgres.py
  -> python -m src.training.lightning_server_app
  -> python -m src.agents.supervisor
  -> python -m src.rewards.scorer
  -> python -m src.training.prepare_grpo_dataset
  -> scripts/gpu/run_02_train_policy_qlora_grpo.sh
  -> python -m src.training.train_policy_qlora_grpo
  -> optional baseline/tuned policy rerun
  -> optional streamlit dashboard
```

## Top-Level Scripts

| Script | Purpose | Default path? |
|---|---|---:|
| `runpod_bootstrap_e2e.sh` | One-command RunPod bootstrap and E2E benchmark runner. This is the main fresh-checkout entrypoint. | Yes |
| `setup_environment.sh` | Creates/uses a venv and installs CPU or GPU dependencies from the requirements files. | Optional |
| `start_demo.sh` | Starts MLflow and Streamlit together for a local demo. Can also start the Lightning server. | Optional |
| `demo_dashboard.py` | Thin Streamlit wrapper that imports `apps/dashboard/src/dashboard_app/main.py`. | Yes, when dashboard runs |
| `ingest_openml_to_postgres.py` | Downloads an OpenML dataset, writes safe registry metadata and execution-layer dataset source rows into PostgreSQL. | Yes |
| `verify_environment.py` | Imports core CPU/GPU packages and prints a JSON health report. Used by `setup_environment.sh`. | Optional |
| `validate_postgres_mcp_learning.py` | Validates PostgreSQL/MCP reward behavior and writes a learning report. | Optional |

## CPU Scripts

These scripts run data generation, Track A rollouts, scoring, grouping, MCP servers, and comparison reports. The fresh OpenML flow in `runpod_bootstrap_e2e.sh` calls the same Python modules directly, while these wrappers remain useful for manual step-by-step runs.

| Script | Purpose |
|---|---|
| `cpu/bootstrap_cpu_pod.sh` | Creates expected local folders and prints the manual CPU run order. |
| `cpu/init_database.sh` | Minimal database connectivity check through the project SQLAlchemy connector. |
| `cpu/start_postgres.sh` | Starts a local PostgreSQL container if no database is reachable. For Prisma/Postgres URL runs, set `DATABASE_URL` instead. |
| `cpu/start_mlflow.sh` | Starts an MLflow server using the configured tracking URI. |
| `cpu/start_mcp_server.sh` | Starts the legacy FastAPI MCP server at `src.mcp_server.server`. |
| `cpu/start_official_mcp_server.sh` | Starts the official stdio MCP server at `src.mcp_official.server`. |
| `cpu/run_01_generate_synthetic.sh` | Generates synthetic benchmark tasks into `data/synthetic`. Legacy/local path. |
| `cpu/run_02_load_synthetic_to_postgres.sh` | Loads synthetic tasks into PostgreSQL. Legacy/local path. |
| `cpu/load_synthetic_to_postgres.py` | Python loader used by the synthetic PostgreSQL wrapper. |
| `cpu/run_03_run_baseline_workflow.sh` | Runs Track A agent rollouts with `src.agents.supervisor`. |
| `cpu/run_04_score_trajectories.sh` | Scores rollout JSON with `src.rewards.scorer`. |
| `cpu/run_05_prepare_grpo_dataset.sh` | Groups scored trajectories into `data/grpo/grouped_rollouts.jsonl`. |
| `cpu/run_05b_export_agent_lightning_transitions.sh` | Exports Agent Lightning transition-style data from trajectories. |
| `cpu/run_05c_agent_lightning_official.sh` | Runs the official Agent Lightning runner path when that dependency is installed. |
| `cpu/run_06_compare_baseline_vs_tuned.sh` | Compares scored baseline and tuned trajectory directories. |
| `cpu/run_06_ruler_score_groups.sh` | Applies RULER/vLLM relative scoring to grouped rollouts. |
| `cpu/run_07_discover_mcp_tools.sh` | Discovers available MCP tools and writes a tool catalog. |
| `cpu/run_08_benchmark_policy_endpoints.sh` | Compares multiple scored policy output directories. |
| `cpu/run_08_generate_tool_scenarios.sh` | Generates tool-learning scenarios. Optional experiment path. |
| `cpu/run_09_run_tool_rollouts.sh` | Runs tool-learning rollouts through the supervisor. Optional experiment path. |
| `cpu/run_10_score_tool_learning.sh` | Scores tool-learning rollouts and builds tool RULER data. Optional experiment path. |
| `cpu/run_pdf_alignment_pipeline.sh` | Runs the older PDF-alignment validation pipeline. Optional documentation experiment. |
| `cpu/validate_architecture.sh` | Shell-level architecture validation checks. |

## GPU Scripts

These scripts start inference endpoints, run Track B training, and create runtime proof bundles.

| Script | Purpose |
|---|---|
| `gpu/run_01_baseline_batch_inference.sh` | Runs batch inference for baseline policy jobs. |
| `gpu/run_02_train_policy_qlora_grpo.sh` | Main Track B launcher. Defaults to `TRAINER=trl_grpo` and calls the Python GRPO trainer. |
| `gpu/run_03_tuned_batch_inference.sh` | Runs batch inference for tuned policy jobs. |
| `gpu/run_03_train_tool_policy_art_ruler.sh` | Trains a tool-policy variant using `ruler_relative` reward data. Optional experiment. |
| `gpu/run_all_trackb_options_one_by_one.sh` | Runs several Track B trainer/reward options sequentially for comparison. Optional experiment. |
| `gpu/run_ruler_trl_handoff.sh` | Scores grouped rollouts with a vLLM RULER judge, then optionally starts TRL GRPO. |
| `gpu/start_baseline_inference.sh` | Starts a baseline OpenAI-compatible vLLM/local inference endpoint. |
| `gpu/start_tuned_inference.sh` | Starts a tuned endpoint with a PEFT adapter checkpoint. |
| `gpu/start_lightning_server.sh` | Starts the GPU-side Lightning server bridge. |
| `gpu/start_verl_training.sh` | Starts veRL GRPO when veRL is installed; can explicitly fall back to TRL only when allowed. |
| `gpu/start_vllm_ruler_judge.sh` | Starts or health-checks an OpenAI-compatible vLLM endpoint used as a RULER judge. |
| `gpu/run_runtime_proof_all.sh` | Runs the full runtime proof bundle. |
| `gpu/run_runtime_proof_vllm_judge_ruler.sh` | Proves live vLLM RULER scoring with no fallback. |
| `gpu/run_runtime_proof_grpo_smoke_train.sh` | Runs a bounded TRL GRPO smoke training job and checks checkpoint files. |
| `gpu/run_runtime_proof_checkpoint_reload_compare.sh` | Reloads a checkpoint and compares baseline versus tuned held-out behavior. |

## Validation Scripts

| Script | Purpose |
|---|---|
| `validate/validate_agent_lightning_strict.py` | Ensures Agent Lightning paths fail fast and do not silently stub required behavior. |
| `validate/validate_monorepo_phase1.py` | Checks package/app/service boundaries compile and import. |
| `validate/validate_strict_mcp_agent_boundary.py` | Enforces the MCP boundary: agents cannot directly access raw DB rows or DB clients. |
| `dev_validate_official_mcp_server.py` | Validates official MCP server tool registration and basic tool calls. |
| `dev_validate_trackb_wiring.py` | Validates reward functions, training CLI help, and GPU shell syntax. |
| `dev_validate_dashboard_refactor.py` | Validates the modular Streamlit dashboard imports and render functions. |
| `dev_validate_official_art_ruler.py` | Checks optional ART/RULER integration availability and wiring. |
| `dev_validate_pdf_alignment.py` | Validates the older PDF-alignment runbook artifacts. |
| `dev_smoke_official_mcp_stdio.py` | Small stdio smoke test for official MCP client/tool calls. |
| `dev_summarize_validation.py` | Summarizes generated validation outputs. |

## TRL GRPO Drill-Down

The Track B GRPO training path is:

```text
scripts/runpod_bootstrap_e2e.sh
  -> TRAINER=trl_grpo bash scripts/gpu/run_02_train_policy_qlora_grpo.sh
    -> python -m src.training.train_policy_qlora_grpo
      -> train_trl_grpo()
        -> trl.GRPOConfig
        -> trl.GRPOTrainer
        -> trainer.train()
        -> trainer.save_model(...)
```

Key files:

| File | What it does |
|---|---|
| `scripts/runpod_bootstrap_e2e.sh` | Builds `data/grpo/grouped_rollouts.jsonl` from Track A scored trajectories and invokes Track B. |
| `scripts/gpu/run_02_train_policy_qlora_grpo.sh` | Converts shell env vars into CLI args for the trainer. Important env vars: `TRAINER`, `REWARD_MODE`, `NUM_GENERATIONS`, `MODEL_NAME`, `GRPO_DATASET_PATH`, and `POLICY_OUTPUT_DIR`. |
| `src/training/prepare_grpo_dataset.py` | Groups scored trajectories by task so GRPO can compare multiple rollouts for the same task. |
| `src/training/train_policy_qlora_grpo.py` | Contains the actual TRL GRPO implementation. |
| `src/rewards/policy_reward.py` | Provides TRL-compatible reward functions such as `hybrid`, `workflow_policy`, `trajectory_reward`, and `ruler_relative`. |

Inside `src/training/train_policy_qlora_grpo.py`, the important sequence is:

```text
build_training_examples(...)
  -> converts grouped rollout trajectories into prompt/completion/reward records

train_trl_grpo(...)
  -> imports GRPOConfig and GRPOTrainer from TRL
  -> builds a Hugging Face Dataset
  -> creates a LoRA config
  -> optionally configures 4-bit loading
  -> creates GRPOConfig
  -> selects a reward function from src.rewards.policy_reward
  -> instantiates GRPOTrainer
  -> calls trainer.train()
  -> saves the adapter checkpoint
```

The default training mode is not SFT. It is:

```bash
TRAINER=trl_grpo
REWARD_MODE=hybrid
NUM_GENERATIONS=4
bash scripts/gpu/run_02_train_policy_qlora_grpo.sh
```

## Cleanup Notes

The scripts directory intentionally keeps optional experiment wrappers, but removes one-off patch generators, old hardcoded OpenML seed scripts, generated caches, and the old monolithic Streamlit backup. The current OpenML database path is `scripts/ingest_openml_to_postgres.py`, and the current dashboard path is `scripts/demo_dashboard.py` -> `apps/dashboard/src/dashboard_app/main.py`.
