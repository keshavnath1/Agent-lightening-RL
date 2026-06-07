# Self-Improving Tabular ML Agent on RunPod

This repository is a RunPod-ready MVP for demonstrating a **multi-agent tabular ML workflow** that improves its LLM decision policy through reinforcement-style fine-tuning. The architecture intentionally separates the **CPU tool/control plane** from the **GPU policy/training plane**.

The key implementation principle is that there are two distinct training tracks. **Track A** trains or benchmarks tabular task models such as XGBoost and LightGBM. **Track B** trains the agent policy, using scored multi-agent trajectories and grouped rollout data with TRL GRPO-style policy optimization.

## Handoff documents

See `docs/implementation_summary_and_runpod_instructions.md` for the original implementation summary and MVP execution sequence. See `docs/runpod_cpu_gpu_setup_and_connection_guide.md` for the updated answer to the RunPod setup question, including when to use SSH, VS Code Remote-SSH, and JupyterLab. See `docs/agent_lightning_assessment_and_migration.md` for the Microsoft Agent Lightning assessment and migration path.

## Fresh RunPod E2E Quickstart

For a new RunPod with an empty network volume, the intended path is clone branch
`E2E_grpo_4openml_streamlit_evidence`, provide runtime secrets as environment variables, and run one
bootstrap script.

```bash
cd /workspace
git clone https://github.com/keshavnath1/Agent-lightening-RL.git
cd Agent-lightening-RL
git checkout E2E_grpo_4openml_streamlit_evidence

export DATABASE_URL='postgres://USER:PASSWORD@HOST:PORT/DB?sslmode=require'
# Optional, but recommended for Hugging Face model downloads.
export HF_TOKEN='hf_...'
# Recommended on RunPod network volumes so package install happens on local disk.
export VENV_DIR=/root/agent-lightening-rl-grpo-venv
# TRL GRPO requires a torch build with torch.distributed.fsdp.FSDPModule.
export PYTORCH_VERSION=2.8.0

bash scripts/runpod_bootstrap_e2e.sh
```

To include the post-Track-B live endpoint redeploy comparison and dashboard:

```bash
RUN_LIVE_POLICY=1 START_DASHBOARD=1 DASHBOARD_PORT=8503 bash scripts/runpod_bootstrap_e2e.sh
```

Use `DASHBOARD_PORT=8888` instead if that port is free and exposed on your pod. On the current RunPod validation pod, Jupyter owns `8888`, so Streamlit evidence was captured from `8503`.

The script creates `.venv`, installs dependencies, ingests four OpenML tasks into the
PostgreSQL ML contract, runs Track A, scores/groups rollouts, runs Track B TRL GRPO,
writes benchmark reports under `reports/`, and optionally starts Streamlit.
See `.github/prompts/fresh-runpod-e2e.prompt.md` for the Codex-facing operating
prompt. Do not commit `.env`, database URLs, Hugging Face tokens, checkpoints,
model weights, database dumps, or raw dataset rows. Sanitized trajectory JSONL,
run logs, grouped rollout data, and benchmark reports are allowed on this branch
so a fresh checkout can run Streamlit and inspect evidence immediately.

## Streamlit Evidence After Checkout

This branch intentionally keeps sanitized demo evidence in git:

- `reports/*.md` and `reports/*.json` for benchmark summaries
- `reports/run_logs/*.log` and `reports/service_logs/*.log` for Track A/B operational logs
- `trajectories/**/*.jsonl` for multi-agent traces and scored rollouts
- `data/grpo/grouped_rollouts.jsonl` for GRPO grouped rollout evidence

Run:

```bash
LIGHTNING_SERVER_URL=http://localhost:19124 \
VLLM_BASE_URL=http://localhost:18180 \
MLFLOW_TRACKING_URI=/workspace/Agent-lightening-RL/mlruns \
MLFLOW_ALLOW_FILE_STORE=true \
streamlit run scripts/demo_dashboard.py --server.port 8503 --server.address 0.0.0.0
```

Then open the Results page to inspect reports, logs, trajectory steps, tool calls, rewards, and baseline-vs-tuned comparisons.

### Streamlit screenshot evidence

The screenshots below were captured from the current RunPod dashboard with the service health line showing `OK Lightning OK vLLM OK MLflow`. They are committed as lightweight evidence that a fresh checkout can render the dashboard views from the checked-in reports, trajectories, grouped rollouts, and logs.

| Dashboard view | Evidence |
|---|---|
| Overview and service health | ![Streamlit overview health](docs/evidence/streamlit/streamlit-overview-health.jpg) |
| Track B TRL GRPO configuration | ![Streamlit Track B GRPO](docs/evidence/streamlit/streamlit-trackb-grpo.jpg) |
| Results benchmark comparison | ![Streamlit results benchmark](docs/evidence/streamlit/streamlit-results-benchmark.jpg) |
| Monorepo health inventory | ![Streamlit monorepo health](docs/evidence/streamlit/streamlit-monorepo-health.jpg) |

## Current GRPO Branch Benchmark Goal

This branch replaces the previous supervised Track B demo with a TRL GRPO-first path.
The target benchmark is:

| Stage | Evidence artifact |
|---|---|
| Four OpenML ingests | `reports/run_logs/openml_ingest_latest.log` |
| Initial Track A rollouts | `trajectories/tracka_initial/` |
| Scored Track A rollouts | `trajectories/tracka_initial_scored/` |
| GRPO grouped data | `data/grpo/grouped_rollouts.jsonl` |
| Track B TRL GRPO logs | `reports/run_logs/trackb_trl_grpo_latest.log` |
| Baseline vs tuned comparison | `reports/tracka_initial_vs_baseline_vs_trackb_redeploy.md` |

After each run, sanitize and commit the text/JSON evidence above so Streamlit can render it from a fresh checkout.

## Latest feedback implementation pass

The latest pass implements the highest-priority Track A feedback: real GBM benchmarking, profile-driven preprocessing, high-cardinality-aware benchmark planning, MLflow-compatible tracking with JSON fallback, guardrail validation, stronger reward scoring, multiple rollouts per task, and a Docker execution path with safe local fallback. Validation evidence is saved at `reports/feedback_validation_summary.txt`.

## MVP Flow

```text
CPU plane: synthetic tasks → PostgreSQL/MCP tools → multi-agent workflow → telemetry → reward scoring
GPU plane: baseline inference → RL policy training → tuned inference → held-out evaluation
```

## Main Commands

```bash
# CPU Pod
export WORKSPACE_DIR=/workspace/self-improving-ml-agent
export TASK_COUNT=20
export TASK_LIMIT=2
export ROLLOUTS_PER_TASK=2
bash scripts/cpu/run_01_generate_synthetic.sh
bash scripts/cpu/run_03_run_baseline_workflow.sh
bash scripts/cpu/run_04_score_trajectories.sh
bash scripts/cpu/run_05_prepare_grpo_dataset.sh
bash scripts/cpu/run_05b_export_agent_lightning_transitions.sh

# GPU Pod
bash scripts/gpu/run_01_baseline_batch_inference.sh
bash scripts/gpu/run_02_train_policy_qlora_grpo.sh
bash scripts/gpu/run_03_tuned_batch_inference.sh

# CPU Pod after tuned results are available
bash scripts/cpu/run_06_compare_baseline_vs_tuned.sh
```

## Guardrails

The demo enforces a **no raw rows in LLM context** design. Agents receive schema metadata, profile summaries, and artifact references, not raw dataframe rows. Code execution is routed through a controlled interpreter wrapper with Docker support and safe local fallback, all meaningful tool calls are logged as trajectories, and reviewer guardrails validate generated artifacts before rewards are finalized.


## Official ART + RULER Integration

This project now supports an official ART + RULER training path in addition to the existing TRL/veRL/Agent-Lightning-style paths.

The official ART path uses openpipe-art's LangGraph integration to wrap ML-agent rollouts, capture them as `art.Trajectory` objects, group multiple rollouts as `art.TrajectoryGroup`, score them using official RULER when available, and train through `art.LocalBackend`.

When ART/RULER is not installed, the project falls back only to clearly labeled local compatibility modes and does not claim official ART execution.

The project also includes a local compatibility path: `src/telemetry/art_compat.py` converts custom ML-agent trajectories into ART-like structures, and `src/rewards/ruler_like_judge.py` supports heuristic RULER-like ranking for offline demonstrations.


## Optional integrations

Experimental official Agent Lightning and ART/RULER integrations are isolated in `requirements-optional.txt`. Install them with `bash scripts/setup_environment.sh --target cpu --with-optional` only when those integrations are required.


## RULER vLLM judge to TRL handoff

The repository now includes a safe handoff path for the pasted RULER/vLLM/TRL workflow. Use `bash scripts/gpu/start_vllm_ruler_judge.sh --health` to inspect a local OpenAI-compatible judge endpoint, then run `bash scripts/gpu/run_ruler_trl_handoff.sh --dry-run` to score grouped rollouts, validate `ruler_relative` reward metadata, and print the exact GPU training command. The handoff launches training only when called with `--train` or `RUN_TRAINING=1`. See `docs/ruler_vllm_trl_runbook.md` for the one-step-at-a-time operating table.

## Official Project MCP Server

This repository now includes an **Official Project MCP Server** in `src/mcp_official/`. The server uses the official Python MCP SDK to expose safe project tools for the ML-agent tool layer, while leaving the older REST-style `src/mcp_server/` implementation in place for compatibility.

| Capability | Implementation | Safety posture |
|---|---|---|
| Safe project tools | `postgres_get_task_metadata`, `postgres_get_dataset_schema`, `postgres_get_dataset_summary`, `postgres_get_artifact_manifest`, `postgres_get_rollout_status`, and `postgres_get_reward_history` | Designed for metadata, summaries, manifests, and trajectory/reward summaries; ordinary tools do not expose raw dataset rows |
| Developer SQL diagnostics | `postgres_debug_readonly_sql` | Disabled unless `ALLOW_RAW_SQL_TOOL=1`; guarded by `validate_sql`, which blocks `SELECT *`, write operations, comments, semicolons, and blocked raw-row tables |
| Tool-call telemetry | `reports/mcp_official_tool_calls.jsonl` | Every official MCP tool call records timestamp, arguments, status, error, duration, `official_mcp=true`, and a compact result summary |
| Agent integration | `src/tools/official_mcp_client.py`, `src/tools/langchain_mcp_bridge.py`, and `src/tools/schema_from_function.py` | Agents can discover tools and schemas dynamically without embedding credentials or raw data |

Start the server from the project root with:

```bash
bash scripts/cpu/start_official_mcp_server.sh
```

A placeholder Cursor/Claude configuration is available at `.cursor/mcp.example.json`. It intentionally uses `postgresql+psycopg2://USER:PASSWORD@HOST:PORT/DBNAME` and must not be replaced with real credentials in source control.

The Project MCP server is separate from RunPod MCP. RunPod MCP is infrastructure-only; Project MCP is the ML-agent tool-learning interface. Its logged tool interactions can become trajectory data, RULER can judge whether the agent selected and used tools correctly, and TRL GRPO can train the policy to improve future tool choices while keeping the existing RULER/vLLM, TRL, QLoRA, and Agent-Lightning-style pipeline intact.

## Lightweight Monorepo Phase 1

This repository now includes a **Phase 1 lightweight monorepo layout**. The change follows a no-big-bang migration strategy: the existing `src/` code and `scripts/` remain operational, while new `packages/`, `services/`, and `apps/` directories provide clean boundaries through compatibility wrappers.

| New boundary | Path | Current source of truth |
|---|---|---|
| Shared contracts | `packages/contracts` | New dataclass schemas. |
| Safe ML/PostgreSQL tools | `packages/ml_tools` | `src.tools`. |
| Reward and RULER scoring | `packages/rewards` | `src.rewards`. |
| MCP client bridge | `packages/mcp_client_bridge` | `src.tools.official_mcp_client`, `src.tools.langchain_mcp_bridge`, and schema helpers. |
| Agent Lightning bridge | `packages/lightning_bridge` | `src.training` bridge modules. |
| Official project MCP server | `services/project_mcp_server` | `src.mcp_official`. |
| Runnable apps | `apps/dashboard`, `apps/rollout_worker`, `apps/ruler_scorer`, `apps/trainer` | Existing dashboard, rollout, RULER, and trainer modules. |

The new root `pyproject.toml` defines a `uv` workspace and pytest Python paths for these packages. This keeps CPU/GPU, MCP, RULER, and TRL/GRPO concerns separated while preserving the existing Track A and Track B benchmark commands.

See `docs/monorepo_phase1.md` for the validation checklist.


## Phase 2 monorepo hardening setup

Phase 2 keeps the existing `src/` implementation as the runtime source of truth while making the new monorepo package boundaries easier to install, inspect, and validate. The most important operational rule is that lightweight imports should not require GPU, MCP, LangGraph, PostgreSQL, or Streamlit dependencies unless the concrete submodule needs them.

| Goal | Command |
|---|---|
| Install the CPU/reporting environment | `python3 -m pip install -r requirements-cpu.txt` |
| Validate monorepo structure and compatibility wrappers | `make validate-monorepo` |
| Run targeted monorepo compatibility tests | `make test-monorepo` |
| Validate the official project MCP server | `make validate-mcp` |
| Validate ART/RULER/Track B wiring | `make validate-ruler` |
| Start the dashboard wrapper | `make dashboard` |
| Run Track A benchmark wrapper | `make track-a` |
| Run Track B benchmark wrapper | `make track-b` |

For package-level work, import concrete modules rather than package roots when a runtime dependency is expected. For example, use `from project_mcp_server.server import mcp` to start the MCP service, `from rollout_worker.run_track_a import run_tasks` for rollouts, and `from ml_tools.sql_safety import validate_sql` for lightweight SQL validation.

## Fresh GPU Pod Setup

When creating a new GPU pod with the same network volume, use the persistent setup prompt at:

```text
.github/prompts/gpu-pod-project-setup.prompt.md
```

This prompt helps coding agents understand the project structure, install only missing dependencies, validate the monorepo, check Track B, inspect artifacts, and prepare the environment for updates or testing. The always-on Copilot instructions are stored at `.github/copilot-instructions.md`, and the short safe-change checklist is stored at `.github/prompts/project-update-checklist.prompt.md`.

Important validation commands:

```bash
python scripts/validate/validate_monorepo_phase1.py
python scripts/dev_validate_official_mcp_server.py
python scripts/dev_validate_trackb_wiring.py
python scripts/dev_validate_dashboard_refactor.py
```

Dashboard command:

```bash
streamlit run scripts/demo_dashboard.py   --server.port 8501   --server.address 0.0.0.0   --server.headless true   --server.enableCORS false   --server.enableXsrfProtection false
```

The GPU pod setup instructions explicitly preserve the monorepo structure, official Project MCP server, RULER/vLLM judge path, TRL GRPO path, Agent Lightning control plane, Streamlit dashboard entry, and CPU/GPU handoff model. They also prohibit secrets, raw data leakage, raw SQL exposure by default, and expensive training runs unless explicitly requested.
