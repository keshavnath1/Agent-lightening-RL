
## CPU architecture diagnostic snapshot — 2026-05-31 01:45 UTC

The CPU RunPod repository is located at `/workspace/self-improving-ml-agent`. The source dependencies required for the CPU workflow are importable, including FastAPI, uvicorn, SQLAlchemy, psycopg2, MLflow, DVC, pandas, pyarrow, scikit-learn, XGBoost, LightGBM, ydata-profiling, and requests. No PostgreSQL, MLflow, MCP, Docker, vLLM, or Python service processes were running at diagnostic time except the diagnostic command itself. No listeners were present on ports 5432, 5000, 8090, or 8000. Docker is not installed or unavailable on the CPU pod; `pg_isready` was unavailable or failed, and the repository database connector failed against `postgresql+psycopg2://agent:change_me@localhost:5432/agentic_ml` with connection refused. MCP health on `127.0.0.1:8090/health` and MLflow health on `127.0.0.1:5000/health` were not reachable.

Artifacts exist from the smoke path: `data/synthetic/tasks.jsonl` has 6 lines; `data/grpo/grouped_rollouts.jsonl` has 4 lines; `data/grpo/agent_lightning_transitions.jsonl` has 20 lines; `reports/baseline_vs_rl_tuned.md` exists; `reports/feedback_validation_summary.txt` exists; baseline/scored trajectory directories each contain 12 files; `reports/mlflow_runs` has 8 JSON fallback metadata files; `mlruns` has 162 files; `checkpoints` contains 1 file; and `data/synthetic/datasets` contains 20 files. `reports/runpod_cpu_validation_summary.txt` is missing. The CPU-to-GPU policy endpoint test passed using `https://86k136u2po46i5-8000.proxy.runpod.net/v1/chat/completions`, model `qwen2.5-coder-32b-instruct`, returning `ARCH_OK`.

## CPU support-service validation — 2026-05-31 01:47 UTC

MCP and MLflow were not initially running, so they were started from the repository launch scripts. MCP now listens on `0.0.0.0:8090` via `uvicorn src.mcp_server.server:app`, and `GET http://127.0.0.1:8090/health` returned `{"status":"ok"}`. MLflow now listens on `0.0.0.0:5000` using a file-backed store at `file:/workspace/self-improving-ml-agent/mlruns`, and `GET http://127.0.0.1:5000/health` returned `OK`. Logs are under `/workspace/logs/mcp_server.log` and `/workspace/logs/mlflow_server.log`, with PID files under `/workspace/logs/`.

PostgreSQL remains missing. Docker is not available on the CPU pod, `pg_isready` is unavailable or not ready, and the SQLAlchemy connector still fails against the default URL `postgresql+psycopg2://agent:change_me@localhost:5432/agentic_ml` with connection refused. This confirms that the code path and client dependency exist, but the CPU database service itself has not yet been provisioned.

## MCP endpoint validation — 2026-05-31 01:48 UTC

The MCP-style server is reachable on the CPU pod at `http://127.0.0.1:8090`. A Python-based smoke test confirmed `/health` returned HTTP 200 with `{"status":"ok"}`. The profile endpoint `/tools/profile/parquet` also returned HTTP 200 against `/workspace/self-improving-ml-agent/data/synthetic/datasets/task_0000.parquet`, producing a summary with `row_count=500`, `column_count=8`, and target candidate `target`, and writing `reports/mcp_profile_validation/profile_summary.json`.

The SQL endpoint `/tools/sql/query` returned HTTP 500 for `select 1 as ok`, which is expected while PostgreSQL is not running. This confirms the MCP server itself is functional, but database-backed MCP capability remains blocked by the missing PostgreSQL runtime.

## CPU-to-GPU inference validation — 2026-05-31 01:49 UTC

The CPU pod successfully reached the GPU vLLM endpoint via `https://86k136u2po46i5-8000.proxy.runpod.net/v1`. The `/models` endpoint returned HTTP 200 and listed model id `qwen2.5-coder-32b-instruct` with root `Qwen/Qwen2.5-Coder-32B-Instruct` and `max_model_len=4096`. A chat completion request asking for exactly `CPU_GPU_OK` returned HTTP 200 with assistant content `CPU_GPU_OK`. This validates CPU-to-GPU policy inference connectivity through the OpenAI-compatible API.

## CPU runtime revalidation after support-service start — 2026-05-31 01:54 UTC

A follow-up CPU RunPod validation confirmed that MCP and MLflow remained reachable after startup. MCP health returned HTTP 200 with `{"status":"ok"}`, and MLflow health returned HTTP 200 `OK`. PostgreSQL remained unavailable because `pg_isready` is not installed on the CPU pod and no local PostgreSQL listener was confirmed.

The same revalidation exposed a normalization issue in `/workspace/gpu_qwen_endpoint.env`: `GPU_POLICY_BASE_URL` resolved to `https://86k136u2po46i5-8000.proxy.runpod.net` without the required OpenAI-compatible `/v1` prefix, so `GET /models` returned HTTP 404. The previously validated endpoint is `https://86k136u2po46i5-8000.proxy.runpod.net/v1`. The CPU pod environment file should therefore store the base URL with `/v1`.

## CPU GPU endpoint environment correction — 2026-05-31 01:54 UTC

`/workspace/gpu_qwen_endpoint.env` on the CPU pod was corrected so `GPU_POLICY_BASE_URL` now includes the required `/v1` prefix and `BASELINE_POLICY_URL` points to `/v1/chat/completions`. After sourcing the corrected environment file, `GET ${GPU_POLICY_BASE_URL}/models` returned HTTP 200 and listed model id `qwen2.5-coder-32b-instruct`, confirming that the CPU pod configuration again points at the active GPU vLLM OpenAI-compatible endpoint.
