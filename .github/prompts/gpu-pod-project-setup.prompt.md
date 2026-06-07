# GPU Pod Project Setup Prompt

You are working inside a fresh GPU pod that has mounted the same network volume for the `self-improving-ml-agent` project. Your job is to make the project ready for updates, validation, dashboard use, and Track B GPU testing.

Do not assume the environment is already ready. Inspect first, then install only what is missing.

## Project Context

This is a self-improving ML-agent monorepo with Agent-Lightning-style trajectory collection, an official Project MCP Server, PostgreSQL safe metadata tools, tool-call logging, RULER/vLLM judge scoring, TRL GRPO/QLoRA policy training, optional ART/RULER integration, a Streamlit dashboard, and CPU/GPU handoff through a shared network volume.

Core mental model:

```text
Track A CPU:
  rollouts -> tool calls -> trajectories -> rewards -> grouped_rollouts.jsonl

Track B GPU:
  vLLM judge/policy -> RULER scores -> TRL GRPO -> LoRA checkpoint
```

## Step 1: Identify Project Root

Run:

```bash
pwd
ls -la
find . -maxdepth 2 -type f \( -name "pyproject.toml" -o -name "requirements-cpu.txt" -o -name "requirements-gpu.txt" \)
```

Confirm this repository has:

```text
apps/
services/
packages/
scripts/
configs/
docs/
tests/
```

If the current directory is wrong, locate the repository on the mounted network volume.

## Step 2: Inspect Environment

Run:

```bash
python --version
which python
pip --version
nvidia-smi || true
```

Check installed key packages:

```bash
python - <<PY
mods = [
    "torch",
    "transformers",
    "trl",
    "peft",
    "vllm",
    "streamlit",
    "mcp",
    "sqlalchemy",
    "langgraph",
]
for m in mods:
    try:
        mod = __import__(m)
        print(f"OK {m}: {getattr(mod, __version__, unknown)}")
    except Exception as e:
        print(f"MISSING {m}: {e}")
PY
```

## Step 3: Install Dependencies

If this is a CPU-only validation task, run:

```bash
python -m pip install --upgrade pip
pip install -r requirements-cpu.txt
```

If this is GPU/Track B work, run:

```bash
python -m pip install --upgrade pip
pip install -r requirements-cpu.txt
pip install -r requirements-gpu.txt
```

If `uv.lock` and workspace are configured and `uv` is available, run:

```bash
uv sync --all-packages
```

Do not install huge GPU dependencies repeatedly if they are already available and imports work.

## Step 4: Validate Monorepo

Run:

```bash
python scripts/validate/validate_monorepo_phase1.py
```

Then run:

```bash
python scripts/dev_validate_official_mcp_server.py
python scripts/dev_validate_trackb_wiring.py
python scripts/dev_validate_dashboard_refactor.py
```

If a validation fails from a missing dependency, install only that missing dependency and rerun. If validation fails from code, inspect the failure and make the smallest fix.

## Step 5: Validate Shell Scripts

Run:

```bash
find scripts -name "*.sh" -print0 | xargs -0 -I{} bash -n {}
```

Do not execute training scripts yet. Only syntax-check them.

## Step 6: Check Project Artifacts

Inspect:

```bash
ls -lh data/grpo || true
ls -lh trajectories || true
ls -lh checkpoints || true
ls -lh reports || true
```

Expected important files may include:

```text
data/grpo/grouped_rollouts.jsonl
data/grpo/ruler_scored_groups.jsonl
reports/postgres_mcp_learning_report.md
reports/ruler_scoring_summary.md
reports/mcp_official_tool_calls.jsonl
```

If files are missing, do not fabricate them. Report what is missing and which script likely generates them.

## Step 7: Validate Official Project MCP Server

Run:

```bash
python scripts/dev_validate_official_mcp_server.py
```

Optional server start:

```bash
bash scripts/cpu/start_official_mcp_server.sh
```

Remember that Project MCP is for ML-agent tools, RunPod MCP is infra-only, and raw SQL must not be exposed unless `ALLOW_RAW_SQL_TOOL=1`.

## Step 8: Validate RULER/vLLM/TRL Setup

Check the GPU:

```bash
nvidia-smi
```

Check Track B wiring:

```bash
python scripts/dev_validate_trackb_wiring.py
```

If RULER/vLLM scripts exist, syntax-check them:

```bash
bash -n scripts/gpu/start_vllm_ruler_judge.sh || true
bash -n scripts/gpu/run_04_ruler_then_trl_grpo.sh || true
```

Expected defaults:

```text
Judge endpoint: http://localhost:8001/v1
Judge model: local-ruler-judge
Policy endpoint: http://localhost:8000/v1
Reward mode: ruler_relative
Trainer: trl_grpo
```

Do not start a long training run unless explicitly requested.

## Step 9: Start Dashboard Only If Requested

Dashboard command:

```bash
streamlit run scripts/demo_dashboard.py \
  --server.port 8501 \
  --server.address 0.0.0.0 \
  --server.headless true \
  --server.enableCORS false \
  --server.enableXsrfProtection false
```

If the browser cannot open the dashboard, check:

```bash
ss -ltnp | grep 8501 || true
curl -I http://localhost:8501 || true
hostname -I
```

In RunPod, the issue is usually port exposure, not Streamlit itself.

## Step 10: Make Project Ready For Updates

Before making code changes, run validations, identify the failing area, make the smallest targeted change, rerun relevant validation, and summarize the result.

When finished, report:

```text
Environment:
- Python version
- GPU detected
- Key packages installed/missing

Validation:
- monorepo validation
- official MCP validation
- Track B validation
- dashboard validation
- shell script syntax

Artifacts:
- grouped rollouts present/missing
- RULER scored groups present/missing
- checkpoints present/missing

Changes made:
- files changed
- reason
- validation after change

Next step:
- recommended next command
```

## Strict Rules

Do not commit secrets, hardcode a real `DATABASE_URL`, expose raw dataset rows, enable raw SQL by default, start expensive training without request, delete compatibility wrappers during Phase 1, replace TRL GRPO, replace Agent Lightning, or confuse RunPod MCP with Project MCP.

Always preserve the monorepo structure, official Project MCP server, safe PostgreSQL tools, RULER/vLLM judge path, TRL GRPO path, Streamlit dashboard entry, and validation scripts.
