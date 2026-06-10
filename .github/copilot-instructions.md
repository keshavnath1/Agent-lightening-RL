# Copilot Instructions — Self-Improving ML Agent

## Project Identity

This repository is a **self-improving ML-agent monorepo**. The project combines Agent-Lightning-style trajectory collection, an official Project MCP Server for safe ML tools, PostgreSQL-backed task/data/reward metadata, tool-call logging and tool-learning rewards, RULER/vLLM judge scoring, TRL GRPO with PEFT LoRA adapters, optional ART/RULER integration, a Streamlit dashboard for demo and validation, and CPU/GPU handoff through a shared network volume.

Do not treat this repository as a normal single-script ML project. It is an **agentic ML workflow platform** with multiple cooperating packages, services, applications, validation scripts, and runtime handoff paths.

## Monorepo Structure

The expected high-level layout is:

```text
apps/
  dashboard/
  rollout_worker/
  ruler_scorer/
  trainer/

services/
  project_mcp_server/

packages/
  contracts/
  ml_tools/
  rewards/
  lightning_bridge/
  mcp_client_bridge/

scripts/
  cpu/
  gpu/
  validate/

configs/
data/
trajectories/
checkpoints/
reports/
docs/
tests/
```

Older `src/` modules may still exist as the compatibility source of truth during the Phase 1 and Phase 2 migration period. Do not delete them unless the migration is explicitly completed and tests prove compatibility.

## Core Architecture

Use this mental model when making changes:

```text
Track A CPU
  Agent rollout
    -> Official Project MCP tools
    -> PostgreSQL metadata/tools
    -> ToolCallRecord logs
    -> Trajectories
    -> Reward scoring
    -> grouped_rollouts.jsonl

Track B GPU
  vLLM policy server
  vLLM RULER judge server
  RULER scores grouped trajectories
    -> ruler_scored_groups.jsonl
  TRL GRPO trains LoRA adapter
    -> checkpoints/
  Agent Lightning bridge registers/reloads checkpoint
```

Component roles are:

| Component | Role |
|---|---|
| Agent Lightning | Trajectory/control plane |
| Official Project MCP Server | Safe tool interface |
| RULER | Judge/reward scoring plane |
| TRL GRPO | Optimizer and fine-tuning plane |
| vLLM | Inference plane |
| PostgreSQL | Structured state and metadata layer |
| Streamlit | Demo and evidence console |

## Important Rule: Do Not Confuse MCP Types

RunPod MCP is **infra-only** and is not core to the learning loop. The Project MCP Server is core. It exposes safe ML-agent tools such as `postgres_get_task_metadata`, `postgres_get_dataset_schema`, `postgres_get_dataset_summary`, `postgres_get_artifact_manifest`, `postgres_get_rollout_status`, and `postgres_get_reward_history`.

Do not route ML-agent task, data, or reward access through RunPod MCP.

## Safety Rules

Never expose raw dataset rows to the LLM, RULER judge, or dashboard by default. Raw SQL must remain developer-only and must be gated explicitly:

```bash
ALLOW_RAW_SQL_TOOL=1
```

If raw SQL is disabled, tools must instruct the user to use safe PostgreSQL metadata tools instead. Do not commit API keys, real `DATABASE_URL` values, Hugging Face tokens, OpenAI/OpenRouter keys, RunPod API keys, generated checkpoints, large datasets, or raw trajectory dumps unless intentionally sampled and approved.

## Dependency Setup on New Pods

On a new CPU or GPU pod, first inspect the project root and runtime environment:

```bash
pwd
ls -la
python --version
```

For CPU validation work, prefer:

```bash
python -m pip install --upgrade pip
pip install -r requirements-cpu.txt
```

For GPU/Track B work, install GPU requirements only when needed:

```bash
pip install -r requirements-gpu.txt
```

If `uv.lock` and workspace files exist and `uv` is available, use:

```bash
uv sync --all-packages
```

Do not blindly reinstall heavy GPU packages if they are already installed and working.

For a clone-only fresh RunPod E2E setup, follow:

```text
.github/prompts/fresh-runpod-e2e.prompt.md
```

The hard required runtime secret is `DATABASE_URL`. A Hugging Face token is optional
but recommended for model downloads:

```bash
export HF_TOKEN=hf_...
# or
export HUGGINGFACE_HUB_TOKEN=hf_...
```

Never commit or print those values. The fast-path command is:

```bash
bash scripts/runpod_bootstrap_e2e.sh
```

## Validation Commands

After setup, run lightweight validations first:

```bash
python scripts/validate/validate_monorepo_phase1.py
python scripts/dev_validate_official_mcp_server.py
python scripts/dev_validate_trackb_wiring.py
python scripts/dev_validate_dashboard_refactor.py
```

For tests, prefer:

```bash
make test-monorepo
```

or, when appropriate:

```bash
pytest -q
```

If dependencies are missing, install the smallest required package set and retry.

## Dashboard

Primary dashboard command:

```bash
streamlit run scripts/demo_dashboard.py \
  --server.port 8501 \
  --server.address 0.0.0.0 \
  --server.headless true \
  --server.enableCORS false \
  --server.enableXsrfProtection false
```

The dashboard should explain the monorepo structure, official MCP server, Tool Learning flywheel, Track A rollouts, RULER/vLLM judge, TRL GRPO training, Agent Lightning control plane, and validation evidence. Do not add arbitrary shell execution in Streamlit. Only use allowlisted service commands.

## Official Project MCP Server

Start command:

```bash
bash scripts/cpu/start_official_mcp_server.sh
```

MCP tool-call log:

```text
reports/mcp_official_tool_calls.jsonl
```

When implementing MCP changes, preserve safe PostgreSQL tools, raw SQL gating, tool-call logging, resources, prompts, and validation scripts.

## RULER and TRL GRPO

RULER judge via local vLLM usually runs on port `8001`. The policy vLLM usually runs on port `8000`. RULER scoring output should be written to:

```text
data/grpo/ruler_scored_groups.jsonl
```

TRL GRPO should consume RULER scores using:

```bash
REWARD_MODE=ruler_relative
```

The judge model should remain fixed, the policy model is trainable, reward scores must not be inserted into prompt text, and reward scores should remain dataset metadata columns.

## Common GPU Pod Workflow

When a new GPU pod is created using the same network volume, confirm the project root and mounted volume, install missing dependencies, validate the monorepo, validate Track B wiring, check the GPU, start vLLM policy or judge only as needed, run dry-runs before real training, and write validation evidence to `reports/`.

Suggested commands:

```bash
nvidia-smi
python scripts/validate/validate_monorepo_phase1.py
python scripts/dev_validate_trackb_wiring.py
python scripts/dev_validate_official_mcp_server.py
```

## Do Not Break Existing Behavior

When making changes, keep old scripts working unless explicitly migrated. Keep `scripts/demo_dashboard.py` working. Keep Track B validation, official MCP validation, and monorepo validation passing. Preserve compatibility wrappers during Phase 1 and Phase 2. Prefer small targeted changes over broad rewrites.

## What To Do When Asked To Update The Project

Before editing, inspect relevant files, understand current structure, make the smallest safe change, run validation, and summarize exactly what changed and what passed.

Always report:

```text
Changed files
Validation commands run
Pass/fail status
Known limitations
Next recommended step
```
