# Agent Lightning Implementation — Session Notes (2026-05-31)

## Overview

This session completed the Agent Lightning architecture refactor, fixed three critical data-flow bugs in the Optimization Framework pipeline, evaluated parity against the official Microsoft Agent Lightning v0.3.1 repo, implemented three gap-closing fixes, and built a Streamlit demo dashboard for end-to-end showcase.

---

## 1. LangGraph Agent Refactor

### `src/agents/graph.py` — NEW
Replaced the hand-rolled sequential agent loop with a LangGraph `StateGraph`.

**`AgentGraphState`** (TypedDict):
```
task, policy_version, artifacts, tool_outputs, steps, require_live_policy, error
```

**Graph topology:**
```
START → policy_router → data_engineer → gbm_specialist
      → sandbox_execution → experiment_tracking → reviewer_critic → END
```

Key design decisions:
- First-error-wins pattern: nodes check `state['error']` and return `{}` if set
- Factory-based nodes (`_make_agent_node`, `_build_policy_router`) so sidecar closes over the graph instance
- `_build_policy_router` is a no-op when `require_live_policy=False`

### `src/agents/supervisor.py` — REWRITTEN
Orchestrator now uses LangGraph graph per task and manages the full sidecar lifecycle.

Key methods:
- `_make_sidecar(policy_version)` — creates `PolicyClient` if live policy required, wraps in `LightningClientSidecar`
- `_report_to_lightning(trajectory, sidecar)` — scores trajectory, converts to transitions, calls official `agl.emit_reward()` + `agl.emit_trace()` (best-effort), POSTs `RolloutReport` to server
- `run_task(task, policy_version)` — builds `AgentGraphState`, invokes compiled graph, reconstructs `Trajectory` from steps
- `run_tasks()` — tries Lightning Server task pull first, falls back to JSONL

CLI: `--lightning-server-url` arg added.

---

## 2. Lightning Sidecar (CPU-side trace collection)

### `src/inference/lightning_sidecar.py` — NEW

Non-intrusive CPU-side sidecar implementing Stage 2 of Agent Lightning (trace collection without modifying agent code).

**Key classes:**

`LLMSpan` — captures each LLM call: span_id, messages, response, error, latency_ms, model_name, started_at, completed_at

`RolloutReport` — full rollout data: task_id, trajectory_id, policy_version, llm_spans, transitions, final_reward, reward_metadata, error_types, **trajectory_dict** (full `Trajectory.to_dict()`)

`LightningClientSidecar`:
- `__init__` monkey-patches `policy_client.chat` at instance level — intercepts ALL LLM calls non-intrusively
- `chat()` uses `self._original_chat` to avoid recursion
- `report_rollout()` → `POST /api/rollouts/report`
- `pull_task()` → `GET /api/tasks/pull`
- Enabled only when `LIGHTNING_SERVER_URL` env var is set

---

## 3. Lightning Server (GPU-side bridge)

### `src/training/lightning_server_app.py` — NEW + UPDATED

FastAPI application implementing all three Agent Lightning stages on the GPU pod.

**Configuration:**
```
PORT=19123
VLLM_BASE_URL=http://localhost:8000
CHECKPOINT_DIR=checkpoints/qwen25-3b-agent-lora
MIN_ROLLOUTS=4
```

**Endpoints:**

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Server health, reload status |
| POST | `/api/tasks/load` | Load tasks JSONL into pool |
| GET | `/api/tasks/pull` | Pop next task (returns `{}` when empty) |
| POST | `/api/rollouts/report` | Accept `RolloutReport` from sidecar |
| GET | `/api/rollouts` | List collected rollouts |
| GET | `/api/transitions` | List written transitions |
| GET | `/api/training/status` | Training subprocess status |
| POST | `/api/training/trigger` | Launch GRPO/QLoRA/veRL training |
| POST | `/api/inference/reload` | Hot-reload vLLM LoRA adapter |

**Interactive API docs (Swagger UI):** `http://localhost:19123/docs`

---

## 4. Three Data-Flow Bug Fixes

### Bug 1 — Wrong CLI argument
`trigger_training` was passing `--input` to the training script but it only accepts `--dataset`.

**Fix:** Changed subprocess command to use `--dataset`.

### Bug 2 — Training data format mismatch
Server was writing flat transitions JSONL but `build_training_examples()` needs grouped rollouts format:
```json
{"task_id": "...", "group_size": 4, "ranked_trajectories": [{"trajectory_id": "...", "reward": 0.8, "policy_version": "v1", "steps": [...]}]}
```

**Fix:** `trigger_training` now writes `lightning_grouped_rollouts.jsonl` by grouping rollouts per task_id, sorted by reward descending.

### Bug 3 — Missing trajectory data
`RolloutReport` had no full trajectory — server couldn't reconstruct grouped rollouts for training.

**Fix:** Added `trajectory_dict` field to both `RolloutReport` dataclass and `RolloutReportRequest` Pydantic model. Supervisor passes `trajectory_dict=traj_dict`.

---

## 5. vLLM Hot-Reload (Closes the feedback loop)

### New functions in `lightning_server_app.py`:

`_reload_vllm_adapter()` — POSTs `{lora_name, lora_path}` to `{VLLM_BASE_URL}/v1/load_lora_adapter`. Returns status string, never propagates errors.

`_watch_training_and_reload(proc)` — background daemon thread. Blocks on `proc.wait()`, then fires reload if `returncode == 0`. Updates `_last_reload_at` and `_last_reload_status`.

**Thread started in `trigger_training()`:**
```python
threading.Thread(
    target=_watch_training_and_reload,
    args=(_proc_ref,),
    daemon=True,
    name='training-reload-watcher',
).start()
```

**Full feedback loop:**
```
rollout → sidecar → server → GRPO training → _watch_training_and_reload → vLLM reload → next rollout uses updated weights
```

`/health` endpoint now surfaces `last_reload_at` and `last_reload_status`.

---

## 6. Official Agent Lightning Emit API (v0.3+)

Added to `supervisor._report_to_lightning()`:
```python
import agentlightning as agl
agl.emit_reward(reward)
agl.emit_trace({task_id, trajectory_id, transitions, reward, error_types, llm_spans})
```
Both calls are best-effort — silently skipped if `agentlightning` not installed or called outside a `Trainer` context.

`agentlightning>=0.3.1` added to `requirements-cpu.txt`.

---

## 7. veRL Training Script

### `scripts/gpu/start_verl_training.sh` — NEW

Real veRL GRPO training configuration with Hydra-style overrides:

```bash
python -m verl.trainer.main_grpo \
  data.train_files=<grouped_rollouts.jsonl> \
  actor_rollout_ref.model.path=Qwen/Qwen2.5-3B-Instruct \
  actor_rollout_ref.model.lora_rank=16 \
  actor_rollout_ref.rollout.n=4 \
  trainer.total_epochs=1 \
  trainer.default_local_dir=checkpoints/qwen25-3b-agent-lora \
  ...
```

Falls back to `train_policy_qlora_grpo.py --trainer trl_grpo` if veRL not installed.

After training completes:
1. Fires `curl POST {VLLM_BASE_URL}/v1/load_lora_adapter`
2. Notifies `{LIGHTNING_SERVER_URL}/api/inference/reload` if set

`scripts/gpu/start_lightning_server.sh` updated to export `VERL_TRAIN_CMD` pointing at this script.

---

## 8. Demo Dashboard

### `scripts/demo_dashboard.py` — NEW
### `scripts/start_demo.sh` — NEW

Streamlit UI for end-to-end showcase. Three tabs:

**Tab 1 — Run Agent**
- Task description input + dataset path
- **▶ Run Task** executes the full LangGraph pipeline inline
- Each step expands: agent name, reasoning summary, tool calls with ✅/❌
- "Load tasks into server pool" panel for pre-loading JSONL
- Switchable `LIGHTNING_SERVER_URL` — swap between local CPU and RunPod GPU

**Tab 2 — Trajectories & Rewards**
- Colour-coded reward table (red→green gradient)
- Learning curve: scatter + 5-rollout rolling average line (shows improvement over time)
- Policy comparison bar chart (baseline vs fine-tuned side by side)
- Row click → full `trajectory_dict` JSON + `reward_metadata`
- Raw transitions viewer (state_t, action_t, reward_t, state_t+1)

**Tab 3 — Fine-tune & Reload**
- Live metrics: rollouts collected, transitions written, training active, ready-to-train
- Trigger Training with trainer/model/threshold selectors
- Auto-polls training status while active
- ♻ Reload vLLM Adapter button with custom adapter path input
- MLflow recent runs table (reads local `mlruns/`)

**How to run:**
```bash
pip install streamlit>=1.35 altair>=5.3

# Dashboard + MLflow only
bash scripts/start_demo.sh

# With Lightning Server
START_LIGHTNING_SERVER=1 bash scripts/start_demo.sh

# Point at RunPod GPU pod
LIGHTNING_SERVER_URL=https://<pod-id>-19123.proxy.runpod.net bash scripts/start_demo.sh
```

URLs:
- Dashboard: `http://localhost:8501`
- API docs (Swagger): `http://localhost:19123/docs`
- MLflow: `http://localhost:5000`

---

## 9. Requirements Updates

### `requirements-cpu.txt`
```
langgraph>=0.2
agentlightning>=0.3.1
streamlit>=1.35
altair>=5.3
```

### `requirements-gpu.txt`
```
fastapi>=0.111
uvicorn[standard]>=0.30
pydantic>=2.7
```

---

## 10. Architecture Parity vs Microsoft Agent Lightning v0.3.1

| Component | Before | After |
|-----------|--------|-------|
| Agent graph | Hand-rolled sequential loop | LangGraph StateGraph ✅ |
| Trace collection | None | `LightningClientSidecar` monkey-patch ✅ |
| Lightning Server | None | Full FastAPI with all 3 stages ✅ |
| Grouped rollouts format | Wrong (flat) | Correct (task_id + ranked_trajectories) ✅ |
| Training CLI arg | `--input` (broken) | `--dataset` (fixed) ✅ |
| trajectory_dict passthrough | Missing | Added to RolloutReport ✅ |
| vLLM hot-reload | Never triggered | `_watch_training_and_reload` thread ✅ |
| Official emit API | None | `agl.emit_reward` + `agl.emit_trace` ✅ |
| veRL integration | Stub only | Real Hydra config + fallback ✅ |
| Demo UI | None | Streamlit 3-tab dashboard ✅ |

**Estimated parity: ~80%**

Remaining gaps:
- `LightningStore`/`Algorithm` interface depth (20%)
- Dashboard UI matching official pattern

---

## 11. Demo Narrative (Presentation Flow)

```
1. Tab 1 → type a task → Run Task
   Audience sees LangGraph nodes executing step by step with reasoning + tool calls

2. Run 4+ tasks → Tab 2
   Table fills with trajectory rows + reward scores
   Learning curve shows reward improving over rollouts

3. Tab 2 → Policy comparison bar chart
   baseline vs v2_finetuned reward bars side by side

4. Tab 3 → Trigger Training
   Progress polling → "Training complete! vLLM reload triggered"

5. Sidebar → change LIGHTNING_SERVER_URL to RunPod URL → re-run same task
   Same workflow, now running on GPU pod with updated LoRA weights
```
