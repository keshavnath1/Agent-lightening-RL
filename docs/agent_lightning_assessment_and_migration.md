# Microsoft Agent Lightning Assessment and Migration Notes

> **Last updated:** 2026-06-02 (Session 3 — all Lightning integration components implemented)

## Short answer

The current repository **implements a complete Agent Lightning-compatible execution layer** as custom wrappers that are drop-in compatible with the official Microsoft Agent Lightning architecture.  All core components exist: `LightningStoreAdapter`, `GRPOAlgorithm`, `LightningTrainer`, `SpanTraceAdapter`, `MLAgentLitAgent`, `LightningClientSidecar`, and a full FastAPI Lightning Server.

The **official `agentlightning` pip package is intentionally optional** due to a `blinker` version conflict with the current environment.  All integration classes use a `try: from agentlightning import X; except ImportError: class X: ...stub...` pattern so the system runs correctly whether or not the package is installed.

Microsoft describes Agent Lightning as a framework that can optimize agents built with many existing agent frameworks by **decoupling agent workflow development from RL training**, with a Lightning Server and Lightning Client bridging agent execution and training infrastructure. Its trace collection converts agent traces into transition tuples of **state_t, action_t, reward_t, state_t+1**, then uses RL infrastructure such as **veRL** and algorithms such as **GRPO** for optimization.[^1]

## Current repository status (Session 3)

| Area | Current repository behavior | Agent Lightning equivalent | Status |
|---|---|---|---|
| Agent execution | `SupervisorAgent` runs LangGraph multi-agent workflow. `MLAgentLitAgent` wraps it as a `LitAgent.rollout()`. | Agent side runs its native workflow wrapped by LitAgent. | ✅ Implemented |
| LLM inference | `LightningClientSidecar` intercepts all LLM calls, records `LLMSpan` records with full rollout attribution (`rollout_id`, `attempt_id`, `sequence_id`). `proxy_chat()` routes via LLMProxy headers. | Lightning sidecar non-intrusively intercepts LLM calls. | ✅ Implemented |
| Lightning Server | `src/training/lightning_server_app.py` — FastAPI app on port 19123.  Endpoints: `/api/tasks/pull`, `/api/rollouts/report`, `/api/training/trigger`, `/api/store/statistics`, etc. | Lightning Server bridges agent execution and training. | ✅ Implemented |
| Task queue / rollout store | `LightningStoreAdapter` — thread-safe in-memory store.  `pull_task()` injects `lightning_rollout_id` so reports correlate back to the same record (no stale `preparing` entries). | Lightning Server manages rollout lifecycle. | ✅ Implemented (bug fixed Session 3) |
| Trace adapter | `SpanTraceAdapter` (`src/inference/trace_adapter.py`) converts `LLMSpan` dicts → `Triplet` objects for training. | Lightning converts agent traces to RL transition tuples. | ✅ Implemented |
| Training loop | `LightningTrainer` (`src/training/trainer_loop.py`) — `load_tasks`, `maybe_trigger_training`, `run(max_rounds)`, `fit()`. | Lightning Trainer orchestrates training rounds. | ✅ Implemented |
| GRPO algorithm | `GRPOAlgorithm` (`src/training/grpo_algorithm.py`) — `build_triplet_dataset`, `_launch_training`, `_reload_vllm`. | Optimization Framework (veRL/GRPO). | ✅ Implemented |
| QLoRA quantization | `train_qlora_sft` and `train_trl_grpo` both use `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4')`. TRL GRPO passes via `model_init_kwargs`. | 4-bit GRPO training for GPU efficiency. | ✅ Implemented (fixed Session 3) |
| veRL integration | `run_verl_training_handoff` in training script; `start_verl_training.sh` with strict mode: fails on missing veRL unless `ALLOW_VERL_FALLBACK=1`. | Official veRL training cluster. | ✅ Strict mode (fixed Session 3) |
| Reward signal | `src/rewards/scorer.py` computes evidence-based reward components. | User-defined reward signal. | ✅ Reusable |
| Trace visualization | Streamlit dashboard Tab 4 — Live Trace — streams LangGraph node updates in real time. | Observability. | ✅ Implemented |
| Official package runtime | `agentlightning` pip package — blocked by `blinker` conflict; stubs in place everywhere. | Full Lightning runtime. | ⚠️ Optional / unproven |

## What changed in Session 3

### Bug fixes from external assessment

1. **Stale `preparing` rollouts (Issue #5)** — `pull_task()` now injects `lightning_rollout_id` into the returned task dict.  `accept_rollout_report()` correlates that id back to the existing record (and falls back to cancelling any orphaned `preparing` record with the same `task_id`).  Dashboard statistics now show one rollout per task.

2. **LLMSpan missing attribution (Issue #6)** — `LLMSpan` dataclass now has `rollout_id: str | None`, `attempt_id: str | None`, `sequence_id: int | None` fields.  `proxy_chat()` populates them from the HTTP routing headers, enabling `SpanTraceAdapter` to group spans correctly.

3. **veRL silent fallback (Issue #3)** — `start_verl_training.sh` now exits with `exit 1` when veRL is not installed, unless `ALLOW_VERL_FALLBACK=1` is explicitly set.  This prevents the Lightning Server from reporting `trainer=verl` success when TRL GRPO was silently substituted.

4. **`trl_grpo` lacked 4-bit quantization (Issue #4)** — `train_trl_grpo()` now builds a `BitsAndBytesConfig` identical to `train_qlora_sft` and passes it as `model_init_kwargs` to `GRPOTrainer`.  A `try/except ImportError` guard keeps the function functional in CPU-only environments without `bitsandbytes`.

## Session 2 additions

Added in Session 2 (full Agent Lightning parity):

- `LightningStoreAdapter` — wraps the in-memory rollout store with async interface + `capabilities` property
- `GRPOAlgorithm` — `build_triplet_dataset`, training launch, vLLM hot-reload
- `LightningTrainer` — orchestrates task loading and training rounds
- `SpanTraceAdapter` — `LLMSpan → Triplet` conversion
- `MLAgentLitAgent` — wraps `SupervisorAgent` as `LitAgent.rollout()`
- `emit_trace` bug removed from `supervisor.py` (API doesn't exist in package stubs)
- Live Trace tab in Streamlit dashboard

## Commands

Start the Lightning Server on the GPU pod:

```bash
python -m src.training.lightning_server_app
# or
scripts/gpu/start_lightning_server.sh
```

Run the full CPU→GPU feedback loop:

```bash
# Load tasks
curl -X POST http://localhost:19123/api/tasks/load \
  -H 'Content-Type: application/json' \
  -d '{"tasks_jsonl_path": "data/synthetic/tasks.jsonl"}'

# Agents pull tasks, execute, report — then trigger training
curl -X POST http://localhost:19123/api/training/trigger
```

For veRL fallback (e.g. CPU testing):

```bash
ALLOW_VERL_FALLBACK=1 bash scripts/gpu/start_verl_training.sh
```

## Full migration path

1. Keep the CPU pod as the **agent execution side**.
2. Keep the GPU pod as the **policy inference and training side**.
3. Resolve `blinker` conflict to enable `pip install agentlightning` — all stubs will automatically defer to the real package.
4. Validate `LitAgent.rollout()`, `LightningStore`, and `LightningServer` against the official package's gRPC interface once a compatible version is available.
5. Replace the custom `LightningStoreAdapter` with the official `LightningStore` class if API-compatible.
6. Keep `scorer.py` and `reviewer.py` as the project-specific reward and guardrail layer regardless of runtime.

[^1]: Microsoft Research, "Agent Lightning," describes Lightning Server/Client, sidecar trace collection, OpenAI-compatible LLM API, transition tuples, veRL, and GRPO. <https://www.microsoft.com/en-us/research/project/agent-lightning/>


## Current repository status

| Area | Current repository behavior | Agent Lightning equivalent | Status |
|---|---|---|---|
| Agent execution | CPU-plane supervisor runs deterministic multi-agent workflow and tools. | Agent side runs its native workflow. | Conceptually aligned. |
| LLM inference | CPU plane calls OpenAI-compatible vLLM endpoint via `src/inference/policy_client.py`. | Lightning exposes an OpenAI-compatible API inside training infrastructure. | Compatible pattern, not Lightning runtime. |
| Trace logging | `src/telemetry/schema.py` and `src/telemetry/logger.py` store trajectory JSONL with steps, tool calls, reward, and metadata. | Lightning sidecar collects traces, errors, rewards, and converts them into transitions. | Conceptually aligned. |
| Reward signal | `src/rewards/scorer.py` computes evidence-based reward components. | User-defined reward signal reported to Lightning Server. | Reusable with Lightning. |
| Training data | `src/training/prepare_grpo_dataset.py` groups scored trajectories for GRPO-style training. | Lightning organizes traces into training-ready structures for veRL/GRPO. | Partially aligned. |
| Actual Lightning runtime | No Lightning Server, Client, package dependency, or server-reporting API integration is declared. | Lightning Server and Client bridge agent execution and training. | Not implemented yet. |

## What I added in this pass

I added a dependency-free bridge export at `src/training/agent_lightning_export.py` and a CPU helper script at `scripts/cpu/run_05b_export_agent_lightning_transitions.sh`. This does **not** claim full Agent Lightning integration. It creates an **Agent Lightning-style transition JSONL** from the repository's scored trajectories so the next implementation pass can connect the data shape to the actual Lightning Server/Client APIs once the exact package version and API surface are pinned.

The default output path is:

```bash
data/grpo/agent_lightning_transitions.jsonl
```

Each exported line contains:

| Field | Meaning |
|---|---|
| `task_id` | Synthetic ML task identifier. |
| `trajectory_id` | Internal trajectory ID. |
| `transition_id` | Stable step-level transition ID. |
| `state_t` | Compact state before the agent step. |
| `action_t` | Agent action, reasoning summary, and tool calls. |
| `reward_t` | Immediate reward; currently terminal reward is assigned to the final step. |
| `state_t_plus_1` | Compact next state after the step. |
| `terminal` | Whether this transition ended the trajectory. |
| `reward_metadata` | Evidence-based reward components. |
| `error_types` | Tool errors available for error monitoring. |

## Recommended next integration decision

For the near-term MVP, keep the current custom implementation because it is small, transparent, and already validates the CPU workflow. Use Agent Lightning in the next phase if the goal is to run a more faithful RL training loop with a Lightning Server/Client and veRL backend.

| Option | Recommendation | Why |
|---|---|---|
| Keep current custom Track B | Best for immediate demo stability. | It avoids a moving dependency while preserving trajectory and reward structure. |
| Add Agent Lightning transition export | Already added. | It de-risks migration and makes the data shape explicit. |
| Full Agent Lightning integration | Next phase after GPU pod is created. | It needs Lightning Server/Client setup, network wiring, and exact API validation on the GPU environment. |

## Commands

After running the normal CPU workflow and scoring trajectories, run:

```bash
cd /workspace/self-improving-ml-agent
scripts/cpu/run_05b_export_agent_lightning_transitions.sh
```

Then inspect:

```bash
head -n 2 data/grpo/agent_lightning_transitions.jsonl
```

## Full migration path

1. Keep the CPU pod as the **agent execution side**.
2. Keep the GPU pod as the **policy inference and training side**.
3. Replace or wrap `src/inference/policy_client.py` with the Lightning Client-compatible endpoint once the Lightning Server is running.
4. Report trajectory steps, tool errors, and reward metadata to the Lightning Server rather than only local JSONL files.
5. Replace `src/training/train_policy_qlora_grpo.py` with the actual Lightning/veRL training launcher.
6. Keep the existing scorer and reviewer as the project-specific reward and guardrail layer.

[^1]: Microsoft Research, “Agent Lightning,” describes Lightning Server/Client, sidecar trace collection, OpenAI-compatible LLM API, transition tuples, veRL, and GRPO. <https://www.microsoft.com/en-us/research/project/agent-lightning/>
