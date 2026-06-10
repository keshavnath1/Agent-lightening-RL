# Agent Lightning Strict Migration Checklist

This checklist tracks the strict Agent Lightning guardrails that must remain true as the repository evolves.

## Current strict defaults completed

- Track A make target points to official runner script.
- Dashboard command hints now point to `python -m src.training.agent_lightning_official_runner`.
- Official runner uses strict `Trainer.fit(agent=..., train_dataset=...)` only.
- Sidecar task/report API paths fail fast with explicit errors.
- Rollout runner paths fail fast when server URL is missing or reporting fails.

## Remaining custom surfaces to keep explicit

1. Duplicate rollout entrypoints still exist.
- `apps/rollout_worker/src/rollout_worker/run_track_a.py`
- `apps/rollout_worker/src/rollout_worker/langgraph_workflow.py`
- `src/agents/supervisor.py`

Status:
- Compatibility wrappers are allowed only when they delegate to the strict official runner path or enforce the same fail-fast behavior.

2. Local compatibility abstractions still exist in training/adapter layers.
- `src/training/grpo_algorithm.py`
- `src/training/trainer_loop.py`
- `src/inference/trace_adapter.py`

Status:
- Strict validation requires direct official imports and rejects ImportError fallback stubs.

3. Store/server integration is still custom.
- `src/training/lightning_store.py`
- `src/training/lightning_server_app.py`

Status:
- Keep this integration explicit. It should fail closed when required reporting, task-pull, or training-trigger behavior is unavailable.

## Operational safeguards

- Run `make validate-agent-lightning-strict` before merge.
- Fail CI if strict validator fails.
- Use only this command for Track A:

```bash
python -m src.training.agent_lightning_official_runner
```

## Acceptance criteria for strict mode

- No silent `except ...: pass` blocks around Agent Lightning reporting paths.
- No signature-probing or positional fallback invocation for `Trainer.fit`.
- No local JSONL task fallback in rollout execution path.
- One canonical Track A entrypoint documented and enforced.
- `python scripts/validate/validate_agent_lightning_strict.py` passes.
