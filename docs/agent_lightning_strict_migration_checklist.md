# Agent Lightning Strict Migration Checklist

This checklist tracks remaining work to enforce official Agent Lightning usage end-to-end.

## Current strict defaults completed

- Track A make target points to official runner script.
- Dashboard command hints now point to `python -m src.training.agent_lightning_official_runner`.
- Official runner uses strict `Trainer.fit(agent=..., train_dataset=...)` only.
- Sidecar task/report API paths fail fast with explicit errors.
- Rollout runner paths fail fast when server URL is missing or reporting fails.

## Remaining non-official/custom surfaces to retire

1. Duplicate rollout entrypoints still exist.
- `apps/rollout_worker/src/rollout_worker/run_track_a.py`
- `apps/rollout_worker/src/rollout_worker/langgraph_workflow.py`
- `src/agents/supervisor.py`

Action:
- Keep only one official execution path and convert other modules to wrappers that raise deprecation errors pointing to the official runner.

2. Local compatibility abstractions still exist in training/adapter layers.
- `src/training/grpo_algorithm.py`
- `src/training/trainer_loop.py`
- `src/inference/trace_adapter.py`

Action:
- Remove fallback base classes and require official Agent Lightning imports directly.
- Delete local fallback stubs for Algorithm, Trainer, TraceAdapter, and Triplet.

3. Store/server integration is still custom.
- `src/training/lightning_store.py`
- `src/training/lightning_server_app.py`

Action:
- Replace custom report ingestion behavior with official runtime-managed storage/reporting path where available.
- Keep only compatibility required by official package contracts.

## Operational safeguards

- Run `make validate-agent-lightning-strict` before merge.
- Fail CI if strict validator fails.
- Use only this command for Track A:

```bash
python -m src.training.agent_lightning_official_runner
```

## Acceptance criteria for full strict mode

- No silent `except ...: pass` blocks around Agent Lightning reporting paths.
- No signature-probing or positional fallback invocation for `Trainer.fit`.
- No local JSONL task fallback in rollout execution path.
- One canonical Track A entrypoint documented and enforced.
