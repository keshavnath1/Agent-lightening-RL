# Phase 2 Monorepo Hardening Runbook

Phase 2 strengthens the lightweight monorepo structure introduced in Phase 1. It does not move the source-of-truth implementation out of `src/`; instead, it makes package imports safer, records package-level dependencies, and clarifies validation commands for Git review.

| Step | Command | Expected result |
|---|---|---|
| Install CPU dependencies | `python3 -m pip install -r requirements-cpu.txt` | CPU/reporting dependencies are available. |
| Validate structure | `make validate-monorepo` | Package boundaries, scripts, and imports are valid. |
| Run targeted tests | `make test-monorepo` | Compatibility-wrapper tests pass. |
| Validate MCP | `make validate-mcp` | Official Project MCP server checks pass. |
| Validate RULER/Track B wiring | `make validate-ruler` | RULER/TRAINER wiring checks pass, with optional dependency warnings allowed. |

## Import policy

Package root imports should stay lightweight. Runtime dependencies should be pulled only by concrete submodules, such as `project_mcp_server.server`, `rollout_worker.run_track_a`, `ruler_scorer.vllm_judge`, or `trainer_app.train_policy_qlora_grpo`.

## Benchmark policy

Because Phase 2 does not intentionally change policy generation or reward scoring, full Track A and Track B benchmarks are regression checks rather than required implementation inputs. A bounded smoke benchmark is sufficient unless validation shows that a runtime path changed.
