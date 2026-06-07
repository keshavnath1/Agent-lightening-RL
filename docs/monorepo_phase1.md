# Lightweight Monorepo Phase 1

This repository now has a **Phase 1 lightweight monorepo layout**. The change is intentionally conservative: the existing `src/` implementation remains the working source of truth, while `packages/`, `services/`, and `apps/` introduce clear package boundaries through compatibility wrappers.

| Boundary | Path | Purpose |
|---|---|---|
| Contracts | `packages/contracts` | Shared dataclass schemas for trajectories, tool calls, rewards, RULER groups, and checkpoints. |
| Tools | `packages/ml_tools` | Safe PostgreSQL/tooling wrappers over `src.tools`. |
| Rewards | `packages/rewards` | Reward and RULER wrappers over `src.rewards`. |
| MCP bridge | `packages/mcp_client_bridge` | Official MCP client and LangChain adapter wrappers. |
| Lightning bridge | `packages/lightning_bridge` | Agent Lightning, GRPO, and checkpoint registry wrappers. |
| Project MCP server | `services/project_mcp_server` | Official MCP service wrapper over `src.mcp_official`. |
| Apps | `apps/*` | Dashboard, rollout worker, RULER scorer, and trainer entry boundaries. |

The goal is to make the project easier to review, test, and eventually package without breaking Track A/Track B benchmark paths. A later phase can move implementation files into these packages once tests and CI are stable.

## Validation

Run the Phase 1 validation with:

```bash
make validate-monorepo
make test-monorepo
make validate-mcp
make validate-ruler
```

The refactor is considered safe only if these checks pass and the E2E Track A/Track B benchmark still validates.


## Phase 2 hardening

Phase 2 adds package-level dependency metadata and lazy package initializers. This is intentionally a hardening pass rather than a full source migration. The existing `src/` modules remain the implementation source of truth, and the app/service/package directories remain compatibility boundaries around those modules.

| Boundary | Phase 2 behavior |
|---|---|
| `packages/ml_tools` | Exposes lightweight SQL-safety symbols lazily and avoids importing PostgreSQL/profiling tools from the package initializer. |
| `packages/rewards` | Avoids eager reward, RULER, or judge imports from the package initializer. Concrete reward modules remain importable from submodules. |
| `packages/mcp_client_bridge` | Exposes schema extraction lazily and keeps MCP/LangChain bridge imports in concrete submodules. |
| `services/project_mcp_server` | Avoids importing the FastMCP server object from package import time. Start with `project_mcp_server.server`. |
| `apps/rollout_worker` | Avoids loading LangGraph/policy runtime dependencies until the rollout runner is explicitly imported. |

The validation sequence for this phase is:

```bash
python3 -m pip install -r requirements-cpu.txt
make validate-monorepo
make test-monorepo
make validate-mcp
make validate-ruler
```

The GPU E2E benchmark scripts should only be rerun after these checks pass, because Phase 2 is expected to be import/packaging hardening with no change to policy output, reward computation, or benchmark scoring logic.

