from __future__ import annotations

from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    status = "[PASS]" if ok else "[FAIL]"
    print(f"{status} {name}" + (f" - {detail}" if detail else ""))
    if not ok:
        failures.append(name)


def read(rel_path: str) -> str:
    return (ROOT / rel_path).read_text(encoding="utf-8")


# 1) Official runner should use strict fit signature only.
runner_path = "src/training/agent_lightning_official_runner.py"
runner = read(runner_path)
check(f"exists {runner_path}", (ROOT / runner_path).exists())
check(
    "official runner uses explicit fit(agent=..., train_dataset=...)",
    "trainer.fit(agent=agent, train_dataset=tasks)" in runner,
)
check(
    "official runner has no fit kwargs fallback",
    "trainer.fit(**kwargs)" not in runner,
)
check(
    "official runner has no positional fit fallback",
    "trainer.fit(agent, tasks)" not in runner,
)
check(
    "official runner has no dynamic signature probing",
    "inspect.signature(trainer.fit)" not in runner,
)

# 2) Sidecar reporting and task pulling should fail fast.
sidecar_path = "src/inference/lightning_sidecar.py"
sidecar = read(sidecar_path)
check(f"exists {sidecar_path}", (ROOT / sidecar_path).exists())
check(
    "sidecar report_rollout fails on disabled/unreachable",
    "report_rollout() failed" in sidecar and "return False" not in sidecar.split("def report_rollout", 1)[1].split("def pull_task", 1)[0],
)
check(
    "sidecar pull_task fails on config/connectivity errors",
    "pull_task() failed" in sidecar,
)

# 3) Strict rollout paths should not silently swallow reporting errors.
strict_files = [
    "src/agents/supervisor.py",
    "apps/rollout_worker/src/rollout_worker/run_track_a.py",
    "apps/rollout_worker/src/rollout_worker/langgraph_workflow.py",
]

for rel in strict_files:
    text = read(rel)
    check(f"exists {rel}", (ROOT / rel).exists())

    if rel.endswith("run_track_a.py") or rel.endswith("langgraph_workflow.py"):
        check(
            f"{rel} delegates to official runner",
            "run_official_agent_lightning" in text or "from rollout_worker.run_track_a import run_tasks" in text,
        )
        check(
            f"{rel} has no inline supervisor implementation",
            "class SupervisorAgent:" not in text,
        )
    else:
        check(
            f"{rel} requires server URL in strict mode",
            "Strict Agent Lightning mode requires --lightning-server-url" in text
            or "requires LIGHTNING_SERVER_URL in strict mode" in text,
        )
        check(
            f"{rel} does not swallow report failures with pass",
            not re.search(r"except Exception:\\n\\s+pass\\s+# reporting", text),
        )
        check(
            f"{rel} calls official emit_reward",
            "agl.emit_reward(reward)" in text,
        )

# 4) Algorithm/Trainer/Adapter modules must not contain ImportError fallback stubs.
strict_import_files = {
    "src/training/grpo_algorithm.py": "from agentlightning import Algorithm as _BASE",
    "src/training/trainer_loop.py": "from agentlightning import Trainer as _TRAINER_BASE",
    "src/inference/trace_adapter.py": "from agentlightning import TraceAdapter as _ADAPTER_BASE",
    "packages/lightning_bridge/src/lightning_bridge/grpo_algorithm.py": "from agentlightning import Algorithm as _BASE",
    "apps/trainer/src/trainer_app/train_sft.py": "from agentlightning import Trainer as _TRAINER_BASE",
}

for rel, required_import in strict_import_files.items():
    text = read(rel)
    check(f"exists {rel}", (ROOT / rel).exists())
    check(
        f"{rel} uses direct official import",
        required_import in text,
    )
    check(
        f"{rel} has no ImportError fallback stub",
        "except ImportError" not in text and "Minimal stub" not in text,
    )

# 5) Agent layer must not import/use direct database clients.
forbidden_tokens = [
    "sqlalchemy",
    "psycopg2",
    "asyncpg",
    "DATABASE_URL",
    "SQLAlchemyConnector",
    "read_sql",
]

agent_boundaries = [
    "src/agents",
    "apps/rollout_worker/src/rollout_worker",
]

for base in agent_boundaries:
    base_path = ROOT / base
    if not base_path.exists():
        continue
    for py in base_path.rglob("*.py"):
        rel = str(py.relative_to(ROOT))
        text = py.read_text(encoding="utf-8")
        violations = [token for token in forbidden_tokens if token in text]
        check(
            f"{rel} has no direct DB imports/usages",
            not violations,
            detail=(f"forbidden tokens: {', '.join(violations)}" if violations else ""),
        )

if failures:
    print(f"Agent Lightning strict validation FAILED: {len(failures)} issue(s).")
    sys.exit(1)

print("Agent Lightning strict validation passed.")
