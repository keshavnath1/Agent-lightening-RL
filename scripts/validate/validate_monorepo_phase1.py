from __future__ import annotations

import importlib
import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PATHS = [
    ROOT,
    ROOT / "packages/contracts/src",
    ROOT / "packages/ml_tools/src",
    ROOT / "packages/rewards/src",
    ROOT / "packages/lightning_bridge/src",
    ROOT / "packages/mcp_client_bridge/src",
    ROOT / "services/project_mcp_server/src",
    ROOT / "apps/dashboard/src",
    ROOT / "apps/rollout_worker/src",
    ROOT / "apps/ruler_scorer/src",
    ROOT / "apps/trainer/src",
]
for p in reversed(PATHS):
    sys.path.insert(0, str(p))

failures: list[str] = []

def check(name: str, ok: bool, detail: str = "") -> None:
    status = "[PASS]" if ok else "[FAIL]"
    print(f"{status} {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        failures.append(name)

expected_paths = [
    "packages/contracts/src/agent_contracts/__init__.py",
    "packages/ml_tools/src/ml_tools/sql_safety.py",
    "packages/rewards/src/agent_rewards/policy_reward.py",
    "services/project_mcp_server/src/project_mcp_server/server.py",
    "apps/ruler_scorer/src/ruler_scorer/vllm_judge.py",
    "apps/trainer/src/trainer_app/train_policy_qlora_grpo.py",
    "configs/ruler.yaml",
    "docs/monorepo_phase1.md",
]
for rel in expected_paths:
    check(f"exists {rel}", (ROOT / rel).exists())

py_files = [p for rel in ["packages", "services", "apps", "tests", "scripts/validate"] for p in (ROOT / rel).rglob("*.py")]
for py_file in py_files:
    try:
        py_compile.compile(str(py_file), doraise=True)
        check(f"compile {py_file.relative_to(ROOT)}", True)
    except Exception as exc:
        check(f"compile {py_file.relative_to(ROOT)}", False, str(exc))

imports = [
    "agent_contracts",
    "ml_tools.sql_safety",
    "ml_tools.postgres_tooling",
    "agent_rewards.policy_reward",
    "agent_rewards.ruler_vllm_judge",
    "mcp_client_bridge.schema_from_function",
    "project_mcp_server.server",
    "rollout_worker.run_track_a",
    "ruler_scorer.vllm_judge",
    "trainer_app.train_policy_qlora_grpo",
]
for module in imports:
    try:
        importlib.import_module(module)
        check(f"import {module}", True)
    except Exception as exc:
        check(f"import {module}", False, repr(exc))

try:
    from agent_contracts import RewardBreakdown, Trajectory, validate_reward_breakdown

    reward = validate_reward_breakdown({"final_reward": 1.0, "valid_json": 1.0})
    traj = Trajectory(task_id="smoke", policy="test", reward=reward)
    check("contracts round-trip", Trajectory.from_dict(traj.to_dict()).reward.final_reward == 1.0)
except Exception as exc:
    check("contracts round-trip", False, repr(exc))

if failures:
    print(f"Phase 1 monorepo validation FAILED: {len(failures)} issue(s).")
    raise SystemExit(1)
print("Phase 1 monorepo validation passed.")
