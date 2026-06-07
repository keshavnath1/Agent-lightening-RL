#!/usr/bin/env python3
"""
dev_validate_trackb_wiring.py — Track B E2E wiring validation.

Checks:
  1. All main Track B source files compile
  2. All 4 reward modes are registered in REWARD_MODES
  3. Each reward mode can be called with (prompts, completions) signature
  4. trajectory_reward returns passed-in reward column values (batch kwarg path)
  5. json_validity_reward handles legacy (completions) call style
  6. train_policy_qlora_grpo --help contains required CLI flags
  7. run_all_trackb_options_one_by_one.sh does NOT contain --tasks-path
  8. run_all_trackb_options_one_by_one.sh contains --agent-lightning-tasks
  9. Shell scripts pass bash -n syntax check

Run:
    python scripts/dev_validate_trackb_wiring.py

Expected output:
    Track B validation passed.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PASS = "\033[32m[PASS]\033[0m"
FAIL = "\033[31m[FAIL]\033[0m"

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    if ok:
        print(f"  {PASS}  {name}")
    else:
        print(f"  {FAIL}  {name}" + (f" — {detail}" if detail else ""))
        failures.append(name)


# ─── 1. Compile checks ────────────────────────────────────────────────────────
print("\n── 1. Compile checks ────────────────────────────────────────────────")
COMPILE_FILES = [
    "src/rewards/policy_reward.py",
    "src/training/split_policy_dataset.py",
    "src/training/train_policy_qlora_grpo.py",
    "src/training/grpo_algorithm.py",
    "src/training/lightning_server_app.py",
    "src/training/trainer_loop.py",
    "scripts/demo_dashboard.py",
]
for rel in COMPILE_FILES:
    path = ROOT / rel
    if not path.exists():
        check(f"exists: {rel}", False, "file not found")
        continue
    result = subprocess.run(
        [sys.executable, "-m", "py_compile", str(path)],
        capture_output=True,
        text=True,
    )
    check(f"compile: {rel}", result.returncode == 0, result.stderr.strip())


# ─── 2. Reward modes registered ───────────────────────────────────────────────
print("\n── 2. Reward mode registry ──────────────────────────────────────────")
try:
    sys.path.insert(0, str(ROOT))
    from src.rewards.policy_reward import REWARD_MODES, get_reward_fn  # noqa: E402

    EXPECTED_MODES = {"json_validity", "workflow_policy", "trajectory_reward", "hybrid"}
    for mode in EXPECTED_MODES:
        check(f"registered: {mode}", mode in REWARD_MODES)
except Exception as exc:
    check("import policy_reward", False, str(exc))


# ─── 3. Each reward mode callable with (prompts, completions) ─────────────────
print("\n── 3. Reward fn signatures ──────────────────────────────────────────")
_PROMPT      = ['{"task_id": "t0", "instruction": "test"}']
_COMPLETION  = ['{"agent_name": "data_engineer", "action": "profile_data", '
                '"reasoning_summary": "baseline", "expected_tool_calls": []}']

for mode in ["json_validity", "workflow_policy", "hybrid", "trajectory_reward"]:
    try:
        fn = get_reward_fn(mode)
        result = fn(_PROMPT, _COMPLETION)
        ok = isinstance(result, list) and len(result) == 1 and isinstance(result[0], float)
        check(f"callable (prompts, completions): {mode}", ok, str(result) if not ok else "")
    except Exception as exc:
        check(f"callable (prompts, completions): {mode}", False, str(exc))


# ─── 4. json_validity_reward legacy call style ────────────────────────────────
print("\n── 4. json_validity_reward legacy call style ────────────────────────")
try:
    from src.rewards.policy_reward import json_validity_reward  # noqa: E402

    # Legacy: positional completions only (no prompts arg)
    r1 = json_validity_reward(_COMPLETION)
    check("legacy: json_validity_reward(completions)", isinstance(r1, list) and len(r1) == 1)

    # Keyword
    r2 = json_validity_reward(completions=_COMPLETION)
    check("keyword: json_validity_reward(completions=[...])", isinstance(r2, list) and len(r2) == 1)

    # Standard TRL positional
    r3 = json_validity_reward(_PROMPT, _COMPLETION)
    check("standard: json_validity_reward(prompts, completions)", isinstance(r3, list) and len(r3) == 1)
except Exception as exc:
    check("json_validity_reward multi-style", False, str(exc))


# ─── 5. trajectory_reward uses batch 'reward' kwarg ──────────────────────────
print("\n── 5. trajectory_reward batch column ────────────────────────────────")
try:
    from src.rewards.policy_reward import trajectory_reward  # noqa: E402

    expected = 0.75
    result = trajectory_reward(_PROMPT, _COMPLETION, reward=[expected])
    got = result[0] if result else None
    check(
        "trajectory_reward uses kwargs['reward'] batch column",
        abs((got or 0) - expected) < 0.01,
        f"expected {expected}, got {got}",
    )
except Exception as exc:
    check("trajectory_reward batch column", False, str(exc))


# ─── 6. train_policy_qlora_grpo --help ───────────────────────────────────────
print("\n── 6. CLI flags in train_policy_qlora_grpo ──────────────────────────")
try:
    result = subprocess.run(
        [sys.executable, "-m", "src.training.train_policy_qlora_grpo", "--help"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    help_text = result.stdout + result.stderr
    for flag in ["--reward-mode", "--num-generations", "--agent-lightning-tasks"]:
        check(f"--help contains: {flag}", flag in help_text)
except Exception as exc:
    check("train_policy_qlora_grpo --help", False, str(exc))


# ─── 7 + 8. Shell script content checks ──────────────────────────────────────
print("\n── 7+8. Shell script content ────────────────────────────────────────")
SH = ROOT / "scripts" / "gpu" / "run_all_trackb_options_one_by_one.sh"
if SH.exists():
    sh_text = SH.read_text()
    check("shell: no --tasks-path",           "--tasks-path" not in sh_text,
          "--tasks-path still present")
    check("shell: has --agent-lightning-tasks", "--agent-lightning-tasks" in sh_text)
else:
    check("shell script exists", False, str(SH))


# ─── 9. bash -n syntax check on shell scripts ────────────────────────────────
print("\n── 9. Bash syntax checks ────────────────────────────────────────────")
SHELL_SCRIPTS = list((ROOT / "scripts" / "gpu").glob("*.sh"))
for sh in sorted(SHELL_SCRIPTS):
    r = subprocess.run(["bash", "-n", str(sh)], capture_output=True, text=True)
    check(f"bash -n: {sh.name}", r.returncode == 0, r.stderr.strip())


# ─── Summary ──────────────────────────────────────────────────────────────────
print()
if failures:
    print(f"\033[31mTrack B validation FAILED — {len(failures)} issue(s):\033[0m")
    for f in failures:
        print(f"  • {f}")
    sys.exit(1)
else:
    print("\033[32mTrack B validation passed.\033[0m")
