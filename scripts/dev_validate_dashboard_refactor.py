"""Validate the modular Streamlit dashboard wiring.

The validator intentionally checks importability and wiring without starting a
Streamlit server. It covers the app package entrypoint, router, page modules,
component modules, trainer configuration keys, checkpoint selection, and command
construction.
"""
from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DASHBOARD_SRC = ROOT / "apps" / "dashboard" / "src"
for candidate in [ROOT, DASHBOARD_SRC]:
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

try:
    import streamlit  # noqa: F401
except ImportError:
    print("\nstreamlit is not installed. Run: pip install -r requirements-cpu.txt\n")
    sys.exit(2)

PASS = "PASS"
FAIL = "FAIL"
_results: list[tuple[str, bool, str]] = []


def check(label: str, fn):
    try:
        fn()
        _results.append((label, True, ""))
    except Exception as exc:
        _results.append((label, False, str(exc)))


def _import(name: str):
    return importlib.import_module(name)


# Component + view-model imports.
for module in [
    "src.ui.view_models.dashboard_state",
    "src.ui.components.status_cards",
    "src.ui.components.trainer_cards",
    "src.ui.components.reward_viz",
    "src.ui.components.command_box",
    "src.ui.components.evidence_panel",
    "src.ui.components.validation_panel",
]:
    check(f"Import {module}", lambda m=module: _import(m))

# Page imports.
for module in [
    "src.ui.pages.overview",
    "src.ui.pages.architecture",
    "src.ui.pages.setup",
    "src.ui.pages.track_a",
    "src.ui.pages.tool_learning",
    "src.ui.pages.track_b",
    "src.ui.pages.results",
    "src.ui.pages.monorepo_health",
    "src.ui.pages.debug",
]:
    check(f"Import {module}", lambda m=module: _import(m))


def _check_python_syntax(rel: str) -> None:
    path = ROOT / rel
    assert path.exists(), f"{rel} not found"
    ast.parse(path.read_text(), filename=str(path))


for rel in [
    "scripts/demo_dashboard.py",
    "apps/dashboard/src/dashboard_app/main.py",
    "apps/dashboard/src/dashboard_app/router.py",
]:
    check(f"{rel} syntax valid", lambda r=rel: _check_python_syntax(r))


def _check_state_exports() -> None:
    m = _import("src.ui.view_models.dashboard_state")
    for sym in [
        "DataStatus",
        "GPUStatus",
        "ServiceHealth",
        "TrainingConfig",
        "load_data_status",
        "load_gpu_status",
        "load_service_health",
        "load_task_results",
        "load_grouped_rollouts_for_grpo",
        "get_checkpoint_path",
        "get_shell_command",
    ]:
        assert hasattr(m, sym), f"Missing: {sym}"


check("dashboard_state exports all symbols", _check_state_exports)


def _check_trainer_configs() -> None:
    from src.ui.components.trainer_cards import TRAINER_CONFIGS

    expected_keys = {
        "qlora_sft",
        "trl_grpo",
        "verl",
        "agent_lightning_official",
        "official_art_ruler",
    }
    missing = expected_keys.difference(TRAINER_CONFIGS)
    assert not missing, f"Missing trainer keys: {sorted(missing)}"
    assert len(TRAINER_CONFIGS) >= len(expected_keys), f"Expected at least {len(expected_keys)} trainers, got {len(TRAINER_CONFIGS)}"


check("trainer_cards has expected trainer configs", _check_trainer_configs)

# Component callables.
for module, attr in [
    ("src.ui.components.trainer_cards", "render_trainer_selector"),
    ("src.ui.components.reward_viz", "render_grpo_explanation_table"),
    ("src.ui.components.status_cards", "render_stage_card"),
    ("src.ui.components.command_box", "render_command"),
    ("src.ui.components.evidence_panel", "render_adapter_evidence"),
    ("src.ui.components.validation_panel", "render_validation_panel"),
]:
    check(f"{module}.{attr} is callable", lambda m=module, a=attr: callable(getattr(_import(m), a)))

# Page render callables.
for page_name in ["overview", "architecture", "setup", "track_a", "tool_learning", "track_b", "results", "monorepo_health", "debug"]:
    check(f"pages.{page_name}.render is callable", lambda n=page_name: callable(_import(f"src.ui.pages.{n}").render))


def _check_dashboard_app_entrypoints() -> None:
    main = _import("dashboard_app.main")
    router = _import("dashboard_app.router")
    assert callable(main.run), "dashboard_app.main.run is not callable"
    assert callable(router.render_dashboard), "dashboard_app.router.render_dashboard is not callable"
    script = (ROOT / "scripts" / "demo_dashboard.py").read_text()
    assert "from dashboard_app.main import run" in script, "demo_dashboard.py does not import dashboard_app.main.run"
    assert "scripts.demo_dashboard" not in (ROOT / "apps/dashboard/src/dashboard_app/main.py").read_text(), "main.py still imports scripts.demo_dashboard"


check("dashboard app entrypoints are wired", _check_dashboard_app_entrypoints)


def _check_checkpoint_path() -> None:
    from src.ui.view_models.dashboard_state import get_checkpoint_path

    assert get_checkpoint_path("trl_grpo", "ruler_relative") == "checkpoints/trackb_trl_grpo_ruler"
    assert get_checkpoint_path("trl_grpo", "hybrid") == "checkpoints/trackb_trl_grpo_hybrid"


check("get_checkpoint_path handles ruler_relative", _check_checkpoint_path)


def _check_shell_command() -> None:
    from src.ui.view_models.dashboard_state import TrainingConfig, get_shell_command

    cfg = TrainingConfig(
        trainer="qlora_sft",
        reward_mode="hybrid",
        num_generations=4,
        checkpoint_path="checkpoints/trackb_qlora_sft",
    )
    cmd = get_shell_command(cfg)
    assert "TRAINER=qlora_sft" in cmd, f"TRAINER missing in: {cmd}"
    assert "REWARD_MODE=hybrid" in cmd, f"REWARD_MODE missing in: {cmd}"
    assert "NUM_GENERATIONS=4" in cmd, f"NUM_GENERATIONS missing in: {cmd}"
    assert "run_02_train_policy_qlora_grpo.sh" in cmd, f"script name missing in: {cmd}"


check("get_shell_command produces correct command", _check_shell_command)

print()
print("=" * 60)
print("Dashboard Refactor Validation")
print("=" * 60)
passed = 0
failed = 0
for label, ok, msg in _results:
    icon = PASS if ok else FAIL
    print(f"  {icon} {label}" + (f"\n     {msg}" if msg else ""))
    if ok:
        passed += 1
    else:
        failed += 1

print()
print(f"Results: {passed}/{len(_results)} passed, {failed} failed")
print()

if failed:
    print("Dashboard refactor validation FAILED.")
    sys.exit(1)

print("Dashboard refactor validation passed.")
sys.exit(0)
