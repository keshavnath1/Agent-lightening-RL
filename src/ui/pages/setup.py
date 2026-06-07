"""setup.py — Setup & Handoff: Environment / Data / Next Command."""
from __future__ import annotations
import sys
import os
import streamlit as st
from pathlib import Path
from src.ui.view_models.dashboard_state import (
    DataStatus, GPUStatus, ServiceHealth, ROOT,
)
from src.ui.components.status_cards import status_badge
from src.ui.components.command_box import render_command_card


def render(ds: DataStatus, gs: GPUStatus, sh: ServiceHealth) -> None:
    st.header("🔧 Setup & Handoff")
    st.caption("Three things must be true before GPU training: environment ✓ · data ✓ · command ✓")

    # ── A. Environment ────────────────────────────────────────────────────────
    st.subheader("A. Environment")

    py_ok = sys.version_info >= (3, 10)
    status_badge(py_ok, f"Python {sys.version.split()[0]}", severity="required")
    status_badge(gs.available, f"GPU: {gs.name} {gs.vram_gb} GB × {gs.count}" if gs.available else "GPU not detected",
                 detail="Required for training" if not gs.available else "", severity="required")

    # Optional services — shown as warning, not error
    status_badge(sh.lightning_ok, "Lightning Server",
                 detail="Start: `bash scripts/gpu/start_lightning_server.sh`" if not sh.lightning_ok else "online",
                 severity="optional")
    status_badge(sh.vllm_ok, "vLLM",
                 detail="Required only for inference / hot-reload" if not sh.vllm_ok else "online",
                 severity="optional")
    status_badge(sh.mlflow_ok, "MLflow",
                 detail="Optional evidence tracking" if not sh.mlflow_ok else "online",
                 severity="optional")

    # Docker
    try:
        import subprocess
        docker_ok = subprocess.run(["docker", "info"], capture_output=True, timeout=3).returncode == 0
    except Exception:
        docker_ok = False
    status_badge(docker_ok, "Docker",
                 detail="Optional for local demo — not needed for GPU training" if not docker_ok else "running",
                 severity="optional")

    # Key packages
    st.markdown("")
    _pkg_checks = [
        ("torch",          "PyTorch",     "required"),
        ("peft",           "PEFT / LoRA", "required"),
        ("transformers",   "Transformers","required"),
        ("trl",            "TRL (GRPO)",  "required"),
        ("bitsandbytes",   "bitsandbytes","optional"),
        ("verl",           "veRL",        "advanced"),
        ("agentlightning", "Agent Lightning", "advanced"),
    ]
    pkg_rows = []
    for pkg, label, sev in _pkg_checks:
        try:
            __import__(pkg)
            ok = True
            ver = ""
            try:
                import importlib.metadata
                ver = importlib.metadata.version(pkg)
            except Exception:
                pass
        except ImportError:
            ok = False
            ver = ""
        pkg_rows.append({"label": label, "ok": ok, "ver": ver, "sev": sev})

    req_missing  = [r for r in pkg_rows if not r["ok"] and r["sev"] == "required"]
    with st.expander(f"Package checks {'✅' if not req_missing else f'❌ {len(req_missing)} required missing'}"):
        for r in pkg_rows:
            icon = "✅" if r["ok"] else ("❌" if r["sev"]=="required" else "⬜")
            tag  = {"required":"", "optional":" *(optional)*", "advanced":" *(advanced)*"}[r["sev"]]
            ver_str = f" `{r['ver']}`" if r["ver"] else ""
            fix = f" — `pip install {r['label'].lower().replace(' ','')}`" if not r["ok"] else ""
            st.markdown(f"{icon} **{r['label']}**{tag}{ver_str}{fix}")

    # ── B. Required Data ──────────────────────────────────────────────────────
    st.divider()
    st.subheader("B. Required Data")

    tasks_path = ROOT / "data" / "synthetic" / "tasks.jsonl"
    status_badge(tasks_path.exists(), "tasks.jsonl",
                 detail=f"{ds.tasks_loaded} tasks" if tasks_path.exists() else "Missing — create synthetic tasks",
                 severity="optional")

    grp_path = ROOT / "data" / "grpo" / "grouped_rollouts.jsonl"
    status_badge(grp_path.exists(), "grouped_rollouts.jsonl",
                 detail=f"{ds.grouped_rollouts} task group(s)" if grp_path.exists() else "Run Track A first",
                 severity="required")

    scored_dir = ROOT / "trajectories" / "scored"
    scored_ok  = scored_dir.exists() and ds.scored_trajectories > 0
    status_badge(scored_ok, "Scored trajectories",
                 detail=f"{ds.scored_trajectories} found" if scored_ok else "Run Track A + scorer",
                 severity="optional")

    artifacts_ok = len(ds.artifacts) > 0
    status_badge(artifacts_ok, "Champion model artifacts",
                 detail=f"{len(ds.artifacts)} found" if artifacts_ok else "Run Track A pipeline",
                 severity="optional")

    # ── C. Recommended Next Command ───────────────────────────────────────────
    st.divider()
    st.subheader("C. Recommended Next Command")

    if not grp_path.exists():
        render_command_card(
            "python -m src.training.agent_lightning_official_runner",
            "Run Track A pipeline (collect rollouts)",
            "This produces grouped_rollouts.jsonl",
        )
    elif not gs.available:
        render_command_card(
            "# GPU required for training\n"
            "# Connect to GPU pod and run:\n"
            "bash scripts/setup_environment.sh --target gpu\n"
            "source .venv-gpu/bin/activate\n"
            "TRAINER=qlora_sft bash scripts/gpu/run_02_train_policy_qlora_grpo.sh",
            "GPU environment + training command",
            "Run on the GPU pod",
        )
    else:
        render_command_card(
            "bash scripts/gpu/run_all_trackb_options_one_by_one.sh --dry-run",
            "Dry run — check all 5 Track B options",
            "Safe to run, no training yet",
        )
        render_command_card(
            "bash scripts/gpu/run_all_trackb_options_one_by_one.sh",
            "Run all Track B options sequentially",
            "Trains 5 options, skips unavailable ones",
        )

    with st.expander("All available commands"):
        st.code(
            "# Dry run with explicit trainer\n"
            "python -m src.training.train_policy_qlora_grpo \\\n"
            "  --dataset data/grpo/grouped_rollouts.jsonl \\\n"
            "  --trainer qlora_sft --dry-run\n\n"
            "# Single trainer\n"
            "TRAINER=trl_grpo REWARD_MODE=hybrid \\\n"
            "  bash scripts/gpu/run_02_train_policy_qlora_grpo.sh\n\n"
            "# All options (blocking, sequential)\n"
            "bash scripts/gpu/run_all_trackb_options_one_by_one.sh\n\n"
            "# Start Lightning Server\n"
            "bash scripts/gpu/start_lightning_server.sh\n\n"
            "# Start vLLM\n"
            "bash scripts/gpu/start_baseline_inference.sh",
            language="bash",
        )
