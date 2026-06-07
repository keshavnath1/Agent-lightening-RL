"""track_b.py — Track B policy optimisation: trainer selection, GRPO explanation, run controls."""
from __future__ import annotations
import streamlit as st
from typing import Callable

from src.ui.view_models.dashboard_state import (
    DataStatus, ServiceHealth, TrainingConfig,
    load_grouped_rollouts_for_grpo, get_checkpoint_path, get_shell_command,
)
from src.ui.components.trainer_cards import TRAINER_CONFIGS, render_trainer_selector, render_trainer_detail
from src.ui.components.reward_viz import render_grpo_explanation_table, render_reward_mode_summary
from src.ui.components.command_box import render_command_card
from src.ui.components.evidence_panel import render_adapter_evidence


def render(
    ds: DataStatus,
    sh: ServiceHealth,
    post_fn: Callable,      # _post(path, payload) → dict
    get_fn:  Callable,      # _get(path) → dict
) -> None:
    st.header("🚀 Track B — Policy Optimisation")
    st.caption(
        "Choose a trainer, configure rewards, and run GPU training to update the policy. The current RULER path scores grouped rollouts with a required local vLLM judge, validates `ruler_relative` reward metadata, and fails closed if the judge endpoint is unavailable."
    )

    if ds.grouped_rollouts < 1:
        st.error(
            "No grouped rollouts found. Complete Track A first to generate training data."
        )
        st.code("python -m src.training.agent_lightning_official_runner", language="bash")
        return

    # ── Section 1: Trainer Selection ─────────────────────────────────────────
    st.subheader("① Choose Training Option")
    selected_trainer = render_trainer_selector()
    render_trainer_detail(selected_trainer)

    # ── Section 2: Reward / GRPO Explanation ─────────────────────────────────
    st.divider()
    st.subheader("② Reward & GRPO Overview")

    cfg_entry = TRAINER_CONFIGS.get(selected_trainer, {})
    available_rms = cfg_entry.get("reward_modes", ["hybrid"])
    default_rm    = cfg_entry.get("default_rm", "hybrid")

    col_rm, col_ng = st.columns(2)
    with col_rm:
        reward_mode = st.selectbox(
            "Reward mode",
            available_rms,
            index=available_rms.index(default_rm) if default_rm in available_rms else 0,
            key="tb_reward_mode",
        )
    with col_ng:
        num_generations = st.slider(
            "Rollouts per prompt (GRPO group size)",
            min_value=2, max_value=8,
            value=st.session_state.get("tb_num_gen", 4),
            step=2,
            key="tb_num_gen",
            help="Only applies to TRL GRPO / veRL / Agent Lightning",
        )

    render_reward_mode_summary(reward_mode)

    groups = load_grouped_rollouts_for_grpo()
    if groups:
        with st.expander(f"GRPO advantage table — first group ({len(groups)} total)", expanded=True):
            render_grpo_explanation_table(groups)

    # ── Section 3: Run & Evidence ─────────────────────────────────────────────
    st.divider()
    st.subheader("③ Run & Evidence")

    checkpoint_path = get_checkpoint_path(selected_trainer, reward_mode)
    cfg = TrainingConfig(
        trainer=selected_trainer,
        reward_mode=reward_mode,
        num_generations=num_generations,
        checkpoint_path=checkpoint_path,
    )

    # Config preview
    with st.expander("Training configuration"):
        import json
        st.json({
            "trainer":         cfg.trainer,
            "reward_mode":     cfg.reward_mode,
            "num_generations": cfg.num_generations,
            "checkpoint_path": cfg.checkpoint_path,
            "model_name":      cfg.model_name,
            "min_rollouts":    cfg.min_rollouts,
            "reload_after":    cfg.reload_after,
        })

    # Run buttons
    btn_col1, btn_col2, btn_col3 = st.columns(3)

    with btn_col1:
        if st.button("🔍 Dry Run", use_container_width=True, help="Validates config — no GPU training"):
            _do_dry_run(cfg)

    with btn_col2:
        if st.button("▶️ Run Selected", use_container_width=True, type="primary",
                     disabled=not sh.lightning_ok,
                     help="Sends training request to Lightning Server" if sh.lightning_ok else "Lightning Server offline"):
            _do_run(cfg, post_fn, get_fn)

    with btn_col3:
        if st.button("📋 Shell Command", use_container_width=True):
            st.session_state["tb_show_shell"] = not st.session_state.get("tb_show_shell", False)

    if not sh.lightning_ok:
        st.caption("ℹ️ Lightning Server offline — use Shell Command to run directly on GPU.")

    if st.session_state.get("tb_show_shell"):
        render_command_card(
            get_shell_command(cfg),
            "Shell command — copy and run on GPU pod",
            "Or use run_all_trackb_options_one_by_one.sh for all 5 options",
        )
        render_command_card(
            "bash scripts/gpu/run_all_trackb_options_one_by_one.sh",
            "Run all Track B options sequentially",
            "Logs to logs/trackb_option_runs/",
        )
        render_command_card(
            "RULER_MODE=vllm_judge bash scripts/gpu/run_ruler_trl_handoff.sh --dry-run",
            "RULER → TRL handoff dry-run",
            "Scores grouped rollouts, validates RULER reward metadata, and prints the exact training command without launching GPU training.",
        )

    # Evidence
    st.divider()
    st.subheader("Evidence")
    st.info("For RULER/vLLM/TRL handoff evidence, review `docs/ruler_vllm_trl_runbook.md`, `reports/ruler_vllm_scoring_summary.md`, and the dry-run output from `scripts/gpu/run_ruler_trl_handoff.sh`.")
    render_adapter_evidence(checkpoint_path)


# ── Helper actions ────────────────────────────────────────────────────────────

def _do_dry_run(cfg: TrainingConfig) -> None:
    import subprocess, sys
    cmd = [
        sys.executable, "-m", "src.training.train_policy_qlora_grpo",
        "--dataset", "data/grpo/grouped_rollouts.jsonl",
        "--trainer", cfg.trainer,
        "--reward-mode", cfg.reward_mode,
        "--num-generations", str(cfg.num_generations),
        "--output-dir", cfg.checkpoint_path,
        "--dry-run",
    ]
    with st.spinner("Running dry run…"):
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode == 0:
        st.success("✅ Dry run passed — configuration is valid")
    else:
        st.error("❌ Dry run failed")
    with st.expander("Dry run output"):
        st.text(result.stdout or result.stderr or "(no output)")


def _do_run(cfg: TrainingConfig, post_fn: Callable, get_fn: Callable) -> None:
    payload = {
        "trainer":         cfg.trainer,
        "reward_mode":     cfg.reward_mode,
        "num_generations": cfg.num_generations,
        "checkpoint_path": cfg.checkpoint_path,
        "model_name":      cfg.model_name,
        "min_rollouts":    cfg.min_rollouts,
        "reload_after":    cfg.reload_after,
    }
    with st.spinner("Sending training request…"):
        resp = post_fn("/api/training/trigger", payload)

    status = resp.get("status", "")
    if status == "triggered":
        st.success("✅ Training triggered — poll `/api/training/status` for progress")
        _wait_for_idle(get_fn)
    elif status == "already_running":
        st.warning("⚠️ Training is already running. Wait for it to finish.")
    elif "error" in resp:
        st.error(f"❌ Error: {resp['error']}")
    else:
        st.info(f"Response: {resp}")


def _wait_for_idle(get_fn: Callable) -> None:
    """Poll /api/training/status every 20 s for up to ~3 minutes (9 polls)."""
    import time
    max_polls = 9  # 9 × 20 s = 3 minutes
    bar = st.progress(0, text="Waiting for training to complete…")
    for i in range(max_polls):
        time.sleep(20)
        try:
            resp = get_fn("/api/training/status")
            if not resp.get("training_active", True):
                bar.progress(1.0, text="✅ Training complete")
                return
        except Exception:
            pass
        bar.progress((i + 1) / max_polls, text=f"Training active… poll {i+1}/{max_polls}")
    bar.empty()
    st.info("Polling stopped (3-min timeout). Check status manually or look at the evidence panel.")
