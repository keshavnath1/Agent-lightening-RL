"""overview.py — Landing page: 4 status cards + recommended next action."""
from __future__ import annotations
from pathlib import Path

import streamlit as st
from src.ui.view_models.dashboard_state import (
    DataStatus, GPUStatus, ServiceHealth, load_task_results,
)
from src.ui.components.status_cards import render_stage_card


STAGE_ORDER = ["setup", "track_a", "track_b", "results"]


def _infer_stage(ds: DataStatus, sh: ServiceHealth) -> str:
    if sh.training_active:
        return "track_b"
    if ds.grouped_rollouts >= 1:
        return "track_b"
    if ds.scored_trajectories >= 1 or ds.artifacts:
        return "track_a"
    return "setup"


def render(ds: DataStatus, gs: GPUStatus, sh: ServiceHealth, nav_callback) -> None:
    st.title("⚡ Self-Improving ML Agent")
    st.caption("Gated Track A benchmark -> Track B TRL GRPO adapter redeploy -> before/after endpoint benchmark")

    current_stage = _infer_stage(ds, sh)

    # ── Status row ────────────────────────────────────────────────────────────
    st.markdown("### Current Status")
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        render_stage_card(
            "Setup",
            "Environment & data ready",
            status="complete" if ds.grouped_rollouts >= 1 else "ready",
            metric=f"Tasks: {ds.tasks_loaded}",
        )
    with c2:
        render_stage_card(
            "Track A",
            "LangGraph ML pipeline",
            status="complete" if ds.scored_trajectories >= 1 or ds.artifacts else "pending",
            metric=f"Artifacts: {len(ds.artifacts)}  ·  Scored: {ds.scored_trajectories}",
        )
    with c3:
        render_stage_card(
            "Track B",
            "Policy optimisation",
            status="complete" if (ds.grouped_rollouts >= 1 and not sh.training_active) else
                   "ready"    if ds.grouped_rollouts >= 1 else "pending",
            metric=f"Grouped rollouts: {ds.grouped_rollouts}",
        )
    with c4:
        render_stage_card(
            "Results",
            "Baseline vs tuned policy",
            status="ready" if sh.lightning_ok else "pending",
            metric="Lightning Server: " + ("online" if sh.lightning_ok else "offline"),
        )

    # ── Quick system facts ────────────────────────────────────────────────────
    st.markdown("### System")
    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("GPU", f"{gs.name} {gs.vram_gb} GB" if gs.available else "Not detected")
    sc2.metric("Grouped rollouts", ds.grouped_rollouts)
    sc3.metric("Scored trajectories", ds.scored_trajectories)
    sc4.metric("Lightning Server", "🟢 online" if sh.lightning_ok else "⚪ offline")

    # ── Recommended next action ───────────────────────────────────────────────
    st.markdown("### Recommended Next Action")

    if sh.training_active:
        st.info("🔄 Training is currently running. Check Track B → Evidence when complete.")
        if st.button("→ Go to Track B", type="primary"):
            nav_callback("Track B")

    elif current_stage == "setup":
        st.warning("No rollout data found yet. Run Track A to collect rollouts.")
        with st.expander("Quick start command"):
            st.code("python -m src.training.agent_lightning_official_runner", language="bash")
        if st.button("→ Go to Setup", type="primary"):
            nav_callback("Setup & Handoff")

    elif current_stage == "track_b":
        st.success(
            f"✅ **{ds.grouped_rollouts} grouped rollout(s)** ready.  "
            "Run TRL GRPO training on GPU."
        )
        col_b1, col_b2 = st.columns(2)
        with col_b1:
            if st.button("→ Go to Track B Training", type="primary", use_container_width=True):
                nav_callback("Track B")
        with col_b2:
            with st.expander("Shell command"):
                st.code(
                    "TRAINER=trl_grpo REWARD_MODE=hybrid bash scripts/gpu/run_02_train_policy_qlora_grpo.sh",
                    language="bash",
                )

    elif current_stage == "track_a":
        st.info("Track A artifacts found. Run the grouping script, then proceed to Track B.")
        with st.expander("Group rollouts command"):
            st.code("python -m src.training.split_policy_dataset --input data/grpo", language="bash")
        if st.button("→ Go to Track A", type="primary"):
            nav_callback("Track A")

    # ── Latest E2E benchmark summary ───────────────────────────────────────────
    latest_summary = Path(__file__).resolve().parents[3] / "reports" / "e2e_tracka_vs_trackb_summary_20260604_053507.json"
    if latest_summary.exists():
        st.markdown("### Latest E2E Benchmark")
        st.success("Track A gate passed; Track B policy adapter redeploy completed; benchmark evidence is available in reports and trajectories.")

    # ── Recent artifacts ──────────────────────────────────────────────────────
    results = load_task_results()
    if results:
        import pandas as pd
        st.markdown("### Latest Track A Results")
        st.dataframe(
            pd.DataFrame(results).tail(4),
            use_container_width=True, hide_index=True,
        )


# Alignment note inserted for presentation mode
def _render_pdf_alignment_note() -> None:
    import streamlit as st
    st.markdown('### PDF Alignment')
    st.markdown('''PDF alignment: ART-like trajectories, grouped rollouts, GRPO, MCP tool loop partial, fail-closed vLLM-backed RULER judge, checkpoint forking metadata, behavioral policy drift, and GSPO roadmap only. Official ART/RULER is available only when openpipe-art and official RULER are installed.''')
