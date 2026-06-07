"""track_a.py — Track A pipeline status and results table."""
from __future__ import annotations
import streamlit as st
from src.ui.view_models.dashboard_state import DataStatus, load_task_results


PIPELINE_STAGES = [
    ("📋 Task",              "Load metadata-only task from PostgreSQL/MCP"),
    ("🔬 Data Engineer",     "Build safe schema/profile artifacts without raw rows"),
    ("📊 GBM Specialist",    "LightGBM baseline · hyperparameter search · cross-val"),
    ("🏗 Sandbox",           "Execute code · validate output · score metric"),
    ("📈 Tracking",          "Log experiment · record params & metrics to MLflow"),
    ("🔍 Reviewer",          "Assess evidence · assign reward · write judgement"),
    ("✅ Scored Trajectory",  "Store action sequence + reward in scored/"),
]


def render(ds: DataStatus) -> None:
    st.header("🤖 Track A — LangGraph ML Pipeline")
    st.caption(
        "Track A agents collaborate to train a GBM model on each task and produce a **scored trajectory** "
        "used by Track B for policy learning."
    )

    # ── Pipeline flow diagram ─────────────────────────────────────────────────
    st.subheader("Pipeline Flow")
    stage_html = " &nbsp;→&nbsp; ".join(
        f"<span style='border:1px solid #444; padding:3px 8px; border-radius:4px; "
        f"font-size:0.82em;'>{name}</span>"
        for name, _ in PIPELINE_STAGES
    )
    st.markdown(f"<div style='line-height:2.5;'>{stage_html}</div>", unsafe_allow_html=True)

    with st.expander("Stage descriptions"):
        for name, desc in PIPELINE_STAGES:
            st.markdown(f"**{name}** — {desc}")

    # ── Counts ────────────────────────────────────────────────────────────────
    st.divider()
    col1, col2, col3 = st.columns(3)
    col1.metric("Artifacts completed",  len(ds.artifacts))
    col2.metric("Scored trajectories",  ds.scored_trajectories)
    col3.metric("Grouped GRPO rollouts",ds.grouped_rollouts)

    # ── Results table ─────────────────────────────────────────────────────────
    st.subheader("Champion Model Results")
    results = load_task_results()
    if not results:
        st.info(
            "No champion_model.json files found in `artifacts/task_*/`.  "
            "Run the Track A pipeline:  \n"
            "`python -m src.agents.supervisor --task-source postgres_mcp --lightning-server-url http://127.0.0.1:19124`"
        )
        return

    import pandas as pd
    df = pd.DataFrame(results)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # ── Rollout readiness ─────────────────────────────────────────────────────
    st.divider()
    if ds.grouped_rollouts >= 1:
        st.success(
            f"✅ **{ds.grouped_rollouts} grouped rollout(s)** ready for Track B policy training."
        )
    else:
        st.warning(
            "Grouped rollouts not yet available.  "
            "If scored trajectories exist, run the grouping script:"
        )
        st.code(
            "python -m src.training.split_policy_dataset --input data/grpo",
            language="bash",
        )
