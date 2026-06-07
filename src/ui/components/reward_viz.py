"""reward_viz.py — GRPO reward and advantage visualisation components."""
from __future__ import annotations
import streamlit as st
from typing import Any


def render_grpo_explanation_table(groups: list[dict[str, Any]]) -> None:
    """
    Render one GRPO advantage table from grouped rollout data.
    Takes the first group with ≥ 2 trajectories.
    """
    import pandas as pd

    group = next(
        (g for g in groups if len(g.get("ranked_trajectories", [])) >= 2),
        None,
    )
    if not group:
        st.info("No grouped rollouts with ≥ 2 trajectories found. Run Track A first.")
        return

    trajs = group.get("ranked_trajectories", [])
    rewards = [float(t.get("reward") or 0.0) for t in trajs]
    mean_r  = sum(rewards) / len(rewards) if rewards else 0.0

    rows = []
    for i, (t, r) in enumerate(zip(trajs, rewards)):
        adv = r - mean_r
        rows.append({
            "Rollout":        chr(65 + i),                   # A, B, C …
            "Trajectory ID":  (t.get("trajectory_id") or "")[:12] + "…",
            "Reward":         round(r, 4),
            "Group mean":     round(mean_r, 4),
            "Advantage":      round(adv, 4),
            "GRPO update":    "⬆ increase prob" if adv > 0 else "⬇ decrease prob",
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown(
        "> **GRPO** increases the probability of actions from above-average rollouts "
        "and decreases actions from below-average rollouts — no value function needed."
    )

    # Simple bar chart
    try:
        import altair as alt
        chart = (
            alt.Chart(df)
            .mark_bar()
            .encode(
                x=alt.X("Rollout:N"),
                y=alt.Y("Advantage:Q", title="Advantage (reward − group mean)"),
                color=alt.condition(
                    "datum.Advantage > 0",
                    alt.value("#28a745"),
                    alt.value("#dc3545"),
                ),
                tooltip=["Rollout", "Reward", "Group mean", "Advantage", "GRPO update"],
            )
            .properties(height=200)
        )
        st.altair_chart(chart, use_container_width=True)
    except ImportError:
        pass


def render_reward_mode_summary(mode: str) -> None:
    """One-line summary of what a reward mode measures."""
    SUMMARIES = {
        "hybrid":           "**Hybrid (default):** 30% JSON validity + 70% workflow quality",
        "workflow_policy":  "**Workflow policy:** Semantic scoring — rewards schema/GBM/MLflow/sandbox mentions; penalises data leakage and fake metrics",
        "json_validity":    "**JSON validity:** Structural check only — valid JSON + required keys",
        "trajectory_reward":"**Trajectory reward:** Uses pre-computed rollout scores from scored trajectories",
    }
    st.info(SUMMARIES.get(mode, f"Reward mode: `{mode}`"))
