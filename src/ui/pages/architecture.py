"""Architecture page for the dashboard."""
from __future__ import annotations

import streamlit as st


def render() -> None:
    """Render the Track A/Track B architecture overview."""
    st.header("Architecture")
    st.caption("Control-plane, reward-plane, optimizer, and inference responsibilities for the self-improving ML agent.")

    st.subheader("Track A CPU control plane")
    st.markdown(
        "Agent rollout → Official Project MCP tools → Trajectory store → Reward scorer. "
        "Track A focuses on collecting tool-using trajectories, scoring them, and preparing grouped rollout data."
    )

    st.subheader("Track B GPU optimization plane")
    st.markdown(
        "RULER/vLLM judge → ruler_scored_groups.jsonl → TRL GRPO/QLoRA → LoRA checkpoint → vLLM policy server. "
        "Track B converts grouped rollouts into judged training data and updates the policy adapter behind an explicit GPU handoff."
    )

    st.subheader("Component responsibilities")
    rows = [
        ("Agent Lightning", "Trajectory and control plane"),
        ("Official Project MCP", "Tool interface exposed to the rollout agent"),
        ("RULER", "Judge and reward plane for relative scoring"),
        ("TRL GRPO", "Policy optimizer for grouped rollouts"),
        ("vLLM", "Inference plane for policy serving and local judging"),
    ]
    st.table({"Component": [r[0] for r in rows], "Role": [r[1] for r in rows]})

    st.subheader("End-to-end flow")
    st.code(
        "Track A: agent rollout -> MCP tools -> trajectories -> rewards -> grouped_rollouts.jsonl\n"
        "Track B: grouped_rollouts.jsonl -> RULER/vLLM scoring -> TRL GRPO -> LoRA adapter -> vLLM policy server",
        language="text",
    )
