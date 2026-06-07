"""evidence_panel.py — adapter evidence and comparison panel."""
from __future__ import annotations
import json
import streamlit as st
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def render_adapter_evidence(checkpoint_path: str) -> None:
    """Show key files from a checkpoint directory."""
    ckpt = ROOT / checkpoint_path
    meta = ckpt / "adapter_metadata.json"

    if meta.exists():
        try:
            data = json.loads(meta.read_text())
            st.success(f"✅ Adapter trained — `{checkpoint_path}`")
            c1, c2, c3 = st.columns(3)
            c1.metric("Trainer",   data.get("trainer", "—"))
            c1.metric("Status",    data.get("adapter_status", "—"))
            c2.metric("Examples",  data.get("num_examples", "—"))
            c3.metric("Reward mode", data.get("reward_mode", "—"))
            with st.expander("Full metadata"):
                st.json(data)
            return
        except Exception:
            pass

    FILE_CHECKS = [
        ("adapter_config.json",        "LoRA config"),
        ("adapter_model.safetensors",  "Model weights"),
        ("tokenizer_config.json",      "Tokeniser"),
        ("adapter_metadata.json",      "Training metadata"),
    ]
    any_found = any((ckpt / f).exists() for f, _ in FILE_CHECKS)
    if any_found:
        st.warning(f"Partial checkpoint at `{checkpoint_path}`")
        for fname, desc in FILE_CHECKS:
            icon = "✅" if (ckpt / fname).exists() else "⬜"
            st.markdown(f"{icon} `{fname}` — {desc}")
    else:
        st.info(
            f"No checkpoint at `{checkpoint_path}` yet.  "
            "Run training first."
        )


def render_comparison_summary(server_url: str) -> None:
    """Fetch rollout data and show baseline vs tuned summary."""
    import requests as _req
    try:
        r = _req.get(f"{server_url}/api/rollouts?limit=500", timeout=4)
        recs = r.json().get("rollouts", []) if r.status_code == 200 else []
    except Exception:
        recs = []

    if not recs:
        # Fall back to local report files
        for pat in ["baseline_vs_rl_tuned.md", "baseline_v2_tuned_benchmark.md"]:
            p = ROOT / "reports" / pat
            if p.exists():
                st.markdown(p.read_text())
                return
        st.info("No comparison data yet. Run baseline and tuned inference then compare.")
        return

    import pandas as pd
    df = pd.DataFrame([{
        "policy": r.get("policy_version", "unknown"),
        "reward": float(r.get("final_reward") or 0.0),
    } for r in recs])

    if df["policy"].nunique() < 2:
        st.info("Need at least 2 policy versions to compare. Train Track B and run new rollouts.")
        return

    summary = (
        df.groupby("policy")["reward"]
        .agg(mean="mean", std="std", count="count")
        .reset_index()
        .sort_values("mean")
    )
    policies = summary["policy"].tolist()
    baseline_mean = float(summary[summary["policy"] == policies[0]]["mean"].iloc[0])
    tuned_mean    = float(summary[summary["policy"] == policies[-1]]["mean"].iloc[0])
    win_rate      = 100 * sum(1 for r in recs if float(r.get("final_reward",0)) > baseline_mean) / len(recs)

    c1, c2, c3 = st.columns(3)
    c1.metric("Baseline mean reward", f"{baseline_mean:.4f}")
    c2.metric("Tuned mean reward",    f"{tuned_mean:.4f}", delta=f"{tuned_mean-baseline_mean:+.4f}")
    c3.metric("Win rate",             f"{win_rate:.1f}%")
