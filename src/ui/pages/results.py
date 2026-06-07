"""results.py — Baseline vs tuned policy comparison."""
from __future__ import annotations
import json
import streamlit as st
from pathlib import Path
from src.ui.view_models.dashboard_state import ServiceHealth, ROOT


def render(sh: ServiceHealth, server_url: str) -> None:
    st.header("📊 Results — Policy Improvement")
    st.caption("Compare the gated Track A baseline against the Track B TRL GRPO adapter redeploy.")

    # ── Attempt to load comparison data from multiple sources ─────────────────
    shown = _try_live(sh, server_url)
    shown = _try_local_reports() or shown
    if not shown:
        _show_empty_state()
    _show_trajectory_traces()
    _show_recent_logs()


# ── Private helpers ───────────────────────────────────────────────────────────

def _try_live(sh: ServiceHealth, server_url: str) -> bool:
    """Try to fetch rollout data from the Lightning Server."""
    if not sh.lightning_ok:
        return False
    import requests as _req
    try:
        r = _req.get(f"{server_url}/api/rollouts?limit=1000", timeout=5)
        if r.status_code != 200:
            return False
        recs = r.json().get("rollouts", [])
    except Exception:
        return False

    if not recs:
        return False

    import pandas as pd
    df = pd.DataFrame([{
        "policy_version": r.get("policy_version", "unknown"),
        "reward":         float(r.get("final_reward") or 0.0),
        "task_id":        r.get("task_id", ""),
    } for r in recs])

    if df["policy_version"].nunique() < 2:
        st.info(
            f"{len(recs)} rollout(s) found, but only one policy version.  "
            "Train a new checkpoint with Track B to see a comparison."
        )
        _show_single_policy_summary(df)
        return True

    _show_comparison(df)
    return True


def _show_single_policy_summary(df) -> None:
    m = df["reward"].mean()
    s = df["reward"].std()
    st.metric("Mean reward", f"{m:.4f}", delta=f"±{s:.4f} std")
    with st.expander("Reward distribution"):
        try:
            import altair as alt
            chart = alt.Chart(df).mark_bar().encode(
                alt.X("reward:Q", bin=alt.Bin(maxbins=20)), alt.Y("count()"),
            ).properties(height=180)
            st.altair_chart(chart, use_container_width=True)
        except ImportError:
            st.dataframe(df[["reward"]].describe())


def _show_comparison(df) -> None:
    import pandas as pd
    summary = (
        df.groupby("policy_version")["reward"]
        .agg(mean="mean", std="std", count="count")
        .reset_index()
        .sort_values("mean")
    )

    policies = summary["policy_version"].tolist()
    base_mean  = float(summary.iloc[0]["mean"])
    tuned_mean = float(summary.iloc[-1]["mean"])
    win_count  = sum(1 for r in df["reward"] if r > base_mean)
    win_rate   = 100 * win_count / len(df)

    st.subheader("Before vs After")
    c1, c2, c3 = st.columns(3)
    c1.metric("Baseline mean reward",  f"{base_mean:.4f}")
    c2.metric("Tuned mean reward",     f"{tuned_mean:.4f}", delta=f"{tuned_mean - base_mean:+.4f}")
    c3.metric("Win rate",              f"{win_rate:.1f}%")

    # Per-policy breakdown table
    st.markdown("### Per-Policy Breakdown")
    st.dataframe(summary.rename(columns={"policy_version": "Policy"}), use_container_width=True, hide_index=True)

    # Side-by-side histogram
    try:
        import altair as alt
        chart = (
            alt.Chart(df)
            .mark_bar(opacity=0.7)
            .encode(
                alt.X("reward:Q", bin=alt.Bin(maxbins=25), title="Reward"),
                alt.Y("count()"),
                alt.Color("policy_version:N", title="Policy"),
                alt.Column("policy_version:N"),
            )
            .properties(height=200)
        )
        st.altair_chart(chart)
    except ImportError:
        pass


def _try_local_reports() -> bool:
    """Fall back to pre-rendered markdown reports."""
    report_files = [
        ROOT / "reports" / "tracka_initial_vs_baseline_vs_trackb_redeploy.md",
        ROOT / "reports" / "tracka_initial_benchmark.md",
        ROOT / "reports" / "e2e_tracka_vs_trackb_adapter_first4_20260604_053507.md",
        ROOT / "reports" / "e2e_tracka_baseline_vs_trackb_full20_20260604_053507.md",
        ROOT / "reports" / "e2e_trackb_adapter_20260604_053507.md",
        ROOT / "reports" / "e2e_tracka_baseline_20260604_052534.md",
        ROOT / "reports" / "baseline_vs_rl_tuned.md",
        ROOT / "reports" / "baseline_v2_tuned_benchmark.md",
    ]
    found = [p for p in report_files if p.exists()]
    if not found:
        return False
    for p in found:
        with st.expander(p.name, expanded=True):
            st.markdown(p.read_text())
    return True


def _show_trajectory_traces() -> None:
    """Render committed or freshly generated multi-agent trajectory traces."""
    trajectory_root = ROOT / "trajectories"
    if not trajectory_root.exists():
        return

    trace_files = sorted(
        trajectory_root.glob("**/*.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not trace_files:
        return

    st.markdown("### Multi-Agent Trajectory Traces")
    st.caption("These JSONL traces are the Track A evidence used for scoring, grouping, and Track B policy training.")

    labels = [str(p.relative_to(ROOT)) for p in trace_files[:50]]
    chosen = st.selectbox("Trajectory file", labels, key="results_trajectory_file")
    path = ROOT / chosen

    records = []
    for line in path.read_text(errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except Exception:
            continue
    if not records:
        st.warning("Selected trajectory file did not contain parseable JSONL records.")
        return

    record = records[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Task", str(record.get("task_id", "unknown"))[:28])
    c2.metric("Policy", str(record.get("policy_version", "unknown"))[:28])
    c3.metric("Reward", f"{float(record.get('reward') or 0.0):.4f}")
    c4.metric("Status", record.get("final_status", "unknown"))

    steps = record.get("steps", []) or []
    if steps:
        import pandas as pd

        rows = []
        for idx, step in enumerate(steps):
            tool_calls = step.get("tool_calls", []) or []
            rows.append({
                "step": idx,
                "agent": step.get("agent_name"),
                "action": step.get("action"),
                "tool_calls": len(tool_calls),
                "failed_tools": sum(1 for call in tool_calls if call.get("status") not in {"success", "ok", "completed"}),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with st.expander("Raw trajectory JSON", expanded=False):
        st.json(record)


def _show_recent_logs() -> None:
    log_roots = [ROOT / "reports" / "run_logs", ROOT / "reports" / "service_logs"]
    log_files = sorted(
        [p for root in log_roots if root.exists() for p in root.glob("*.log")],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not log_files:
        return
    with st.expander("Run and Service Logs", expanded=True):
        labels = [str(p.relative_to(ROOT)) for p in log_files[:20]]
        chosen = st.selectbox("Log file", labels, key="results_log_file")
        lines = (ROOT / chosen).read_text(errors="replace").splitlines()
        st.text("\n".join(lines[-140:]))


def _show_empty_state() -> None:
    st.info(
        "No comparison data available yet.  \n\n"
        "**Required steps:**  \n"
        "1. Run Track A to collect rollouts  \n"
        "2. Run Track B to train a new policy  \n"
        "3. Run inference again with the new checkpoint  \n"
        "4. Return here to see the before/after comparison"
    )
    with st.expander("Comparison script"):
        st.code(
            "# Run baseline inference\n"
            "bash scripts/gpu/start_baseline_inference.sh\n\n"
            "# Train policy (Track B)\n"
            "TRAINER=trl_grpo REWARD_MODE=hybrid bash scripts/gpu/run_02_train_policy_qlora_grpo.sh\n\n"
            "# Run tuned inference\n"
            "bash scripts/gpu/start_tuned_inference.sh",
            language="bash",
        )


# Alignment note inserted for presentation mode
def _render_pdf_alignment_note() -> None:
    import streamlit as st
    st.markdown('### RULER / Stability Results')
    st.markdown('''PDF alignment: ART-like trajectories, grouped rollouts, GRPO, MCP tool loop partial, fail-closed vLLM-backed RULER judge, checkpoint forking metadata, behavioral policy drift, and GSPO roadmap only. Official ART/RULER is available only when openpipe-art and official RULER are installed.''')
