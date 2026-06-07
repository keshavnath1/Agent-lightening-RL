from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
TRAJ_DIR = ROOT / "trajectories" / "mcp_boundary_four_rollout_20260606_052406"
GROUP_PATH = ROOT / "data" / "grpo" / "mcp_boundary_four_rollout_grouped_20260606_052406.jsonl"
SCORED_PATH = ROOT / "data" / "grpo" / "mcp_boundary_four_rollout_ruler_scored_20260606_052406.jsonl"
REPORT_PATH = ROOT / "reports" / "mcp_boundary_four_rollout_ruler_summary_20260606_052406.md"
MANIFEST_PATH = ROOT / "reports" / "mcp_boundary_four_rollout_manifest_20260606_052406.json"

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

def load_trajectories() -> list[dict[str, Any]]:
    rows = []
    for p in sorted(TRAJ_DIR.glob("*.jsonl")):
        items = read_jsonl(p)
        if items:
            row = items[0]
            row["_file"] = str(p.relative_to(ROOT))
            rows.append(row)
    return rows

def boundary_status(row: dict[str, Any]) -> str:
    text = json.dumps(row, ensure_ascii=False).lower()
    if "raw_rows_exposed_to_llm\": true" in text or "raw_rows_loaded_into_agent_context\": true" in text:
        return "violation"
    if "raw_rows_exposed_to_llm\": false" in text and "registryartifactmaterializer" in text:
        return "passed"
    return "needs review"

st.set_page_config(page_title="Task Rollout Explorer", layout="wide")
st.title("Task Rollout Explorer: PostgreSQL Registry Four-Rollout Run")
st.caption("Single OpenML task, four policy rollouts, GRPO grouping, and RULER-style scoring with raw-row boundary evidence.")

trajectories = load_trajectories()
groups = read_jsonl(GROUP_PATH)
scored = read_jsonl(SCORED_PATH)
manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8")) if MANIFEST_PATH.exists() else {}

metric_cols = st.columns(5)
metric_cols[0].metric("Task", manifest.get("task_id", "openml_31_german_credit"))
metric_cols[1].metric("Trajectories", len(trajectories))
metric_cols[2].metric("GRPO Groups", len(groups))
metric_cols[3].metric("Group Size", groups[0].get("group_size") if groups else "NA")
metric_cols[4].metric("Boundary", manifest.get("raw_row_boundary", "unknown"))

rows = []
for row in trajectories:
    rows.append({
        "trajectory_id": row.get("trajectory_id"),
        "policy_version": row.get("policy_version"),
        "task_id": row.get("task_id"),
        "final_status": row.get("final_status"),
        "reward": row.get("reward", 0.0),
        "boundary_status": boundary_status(row),
        "file": row.get("_file"),
    })
summary_df = pd.DataFrame(rows)
st.subheader("Rollout Summary")
st.dataframe(summary_df, use_container_width=True, hide_index=True)

st.subheader("GRPO Ranked Rollouts")
if groups:
    ranked_rows = []
    for rank, item in enumerate(groups[0].get("ranked_trajectories", []), start=1):
        ranked_rows.append({"rank": rank, "trajectory_id": item.get("trajectory_id"), "policy_version": item.get("policy_version"), "reward": item.get("reward", 0.0)})
    st.dataframe(pd.DataFrame(ranked_rows), use_container_width=True, hide_index=True)
else:
    st.warning("No grouped rollout artifact found.")

st.subheader("RULER Scoring")
if scored:
    st.json(scored[0], expanded=False)
else:
    st.warning("No RULER-scored artifact found.")

st.subheader("Evidence Manifest")
st.json(manifest, expanded=False)
if REPORT_PATH.exists():
    st.subheader("RULER Summary Report")
    st.markdown(REPORT_PATH.read_text(encoding="utf-8"))
