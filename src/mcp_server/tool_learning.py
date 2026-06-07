from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[3]


def _json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except Exception:
        return []


def _latest(pattern: str) -> Path | None:
    files = sorted(ROOT.glob(pattern))
    return files[-1] if files else None


def render() -> None:
    st.title("Tool Learning Flywheel")
    st.caption("MCP tool discovery → tool scenarios → agent rollouts → tool metrics → RULER ranking → policy improvement.")

    catalog = _json(ROOT / "data/mcp/tool_catalog.json", {"tools": []})
    scenario_rows = _jsonl(ROOT / "data/synthetic/tool_scenarios.jsonl") or _jsonl(ROOT / "data/synthetic/section5_tool_scenarios.jsonl")
    evidence_path = _latest("reports/section5_flywheel_evidence_*.json")
    evidence = _json(evidence_path, {}) if evidence_path else {}

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("MCP tools discovered", catalog.get("tool_count", evidence.get("mcp_tool_count", 0)))
    c2.metric("Scenarios generated", len(scenario_rows) or evidence.get("scenario_count", 0))
    c3.metric("Rollouts collected", evidence.get("trajectory_count", 0))
    c4.metric("Tool calls captured", evidence.get("tool_call_count", 0))
    c5.metric("RULER-ranked groups", evidence.get("ruler_group_count", evidence.get("group_count", 0)))

    st.subheader("A. Tool catalog")
    tools = catalog.get("tools", [])
    if tools:
        st.dataframe(pd.DataFrame([{
            "Tool": t.get("name"),
            "Category": t.get("category"),
            "Required Inputs": ", ".join(t.get("required_inputs", [])),
            "Expected Outputs": ", ".join(t.get("expected_outputs", [])),
            "Risk": t.get("risk_level"),
        } for t in tools]), use_container_width=True)
    else:
        st.info("No tool catalog has been exported yet. Run `python -m src.tools.mcp_discovery`.")

    st.subheader("B. Scenario curriculum")
    if scenario_rows:
        counts = Counter((r.get("difficulty", "unknown"), r.get("scenario_type", "unknown")) for r in scenario_rows)
        st.dataframe(pd.DataFrame([{"Difficulty": k[0], "Scenario Type": k[1], "Count": v} for k, v in sorted(counts.items())]), use_container_width=True)
    else:
        st.info("No tool scenarios found. Run `python -m src.training.generate_tool_scenarios`.")

    st.subheader("C. Tool rollout results")
    if evidence:
        st.dataframe(pd.DataFrame([{
            "Scenario/Rollout Dir": evidence.get("raw_rollout_dir"),
            "Used Tools": ", ".join(evidence.get("unique_tool_calls", [])),
            "Success Statuses": ", ".join(evidence.get("tool_call_statuses", [])),
            "Reward Min": evidence.get("reward_min"),
            "Reward Max": evidence.get("reward_max"),
        }]), use_container_width=True)
    else:
        st.info("No rollout evidence file found yet.")

    st.subheader("D. RULER ranking")
    ruler_report = ROOT / evidence.get("ruler_report", "") if evidence.get("ruler_report") else _latest("reports/*tool_ruler*.md")
    if ruler_report and ruler_report.exists():
        st.markdown(ruler_report.read_text(encoding="utf-8"))
    else:
        st.info("No tool RULER report found yet.")

    st.subheader("E. Tool improvement")
    tool_metrics = ROOT / evidence.get("tool_metrics_report", "") if evidence.get("tool_metrics_report") else _latest("reports/*tool_use_metrics*.md")
    flywheel = ROOT / evidence.get("flywheel_report", "") if evidence.get("flywheel_report") else _latest("reports/*flywheel*.md")
    if tool_metrics and tool_metrics.exists():
        st.markdown(tool_metrics.read_text(encoding="utf-8"))
    if flywheel and flywheel.exists():
        st.markdown(flywheel.read_text(encoding="utf-8"))

    st.subheader("F. Official Project MCP Server")
    raw_sql_enabled = os.getenv("ALLOW_RAW_SQL_TOOL", "0") == "1"
    official_log = ROOT / "reports" / "mcp_official_tool_calls.jsonl"
    st.markdown(
        """
The **Official Project MCP Server** is the ML-agent tool-learning interface. It is separate from any RunPod MCP integration: RunPod MCP should be treated as infrastructure control, while Project MCP exposes safe, schema-driven project tools for rollouts, RULER judgment, and TRL GRPO policy improvement.

| Server | Role | Data policy |
|---|---|---|
| RunPod MCP | Infrastructure lifecycle and pod operations only | Do not use for the core learning loop |
| Official Project MCP | Safe task metadata, dataset summaries, column/target profiles, rollout status, and reward history | No raw dataset rows through ordinary tools |

The official server wraps `postgres_get_task_metadata`, `postgres_get_dataset_schema`, `postgres_get_dataset_summary`, `postgres_get_column_profile`, `postgres_get_target_profile`, `postgres_get_rollout_status`, and `postgres_get_reward_history`. The execution-only `postgres_get_execution_dataset_source` contract is reserved for Docker/materializer use. The developer-only `postgres_debug_readonly_sql` tool remains disabled unless `ALLOW_RAW_SQL_TOOL=1`, and SQL safety still blocks `SELECT *`, comments, semicolons, blocked raw-row schemas, and write operations.
"""
    )
    st.code("bash scripts/cpu/start_official_mcp_server.sh", language="bash")
    st.code(
        '{\n  "mcpServers": {\n    "self-improving-ml-agent": {\n      "command": "python",\n      "args": ["-m", "src.mcp_official.server"],\n      "env": {"DATABASE_URL": "postgresql+psycopg2://USER:PASSWORD@HOST:PORT/DBNAME", "ALLOW_RAW_SQL_TOOL": "0"}\n    }\n  }\n}',
        language="json",
    )
    st.markdown(f"**Official MCP log:** `reports/mcp_official_tool_calls.jsonl` — present: `{official_log.exists()}`")
    st.markdown(f"**Raw SQL status:** `{'enabled' if raw_sql_enabled else 'disabled'}`")
