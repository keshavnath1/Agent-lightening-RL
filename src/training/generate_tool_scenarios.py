from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.tools.mcp_discovery import discover_tools

SCENARIO_TYPES = [
    "single_tool",
    "multi_tool_pipeline",
    "invalid_input_recovery",
    "missing_artifact_recovery",
    "tracking_required",
    "sandbox_required",
    "guardrail_edge_case",
    "error_recovery",
]

DIFFICULTY_BY_TYPE = {
    "single_tool": "easy",
    "multi_tool_pipeline": "medium",
    "invalid_input_recovery": "hard",
    "missing_artifact_recovery": "hard",
    "tracking_required": "medium",
    "sandbox_required": "medium",
    "guardrail_edge_case": "hard",
    "error_recovery": "hard",
}


def _tool_name(tool: dict[str, Any]) -> str:
    return str(tool.get("name") or "UnknownTool")


def _pipeline_tools(tools: list[dict[str, Any]]) -> list[str]:
    preferred = ["YDataProfiler", "DockerizedCodeInterpreter", "GBMBenchmark", "MLflowTracking", "ReviewerCritic"]
    names = [_tool_name(t) for t in tools]
    selected = [name for name in preferred if name in names]
    return selected[:5] if selected else names[: min(3, len(names))]


def generate_scenarios(tools: list[dict[str, Any]], num_scenarios: int) -> list[dict[str, Any]]:
    if not tools:
        tools = discover_tools(None)
    rows: list[dict[str, Any]] = []
    names = [_tool_name(t) for t in tools]
    for i in range(num_scenarios):
        scenario_type = SCENARIO_TYPES[i % len(SCENARIO_TYPES)]
        primary = tools[i % len(tools)]
        required = [_tool_name(primary)]
        optional = [n for n in names if n not in required][:2]
        forbidden = ["RawDataPrinter", "UnsupportedShellEscape"]
        failure_modes: list[dict[str, str]] = []
        expected_artifacts = list(primary.get("expected_outputs") or [])
        guardrails = ["no raw row leakage", "no unsupported tools", "schema-valid tool arguments"]
        if scenario_type == "multi_tool_pipeline":
            required = _pipeline_tools(tools)
            expected_artifacts = ["profile_summary.json", "preprocessing_config.json", "benchmark_metrics.json", "tracking_record"]
        elif scenario_type == "invalid_input_recovery":
            failure_modes.append({"tool_name": required[0], "failure_mode": "invalid_argument_type"})
        elif scenario_type == "missing_artifact_recovery":
            failure_modes.append({"tool_name": required[0], "failure_mode": "missing_artifact"})
            expected_artifacts.append("recovery_note.json")
        elif scenario_type == "tracking_required":
            required = list(dict.fromkeys(required + ["MLflowTracking"]))
            expected_artifacts.append("tracking_record")
        elif scenario_type == "sandbox_required":
            required = list(dict.fromkeys(required + ["DockerizedCodeInterpreter"]))
            expected_artifacts.append("execution_log.txt")
        elif scenario_type == "guardrail_edge_case":
            required = list(dict.fromkeys(required + ["ReviewerCritic"]))
            guardrails.append("redact sensitive values before reporting")
        elif scenario_type == "error_recovery":
            failure_modes.append({"tool_name": required[0], "failure_mode": "transient_tool_failure"})
            guardrails.append("do not repeat the same invalid call")
        rows.append({
            "scenario_id": f"tool_scenario_{i + 1:03d}",
            "task_id": f"tool_scenario_{i + 1:03d}",
            "task_description": f"Complete a {scenario_type} tool-learning workflow using {', '.join(required)} while respecting guardrails.",
            "scenario_type": scenario_type,
            "difficulty": DIFFICULTY_BY_TYPE[scenario_type],
            "required_tools": required,
            "optional_tools": optional,
            "forbidden_tools": forbidden,
            "success_criteria": [
                "required tool coverage is satisfied",
                "tool arguments validate against schema",
                "expected artifacts are produced or a recovery artifact explains the failure",
                "no raw data is exposed",
            ],
            "failure_modes": failure_modes,
            "expected_artifacts": sorted(set(expected_artifacts)),
            "guardrails": guardrails,
        })
    return rows


def write_summary(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[tuple[str, str], int] = {}
    for row in rows:
        key = (row.get("difficulty", "unknown"), row.get("scenario_type", "unknown"))
        counts[key] = counts.get(key, 0) + 1
    lines = ["# Tool Scenario Curriculum", "", "| Difficulty | Scenario Type | Count |", "|---|---|---:|"]
    for (difficulty, typ), count in sorted(counts.items()):
        lines.append(f"| {difficulty} | {typ} | {count} |")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate MCP/tool-schema-driven training scenarios.")
    parser.add_argument("--mcp-url", default=None)
    parser.add_argument("--output", default="data/synthetic/tool_scenarios.jsonl")
    parser.add_argument("--report", default="reports/tool_scenario_summary.md")
    parser.add_argument("--num-scenarios", type=int, default=32)
    args = parser.parse_args()
    rows = generate_scenarios(discover_tools(args.mcp_url), args.num_scenarios)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    write_summary(rows, Path(args.report))
    print(f"Wrote {len(rows)} tool scenarios to {out}")
    print(f"Wrote scenario summary to {args.report}")


if __name__ == "__main__":
    main()
