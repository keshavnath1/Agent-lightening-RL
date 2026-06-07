from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _count_jsonl(path: Path) -> int:
    try:
        return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    except Exception:
        return 0


def build_report(catalog_path: Path, scenarios_path: Path, evidence_path: Path, output: Path) -> None:
    catalog = _read_json(catalog_path, {})
    evidence = _read_json(evidence_path, {}) if evidence_path.exists() else {}
    scenario_count = _count_jsonl(scenarios_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Tool Learning Flywheel",
        "",
        "The tool-learning flywheel makes the MCP/tool layer visible above Track A rollouts and Track B optimization. Tools are discovered, converted into scenarios, executed as rollouts, evaluated with explicit tool metrics, ranked within same-scenario groups, and converted into reward-bearing training data.",
        "",
        "| Stage | Evidence |",
        "|---|---:|",
        f"| MCP tools discovered | {catalog.get('tool_count', evidence.get('mcp_tool_count', 0))} |",
        f"| Scenarios generated | {scenario_count or evidence.get('scenario_count', 0)} |",
        f"| Rollouts collected | {evidence.get('trajectory_count', 0)} |",
        f"| Tool calls captured | {evidence.get('tool_call_count', 0)} |",
        f"| RULER-ranked groups | {evidence.get('ruler_group_count', evidence.get('group_count', 0))} |",
        f"| Reward range | {evidence.get('reward_min', '-') } – {evidence.get('reward_max', '-')} |",
        "",
        "## Recommended demo wording",
        "",
        "> The tool-learning flywheel starts by discovering tools from the MCP server. Those tool schemas are used to generate training scenarios such as single-tool tasks, multi-tool ML workflows, error-recovery cases, and guardrail-sensitive tasks. The agent runs multiple rollouts per scenario, and each rollout captures the complete trajectory of tool choices, arguments, results, errors, artifacts, and final outcome. We then score tool behavior using explicit tool-use metrics and RULER-style relative ranking. Track B fine-tunes the policy so future rollouts choose better tools, pass better arguments, recover from failures, and avoid unsafe behavior.",
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the dashboard-ready tool-learning flywheel report.")
    parser.add_argument("--catalog", default="data/mcp/tool_catalog.json")
    parser.add_argument("--scenarios", default="data/synthetic/tool_scenarios.jsonl")
    parser.add_argument("--evidence", default="")
    parser.add_argument("--output", default="reports/tool_learning_flywheel.md")
    args = parser.parse_args()
    evidence_path = Path(args.evidence) if args.evidence else Path("reports/__missing__.json")
    build_report(Path(args.catalog), Path(args.scenarios), evidence_path, Path(args.output))
    print(f"Wrote tool-learning flywheel report to {args.output}")


if __name__ == "__main__":
    main()
