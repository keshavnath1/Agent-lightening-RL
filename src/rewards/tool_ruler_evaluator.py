from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.tools.tool_schema_validator import validate_trajectory


def _iter_records(path: Path):
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield json.loads(line)


def _score_rollout(rollout: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    traj = rollout.get("trajectory") or rollout
    validation = validate_trajectory(traj)
    text = json.dumps(traj, default=str).lower()
    required = rollout.get("required_tools") or traj.get("required_tools") or []
    calls = []
    for step in traj.get("steps", []) or []:
        for call in step.get("tool_calls", []) or []:
            calls.append(call.get("tool_name", ""))
    coverage = 1.0 if not required else len({r for r in required if r in calls or any(r in c for c in calls)}) / max(len(required), 1)
    artifact_success = 1.0 if any(token in text for token in ["artifact", "profile_summary", "benchmark_metrics", "tracking_record", "champion"]) else 0.0
    lexical_guardrail = 0.0 if any(token in text for token in ["raw rows:", "df.head(", "first 5 rows", "synthetic_dataset_rows", "select *"]) else 1.0
    boundary_guardrail = 1.0 if validation.get("boundary_guardrail_passed") else 0.0
    mcp_registry_evidence = 1.0 if validation.get("mcp_registry_evidence") else 0.0
    recovery = 1.0 if any(token in text for token in ["recover", "retry", "fallback", "corrected"]) else 0.5
    efficiency = 1.0 if len(calls) <= max(len(required) + 4, 4) else max(0.0, 1.0 - (len(calls) - len(required) - 4) * 0.05)
    tool_score = (
        0.20 * coverage
        + 0.18 * validation["tool_call_success_rate"]
        + 0.15 * validation["argument_validity_rate"]
        + 0.15 * artifact_success
        + 0.12 * mcp_registry_evidence
        + 0.10 * boundary_guardrail
        + 0.05 * recovery
        + 0.05 * efficiency
        + float(validation.get("schema_penalty", 0.0))
    )
    final_reward = 0.38 * tool_score + 0.20 * artifact_success + 0.18 * boundary_guardrail + 0.12 * mcp_registry_evidence + 0.07 * lexical_guardrail + 0.05 * recovery
    metrics = {
        "required_tool_coverage": round(coverage, 4),
        "argument_validity_rate": round(validation["argument_validity_rate"], 4),
        "tool_call_success_rate": round(validation["tool_call_success_rate"], 4),
        "artifact_success": artifact_success,
        "guardrail": min(lexical_guardrail, boundary_guardrail),
        "lexical_guardrail": lexical_guardrail,
        "boundary_guardrail": boundary_guardrail,
        "mcp_registry_evidence": mcp_registry_evidence,
        "registry_metadata_tool_seen": bool(validation.get("registry_metadata_tool_seen")),
        "execution_layer_materialization_seen": bool(validation.get("execution_layer_materialization_seen")),
        "boundary_errors": validation.get("boundary_errors", []),
        "recovery": recovery,
        "efficiency": round(efficiency, 4),
        "tool_use_score": round(max(0.0, min(1.0, tool_score)), 4),
    }
    return round(max(0.0, min(1.0, final_reward)), 6), metrics


def rank_groups(input_path: Path) -> list[dict[str, Any]]:
    rows = list(_iter_records(input_path))
    output: list[dict[str, Any]] = []
    for group in rows:
        rollouts = group.get("rollouts") or group.get("trajectories") or group.get("ranked_trajectories") or []
        scored = []
        for rollout in rollouts:
            reward, metrics = _score_rollout(rollout)
            r = dict(rollout)
            r["tool_ruler_metrics"] = metrics
            r["tool_ruler_final_reward"] = reward
            scored.append(r)
        scored.sort(key=lambda r: r.get("tool_ruler_final_reward", 0), reverse=True)
        n = len(scored)
        for idx, rollout in enumerate(scored, start=1):
            rollout["tool_ruler_rank"] = idx
            rollout["tool_ruler_relative_score"] = round(1.0 - ((idx - 1) / max(n - 1, 1)), 6)
        row = dict(group)
        row["rollouts"] = scored
        row["tool_ruler_group_size"] = n
        row["tool_ruler_best_trajectory"] = scored[0].get("trajectory_id") or scored[0].get("id") if scored else None
        output.append(row)
    return output


def write_report(groups: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Tool RULER Evaluation", "", "| Scenario group | Group size | Best rollout | Worst rollout | Best reward | Why |", "|---|---:|---|---|---:|---|"]
    for group in groups:
        rollouts = group.get("rollouts", [])
        best = rollouts[0] if rollouts else {}
        worst = rollouts[-1] if rollouts else {}
        best_metrics = best.get("tool_ruler_metrics", {}) or {}
        why = "Best rollout has highest explicit MCP registry evidence, artifact evidence, boundary-guardrail compliance, and tool schema score."
        if best_metrics.get("boundary_errors"):
            why = "Best available rollout still has boundary errors; inspect metrics before promotion."
        lines.append(
            f"| {group.get('task_id') or group.get('scenario_id')} | {len(rollouts)} | {best.get('trajectory_id') or best.get('id','-')} | "
            f"{worst.get('trajectory_id') or worst.get('id','-')} | {best.get('tool_ruler_final_reward', 0):.4f} | {why} |"
        )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank same-scenario tool-use rollout groups with explicit MCP/data-boundary metrics.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="data/grpo/tool_ruler_scored_groups.jsonl")
    parser.add_argument("--report", default="reports/tool_ruler_evaluation.md")
    args = parser.parse_args()
    groups = rank_groups(Path(args.input))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(g) + "\n" for g in groups), encoding="utf-8")
    write_report(groups, Path(args.report))
    print(f"Wrote {len(groups)} tool-RULER groups to {out}")
    print(f"Wrote tool-RULER report to {args.report}")


if __name__ == "__main__":
    main()
