from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.config.ruler_config import RulerJudgeConfig
from src.rewards.ruler_vllm_judge import score_group_with_vllm_judge

SUPPORTED_RULER_MODE = "vllm_judge"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _extract_trajectories(group: dict[str, Any]) -> list[dict[str, Any]]:
    rows = group.get("ranked_trajectories") or group.get("trajectories") or group.get("judged_trajectories") or []
    return [dict(r) for r in rows]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _hard_gate(trajectory: dict[str, Any]) -> float:
    meta = trajectory.get("reward_metadata") or trajectory.get("metrics") or {}
    status = trajectory.get("final_status")
    if status not in (None, "success", "completed", "ok"):
        return 0.0
    if _safe_float(meta.get("R_violation"), 0.0) < 0:
        return 0.0
    return 1.0


def _apply_judgments(trajectories: list[dict[str, Any]], judgment_rows: list[dict[str, Any]], alpha: float, beta: float) -> list[dict[str, Any]]:
    by_id = {str(j.get("trajectory_id")): j for j in judgment_rows}
    expected_ids = {str(traj.get("trajectory_id") or f"trajectory_{idx}") for idx, traj in enumerate(trajectories)}
    missing_ids = expected_ids - set(by_id)
    extra_ids = set(by_id) - expected_ids
    if missing_ids or extra_ids:
        raise RuntimeError(
            "vLLM RULER judge response did not exactly cover the input trajectory set; "
            f"missing={sorted(missing_ids)}, extra={sorted(extra_ids)}"
        )

    judged: list[dict[str, Any]] = []
    for idx, traj in enumerate(trajectories):
        tid = str(traj.get("trajectory_id") or f"trajectory_{idx}")
        j = by_id[tid]
        original_reward = _safe_float(traj.get("reward") or traj.get("original_reward"))
        ruler_relative = _safe_float(j.get("relative_score"))
        final_reward = j.get("final_reward")
        if final_reward is None:
            final_reward = _hard_gate(traj) * (
                alpha * max(0.0, min(1.0, original_reward))
                + beta * max(0.0, min(1.0, ruler_relative))
            )
        row = dict(traj)
        row.update({
            "trajectory_id": tid,
            "original_reward": original_reward,
            "ruler_rank": int(_safe_float(j.get("rank"), 999999)),
            "ruler_relative_score": round(max(0.0, min(1.0, ruler_relative)), 6),
            "final_hybrid_reward": round(float(final_reward), 6),
            "ruler_reason": str(j.get("reason") or "vLLM judge ranking"),
            "ruler_fallback_used": False,
            "reward": round(float(final_reward), 6),
        })
        judged.append(row)
    judged.sort(key=lambda r: (r.get("ruler_rank") or 999999))
    return judged


def score_groups(
    input_path: Path,
    output_path: Path,
    mode: str,
    judge_model: str | None,
    alpha: float,
    beta: float,
    judge_base_url: str | None = None,
    judge_api_key: str | None = None,
) -> list[dict[str, Any]]:
    if mode != SUPPORTED_RULER_MODE:
        raise ValueError(f"RULER scoring is fail-closed and supports only mode={SUPPORTED_RULER_MODE!r}; got {mode!r}")

    cfg = RulerJudgeConfig()
    effective_model = judge_model or cfg.judge_model
    effective_base_url = judge_base_url or cfg.judge_base_url
    effective_api_key = judge_api_key or cfg.judge_api_key
    if not effective_base_url:
        raise ValueError("RULER_JUDGE_BASE_URL / --judge-base-url is required for vLLM RULER scoring")

    groups = _read_jsonl(input_path)
    out_rows: list[dict[str, Any]] = []
    for group in groups:
        task_id = str(group.get("task_id") or "")
        trajectories = _extract_trajectories(group)
        judgment_rows = score_group_with_vllm_judge(
            task_id=task_id,
            trajectories=trajectories,
            judge_model=effective_model,
            judge_base_url=effective_base_url,
            judge_api_key=effective_api_key,
        )
        judged = _apply_judgments(trajectories, judgment_rows, alpha, beta)
        out_rows.append({
            "task_id": task_id,
            "group_size": len(judged),
            "ruler_mode": SUPPORTED_RULER_MODE,
            "judge_model": effective_model,
            "judge_base_url": effective_base_url,
            "alpha": alpha,
            "beta": beta,
            "judged_trajectories": judged,
        })
    _write_jsonl(output_path, out_rows)
    return out_rows


def write_summary(rows: list[dict[str, Any]], report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    group_count = len(rows)
    traj_count = sum(len(r.get("judged_trajectories") or []) for r in rows)
    best = [next(iter(r.get("judged_trajectories") or []), {}) for r in rows]
    avg_best = sum(float(r.get("final_hybrid_reward") or 0.0) for r in best) / max(len(best), 1)
    fallback_count = sum(1 for row in rows for t in (row.get("judged_trajectories") or []) if t.get("ruler_fallback_used"))
    lines = [
        "# RULER vLLM Scoring Summary",
        "",
        "This report describes a fail-closed grouped trajectory scoring pass. The scorer requires a reachable OpenAI-compatible local vLLM judge endpoint and does not provide heuristic or deterministic ranking fallbacks.",
        "",
        f"- Groups scored: {group_count}",
        f"- Trajectories judged: {traj_count}",
        f"- Mean best final hybrid reward: {avg_best:.4f}",
        f"- Fallback-ranked trajectories: {fallback_count}",
        "",
        "| Task ID | Mode | Group Size | Best Trajectory | Best Hybrid Reward | Reason |",
        "|---|---|---:|---|---:|---|",
    ]
    for row in rows[:50]:
        top = next(iter(row.get("judged_trajectories") or []), {})
        reason = str(top.get("ruler_reason") or "").replace("|", "/")[:160]
        lines.append(
            f"| {row.get('task_id')} | {row.get('ruler_mode')} | {row.get('group_size')} | "
            f"{top.get('trajectory_id', '')} | {float(top.get('final_hybrid_reward') or 0.0):.4f} | {reason} |"
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply fail-closed vLLM-backed RULER relative scoring to grouped rollouts.")
    parser.add_argument("--input", default="data/grpo/grouped_rollouts.jsonl")
    parser.add_argument("--output", default="data/grpo/ruler_scored_groups.jsonl")
    parser.add_argument("--mode", choices=[SUPPORTED_RULER_MODE], default=SUPPORTED_RULER_MODE)
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--judge-base-url", default=None)
    parser.add_argument("--judge-api-key", default=None)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--report", default="reports/ruler_scoring_summary.md")
    args = parser.parse_args()
    rows = score_groups(Path(args.input), Path(args.output), args.mode, args.judge_model, args.alpha, args.beta, args.judge_base_url, args.judge_api_key)
    write_summary(rows, Path(args.report))
    print(f"Wrote {len(rows)} RULER-scored groups to {args.output}")
    print(f"Wrote summary to {args.report}")


if __name__ == "__main__":
    main()
