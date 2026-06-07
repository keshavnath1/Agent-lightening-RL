"""RULER-style grouped trajectory scoring through a local vLLM judge.

This module intentionally sends only compact trajectory metadata to the judge.
It does not include raw dataset rows, dataframe previews, or file contents.
When the judge endpoint is unavailable or returns invalid JSON, scoring fails closed.
No deterministic or heuristic fallback ranking is allowed in strict mode.
"""
from __future__ import annotations

from typing import Any
import json
import math
import re

from src.config.ruler_config import RulerJudgeConfig

_FORBIDDEN_RAW_MARKERS = (
    "raw rows:",
    "df.head(",
    "dataframe head",
    "first 5 rows",
    "to_string(",
    "read_csv",
)


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _normalise(values: list[float]) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if math.isclose(lo, hi):
        return [1.0 if v > 0 else 0.0 for v in values]
    return [round((v - lo) / (hi - lo), 6) for v in values]


def _reward_metadata(traj: dict[str, Any]) -> dict[str, Any]:
    meta = traj.get("reward_metadata") or traj.get("metrics") or {}
    if not isinstance(meta, dict):
        return {}
    allowed = {
        "reward", "final_reward", "R_data", "R_model", "R_tracking", "R_sandbox",
        "R_reproducibility", "R_violation", "tool_validity", "model_score",
        "tracking_score", "sandbox_score", "guardrail_score",
    }
    return {str(k): v for k, v in meta.items() if str(k) in allowed or str(k).startswith("R_")}


def _tool_summary(traj: dict[str, Any]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for step in (traj.get("steps") or [])[:12]:
        calls = step.get("tool_calls") or []
        if not isinstance(calls, list):
            calls = []
        summary.append({
            "agent_name": step.get("agent_name"),
            "action": step.get("action"),
            "tool_names": [c.get("tool_name") for c in calls if isinstance(c, dict)],
            "tool_statuses": [c.get("status") for c in calls if isinstance(c, dict)],
            "error_count": sum(1 for c in calls if isinstance(c, dict) and c.get("error")),
        })
    return summary


def compact_trajectory_for_judge(traj: dict[str, Any]) -> dict[str, Any]:
    """Return judge-safe trajectory metadata without raw dataset content."""

    artifacts = traj.get("artifacts") or traj.get("final_artifacts") or {}
    if isinstance(artifacts, dict):
        artifact_names = sorted(str(k) for k in artifacts.keys())[:50]
    elif isinstance(artifacts, list):
        artifact_names = [str(x) for x in artifacts[:50]]
    else:
        artifact_names = []

    compact = {
        "trajectory_id": traj.get("trajectory_id"),
        "final_status": traj.get("final_status"),
        "reward": traj.get("reward"),
        "original_reward": traj.get("original_reward"),
        "reward_components": _reward_metadata(traj),
        "tool_calls_summary": _tool_summary(traj),
        "artifact_names": artifact_names,
        "errors": traj.get("errors") or traj.get("error_summary") or [],
        "guardrail_flags": traj.get("guardrail_flags") or traj.get("violations") or [],
        "model_metrics": traj.get("model_metrics") or {},
        "tracking_status": traj.get("tracking_status") or traj.get("mlflow_status"),
        "sandbox_status": traj.get("sandbox_status"),
    }
    text = json.dumps(compact, ensure_ascii=False, default=str).lower()
    if any(marker in text for marker in _FORBIDDEN_RAW_MARKERS):
        compact["privacy_filter_warning"] = "Forbidden raw-data marker removed from judge payload."
        scrubbed = re.sub("|".join(re.escape(m) for m in _FORBIDDEN_RAW_MARKERS), "[filtered]", text)
        compact["privacy_filter_digest"] = scrubbed[:240]
    return compact


def _parse_json_array(content: str) -> list[dict[str, Any]]:
    content = (content or "").strip()
    try:
        parsed = json.loads(content)
    except Exception:
        match = re.search(r"\[[\s\S]*\]", content)
        if not match:
            raise
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, list):
        raise ValueError("Judge response must be a JSON list")
    rows: list[dict[str, Any]] = []
    for item in parsed:
        if isinstance(item, dict) and item.get("trajectory_id") is not None:
            rows.append(item)
    if not rows:
        raise ValueError("Judge response did not contain trajectory rows")
    return rows


def _judge_failure(task_id: str, reason: str) -> RuntimeError:
    return RuntimeError(f'vLLM RULER judge failed for task_id={task_id}; no fallback ranking is allowed: {reason}')


def score_group_with_vllm_judge(
    task_id: str,
    trajectories: list[dict[str, Any]],
    judge_model: str | None = None,
    judge_base_url: str | None = None,
    judge_api_key: str | None = None,
    timeout_seconds: int | None = None,
    max_tokens: int | None = None,
) -> list[dict[str, Any]]:
    """Score one trajectory group with a local OpenAI-compatible vLLM judge."""

    if not trajectories:
        return []

    cfg = RulerJudgeConfig()
    model = judge_model or cfg.judge_model
    base_url = judge_base_url or cfg.judge_base_url
    api_key = judge_api_key or cfg.judge_api_key
    timeout = timeout_seconds or cfg.timeout_seconds
    max_out = max_tokens or cfg.max_tokens

    compact = [compact_trajectory_for_judge(t) for t in trajectories]
    prompt = (
        "You are a RULER-style evaluator for a self-improving ML workflow. "
        "Rank trajectories for the same task by task success, artifact completeness, "
        "model quality, tracking quality, sandbox/tool correctness, reproducibility, "
        "guardrail compliance, and error recovery. Use only the compact metadata below. "
        "Return JSON only as a list of objects with trajectory_id, rank, relative_score, and reason.\n"
        f"TASK_ID: {task_id}\nCOMPACT_TRAJECTORIES_JSON:\n{json.dumps(compact, ensure_ascii=False, default=str)}"
    )

    try:
        from openai import OpenAI
        client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=max_out,
        )
        content = response.choices[0].message.content or "[]"
        rows = _parse_json_array(content)
    except Exception as exc:
        raise _judge_failure(task_id, f'{exc.__class__.__name__}: {exc}') from exc

    valid_ids = {str(t.get("trajectory_id") or f"{task_id}_{i}") for i, t in enumerate(trajectories)}
    cleaned: list[dict[str, Any]] = []
    for row in rows:
        tid = str(row.get("trajectory_id"))
        if tid not in valid_ids:
            continue
        cleaned.append({
            "trajectory_id": tid,
            "rank": int(_safe_float(row.get("rank"), 999999)),
            "relative_score": _clamp(_safe_float(row.get("relative_score"), 0.0)),
            "reason": str(row.get("reason") or "vLLM judge ranking"),
            "fallback_used": False,
        })
    seen = {r["trajectory_id"] for r in cleaned}
    if seen != valid_ids:
        missing = sorted(valid_ids - seen)
        raise _judge_failure(task_id, f'judge omitted trajectories: {missing}')
    cleaned.sort(key=lambda r: (int(r.get("rank") or 999999), -float(r.get("relative_score") or 0.0)))
    normalised_scores = _normalise([float(r.get("relative_score") or 0.0) for r in cleaned])
    for idx, row in enumerate(cleaned):
        row["rank"] = idx + 1
        row["relative_score"] = _clamp(normalised_scores[idx] if idx < len(normalised_scores) else float(row.get("relative_score") or 0.0))
    return cleaned

