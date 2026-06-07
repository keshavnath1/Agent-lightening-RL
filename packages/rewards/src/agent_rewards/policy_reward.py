"""
Policy reward functions for TRL GRPO training.

Three modes are exposed:

  json_validity_reward      – lightweight structural check (original behaviour)
  workflow_policy_reward    – workflow-aware semantic scoring
  hybrid_grpo_reward        – blend of both with group-relative normalisation

Each function has the signature expected by TRL GRPOTrainer:

    fn(prompts, completions, **kwargs) -> list[float]

or the simpler:

    fn(completions, **kwargs) -> list[float]

All scores are clamped to [0.0, 1.0].
"""
from __future__ import annotations

import json
import re
from typing import Any


# ── keyword lists ─────────────────────────────────────────────────────────────

_SCHEMA_KEYWORDS = [
    'schema', 'profile', 'profile_summary', 'data_quality', 'ydata', 'profiling',
    'feature', 'preprocessing', 'clean', 'missing',
]
_GBM_KEYWORDS = [
    'xgboost', 'lightgbm', 'histgradientboosting', 'gbm', 'gradient_boosting',
    'benchmark', 'champion', 'auc', 'roc_auc', 'accuracy', 'model_selection',
]
_MLFLOW_KEYWORDS = [
    'mlflow', 'tracking', 'experiment', 'run_id', 'log_metric', 'log_param',
    'register', 'artifact',
]
_SANDBOX_KEYWORDS = [
    'sandbox', 'docker', 'interpreter', 'code_interpreter', 'dockerized', 'container',
    'subprocess', 'isolated',
]
_REPRO_KEYWORDS = [
    'reproducibility', 'reproducible', 'seed', 'random_state', 'deterministic',
    'artifact_path', 'artifact_dir', 'output_ref',
]
_GUARDRAIL_KEYWORDS = [
    'guardrail', 'review', 'critic', 'reviewer', 'validation', 'check', 'safety',
    'policy', 'violation',
]
_CHAMPION_KEYWORDS = [
    'champion', 'champion_model', 'champion_artifact', 'register', 'model_registry',
    'best_model', 'model_path',
]

_RAW_DATA_MARKERS = [
    'dataframe head', 'raw rows:', 'first 5 rows', '.head(', 'to_string(',
    'df.head', 'print(df', 'display(df',
]
_FAKE_METRIC_MARKERS = [
    '"auc": 0.99', '"auc": 1.0', '"accuracy": 1.0', '"accuracy": 0.99',
    'fake_metric', 'placeholder_metric', 'hardcoded metric',
]
_UNSUPPORTED_TOOL_MARKERS = [
    'run_sql_query_on_raw_file', 'direct_db_dump', 'execute_shell_unrestricted',
    'read_raw_csv_bytes',
]


# ── helpers ───────────────────────────────────────────────────────────────────

def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _contains_any(text: str, keywords: list[str]) -> bool:
    low = text.lower()
    return any(kw in low for kw in keywords)


def _resolve_completions(
    prompts_or_completions: list[str] | None = None,
    completions: list[str] | None = None,
) -> list[str]:
    """Normalise the two TRL calling conventions into a plain completions list.

    TRL may call reward functions as:
        fn(prompts, completions, **kwargs)   # standard
        fn(completions, **kwargs)            # legacy positional
    """
    return completions if completions is not None else (prompts_or_completions or [])


def _parse_json_safe(text: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


# ── reward functions ──────────────────────────────────────────────────────────

def json_validity_reward(
    prompts_or_completions: list[str] | None = None,
    completions: list[str] | None = None,
    **_: Any,
) -> list[float]:
    """
    Original lightweight reward: checks structural JSON validity only.

    Accepts both TRL calling conventions:
        json_validity_reward(completions=[...])           # keyword
        json_validity_reward(prompts, completions, ...)   # positional (TRL standard)

    Scoring:
        0.20  base for valid JSON
        0.20  each: agent_name, action, reasoning_summary
        0.20  expected_tool_calls present (even if empty list)
    """
    completions = _resolve_completions(prompts_or_completions, completions)
    rewards: list[float] = []
    for completion in completions:
        parsed = _parse_json_safe(completion)
        if parsed is None:
            rewards.append(0.0)
            continue
        score = 0.20
        for key in ('agent_name', 'action', 'reasoning_summary'):
            if parsed.get(key):
                score += 0.20
        if parsed.get('expected_tool_calls') is not None:
            score += 0.20
        rewards.append(_clamp(score))
    return rewards


def workflow_policy_reward(
    prompts: list[str],
    completions: list[str],
    **_: Any,
) -> list[float]:
    """
    Workflow-aware reward that scores completion quality relative to the
    self-improving ML agent workflow.

    Positive contributions
    ----------------------
    format (0.55 total)
        0.15  valid JSON
        0.10  agent_name present
        0.10  action present
        0.10  reasoning_summary present
        0.10  expected_tool_calls present

    workflow quality (0.55 total)
        0.08  mentions schema / profiling / data quality
        0.08  mentions GBM modelling (XGBoost / LightGBM / HGB)
        0.08  mentions MLflow / experiment tracking
        0.08  mentions sandbox / Docker interpreter
        0.08  mentions reproducibility / seed / artifact paths
        0.08  mentions guardrails / review
        0.07  mentions champion model registration

    Penalties
    ---------
        -0.35  raw data row leakage
        -0.25  hardcoded / fake metrics detected
        -0.20  unsupported tool name

    Final score is clamped to [0.0, 1.0].
    """
    rewards: list[float] = []
    for prompt, completion in zip(prompts, completions):
        combined = (prompt + ' ' + completion).lower()
        parsed = _parse_json_safe(completion)

        score = 0.0

        # ── format ────────────────────────────────────────────────────────────
        if parsed is not None:
            score += 0.15
            if parsed.get('agent_name'):
                score += 0.10
            if parsed.get('action'):
                score += 0.10
            if parsed.get('reasoning_summary'):
                score += 0.10
            if parsed.get('expected_tool_calls') is not None:
                score += 0.10

        # ── workflow quality ──────────────────────────────────────────────────
        if _contains_any(combined, _SCHEMA_KEYWORDS):
            score += 0.08
        if _contains_any(combined, _GBM_KEYWORDS):
            score += 0.08
        if _contains_any(combined, _MLFLOW_KEYWORDS):
            score += 0.08
        if _contains_any(combined, _SANDBOX_KEYWORDS):
            score += 0.08
        if _contains_any(combined, _REPRO_KEYWORDS):
            score += 0.08
        if _contains_any(combined, _GUARDRAIL_KEYWORDS):
            score += 0.08
        if _contains_any(combined, _CHAMPION_KEYWORDS):
            score += 0.07

        # ── penalties ─────────────────────────────────────────────────────────
        if _contains_any(completion, _RAW_DATA_MARKERS):
            score -= 0.35
        if _contains_any(completion, _FAKE_METRIC_MARKERS):
            score -= 0.25
        if _contains_any(completion, _UNSUPPORTED_TOOL_MARKERS):
            score -= 0.20

        rewards.append(_clamp(score))
    return rewards


def hybrid_grpo_reward(
    prompts: list[str],
    completions: list[str],
    **kwargs: Any,
) -> list[float]:
    """
    Weighted blend of json_validity_reward (30 %) and workflow_policy_reward (70 %).

    This is the recommended default for GRPO training because it preserves the
    format signal while rewarding workflow-aligned completions more strongly.
    """
    validity = json_validity_reward(completions, **kwargs)
    workflow = workflow_policy_reward(prompts, completions, **kwargs)
    return [_clamp(0.30 * v + 0.70 * w) for v, w in zip(validity, workflow)]


def trajectory_reward(
    prompts: list[str],
    completions: list[str],
    rewards: list[float] | None = None,
    **kwargs: Any,
) -> list[float]:
    """
    Pass-through reward that uses pre-computed trajectory-level rewards from
    the rollout store (R_data, R_model, R_tracking, …).

    Priority order for reward source:
    1. ``rewards`` kwarg — explicitly passed float list (same length as completions).
    2. ``reward`` key inside each element of ``kwargs`` batch columns — the
       TRL GRPOTrainer passes extra dataset columns as kwargs; the TRL dataset
       now includes a ``reward`` column populated from grouped_rollouts.
    3. Falls back to ``hybrid_grpo_reward`` when no pre-computed reward is
       available (e.g. first-run, CPU-only, or missing column).
    """
    # 1. Explicit rewards kwarg
    if rewards is not None and len(rewards) == len(completions):
        return [_clamp(float(r)) for r in rewards]

    # 2. TRL batch column: kwargs['reward'] is a list when the dataset has that column
    batch_rewards = kwargs.get('reward')
    if batch_rewards is not None:
        try:
            if len(batch_rewards) == len(completions):
                return [_clamp(float(r)) for r in batch_rewards]
        except (TypeError, ValueError):
            pass

    # 3. Fallback
    return hybrid_grpo_reward(prompts, completions)



def ruler_relative_reward(
    prompts: list[str],
    completions: list[str],
    ruler_relative_score: list[float] | None = None,
    final_hybrid_reward: list[float] | None = None,
    ruler_rank: list[int] | None = None,
    reward: list[float] | None = None,
    **kwargs: Any,
) -> list[float]:
    """Reward mode that consumes RULER-like relative ranking columns.

    Priority: ``final_hybrid_reward`` → ``ruler_relative_score`` → ``reward`` →
    regular hybrid fallback. This allows TRL GRPO to train from
    ``ruler_scored_groups.jsonl`` while remaining compatible with the original
    grouped rollout dataset.
    """
    def _column(name: str, explicit: list[Any] | None) -> list[Any] | None:
        value = explicit if explicit is not None else kwargs.get(name)
        try:
            if value is not None and len(value) == len(completions):
                return value
        except TypeError:
            return None
        return None

    for name, explicit in [
        ('final_hybrid_reward', final_hybrid_reward),
        ('ruler_relative_score', ruler_relative_score),
        ('reward', reward),
    ]:
        values = _column(name, explicit)
        if values is not None:
            try:
                return [_clamp(float(v)) for v in values]
            except (TypeError, ValueError):
                pass

    rank_values = _column('ruler_rank', ruler_rank)
    if rank_values is not None:
        try:
            ranks = [max(1, int(v)) for v in rank_values]
            max_rank = max(ranks) if ranks else 1
            return [_clamp(1.0 - ((r - 1) / max(max_rank - 1, 1))) for r in ranks]
        except (TypeError, ValueError):
            pass

    return hybrid_grpo_reward(prompts, completions, **kwargs)


# ── reward mode registry ──────────────────────────────────────────────────────

REWARD_MODES: dict[str, Any] = {
    'json_validity':    json_validity_reward,
    'workflow_policy':  workflow_policy_reward,
    'trajectory_reward': trajectory_reward,
    'hybrid':           hybrid_grpo_reward,
    'ruler_relative':   ruler_relative_reward,
}


def get_reward_fn(mode: str):
    """Return the reward function for the given mode name.

    Raises ValueError for unknown modes.
    """
    if mode not in REWARD_MODES:
        raise ValueError(
            f"Unknown reward mode '{mode}'. "
            f"Available: {list(REWARD_MODES.keys())}"
        )
    return REWARD_MODES[mode]


# ── score breakdown helper (for UI) ──────────────────────────────────────────

def score_breakdown(prompt: str, completion: str) -> dict[str, Any]:
    """
    Return a detailed breakdown dict for a single (prompt, completion) pair.

    Useful for the Streamlit reward waterfall chart.
    """
    combined = (prompt + ' ' + completion).lower()
    parsed = _parse_json_safe(completion)

    breakdown: dict[str, Any] = {
        'valid_json': parsed is not None,
        'has_agent_name': bool(parsed and parsed.get('agent_name')),
        'has_action': bool(parsed and parsed.get('action')),
        'has_reasoning': bool(parsed and parsed.get('reasoning_summary')),
        'has_tool_calls': parsed is not None and parsed.get('expected_tool_calls') is not None,
        'mentions_schema': _contains_any(combined, _SCHEMA_KEYWORDS),
        'mentions_gbm': _contains_any(combined, _GBM_KEYWORDS),
        'mentions_mlflow': _contains_any(combined, _MLFLOW_KEYWORDS),
        'mentions_sandbox': _contains_any(combined, _SANDBOX_KEYWORDS),
        'mentions_reproducibility': _contains_any(combined, _REPRO_KEYWORDS),
        'mentions_guardrails': _contains_any(combined, _GUARDRAIL_KEYWORDS),
        'mentions_champion': _contains_any(combined, _CHAMPION_KEYWORDS),
        'raw_data_leakage': _contains_any(completion, _RAW_DATA_MARKERS),
        'fake_metrics': _contains_any(completion, _FAKE_METRIC_MARKERS),
        'unsupported_tool': _contains_any(completion, _UNSUPPORTED_TOOL_MARKERS),
    }

    # Compute component scores
    score = 0.0
    contributions: dict[str, float] = {}

    def _add(key: str, delta: float) -> None:
        nonlocal score
        score += delta
        contributions[key] = delta

    if breakdown['valid_json']:
        _add('valid_json', 0.15)
    if breakdown['has_agent_name']:
        _add('has_agent_name', 0.10)
    if breakdown['has_action']:
        _add('has_action', 0.10)
    if breakdown['has_reasoning']:
        _add('has_reasoning', 0.10)
    if breakdown['has_tool_calls']:
        _add('has_tool_calls', 0.10)
    if breakdown['mentions_schema']:
        _add('mentions_schema', 0.08)
    if breakdown['mentions_gbm']:
        _add('mentions_gbm', 0.08)
    if breakdown['mentions_mlflow']:
        _add('mentions_mlflow', 0.08)
    if breakdown['mentions_sandbox']:
        _add('mentions_sandbox', 0.08)
    if breakdown['mentions_reproducibility']:
        _add('mentions_reproducibility', 0.08)
    if breakdown['mentions_guardrails']:
        _add('mentions_guardrails', 0.08)
    if breakdown['mentions_champion']:
        _add('mentions_champion', 0.07)
    if breakdown['raw_data_leakage']:
        _add('raw_data_leakage', -0.35)
    if breakdown['fake_metrics']:
        _add('fake_metrics', -0.25)
    if breakdown['unsupported_tool']:
        _add('unsupported_tool', -0.20)

    breakdown['contributions'] = contributions
    breakdown['raw_score'] = round(score, 4)
    breakdown['final_score'] = round(_clamp(score), 4)
    return breakdown
