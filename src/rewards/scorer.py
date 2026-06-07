from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from src.telemetry.logger import TrajectoryLogger
from src.rewards.postgres_tool_reward import score_trajectory_postgres_tool

WEIGHTS = {
    'R_data': 0.18,
    'R_model': 0.25,
    'R_tracking': 0.17,
    'R_sandbox': 0.15,
    'R_reproducibility': 0.15,
    'R_violation': 0.09,
    'R_postgres_tool': 0.01,
}

RAW_ROW_MARKERS = ['dataframe head', 'raw rows:', 'first 5 rows', '.head(', 'to_string(']


def _safe_json(path: str | None) -> dict[str, Any]:
    try:
        if path and Path(path).exists():
            return json.loads(Path(path).read_text(encoding='utf-8'))
    except Exception:
        return {}
    return {}


def _collect_output_refs(t: dict) -> list[str]:
    refs: list[str] = []
    for step in t.get('steps', []):
        for tc in step.get('tool_calls', []):
            output_ref = tc.get('output_ref')
            if output_ref:
                refs.append(str(output_ref))
    return refs


def _artifact_map(t: dict) -> dict[str, str]:
    artifacts: dict[str, str] = {}
    for ref in _collect_output_refs(t):
        path = Path(ref)
        if path.name.endswith('.json'):
            payload = _safe_json(str(path))
            if isinstance(payload.get('artifacts'), dict):
                artifacts.update(payload['artifacts'])
            if isinstance(payload.get('mlflow'), dict):
                artifacts['mlflow_run_metadata'] = str(path)
        if path.name == 'benchmark_metrics.json':
            artifacts['benchmark_metrics_json'] = str(path)
        elif path.name == 'champion_model.json':
            artifacts['champion_model_json'] = str(path)
        elif path.name == 'review_report.json':
            artifacts['review_report_json'] = str(path)
        elif path.name == 'profile_summary.json':
            artifacts['profile_summary_json'] = str(path)
        elif path.name == 'preprocessing_config.json':
            artifacts['preprocessing_config_json'] = str(path)
        elif path.name == 'execution_dataset_source.json':
            artifacts['execution_dataset_source_json'] = str(path)
    return artifacts


def _resolve_existing_path(path: str | None) -> Path | None:
    if not path:
        return None
    p = Path(str(path))
    candidates = [p]
    if not p.is_absolute():
        cwd = Path.cwd()
        candidates.extend([
            cwd / p,
            cwd / 'artifacts' / p,
            Path(__file__).resolve().parents[2] / p,
            Path(__file__).resolve().parents[2] / 'artifacts' / p,
        ])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _exists(path: str | None) -> bool:
    return _resolve_existing_path(path) is not None


def _score_model(artifacts: dict[str, str]) -> float:
    metrics_payload = _safe_json(artifacts.get('benchmark_metrics_json'))
    champion = metrics_payload.get('champion') or _safe_json(artifacts.get('champion_model_json'))
    if not champion:
        return 0.0
    metrics = champion.get('metrics', {})
    model_path = champion.get('model_path') or champion.get('absolute_model_path')
    artifact_score = 1.0 if (_exists(model_path) or _exists(champion.get('absolute_model_path'))) else 0.5
    auc = float(metrics.get('roc_auc', metrics.get('accuracy', 0.0)))
    quality_score = min(max((auc - 0.5) / 0.45, 0.0), 1.0) if auc else 0.0
    return round(0.45 * artifact_score + 0.55 * quality_score, 4)


def _score_tracking(artifacts: dict[str, str]) -> float:
    metadata = _safe_json(artifacts.get('mlflow_run_metadata'))
    if not metadata:
        return 0.0
    backend = metadata.get('mlflow', {}).get('backend')
    has_metrics = bool(metadata.get('metrics'))
    if backend == 'mlflow':
        return 1.0 if has_metrics else 0.85
    return 0.75 if has_metrics else 0.6


def _score_sandbox(t: dict) -> float:
    for step in t.get('steps', []):
        for tc in step.get('tool_calls', []):
            if tc.get('tool_name') == 'DockerizedCodeInterpreter':
                if tc.get('status') != 'success':
                    return 0.0
                mode = (tc.get('arguments') or {}).get('execution_mode')
                return 1.0 if mode == 'docker' else 0.75
    return 0.0


def _score_violation(t: dict, artifacts: dict[str, str]) -> float:
    blob = json.dumps(t, default=str).lower()
    if any(marker in blob for marker in RAW_ROW_MARKERS):
        return 0.0
    review = _safe_json(artifacts.get('review_report_json'))
    issues = review.get('issues', []) if review else []
    if any(issue.get('severity') == 'critical' for issue in issues):
        return 0.0
    if any(issue.get('severity') == 'high' for issue in issues):
        return 0.4
    return 1.0


def score_trajectory(t: dict) -> tuple[float, dict]:
    steps = t.get('steps', [])
    tool_calls = [tc for s in steps for tc in s.get('tool_calls', [])]
    valid_tools = sum(1 for tc in tool_calls if tc.get('status') == 'success')
    total_tools = max(len(tool_calls), 1)
    artifacts = _artifact_map(t)

    data_required = ['execution_dataset_source_json', 'schema_metadata_json', 'profile_summary_json', 'preprocessing_config_json']
    data_score = sum(1 for key in data_required if _exists(artifacts.get(key))) / len(data_required)
    reproducibility_required = ['gbm_benchmark_plan', 'benchmark_metrics_json', 'champion_model_json', 'mlflow_run_metadata']
    reproducibility_score = sum(1 for key in reproducibility_required if _exists(artifacts.get(key))) / len(reproducibility_required)

    postgres_score, postgres_detail = score_trajectory_postgres_tool(t)

    metadata = {
        'task_completion': 1.0 if t.get('final_status') == 'completed' else 0.0,
        'tool_validity': valid_tools / total_tools,
        'R_data': round(data_score, 4),
        'R_model': _score_model(artifacts),
        'R_tracking': _score_tracking(artifacts),
        'R_sandbox': _score_sandbox(t),
        'R_reproducibility': round(reproducibility_score, 4),
        'R_violation': _score_violation(t, artifacts),
        'R_postgres_tool': postgres_score,
        'postgres_tool_detail': postgres_detail,
    }
    # Keep completion and tool validity as multiplicative gates so failed workflows
    # cannot receive high rewards from stale artifacts alone.
    base_reward = sum(WEIGHTS[k] * metadata[k] for k in WEIGHTS)
    reward = base_reward * metadata['task_completion'] * max(0.5, metadata['tool_validity'])
    return round(float(reward), 4), metadata


def score_directory(input_dir: str, output_dir: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    records = TrajectoryLogger.read_many(input_dir)
    for t in records:
        reward, metadata = score_trajectory(t)
        t['reward'] = reward
        t['reward_metadata'] = metadata
        path = out / f"{t['task_id']}_{t['policy_version']}_{t['trajectory_id']}.jsonl"
        path.write_text(json.dumps(t) + '\n', encoding='utf-8')
    print(f'Scored {len(records)} trajectories into {output_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', default='trajectories/baseline')
    parser.add_argument('--output-dir', default='trajectories/scored')
    args = parser.parse_args()
    score_directory(args.input_dir, args.output_dir)
