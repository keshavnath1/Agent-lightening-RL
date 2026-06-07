from __future__ import annotations

import json
from pathlib import Path
from typing import Any

RAW_ROW_MARKERS = ['dataframe head', 'raw rows:', 'first 5 rows', '.head(', 'to_string(']
REQUIRED_ARTIFACT_KEYS = [
    'execution_dataset_source_json',
    'schema_metadata_json',
    'profile_summary_json',
    'preprocessing_config_json',
    'gbm_benchmark_plan',
    'benchmark_metrics_json',
    'champion_model_json',
    'mlflow_run_metadata',
]


def _exists(value: str | None) -> bool:
    return bool(value) and Path(str(value)).exists()


def validate_workflow_artifacts(context: Any) -> dict[str, Any]:
    artifacts = context.artifacts
    missing = [key for key in REQUIRED_ARTIFACT_KEYS if not _exists(artifacts.get(key))]
    raw_text = json.dumps({
        'task': context.task,
        'artifacts': context.artifacts,
        'tool_outputs': context.tool_outputs,
    }, default=str).lower()
    raw_leakage_markers = [marker for marker in RAW_ROW_MARKERS if marker in raw_text]

    benchmark_metrics = {}
    champion = {}
    if _exists(artifacts.get('benchmark_metrics_json')):
        benchmark_metrics = json.loads(Path(artifacts['benchmark_metrics_json']).read_text(encoding='utf-8'))
    if _exists(artifacts.get('champion_model_json')):
        champion = json.loads(Path(artifacts['champion_model_json']).read_text(encoding='utf-8'))

    sandbox = context.tool_outputs.get('sandbox_execution', {})
    tracking = context.tool_outputs.get('experiment_tracking', {})
    issues = []
    if missing:
        issues.append({'severity': 'high', 'type': 'missing_artifacts', 'details': missing})
    if raw_leakage_markers:
        issues.append({'severity': 'critical', 'type': 'raw_data_leakage_marker', 'details': raw_leakage_markers})
    if sandbox.get('status') != 'success':
        issues.append({'severity': 'high', 'type': 'benchmark_execution_failed', 'details': sandbox.get('stderr')})
    if not champion.get('model_path') or not _exists(champion.get('model_path')):
        issues.append({'severity': 'high', 'type': 'missing_champion_model_artifact', 'details': champion})

    scores = {
        'data_profiling_quality': 1.0 if all(_exists(artifacts.get(k)) for k in ['schema_metadata_json', 'profile_summary_json', 'preprocessing_config_json']) else 0.0,
        'preprocessing_quality': 1.0 if _exists(artifacts.get('preprocessing_config_json')) else 0.0,
        'schema_integrity': 1.0 if _exists(artifacts.get('schema_metadata_json')) else 0.0,
        'model_selection_correctness': 1.0 if _exists(artifacts.get('gbm_benchmark_plan')) else 0.0,
        'mlflow_tracking_completeness': 1.0 if tracking.get('backend') == 'mlflow' else (0.75 if _exists(artifacts.get('mlflow_run_metadata')) else 0.0),
        'sandbox_compliance': 1.0 if sandbox.get('execution_mode') == 'docker' and sandbox.get('status') == 'success' else (0.75 if sandbox.get('status') == 'success' else 0.0),
        'raw_data_guardrail_compliance': 0.0 if raw_leakage_markers else 1.0,
        'reproducibility': 1.0 if all(_exists(artifacts.get(k)) for k in ['benchmark_metrics_json', 'champion_model_json', 'mlflow_run_metadata']) else 0.5,
        'final_model_quality': float(champion.get('metrics', {}).get('roc_auc', champion.get('metrics', {}).get('accuracy', 0.0))),
    }
    return {
        'approved': not any(issue['severity'] in {'high', 'critical'} for issue in issues),
        'issues': issues,
        'scores': scores,
        'benchmark_summary': {
            'champion': champion,
            'candidate_count': len(benchmark_metrics.get('candidates', [])) if benchmark_metrics else 0,
        },
    }
