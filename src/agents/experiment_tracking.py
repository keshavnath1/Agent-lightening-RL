from __future__ import annotations

import json
from pathlib import Path
from src.agents.base import BaseAgent, AgentContext
from src.telemetry.schema import AgentStep
from src.tools.mlflow_dvc_tracker import MLflowDVCTracker


def _load_metrics(path: str | None) -> dict[str, float]:
    if not path or not Path(path).exists():
        return {}
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    champion = payload.get('champion', {})
    metrics = champion.get('metrics', {})
    return {f'champion_{k}': float(v) for k, v in metrics.items() if isinstance(v, (int, float))}


class ExperimentTrackingAgent(BaseAgent):
    name = 'ExperimentTrackingAgent'

    def step(self, context: AgentContext) -> AgentStep:
        tracker = MLflowDVCTracker('reports/mlflow_runs')
        run_id = f"{context.task['task_id']}_{context.policy_version}"
        metrics = _load_metrics(context.artifacts.get('benchmark_metrics_json'))
        params = {
            'task_id': context.task['task_id'],
            'policy_version': context.policy_version,
            'metric': context.task.get('metric', 'roc_auc'),
            'target_column': context.task.get('target_column'),
        }
        artifact_paths = list(context.artifacts.values())
        metadata = {
            'task_id': context.task['task_id'],
            'policy_version': context.policy_version,
            'artifacts': context.artifacts,
            'metrics': metrics,
            'params': params,
            'dvc_note': tracker.dvc_note(),
        }
        result = tracker.log_experiment(run_id, metadata, params=params, metrics=metrics, artifact_paths=artifact_paths)
        path = result['metadata_path']
        context.artifacts['mlflow_run_metadata'] = str(path)
        if result.get('run_id'):
            context.artifacts['mlflow_run_id'] = str(result['run_id'])
        context.tool_outputs['experiment_tracking'] = result
        return AgentStep(
            agent_name=self.name,
            action='log_experiment_to_mlflow',
            reasoning_summary='Logged reproducibility metadata, benchmark metrics, artifacts, and DVC integration notes through required MLflow tracking with no fallback.',
            tool_calls=[self.tool_success('MLflowDVCTracker', {'run_id': run_id, 'backend': result.get('backend')}, str(path))],
        )
