from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


class MLflowDVCTracker:
    """Strict MLflow tracker.

    MLflow is required in production; failures are raised instead of being
    replaced by a local JSON fallback. A local JSON summary is still written
    only after a successful MLflow log so Streamlit can display evidence.
    """

    def __init__(self, tracking_dir: str | Path = 'reports/mlflow_runs'):
        self.tracking_dir = Path(tracking_dir)
        self.tracking_dir.mkdir(parents=True, exist_ok=True)
        self.backend = 'mlflow'
        self.last_mlflow_run_id: str | None = None

    def _safe_artifact_paths(self, artifact_paths: list[str] | None) -> list[Path]:
        paths: list[Path] = []
        for raw in artifact_paths or []:
            if not raw:
                continue
            path = Path(raw)
            if path.exists():
                paths.append(path)
        return paths

    def log_experiment(
        self,
        run_id: str,
        payload: dict[str, Any],
        params: dict[str, Any] | None = None,
        metrics: dict[str, float] | None = None,
        artifact_paths: list[str] | None = None,
    ) -> dict[str, Any]:
        try:
            import mlflow
        except Exception as exc:
            raise RuntimeError(
                'MLflow is required in strict mode. Install/configure mlflow; no JSON fallback is allowed.'
            ) from exc

        tracking_uri = os.getenv('MLFLOW_TRACKING_URI', f'file:{Path("mlruns").resolve()}')
        experiment_name = os.getenv('MLFLOW_EXPERIMENT_NAME', 'self_improving_tabular_agent')
        metadata_path = self.tracking_dir / f'{run_id}.json'

        try:
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)
            with mlflow.start_run(run_name=run_id) as active_run:
                flat_params = {k: v for k, v in (params or {}).items() if isinstance(v, (str, int, float, bool))}
                if flat_params:
                    mlflow.log_params(flat_params)
                flat_metrics = {k: float(v) for k, v in (metrics or {}).items() if isinstance(v, (int, float))}
                if flat_metrics:
                    mlflow.log_metrics(flat_metrics)
                for artifact in self._safe_artifact_paths(artifact_paths):
                    if artifact.is_file():
                        mlflow.log_artifact(str(artifact))
                mlflow_info = {
                    'backend': 'mlflow',
                    'run_id': active_run.info.run_id,
                    'tracking_uri': tracking_uri,
                    'experiment_name': experiment_name,
                    'strict_no_fallback': True,
                }
                self.last_mlflow_run_id = active_run.info.run_id
        except Exception as exc:
            raise RuntimeError(
                f'MLflow logging failed in strict mode: {exc.__class__.__name__}: {exc}'
            ) from exc

        enriched = dict(payload)
        enriched['mlflow'] = mlflow_info
        metadata_path.write_text(json.dumps(enriched, indent=2), encoding='utf-8')
        return {'metadata_path': str(metadata_path), **mlflow_info}

    def log_run_metadata(self, run_id: str, payload: dict[str, Any]) -> Path:
        result = self.log_experiment(run_id, payload)
        return Path(result['metadata_path'])

    def dvc_note(self) -> str:
        return 'DVC integration point: version real datasets, profile summaries, trajectories, GBM models, and policy checkpoints.'
