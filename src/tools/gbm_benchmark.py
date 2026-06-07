from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class CandidateResult:
    name: str
    status: str
    metrics: dict[str, float]
    model_path: str | None
    error: str | None = None


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _portable_path(path: str | None, base_dir: str | Path | None = None) -> dict[str, str | None]:
    """Return both portable and absolute artifact references.

    RunPod pods commonly mount the repository under /workspace, while local
    validation may use /home/ubuntu or another path. Storing a repository- or
    artifact-relative path keeps champion metadata portable across pods and zip
    extractions, while retaining the absolute path for direct local inspection.
    """
    if not path:
        return {'model_path': None, 'absolute_model_path': None}
    p = Path(path)
    absolute = str(p.resolve())
    candidates: list[Path] = []
    if base_dir:
        candidates.append(Path(base_dir).resolve())
    cwd = Path.cwd().resolve()
    candidates.extend([cwd, cwd / 'artifacts'])
    for root in candidates:
        try:
            relative = p.resolve().relative_to(root)
            prefix = '' if root == cwd else 'artifacts/'
            return {'model_path': f'{prefix}{relative.as_posix()}', 'absolute_model_path': absolute}
        except Exception:
            continue
    return {'model_path': p.name, 'absolute_model_path': absolute}


def _prepare_xy_from_frame(df: pd.DataFrame, schema_path: str | Path, preprocessing_path: str | Path) -> tuple[pd.DataFrame, pd.Series, dict]:
    schema = _load_json(schema_path)
    config = _load_json(preprocessing_path)
    target = schema.get('target_column') or config.get('target_column')
    if not target or target not in df.columns:
        raise ValueError(f'Target column {target!r} was not found in materialized dataset.')

    y = df[target]
    X = df.drop(columns=[target]).copy()

    numeric_columns = [c for c in config.get('numeric_columns', []) if c in X.columns]
    categorical_columns = [c for c in config.get('categorical_columns', []) if c in X.columns]
    datetime_columns = [c for c in config.get('datetime_columns', []) if c in X.columns]

    for col in datetime_columns:
        parsed = pd.to_datetime(X[col], errors='coerce')
        X[f'{col}_year'] = parsed.dt.year.fillna(0).astype(int)
        X[f'{col}_month'] = parsed.dt.month.fillna(0).astype(int)
        X[f'{col}_day'] = parsed.dt.day.fillna(0).astype(int)
        X = X.drop(columns=[col])

    for col in numeric_columns:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
            X[col] = X[col].fillna(X[col].median())

    for col in categorical_columns:
        if col in X.columns:
            mode = X[col].mode(dropna=True)
            fill_value = mode.iloc[0] if len(mode) else '__missing__'
            X[col] = X[col].fillna(fill_value).astype(str)

    X = pd.get_dummies(X, columns=[c for c in categorical_columns if c in X.columns], dummy_na=False)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X, y, config


def _prepare_xy(clean_parquet: str | Path, schema_path: str | Path, preprocessing_path: str | Path) -> tuple[pd.DataFrame, pd.Series, dict]:
    df = pd.read_parquet(clean_parquet)
    return _prepare_xy_from_frame(df, schema_path, preprocessing_path)


def _split_train_valid(X: pd.DataFrame, y: pd.Series):
    from sklearn.model_selection import train_test_split

    stratify = y if y.nunique(dropna=True) == 2 and y.value_counts().min() >= 2 else None
    return train_test_split(X, y, test_size=0.25, random_state=42, stratify=stratify)


def _classification_metrics(model: Any, X_valid: pd.DataFrame, y_valid: pd.Series) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

    classes = list(getattr(model, 'classes_', []))
    proba_for_loss = None
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X_valid)
        proba_for_loss = proba
        positive = proba[:, 1] if proba.ndim == 2 and proba.shape[1] > 1 else proba.ravel()
    elif hasattr(model, 'decision_function'):
        scores = model.decision_function(X_valid)
        positive = 1 / (1 + np.exp(-np.asarray(scores)))
    else:
        positive = np.asarray(model.predict(X_valid), dtype=float)

    if len(classes) >= 2:
        pred = np.where(positive >= 0.5, classes[1], classes[0])
    else:
        pred = (positive >= 0.5).astype(int)

    try:
        loss = (
            log_loss(y_valid, proba_for_loss, labels=classes)
            if proba_for_loss is not None and len(classes) >= 2
            else log_loss(y_valid, np.clip(positive, 1e-6, 1 - 1e-6))
        )
    except Exception:
        loss = float('nan')

    metrics = {
        'accuracy': float(accuracy_score(y_valid, pred)),
        'log_loss': float(loss),
    }
    try:
        metrics['roc_auc'] = float(roc_auc_score(y_valid, positive))
    except Exception:
        metrics['roc_auc'] = 0.5
    return metrics


def _train_sklearn_histgb(X_train, X_valid, y_train, y_valid, output_dir: Path) -> CandidateResult:
    from sklearn.ensemble import HistGradientBoostingClassifier

    model = HistGradientBoostingClassifier(max_iter=80, learning_rate=0.08, random_state=42)
    model.fit(X_train, y_train)
    metrics = _classification_metrics(model, X_valid, y_valid)
    model_path = output_dir / 'sklearn_hist_gradient_boosting_model.pkl'
    with model_path.open('wb') as f:
        pickle.dump(model, f)
    return CandidateResult('sklearn_hist_gradient_boosting', 'trained', metrics, str(model_path))


def _train_xgboost(X_train, X_valid, y_train, y_valid, output_dir: Path) -> CandidateResult:
    try:
        from xgboost import XGBClassifier
    except Exception as exc:
        return CandidateResult('xgboost', 'skipped_dependency_missing', {}, None, str(exc))
    try:
        model = XGBClassifier(
            n_estimators=120,
            max_depth=4,
            learning_rate=0.06,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric='logloss',
            random_state=42,
            n_jobs=2,
        )
        model.fit(X_train, y_train)
        metrics = _classification_metrics(model, X_valid, y_valid)
        model_path = output_dir / 'xgboost_model.pkl'
        with model_path.open('wb') as f:
            pickle.dump(model, f)
        return CandidateResult('xgboost', 'trained', metrics, str(model_path))
    except Exception as exc:
        return CandidateResult('xgboost', 'failed', {}, None, str(exc))


def _train_lightgbm(X_train, X_valid, y_train, y_valid, output_dir: Path) -> CandidateResult:
    try:
        from lightgbm import LGBMClassifier
    except Exception as exc:
        return CandidateResult('lightgbm', 'skipped_dependency_missing', {}, None, str(exc))
    try:
        model = LGBMClassifier(
            n_estimators=120,
            max_depth=-1,
            learning_rate=0.06,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=2,
            verbose=-1,
        )
        model.fit(X_train, y_train)
        metrics = _classification_metrics(model, X_valid, y_valid)
        model_path = output_dir / 'lightgbm_model.bst'
        model.booster_.save_model(str(model_path))
        pickle_path = output_dir / 'lightgbm_model.pkl'
        with pickle_path.open('wb') as f:
            pickle.dump(model, f)
        return CandidateResult('lightgbm', 'trained', metrics, str(model_path))
    except Exception as exc:
        return CandidateResult('lightgbm', 'failed', {}, None, str(exc))


def _select_champion(results: list[CandidateResult], metric: str) -> CandidateResult | None:
    trained = [r for r in results if r.status == 'trained' and r.metrics]
    if not trained:
        return None
    if metric in {'log_loss', 'loss'}:
        return min(trained, key=lambda r: r.metrics.get('log_loss', float('inf')))
    return max(trained, key=lambda r: r.metrics.get(metric, r.metrics.get('roc_auc', 0.0)))


def run_gbm_benchmark(
    clean_parquet: str,
    schema_metadata: str,
    profile_summary: str,
    preprocessing_config: str,
    output_dir: str,
    metric: str = 'roc_auc',
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    profile = _load_json(profile_summary)
    X, y, config = _prepare_xy(clean_parquet, schema_metadata, preprocessing_config)
    X_train, X_valid, y_train, y_valid = _split_train_valid(X, y)

    results = [
        _train_xgboost(X_train, X_valid, y_train, y_valid, output),
        _train_lightgbm(X_train, X_valid, y_train, y_valid, output),
        _train_sklearn_histgb(X_train, X_valid, y_train, y_valid, output),
    ]
    champion = _select_champion(results, metric) or _select_champion(results, 'roc_auc')
    if champion is None:
        raise RuntimeError('No GBM candidate trained successfully.')

    payload = {
        'benchmark_type': 'real_tabular_gbm_benchmark_with_optional_xgboost_lightgbm',
        'metric': metric,
        'row_count': int(profile.get('row_count', len(y))),
        'feature_count_after_preprocessing': int(X.shape[1]),
        'preprocessing_config': config,
        'candidates': [
            {
                'model_name': r.name,
                'status': r.status,
                'metrics': r.metrics,
                **_portable_path(r.model_path, output),
                'error': r.error,
            }
            for r in results
        ],
        'champion': {
            'model_name': champion.name,
            'status': champion.status,
            'metrics': champion.metrics,
            **_portable_path(champion.model_path, output),
        },
    }
    metrics_path = output / 'benchmark_metrics.json'
    metrics_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    champion_path = output / 'champion_model.json'
    champion_path.write_text(json.dumps(payload['champion'], indent=2), encoding='utf-8')
    return payload


def run_gbm_benchmark_from_postgres(
    dataset_source: str | Path | dict[str, Any],
    schema_metadata: str,
    profile_summary: str,
    preprocessing_config: str,
    output_dir: str,
    metric: str = 'roc_auc',
    execution_database_url_env: str = 'EXECUTION_DATABASE_URL',
) -> dict[str, Any]:
    from src.tools.sandbox_postgres_data_loader import load_dataset_frame_from_source

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    source = _load_json(dataset_source) if not isinstance(dataset_source, dict) else dataset_source
    df = load_dataset_frame_from_source(source, secret_env_var=execution_database_url_env)
    profile = _load_json(profile_summary)
    X, y, config = _prepare_xy_from_frame(df, schema_metadata, preprocessing_config)
    X_train, X_valid, y_train, y_valid = _split_train_valid(X, y)

    results = [
        _train_xgboost(X_train, X_valid, y_train, y_valid, output),
        _train_lightgbm(X_train, X_valid, y_train, y_valid, output),
        _train_sklearn_histgb(X_train, X_valid, y_train, y_valid, output),
    ]
    champion = _select_champion(results, metric) or _select_champion(results, 'roc_auc')
    if champion is None:
        raise RuntimeError('No GBM candidate trained successfully.')

    payload = {
        'benchmark_type': 'postgres_execution_dataset_source_gbm_benchmark',
        'metric': metric,
        'dataset_source': {
            'contract_version': source.get('contract_version'),
            'dataset_key': source.get('dataset_key'),
            'storage_backend': source.get('storage_backend'),
            'source_schema': source.get('source_schema'),
            'source_table': source.get('source_table'),
            'execution_only': bool(source.get('execution_only')),
            'raw_rows_exposed_to_llm': False,
        },
        'row_count': int(profile.get('row_count') or len(y)),
        'feature_count_after_preprocessing': int(X.shape[1]),
        'preprocessing_config': config,
        'candidates': [
            {
                'model_name': r.name,
                'status': r.status,
                'metrics': r.metrics,
                **_portable_path(r.model_path, output),
                'error': r.error,
            }
            for r in results
        ],
        'champion': {
            'model_name': champion.name,
            'status': champion.status,
            'metrics': champion.metrics,
            **_portable_path(champion.model_path, output),
        },
        'raw_rows_exposed_to_llm': False,
    }
    metrics_path = output / 'benchmark_metrics.json'
    metrics_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    champion_path = output / 'champion_model.json'
    champion_path.write_text(json.dumps(payload['champion'], indent=2), encoding='utf-8')
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--clean-parquet')
    source.add_argument('--dataset-source')
    parser.add_argument('--schema-metadata', required=True)
    parser.add_argument('--profile-summary', required=True)
    parser.add_argument('--preprocessing-config', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--metric', default='roc_auc')
    args = parser.parse_args()
    if args.dataset_source:
        payload = run_gbm_benchmark_from_postgres(
            dataset_source=args.dataset_source,
            schema_metadata=args.schema_metadata,
            profile_summary=args.profile_summary,
            preprocessing_config=args.preprocessing_config,
            output_dir=args.output_dir,
            metric=args.metric,
        )
    else:
        payload = run_gbm_benchmark(
            clean_parquet=args.clean_parquet,
            schema_metadata=args.schema_metadata,
            profile_summary=args.profile_summary,
            preprocessing_config=args.preprocessing_config,
            output_dir=args.output_dir,
            metric=args.metric,
        )
    print(json.dumps({'champion': payload['champion'], 'metric': payload['metric']}, indent=2))


if __name__ == '__main__':
    main()
