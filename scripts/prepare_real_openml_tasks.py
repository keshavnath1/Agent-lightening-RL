from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import openml
import pandas as pd

REPO = Path('/workspace/self-improving-ml-agent')
DATA_ROOT = REPO / 'data' / 'real_openml'
DATASET_DIR = DATA_ROOT / 'datasets'
TASKS_PATH = DATA_ROOT / 'tasks.jsonl'
MANIFEST_PATH = DATA_ROOT / 'dataset_manifest.json'
README_PATH = DATA_ROOT / 'README.md'

DATASETS: list[dict[str, Any]] = [
    {
        'data_id': 1590,
        'slug': 'adult_income',
        'display_name': 'Adult Census Income',
        'domain': 'labor economics / income prediction',
        'metric': 'roc_auc',
        'positive_labels': ['>50K', '>50K.'],
        'objective': 'Predict whether annual income exceeds 50K using demographic and employment records from the UCI Adult Census benchmark.',
    },
    {
        'data_id': 1461,
        'slug': 'bank_marketing',
        'display_name': 'Bank Marketing',
        'domain': 'financial services / campaign conversion',
        'metric': 'roc_auc',
        'positive_labels': ['yes', '1', 'true'],
        'objective': 'Predict whether a customer subscribes to a term deposit using the UCI Bank Marketing direct-campaign benchmark.',
    },
    {
        'data_id': 31,
        'slug': 'german_credit',
        'display_name': 'German Credit Risk',
        'domain': 'consumer finance / credit risk',
        'metric': 'roc_auc',
        'positive_labels': ['bad', '2'],
        'objective': 'Predict higher-risk credit applicants from the German Credit benchmark using a reproducible binary-classification workflow.',
    },
    {
        'data_id': 44,
        'slug': 'spambase',
        'display_name': 'Spambase Email Classification',
        'domain': 'email security / abuse detection',
        'metric': 'roc_auc',
        'positive_labels': ['1', 'spam', 'true'],
        'objective': 'Predict spam email using the UCI Spambase benchmark and select a champion gradient-boosting model.',
    },
    {
        'data_id': 1489,
        'slug': 'phoneme',
        'display_name': 'Phoneme Speech Recognition',
        'domain': 'speech recognition / signal classification',
        'metric': 'roc_auc',
        'positive_labels': ['1', '2', 'nasal', 'true'],
        'objective': 'Classify phoneme observations from acoustic features using the OpenML Phoneme benchmark.',
    },
]


def _normalise_label(value: Any) -> str:
    return str(value).strip().strip("'").strip('"')


def _encode_binary_target(y: pd.Series, positive_labels: list[str]) -> tuple[pd.Series, dict[str, int]]:
    raw = y.astype('object').map(_normalise_label)
    mask = raw.notna() & (raw != '') & (raw.str.lower() != 'nan')
    raw = raw[mask]
    values = list(pd.unique(raw))
    if len(values) != 2:
        raise ValueError(f'Expected binary target after cleaning, found {len(values)} labels: {values[:10]}')

    pos_lookup = {str(v).lower() for v in positive_labels}
    chosen_positive = None
    for value in values:
        if value.lower() in pos_lookup:
            chosen_positive = value
            break
    if chosen_positive is None:
        counts = raw.value_counts()
        chosen_positive = str(counts.idxmin())  # deterministic minority-positive fallback

    mapping = {str(v): int(str(v) == chosen_positive) for v in values}
    encoded = raw.map(mapping).astype(int)
    return encoded, mapping


def _prepare_features(X: pd.DataFrame, target_index: pd.Index) -> pd.DataFrame:
    X = X.loc[target_index].copy()
    X.columns = [str(c).strip().replace(' ', '_').replace('-', '_') for c in X.columns]
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            X[col] = pd.to_numeric(X[col], errors='coerce')
        else:
            X[col] = X[col].astype('object').where(X[col].notna(), 'missing').map(_normalise_label)
            X[col] = X[col].replace({'?': 'missing', 'nan': 'missing', '': 'missing'})
    return X


def main() -> None:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []

    for spec in DATASETS:
        dataset = openml.datasets.get_dataset(spec['data_id'], download_data=True, download_qualities=True, download_features_meta_data=True)
        target_name = dataset.default_target_attribute
        X, y, categorical_indicator, attribute_names = dataset.get_data(
            dataset_format='dataframe',
            target=target_name,
        )
        encoded_y, mapping = _encode_binary_target(y, spec['positive_labels'])
        X_clean = _prepare_features(X, encoded_y.index)
        df = X_clean.copy()
        df['target'] = encoded_y.values
        df = df.sample(frac=1.0, random_state=20260606).reset_index(drop=True)

        task_id = f"openml_{spec['data_id']}_{spec['slug']}"
        parquet_path = DATASET_DIR / f'{task_id}.parquet'
        df.to_parquet(parquet_path, index=False)

        class_balance = {str(k): int(v) for k, v in df['target'].value_counts().sort_index().items()}
        task_record = {
            'task_id': task_id,
            'objective': spec['objective'],
            'dataset_ref': str(parquet_path),
            'target_column': 'target',
            'metric': spec['metric'],
            'rows': int(df.shape[0]),
            'columns': int(df.shape[1]),
            'split': 'real_benchmark',
            'data_source': 'OpenML',
            'openml_data_id': int(spec['data_id']),
            'openml_name': dataset.name,
            'openml_version': int(dataset.version),
            'openml_url': f"https://www.openml.org/d/{spec['data_id']}",
            'domain': spec['domain'],
            'target_mapping': mapping,
            'class_balance': class_balance,
            'synthetic': False,
        }
        records.append(task_record)
        manifest.append({
            **task_record,
            'dataset_path': str(parquet_path),
            'feature_columns': [c for c in df.columns if c != 'target'],
            'missing_values_after_cleaning': int(df.isna().sum().sum()),
        })
        print(f"Prepared {task_id}: rows={df.shape[0]} cols={df.shape[1]} balance={class_balance}")

    TASKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TASKS_PATH.open('w', encoding='utf-8') as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')

    total_rows = sum(row['rows'] for row in records)
    README_PATH.write_text(
        '# Real OpenML benchmark tasks\n\n'
        'This directory replaces the previous synthetic task set for the Track A / Track B demo. '
        'Datasets are pulled from OpenML benchmark datasets and converted into local Parquet files with a common binary target column named `target`.\n\n'
        f'- Task file: `{TASKS_PATH}`\n'
        f'- Dataset directory: `{DATASET_DIR}`\n'
        f'- Number of tasks: {len(records)}\n'
        f'- Total rows across tasks: {total_rows}\n\n'
        '| Task ID | Domain | Rows | Columns | OpenML |\n'
        '|---|---:|---:|---:|---|\n' +
        ''.join(
            f"| {row['task_id']} | {row['domain']} | {row['rows']} | {row['columns']} | {row['openml_url']} |\n"
            for row in records
        ),
        encoding='utf-8',
    )
    print(f'Wrote {len(records)} real OpenML tasks to {TASKS_PATH}')
    print(f'Wrote manifest to {MANIFEST_PATH}')


if __name__ == '__main__':
    main()
