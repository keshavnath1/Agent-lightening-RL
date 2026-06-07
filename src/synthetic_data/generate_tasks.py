from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def make_dataset(task_id: str, rows: int, cols: int, seed: int, output_dir: Path) -> dict:
    rng = np.random.default_rng(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = {f'num_{i}': rng.normal(size=rows) for i in range(cols)}
    data['cat_segment'] = rng.choice(['A', 'B', 'C', 'D'], size=rows)
    logits = 0.8 * data['num_0'] - 0.4 * data['num_1'] + rng.normal(scale=0.5, size=rows)
    data['target'] = (logits > np.quantile(logits, 0.55)).astype(int)
    df = pd.DataFrame(data)
    parquet_path = output_dir / f'{task_id}.parquet'
    df.to_parquet(parquet_path, index=False)
    return {
        'task_id': task_id,
        'objective': 'Build a reproducible binary classification workflow and select a champion GBM model.',
        'dataset_ref': str(parquet_path),
        'target_column': 'target',
        'metric': 'roc_auc',
        'rows': rows,
        'columns': len(df.columns),
        'split': 'train' if seed % 5 else 'heldout',
    }


def generate_tasks(count: int, output_path: str, dataset_dir: str, seed: int = 42) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    dataset_root = Path(dataset_dir)
    records = []
    for i in range(count):
        task_id = f'task_{i:04d}'
        rows = int(500 + (i % 5) * 250)
        cols = int(6 + (i % 4))
        records.append(make_dataset(task_id, rows, cols, seed + i, dataset_root))
    with out.open('w', encoding='utf-8') as f:
        for rec in records:
            f.write(json.dumps(rec) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--count', type=int, default=20)
    parser.add_argument('--output', default='data/synthetic/tasks.jsonl')
    parser.add_argument('--dataset-dir', default='data/synthetic/datasets')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    generate_tasks(args.count, args.output, args.dataset_dir, args.seed)
    print(f'Wrote {args.count} synthetic tasks to {args.output}')
