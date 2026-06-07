from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

from src.tools.sql_alchemy_connector import default_database_url, normalize_database_url


def jsonable(value):
    if pd.isna(value):
        return None
    if hasattr(value, 'item'):
        try:
            return value.item()
        except Exception:
            pass
    return value


def quote_ident(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def main() -> None:
    repo_dir = Path(os.environ.get('REPO_DIR', Path(__file__).resolve().parents[2]))
    schema = os.environ.get('POSTGRES_SCHEMA', os.environ.get('PG_SCHEMA', 'agentic_ml'))
    tasks_path = repo_dir / 'data' / 'synthetic' / 'tasks.jsonl'
    datasets_dir = repo_dir / 'data' / 'synthetic' / 'datasets'

    if not tasks_path.exists():
        raise SystemExit(f'Synthetic tasks file not found: {tasks_path}')
    if not datasets_dir.exists():
        raise SystemExit(f'Synthetic datasets directory not found: {datasets_dir}')

    engine = create_engine(
        normalize_database_url(default_database_url()),
        future=True,
        pool_pre_ping=True,
        connect_args={'connect_timeout': int(os.getenv('POSTGRES_CONNECT_TIMEOUT', '30'))},
        use_native_hstore=False,
    )
    schema_q = quote_ident(schema)

    with engine.begin() as conn:
        conn.execute(text(f'create schema if not exists {schema_q}'))
        conn.execute(text(f'''
            create table if not exists {schema_q}.synthetic_tasks (
                task_id text primary key,
                objective text not null,
                dataset_ref text not null,
                target_column text not null,
                metric text not null,
                row_count integer not null,
                column_count integer not null,
                split text not null,
                payload jsonb not null,
                loaded_at timestamptz not null default now()
            )
        '''))
        conn.execute(text(f'''
            create table if not exists {schema_q}.synthetic_dataset_rows (
                task_id text not null,
                row_index integer not null,
                split text not null,
                payload jsonb not null,
                loaded_at timestamptz not null default now(),
                primary key (task_id, row_index)
            )
        '''))
        conn.execute(text(f'''
            create index if not exists synthetic_dataset_rows_payload_gin
            on {schema_q}.synthetic_dataset_rows using gin (payload)
        '''))

    tasks = []
    for line in tasks_path.read_text(encoding='utf-8').splitlines():
        if line.strip():
            tasks.append(json.loads(line))

    task_insert = text(f'''
        insert into {schema_q}.synthetic_tasks
            (task_id, objective, dataset_ref, target_column, metric, row_count, column_count, split, payload)
        values
            (:task_id, :objective, :dataset_ref, :target_column, :metric, :row_count, :column_count, :split, cast(:payload as jsonb))
        on conflict (task_id) do update set
            objective = excluded.objective,
            dataset_ref = excluded.dataset_ref,
            target_column = excluded.target_column,
            metric = excluded.metric,
            row_count = excluded.row_count,
            column_count = excluded.column_count,
            split = excluded.split,
            payload = excluded.payload,
            loaded_at = now()
    ''')
    row_insert = text(f'''
        insert into {schema_q}.synthetic_dataset_rows (task_id, row_index, split, payload)
        values (:task_id, :row_index, :split, cast(:payload as jsonb))
        on conflict (task_id, row_index) do update set
            split = excluded.split,
            payload = excluded.payload,
            loaded_at = now()
    ''')

    with engine.begin() as conn:
        for rec in tasks:
            dataset_ref = str(Path('data/synthetic/datasets') / f"{rec['task_id']}.parquet")
            payload = dict(rec)
            payload['dataset_ref'] = dataset_ref
            conn.execute(task_insert, {
                'task_id': rec['task_id'],
                'objective': rec['objective'],
                'dataset_ref': dataset_ref,
                'target_column': rec['target_column'],
                'metric': rec['metric'],
                'row_count': int(rec['rows']),
                'column_count': int(rec['columns']),
                'split': rec['split'],
                'payload': json.dumps(payload),
            })
    print(f'Loaded/upserted {len(tasks)} records into {schema}.synthetic_tasks')

    total_rows = 0
    for rec in tasks:
        task_id = rec['task_id']
        parquet_path = datasets_dir / f'{task_id}.parquet'
        if not parquet_path.exists():
            print(f'WARNING: missing dataset file for {task_id}: {parquet_path}')
            continue
        df = pd.read_parquet(parquet_path)
        batch = []
        for row_index, row in df.iterrows():
            payload = {str(key): jsonable(value) for key, value in row.to_dict().items()}
            batch.append({
                'task_id': task_id,
                'row_index': int(row_index),
                'split': rec['split'],
                'payload': json.dumps(payload),
            })
            if len(batch) >= int(os.getenv('POSTGRES_LOAD_BATCH_SIZE', '1000')):
                with engine.begin() as conn:
                    conn.execute(row_insert, batch)
                total_rows += len(batch)
                batch.clear()
        if batch:
            with engine.begin() as conn:
                conn.execute(row_insert, batch)
            total_rows += len(batch)
        print(f'Loaded dataset rows for {task_id}: {len(df)}')

    with engine.connect() as conn:
        summary = conn.execute(text(f'''
            select
                (select count(*) from {schema_q}.synthetic_tasks) as task_count,
                (select count(*) from {schema_q}.synthetic_dataset_rows) as dataset_row_count,
                (select count(distinct task_id) from {schema_q}.synthetic_dataset_rows) as dataset_task_count
        ''')).mappings().one()
    print('Validation summary:', dict(summary))
    print(f'Current run inserted/updated {total_rows} dataset row payloads')


if __name__ == '__main__':
    main()
