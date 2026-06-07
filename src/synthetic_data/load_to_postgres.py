from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine, text
from src.tools.sql_alchemy_connector import default_database_url


def load_tasks(tasks_path: str, database_url: str | None = None) -> None:
    engine = create_engine(database_url or default_database_url(), future=True)
    with engine.begin() as conn:
        conn.execute(text(
            'create table if not exists synthetic_tasks ('
            'task_id text primary key, '
            'payload jsonb not null, '
            'split text not null)'
        ))
        for line in Path(tasks_path).read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            conn.execute(
                text(
                    'insert into synthetic_tasks(task_id, payload, split) '
                    'values (:task_id, cast(:payload as jsonb), :split) '
                    'on conflict (task_id) do update set payload = excluded.payload, split = excluded.split'
                ),
                {'task_id': rec['task_id'], 'payload': json.dumps(rec), 'split': rec['split']},
            )
    print(f'Loaded tasks from {tasks_path} into PostgreSQL.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks', default='data/synthetic/tasks.jsonl')
    args = parser.parse_args()
    load_tasks(args.tasks)
