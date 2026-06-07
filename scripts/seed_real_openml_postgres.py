from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import psycopg2
from psycopg2.extras import Json

ROOT = Path("/workspace/self-improving-ml-agent")
TASKS = ROOT / "data/real_openml/tasks.jsonl"
REPORT = ROOT / "reports/real_openml_postgres_seed_report.json"
DSN = "host=localhost port=5432 dbname=agentic_ml user=agent password=change_me"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_schema(df: pd.DataFrame, target: str) -> dict:
    columns = []
    for column in df.columns:
        series = df[column]
        columns.append(
            {
                "name": str(column),
                "dtype": str(series.dtype),
                "role": "target" if column == target else "feature",
                "missing_count": int(series.isna().sum()),
                "missing_rate": round(float(series.isna().mean()), 6),
                "nunique": int(series.nunique(dropna=True)) if len(series) else 0,
            }
        )
    return {
        "column_count": int(len(df.columns)),
        "columns": columns,
        "feature_columns": [str(column) for column in df.columns if column != target],
        "target_column": target,
        "raw_rows_exposed": False,
    }


def safe_summary(df: pd.DataFrame, target: str) -> dict:
    target_counts = {}
    if target in df.columns:
        counts = df[target].value_counts(dropna=False).head(20)
        target_counts = {str(key): int(value) for key, value in counts.items()}
    dtype_counts = {str(key): int(value) for key, value in df.dtypes.astype(str).value_counts().items()}
    numeric_columns = [column for column in df.select_dtypes(include="number").columns if column != target][:25]
    numeric_aggregates = {}
    for column in numeric_columns:
        series = df[column]
        clean = series.dropna()
        numeric_aggregates[str(column)] = {
            "mean": None if clean.empty else round(float(series.mean()), 6),
            "std": None if clean.empty else round(float(series.std()), 6),
            "min": None if clean.empty else round(float(series.min()), 6),
            "max": None if clean.empty else round(float(series.max()), 6),
        }
    return {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "target_column": target,
        "class_balance": target_counts,
        "dtype_counts": dtype_counts,
        "missing_cells": int(df.isna().sum().sum()),
        "missing_rate_total": round(float(df.isna().sum().sum() / max(1, df.shape[0] * df.shape[1])), 6),
        "numeric_aggregates": numeric_aggregates,
        "raw_rows_exposed": False,
        "summary_type": "aggregate_only",
    }


def main() -> None:
    tasks = [json.loads(line) for line in TASKS.read_text(encoding="utf-8").splitlines() if line.strip()]
    conn = psycopg2.connect(DSN)
    conn.autocommit = False
    try:
        with conn.cursor() as cur:
            cur.execute("drop table if exists synthetic_dataset_rows cascade")
            cur.execute("drop table if exists synthetic_tasks cascade")
            cur.execute("drop table if exists synthetic_task_registry cascade")
            cur.execute(
                """
                create table if not exists real_benchmark_tasks (
                    task_id text primary key,
                    data_source text not null default 'openml',
                    payload jsonb not null,
                    dataset_ref text not null,
                    target_column text not null,
                    metric text not null,
                    split text,
                    status text not null default 'ready',
                    priority integer not null default 100,
                    created_at timestamptz not null default now(),
                    updated_at timestamptz not null default now()
                )
                """
            )
            cur.execute(
                """
                create table if not exists real_dataset_summaries (
                    task_id text primary key references real_benchmark_tasks(task_id) on delete cascade,
                    schema_json jsonb not null,
                    summary_json jsonb not null,
                    artifact_manifest jsonb not null,
                    raw_rows_exposed boolean not null default false,
                    created_at timestamptz not null default now(),
                    updated_at timestamptz not null default now()
                )
                """
            )
            cur.execute(
                """
                create table if not exists benchmark_run_status (
                    run_id text primary key,
                    task_id text references real_benchmark_tasks(task_id) on delete cascade,
                    track text not null,
                    policy_version text,
                    status text not null,
                    reward numeric,
                    metrics jsonb,
                    artifacts jsonb,
                    created_at timestamptz not null default now(),
                    updated_at timestamptz not null default now()
                )
                """
            )
            cur.execute(
                """
                create table if not exists tool_call_log (
                    id bigserial primary key,
                    timestamp timestamptz not null default now(),
                    task_id text,
                    tool_name text not null,
                    arguments jsonb,
                    status text not null,
                    output_ref text,
                    metadata jsonb
                )
                """
            )
            cur.execute("delete from real_dataset_summaries")
            cur.execute("delete from real_benchmark_tasks")
            loaded = []
            for task in tasks:
                task = dict(task)
                task["synthetic"] = False
                task["registered_in_postgres_at"] = now()
                task_id = task["task_id"]
                dataset_ref = task["dataset_ref"]
                target = task["target_column"]
                frame = pd.read_parquet(dataset_ref)
                schema = safe_schema(frame, target)
                summary = safe_summary(frame, target)
                manifest = {
                    "dataset_ref": dataset_ref,
                    "parquet_bytes": int(Path(dataset_ref).stat().st_size),
                    "approved_for_execution_layer": True,
                    "raw_rows_exposed_to_llm": False,
                    "source_url": task.get("openml_url"),
                }
                cur.execute(
                    """
                    insert into real_benchmark_tasks(task_id, data_source, payload, dataset_ref, target_column, metric, split, status, priority, updated_at)
                    values (%s, %s, %s, %s, %s, %s, %s, 'ready', 100, now())
                    on conflict (task_id) do update set
                      data_source=excluded.data_source,
                      payload=excluded.payload,
                      dataset_ref=excluded.dataset_ref,
                      target_column=excluded.target_column,
                      metric=excluded.metric,
                      split=excluded.split,
                      status='ready',
                      updated_at=now()
                    """,
                    (task_id, "openml", Json(task), dataset_ref, target, task.get("metric", "roc_auc"), task.get("split")),
                )
                cur.execute(
                    """
                    insert into real_dataset_summaries(task_id, schema_json, summary_json, artifact_manifest, raw_rows_exposed, updated_at)
                    values (%s, %s, %s, %s, false, now())
                    on conflict (task_id) do update set
                      schema_json=excluded.schema_json,
                      summary_json=excluded.summary_json,
                      artifact_manifest=excluded.artifact_manifest,
                      raw_rows_exposed=false,
                      updated_at=now()
                    """,
                    (task_id, Json(schema), Json(summary), Json(manifest)),
                )
                loaded.append({"task_id": task_id, "rows": int(len(frame)), "columns": int(len(frame.columns)), "target_column": target})
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        json.dumps({"seeded_at": now(), "task_count": len(loaded), "tasks": loaded, "raw_rows_exposed": False}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"task_count": len(loaded), "report": str(REPORT), "tasks": [item["task_id"] for item in loaded]}, indent=2))


if __name__ == "__main__":
    main()
