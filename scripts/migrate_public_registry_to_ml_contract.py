#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from typing import Any

import pandas as pd
from sqlalchemy import create_engine, text

from src.tools.sql_alchemy_connector import normalize_database_url


DDL = """
CREATE SCHEMA IF NOT EXISTS ml_registry;
CREATE SCHEMA IF NOT EXISTS ml_data;
CREATE SCHEMA IF NOT EXISTS ml_execution;

CREATE TABLE IF NOT EXISTS ml_registry.real_benchmark_tasks (
    task_id TEXT PRIMARY KEY,
    dataset_key TEXT NOT NULL UNIQUE,
    task_name TEXT NOT NULL,
    problem_statement TEXT NOT NULL,
    target_column TEXT NOT NULL,
    problem_type TEXT NOT NULL CHECK (
        problem_type IN ('binary_classification', 'multiclass_classification', 'regression')
    ),
    primary_metric TEXT NOT NULL,
    secondary_metrics JSONB NOT NULL DEFAULT '[]'::jsonb,
    status TEXT NOT NULL DEFAULT 'ready',
    priority INTEGER NOT NULL DEFAULT 100,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS ml_registry.real_dataset_summaries (
    dataset_key TEXT PRIMARY KEY,
    dataset_name TEXT NOT NULL,
    source TEXT NOT NULL,
    source_dataset_id TEXT,
    row_count INTEGER NOT NULL,
    column_count INTEGER NOT NULL,
    schema_json JSONB NOT NULL,
    dataset_summary JSONB NOT NULL,
    column_profiles JSONB NOT NULL,
    target_profile JSONB NOT NULL,
    raw_rows_exposed_to_llm BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS ml_execution.dataset_sources (
    dataset_key TEXT PRIMARY KEY,
    storage_backend TEXT NOT NULL DEFAULT 'postgres_table',
    source_schema TEXT NOT NULL,
    source_table TEXT NOT NULL,
    row_count INTEGER NOT NULL,
    column_count INTEGER NOT NULL,
    checksum_sha256 TEXT,
    raw_rows_exposed_to_llm BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""

IDENTIFIER_RE = re.compile(r"[^A-Za-z0-9_]+")


def database_url(env_name: str) -> str:
    url = os.getenv(env_name) or os.getenv("MCP_DATABASE_URL") or os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError(f"Set {env_name}, MCP_DATABASE_URL, or DATABASE_URL before migration.")
    return normalize_database_url(url)


def safe_identifier(value: str) -> str:
    ident = IDENTIFIER_RE.sub("_", value.strip().lower()).strip("_")
    if not ident:
        raise ValueError("identifier cannot be empty")
    if ident[0].isdigit():
        ident = f"d_{ident}"
    return ident


def jsonable(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def json_obj(value: Any, fallback: dict[str, Any] | None = None) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str) and value.strip():
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return parsed
    return dict(fallback or {})


def json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return list(value)
    if isinstance(value, str) and value.strip():
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return parsed
    return []


def unique_list(values: list[Any]) -> list[Any]:
    seen: set[str] = set()
    out: list[Any] = []
    for value in values:
        key = json.dumps(value, sort_keys=True, default=str)
        if key not in seen:
            seen.add(key)
            out.append(value)
    return out


def install_contract_tables(engine) -> None:
    with engine.begin() as conn:
        for statement in [part.strip() for part in DDL.split(";") if part.strip()]:
            conn.execute(text(statement))


def load_legacy_tasks(engine) -> list[dict[str, Any]]:
    sql = text(
        """
        select
            t.task_id,
            t.dataset_key,
            t.status,
            t.priority,
            t.objective,
            t.problem_statement,
            t.prediction_goal,
            t.target_column,
            t.primary_metric,
            t.secondary_metrics as task_secondary_metrics,
            m.source_system,
            m.source_dataset_id,
            m.source_dataset_name,
            m.display_name,
            m.description,
            m.problem_type,
            m.row_count,
            m.column_count,
            m.feature_count,
            m.secondary_metrics as metadata_secondary_metrics,
            m.schema_json,
            m.column_profiles,
            m.dataset_summary,
            m.raw_rows_exposed_to_llm
        from public.tasks_registry t
        join public.metadata_information m on m.dataset_key = t.dataset_key
        order by coalesce(t.priority, 100) asc, t.task_id asc
        """
    )
    with engine.connect() as conn:
        return [dict(row) for row in conn.execute(sql).mappings().all()]


def load_dataset_frame(engine, dataset_key: str) -> pd.DataFrame:
    sql = text(
        """
        select row_data
        from public.dataset
        where dataset_key = :dataset_key
        order by row_number asc
        """
    )
    with engine.connect() as conn:
        records = [dict(row["row_data"]) for row in conn.execute(sql, {"dataset_key": dataset_key}).mappings()]
    if not records:
        raise RuntimeError(f"No public.dataset rows found for dataset_key={dataset_key!r}.")
    return pd.DataFrame.from_records(records)


def frame_checksum(frame: pd.DataFrame) -> str:
    h = hashlib.sha256()
    for record in frame.astype(object).where(pd.notna(frame), None).to_dict(orient="records"):
        h.update(json.dumps(record, sort_keys=True, default=str).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def normalize_problem_type(problem_type: str | None, frame: pd.DataFrame, target_column: str) -> str:
    requested = (problem_type or "").strip().lower()
    if requested in {"binary_classification", "multiclass_classification", "regression"}:
        return requested
    if requested == "regression":
        return "regression"
    if target_column in frame.columns:
        target = frame[target_column]
        if pd.api.types.is_numeric_dtype(target) and target.nunique(dropna=True) > 20:
            return "regression"
        if target.nunique(dropna=True) <= 2:
            return "binary_classification"
    return "multiclass_classification"


def ensure_schema_json(value: Any, frame: pd.DataFrame, target_column: str) -> dict[str, Any]:
    schema = json_obj(value)
    schema.setdefault(
        "columns",
        [{"name": str(col), "dtype": str(frame[col].dtype)} for col in frame.columns],
    )
    schema["target_column"] = target_column
    schema["row_count"] = int(len(frame))
    schema["column_count"] = int(len(frame.columns))
    schema["raw_rows_exposed_to_llm"] = False
    return schema


def ensure_dataset_summary(
    value: Any,
    frame: pd.DataFrame,
    *,
    dataset_key: str,
    target_column: str,
    problem_type: str,
) -> dict[str, Any]:
    summary = json_obj(value)
    summary.update(
        {
            "dataset_key": dataset_key,
            "row_count": int(len(frame)),
            "column_count": int(len(frame.columns)),
            "feature_count": int(max(len(frame.columns) - 1, 0)),
            "target_column": target_column,
            "problem_type": problem_type,
            "raw_rows_exposed_to_llm": False,
        }
    )
    return summary


def ensure_column_profiles(value: Any, frame: pd.DataFrame) -> list[dict[str, Any]]:
    profiles = [dict(item) for item in json_list(value) if isinstance(item, dict)]
    if not profiles:
        for column in frame.columns:
            series = frame[column]
            profile: dict[str, Any] = {
                "name": str(column),
                "dtype": str(series.dtype),
                "missing_rate": float(series.isna().mean()),
                "cardinality": int(series.nunique(dropna=True)),
            }
            if pd.api.types.is_numeric_dtype(series):
                numeric = pd.to_numeric(series, errors="coerce")
                profile.update(
                    {
                        "min": jsonable(numeric.min()),
                        "max": jsonable(numeric.max()),
                        "mean": jsonable(numeric.mean()),
                        "median": jsonable(numeric.median()),
                        "std": jsonable(numeric.std()),
                    }
                )
            profiles.append(profile)
    for profile in profiles:
        profile["raw_values_exposed"] = False
    return profiles


def ensure_target_profile(
    dataset_summary: dict[str, Any],
    frame: pd.DataFrame,
    *,
    target_column: str,
    problem_type: str,
) -> dict[str, Any]:
    target = frame[target_column] if target_column in frame.columns else pd.Series(dtype=object)
    profile: dict[str, Any] = {
        "target_column": target_column,
        "problem_type": problem_type,
        "missing_rate": float(target.isna().mean()) if len(target) else 0.0,
        "cardinality": int(target.nunique(dropna=True)) if len(target) else 0,
        "raw_values_exposed": False,
    }
    distribution = dataset_summary.get("target_distribution")
    if isinstance(distribution, dict) and problem_type != "regression":
        total = sum(int(v) for v in distribution.values()) or 1
        profile["class_balance"] = {str(k): float(int(v) / total) for k, v in distribution.items()}
    elif problem_type == "regression":
        numeric = pd.to_numeric(target, errors="coerce")
        profile.update(
            {
                "min": jsonable(numeric.min()),
                "max": jsonable(numeric.max()),
                "mean": jsonable(numeric.mean()),
                "median": jsonable(numeric.median()),
                "std": jsonable(numeric.std()),
            }
        )
    else:
        counts = target.astype(str).value_counts(dropna=True)
        total = max(int(counts.sum()), 1)
        profile["class_balance"] = {str(label): float(count / total) for label, count in counts.items()}
    return profile


def write_execution_table(engine, frame: pd.DataFrame, source_table: str, if_exists: str) -> None:
    frame.astype(object).where(pd.notna(frame), None).to_sql(
        source_table,
        engine,
        schema="ml_data",
        if_exists=if_exists,
        index=False,
        method="multi",
        chunksize=500,
    )


def upsert_registry_records(
    engine,
    *,
    legacy: dict[str, Any],
    frame: pd.DataFrame,
    source_table: str,
    problem_type: str,
    checksum: str,
) -> None:
    dataset_key = str(legacy["dataset_key"])
    target_column = str(legacy["target_column"])
    dataset_name = (
        legacy.get("display_name")
        or legacy.get("source_dataset_name")
        or dataset_key
    )
    source = legacy.get("source_system") or "legacy_public_registry"
    task_name = dataset_name
    problem_statement = (
        legacy.get("problem_statement")
        or legacy.get("objective")
        or f"Train and evaluate a supervised ML model for dataset {dataset_key}."
    )
    primary_metric = legacy.get("primary_metric") or ("rmse" if problem_type == "regression" else "roc_auc")
    secondary_metrics = unique_list(
        json_list(legacy.get("task_secondary_metrics")) + json_list(legacy.get("metadata_secondary_metrics"))
    )
    schema_json = ensure_schema_json(legacy.get("schema_json"), frame, target_column)
    dataset_summary = ensure_dataset_summary(
        legacy.get("dataset_summary"),
        frame,
        dataset_key=dataset_key,
        target_column=target_column,
        problem_type=problem_type,
    )
    column_profiles = ensure_column_profiles(legacy.get("column_profiles"), frame)
    target_profile = ensure_target_profile(
        dataset_summary,
        frame,
        target_column=target_column,
        problem_type=problem_type,
    )

    with engine.begin() as conn:
        conn.execute(
            text(
                """
                insert into ml_registry.real_benchmark_tasks (
                    task_id,
                    dataset_key,
                    task_name,
                    problem_statement,
                    target_column,
                    problem_type,
                    primary_metric,
                    secondary_metrics,
                    status,
                    priority
                )
                values (
                    :task_id,
                    :dataset_key,
                    :task_name,
                    :problem_statement,
                    :target_column,
                    :problem_type,
                    :primary_metric,
                    cast(:secondary_metrics as jsonb),
                    :status,
                    :priority
                )
                on conflict (task_id) do update set
                    dataset_key = excluded.dataset_key,
                    task_name = excluded.task_name,
                    problem_statement = excluded.problem_statement,
                    target_column = excluded.target_column,
                    problem_type = excluded.problem_type,
                    primary_metric = excluded.primary_metric,
                    secondary_metrics = excluded.secondary_metrics,
                    status = excluded.status,
                    priority = excluded.priority
                """
            ),
            {
                "task_id": legacy["task_id"],
                "dataset_key": dataset_key,
                "task_name": task_name,
                "problem_statement": problem_statement,
                "target_column": target_column,
                "problem_type": problem_type,
                "primary_metric": primary_metric,
                "secondary_metrics": json.dumps(secondary_metrics, default=str),
                "status": legacy.get("status") or "ready",
                "priority": int(legacy.get("priority") or 100),
            },
        )
        conn.execute(
            text(
                """
                insert into ml_registry.real_dataset_summaries (
                    dataset_key,
                    dataset_name,
                    source,
                    source_dataset_id,
                    row_count,
                    column_count,
                    schema_json,
                    dataset_summary,
                    column_profiles,
                    target_profile,
                    raw_rows_exposed_to_llm
                )
                values (
                    :dataset_key,
                    :dataset_name,
                    :source,
                    :source_dataset_id,
                    :row_count,
                    :column_count,
                    cast(:schema_json as jsonb),
                    cast(:dataset_summary as jsonb),
                    cast(:column_profiles as jsonb),
                    cast(:target_profile as jsonb),
                    false
                )
                on conflict (dataset_key) do update set
                    dataset_name = excluded.dataset_name,
                    source = excluded.source,
                    source_dataset_id = excluded.source_dataset_id,
                    row_count = excluded.row_count,
                    column_count = excluded.column_count,
                    schema_json = excluded.schema_json,
                    dataset_summary = excluded.dataset_summary,
                    column_profiles = excluded.column_profiles,
                    target_profile = excluded.target_profile,
                    raw_rows_exposed_to_llm = false
                """
            ),
            {
                "dataset_key": dataset_key,
                "dataset_name": dataset_name,
                "source": source,
                "source_dataset_id": legacy.get("source_dataset_id"),
                "row_count": int(len(frame)),
                "column_count": int(len(frame.columns)),
                "schema_json": json.dumps(schema_json, default=str),
                "dataset_summary": json.dumps(dataset_summary, default=str),
                "column_profiles": json.dumps(column_profiles, default=str),
                "target_profile": json.dumps(target_profile, default=str),
            },
        )
        conn.execute(
            text(
                """
                insert into ml_execution.dataset_sources (
                    dataset_key,
                    storage_backend,
                    source_schema,
                    source_table,
                    row_count,
                    column_count,
                    checksum_sha256,
                    raw_rows_exposed_to_llm
                )
                values (
                    :dataset_key,
                    'postgres_table',
                    'ml_data',
                    :source_table,
                    :row_count,
                    :column_count,
                    :checksum_sha256,
                    false
                )
                on conflict (dataset_key) do update set
                    storage_backend = 'postgres_table',
                    source_schema = 'ml_data',
                    source_table = excluded.source_table,
                    row_count = excluded.row_count,
                    column_count = excluded.column_count,
                    checksum_sha256 = excluded.checksum_sha256,
                    raw_rows_exposed_to_llm = false
                """
            ),
            {
                "dataset_key": dataset_key,
                "source_table": source_table,
                "row_count": int(len(frame)),
                "column_count": int(len(frame.columns)),
                "checksum_sha256": checksum,
            },
        )


def migrate(args: argparse.Namespace) -> None:
    engine = create_engine(database_url(args.database_url_env), pool_pre_ping=True)
    install_contract_tables(engine)
    legacy_tasks = load_legacy_tasks(engine)
    if not legacy_tasks:
        raise RuntimeError("No legacy public.tasks_registry rows found to migrate.")

    migrated: list[dict[str, Any]] = []
    for legacy in legacy_tasks:
        if legacy.get("raw_rows_exposed_to_llm") is not False:
            raise RuntimeError(
                f"Refusing to migrate dataset_key={legacy['dataset_key']!r}: "
                "metadata_information.raw_rows_exposed_to_llm must be false."
            )
        dataset_key = str(legacy["dataset_key"])
        frame = load_dataset_frame(engine, dataset_key)
        target_column = str(legacy["target_column"])
        problem_type = normalize_problem_type(legacy.get("problem_type"), frame, target_column)
        source_table = safe_identifier(dataset_key)
        write_execution_table(engine, frame, source_table, args.if_exists)
        checksum = frame_checksum(frame)
        upsert_registry_records(
            engine,
            legacy=legacy,
            frame=frame,
            source_table=source_table,
            problem_type=problem_type,
            checksum=checksum,
        )
        migrated.append(
            {
                "task_id": legacy["task_id"],
                "dataset_key": dataset_key,
                "source_table": f"ml_data.{source_table}",
                "row_count": int(len(frame)),
                "column_count": int(len(frame.columns)),
                "problem_type": problem_type,
            }
        )

    print(json.dumps({"migrated": migrated, "count": len(migrated)}, indent=2, default=str))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Migrate legacy public.tasks_registry/metadata_information/dataset rows "
            "into the strict ml_registry/ml_execution/ml_data PostgreSQL contract."
        )
    )
    parser.add_argument("--database-url-env", default="MCP_DATABASE_URL")
    parser.add_argument(
        "--if-exists",
        choices=["fail", "replace", "append"],
        default="fail",
        help="Behavior for generated ml_data tables.",
    )
    return parser


def main() -> None:
    migrate(build_parser().parse_args())


if __name__ == "__main__":
    main()
