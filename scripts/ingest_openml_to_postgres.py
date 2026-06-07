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


IDENTIFIER_RE = re.compile(r"[^A-Za-z0-9_]+")


def safe_identifier(value: str) -> str:
    ident = IDENTIFIER_RE.sub("_", value.strip().lower()).strip("_")
    if not ident:
        raise ValueError("identifier cannot be empty")
    if ident[0].isdigit():
        ident = f"d_{ident}"
    return ident


def database_url(env_name: str) -> str:
    url = os.getenv(env_name) or os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError(f"Set {env_name} or DATABASE_URL before running ingestion.")
    return normalize_database_url(url)


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


def clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out.columns = [safe_identifier(str(c)) for c in out.columns]
    return out.astype(object).where(pd.notna(out), None)


def infer_problem_type(target: pd.Series, requested: str | None) -> str:
    if requested:
        return requested
    if pd.api.types.is_numeric_dtype(target) and target.nunique(dropna=True) > 20:
        return "regression"
    if target.nunique(dropna=True) <= 2:
        return "binary_classification"
    return "multiclass_classification"


def jsonable(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def frame_checksum(frame: pd.DataFrame) -> str:
    h = hashlib.sha256()
    for record in frame.to_dict(orient="records"):
        h.update(json.dumps(record, sort_keys=True, default=str).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def schema_json(frame: pd.DataFrame, target_column: str) -> dict[str, Any]:
    return {
        "target_column": target_column,
        "columns": [{"name": str(col), "dtype": str(frame[col].dtype)} for col in frame.columns],
        "row_count": int(len(frame)),
        "column_count": int(len(frame.columns)),
        "raw_rows_exposed_to_llm": False,
    }


def column_profiles(frame: pd.DataFrame) -> list[dict[str, Any]]:
    profiles: list[dict[str, Any]] = []
    for col in frame.columns:
        series = frame[col]
        missing_rate = float(series.isna().mean())
        profile: dict[str, Any] = {
            "name": str(col),
            "dtype": str(series.dtype),
            "missing_rate": missing_rate,
            "cardinality": int(series.nunique(dropna=True)),
            "raw_values_exposed": False,
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
        else:
            counts = series.astype(str).value_counts(dropna=True).head(10)
            profile["top_category_frequencies"] = [
                {"category": str(idx), "count": int(count)}
                for idx, count in counts.items()
            ]
        profiles.append(profile)
    return profiles


def target_profile(frame: pd.DataFrame, target_column: str, problem_type: str) -> dict[str, Any]:
    target = frame[target_column]
    profile: dict[str, Any] = {
        "target_column": target_column,
        "problem_type": problem_type,
        "missing_rate": float(target.isna().mean()),
        "cardinality": int(target.nunique(dropna=True)),
        "raw_values_exposed": False,
    }
    if problem_type == "regression":
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
        profile["class_balance"] = {
            str(label): float(count / total)
            for label, count in counts.items()
        }
    return profile


def dataset_summary(frame: pd.DataFrame, target_column: str, problem_type: str) -> dict[str, Any]:
    return {
        "row_count": int(len(frame)),
        "column_count": int(len(frame.columns)),
        "feature_count": int(len([c for c in frame.columns if c != target_column])),
        "target_column": target_column,
        "problem_type": problem_type,
        "missing_cells": int(frame.isna().sum().sum()),
        "raw_rows_exposed_to_llm": False,
    }


def _target_candidates(bunch: Any, requested_target_column: str) -> list[str]:
    if requested_target_column.lower() not in {"auto", "default", "__target__"}:
        return [requested_target_column]

    candidates: list[str] = []
    target_names = getattr(bunch, "target_names", None)
    if target_names:
        if isinstance(target_names, str):
            candidates.append(target_names)
        else:
            candidates.extend(str(name) for name in target_names)
    target = getattr(bunch, "target", None)
    target_name = getattr(target, "name", None)
    if target_name:
        candidates.append(str(target_name))
    details = getattr(bunch, "details", {}) or {}
    for key in ("default_target_attribute", "target", "target_attribute"):
        if details.get(key):
            candidates.append(str(details[key]))

    return list(dict.fromkeys(candidates))


def load_openml_frame(
    openml_id: int,
    target_column: str,
    canonical_target_column: str,
) -> tuple[pd.DataFrame, str, str]:
    from sklearn.datasets import fetch_openml

    bunch = fetch_openml(data_id=openml_id, as_frame=True)
    raw_frame = bunch.frame.copy()
    dataset_name = str(getattr(bunch, "details", {}).get("name") or f"openml_{openml_id}")

    frame = clean_frame(raw_frame)
    candidates = [safe_identifier(c) for c in _target_candidates(bunch, target_column)]
    resolved_target = next((candidate for candidate in candidates if candidate in frame.columns), None)

    if resolved_target is None:
        target = getattr(bunch, "target", None)
        if target is not None:
            fallback_target = candidates[0] if candidates else safe_identifier(canonical_target_column)
            frame[fallback_target] = target
            resolved_target = fallback_target
        else:
            raise RuntimeError(
                f"Target column {target_column!r} not found and OpenML target is unavailable."
            )

    canonical = safe_identifier(canonical_target_column)
    if resolved_target != canonical:
        if canonical in frame.columns:
            raise RuntimeError(
                f"Cannot rename OpenML target {resolved_target!r} to {canonical!r}; "
                "canonical target already exists."
            )
        frame = frame.rename(columns={resolved_target: canonical})
        resolved_target = canonical
    return frame, dataset_name, resolved_target


def install_contract_tables(engine) -> None:
    with engine.begin() as conn:
        for statement in [part.strip() for part in DDL.split(";") if part.strip()]:
            conn.execute(text(statement))


def write_registry_records(
    engine,
    *,
    frame: pd.DataFrame,
    dataset_key: str,
    task_id: str,
    task_name: str,
    dataset_name: str,
    openml_id: int,
    target_column: str,
    problem_type: str,
    primary_metric: str,
    secondary_metrics: list[str],
    status: str,
    priority: int,
    source_table: str,
    checksum_sha256: str,
) -> None:
    schema = schema_json(frame, target_column)
    profiles = column_profiles(frame)
    target = target_profile(frame, target_column, problem_type)
    summary = dataset_summary(frame, target_column, problem_type)
    problem_statement = (
        f"Train and evaluate a {problem_type} tabular model for OpenML dataset "
        f"{dataset_name} using target column {target_column}."
    )

    with engine.begin() as conn:
        conn.execute(
            text(
                """
                insert into ml_registry.real_benchmark_tasks(
                    task_id, dataset_key, task_name, problem_statement, target_column,
                    problem_type, primary_metric, secondary_metrics, status, priority
                )
                values (
                    :task_id, :dataset_key, :task_name, :problem_statement, :target_column,
                    :problem_type, :primary_metric, cast(:secondary_metrics as jsonb),
                    :status, :priority
                )
                on conflict (task_id) do update set
                    dataset_key=excluded.dataset_key,
                    task_name=excluded.task_name,
                    problem_statement=excluded.problem_statement,
                    target_column=excluded.target_column,
                    problem_type=excluded.problem_type,
                    primary_metric=excluded.primary_metric,
                    secondary_metrics=excluded.secondary_metrics,
                    status=excluded.status,
                    priority=excluded.priority
                """
            ),
            {
                "task_id": task_id,
                "dataset_key": dataset_key,
                "task_name": task_name,
                "problem_statement": problem_statement,
                "target_column": target_column,
                "problem_type": problem_type,
                "primary_metric": primary_metric,
                "secondary_metrics": json.dumps(secondary_metrics),
                "status": status,
                "priority": priority,
            },
        )
        conn.execute(
            text(
                """
                insert into ml_registry.real_dataset_summaries(
                    dataset_key, dataset_name, source, source_dataset_id, row_count,
                    column_count, schema_json, dataset_summary, column_profiles,
                    target_profile, raw_rows_exposed_to_llm
                )
                values (
                    :dataset_key, :dataset_name, 'openml', :source_dataset_id,
                    :row_count, :column_count, cast(:schema_json as jsonb),
                    cast(:dataset_summary as jsonb), cast(:column_profiles as jsonb),
                    cast(:target_profile as jsonb), false
                )
                on conflict (dataset_key) do update set
                    dataset_name=excluded.dataset_name,
                    source=excluded.source,
                    source_dataset_id=excluded.source_dataset_id,
                    row_count=excluded.row_count,
                    column_count=excluded.column_count,
                    schema_json=excluded.schema_json,
                    dataset_summary=excluded.dataset_summary,
                    column_profiles=excluded.column_profiles,
                    target_profile=excluded.target_profile,
                    raw_rows_exposed_to_llm=false
                """
            ),
            {
                "dataset_key": dataset_key,
                "dataset_name": dataset_name,
                "source_dataset_id": str(openml_id),
                "row_count": int(len(frame)),
                "column_count": int(len(frame.columns)),
                "schema_json": json.dumps(schema),
                "dataset_summary": json.dumps(summary),
                "column_profiles": json.dumps(profiles, default=str),
                "target_profile": json.dumps(target, default=str),
            },
        )
        conn.execute(
            text(
                """
                insert into ml_execution.dataset_sources(
                    dataset_key, storage_backend, source_schema, source_table,
                    row_count, column_count, checksum_sha256, raw_rows_exposed_to_llm
                )
                values (
                    :dataset_key, 'postgres_table', 'ml_data', :source_table,
                    :row_count, :column_count, :checksum_sha256, false
                )
                on conflict (dataset_key) do update set
                    storage_backend=excluded.storage_backend,
                    source_schema=excluded.source_schema,
                    source_table=excluded.source_table,
                    row_count=excluded.row_count,
                    column_count=excluded.column_count,
                    checksum_sha256=excluded.checksum_sha256,
                    raw_rows_exposed_to_llm=false
                """
            ),
            {
                "dataset_key": dataset_key,
                "source_table": source_table,
                "row_count": int(len(frame)),
                "column_count": int(len(frame.columns)),
                "checksum_sha256": checksum_sha256,
            },
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest an OpenML dataset into the strict PostgreSQL ML contract.")
    parser.add_argument("--openml-id", type=int, required=True)
    parser.add_argument("--dataset-key", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--target-column", required=True)
    parser.add_argument(
        "--canonical-target-column",
        default="target",
        help="Column name to expose to Track A after ingestion. Use this to normalize OpenML targets.",
    )
    parser.add_argument("--problem-type", choices=["binary_classification", "multiclass_classification", "regression"])
    parser.add_argument("--primary-metric", required=True)
    parser.add_argument("--secondary-metric", action="append", default=[])
    parser.add_argument("--task-name")
    parser.add_argument("--status", default="ready")
    parser.add_argument("--priority", type=int, default=100)
    parser.add_argument("--database-url-env", default="DATABASE_URL")
    parser.add_argument("--if-exists", choices=["fail", "replace", "append"], default="fail")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_key = safe_identifier(args.dataset_key)
    source_table = safe_identifier(dataset_key)
    task_name = args.task_name or f"{dataset_key}_baseline"
    engine = create_engine(database_url(args.database_url_env), future=True)

    install_contract_tables(engine)
    frame, dataset_name, target_column = load_openml_frame(
        args.openml_id,
        args.target_column,
        args.canonical_target_column,
    )
    problem_type = infer_problem_type(frame[target_column], args.problem_type)
    checksum = frame_checksum(frame)

    frame.to_sql(
        source_table,
        engine,
        schema="ml_data",
        if_exists=args.if_exists,
        index=False,
        method="multi",
        chunksize=1000,
    )
    write_registry_records(
        engine,
        frame=frame,
        dataset_key=dataset_key,
        task_id=args.task_id,
        task_name=task_name,
        dataset_name=dataset_name,
        openml_id=args.openml_id,
        target_column=target_column,
        problem_type=problem_type,
        primary_metric=args.primary_metric,
        secondary_metrics=args.secondary_metric,
        status=args.status,
        priority=args.priority,
        source_table=source_table,
        checksum_sha256=checksum,
    )
    print(
        json.dumps(
            {
                "ok": True,
                "task_id": args.task_id,
                "dataset_key": dataset_key,
                "row_count": int(len(frame)),
                "source": f"ml_data.{source_table}",
                "raw_rows_exposed_to_llm": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
