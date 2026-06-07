from __future__ import annotations

import os
import re
from typing import Any

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from src.tools.sql_alchemy_connector import normalize_database_url


class SandboxPostgresDataError(RuntimeError):
    """Raised when the sandbox cannot safely materialize PostgreSQL dataset rows."""


_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _database_url(secret_env_var: str = "EXECUTION_DATABASE_URL") -> str:
    url = os.getenv(secret_env_var)
    if not url:
        raise SandboxPostgresDataError(
            f"PostgreSQL connection is not configured in the execution environment. Missing {secret_env_var}."
        )
    return normalize_database_url(url)


def _quote_identifier(value: str, *, field: str) -> str:
    if not _IDENTIFIER_RE.match(value):
        raise SandboxPostgresDataError(f"Invalid {field} in execution dataset source contract.")
    return '"' + value.replace('"', '""') + '"'


def make_engine(secret_env_var: str = "EXECUTION_DATABASE_URL") -> Engine:
    try:
        return create_engine(_database_url(secret_env_var), pool_pre_ping=True, future=True)
    except SandboxPostgresDataError:
        raise
    except Exception as exc:
        raise SandboxPostgresDataError(
            "Failed to initialize PostgreSQL engine in execution sandbox."
        ) from exc


def _validated_source(dataset_source: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(dataset_source, dict):
        raise SandboxPostgresDataError("dataset_source must be a dictionary.")
    if dataset_source.get("storage_backend") != "postgres_table":
        raise SandboxPostgresDataError(
            f"Unsupported storage_backend={dataset_source.get('storage_backend')!r}."
        )
    if dataset_source.get("raw_rows_exposed_to_llm") is not False:
        raise SandboxPostgresDataError("dataset_source raw_rows_exposed_to_llm must be false.")
    if dataset_source.get("execution_only") is not True:
        raise SandboxPostgresDataError("dataset_source must be marked execution_only=true.")

    source_schema = str(dataset_source.get("source_schema") or "")
    source_table = str(dataset_source.get("source_table") or "")
    return {
        **dataset_source,
        "source_schema": source_schema,
        "source_table": source_table,
        "quoted_schema": _quote_identifier(source_schema, field="source_schema"),
        "quoted_table": _quote_identifier(source_table, field="source_table"),
    }


def load_dataset_records_from_source(
    dataset_source: dict[str, Any],
    *,
    limit: int | None = None,
    secret_env_var: str = "EXECUTION_DATABASE_URL",
    engine: Engine | None = None,
) -> list[dict[str, Any]]:
    """Load rows from the trusted execution dataset source contract."""
    source = _validated_source(dataset_source)
    if limit is not None and limit <= 0:
        raise SandboxPostgresDataError("limit must be a positive integer when provided.")

    own_engine = engine is None
    engine = engine or make_engine(secret_env_var)
    query = f"SELECT * FROM {source['quoted_schema']}.{source['quoted_table']}"
    params: dict[str, Any] = {}
    if limit is not None:
        query += " LIMIT :limit"
        params["limit"] = limit

    try:
        with engine.begin() as conn:
            rows = conn.execute(text(query), params).mappings().all()
    except Exception as exc:
        raise SandboxPostgresDataError(
            f"Failed to load dataset rows for dataset_key={source.get('dataset_key')}."
        ) from exc
    finally:
        if own_engine:
            engine.dispose()

    if not rows:
        raise SandboxPostgresDataError(
            f"No dataset rows found for dataset_key={source.get('dataset_key')}."
        )
    return [dict(row) for row in rows]


def load_dataset_frame_from_source(
    dataset_source: dict[str, Any],
    *,
    limit: int | None = None,
    secret_env_var: str = "EXECUTION_DATABASE_URL",
    engine: Engine | None = None,
) -> pd.DataFrame:
    """Load dataset rows into a pandas DataFrame for sandbox execution."""
    records = load_dataset_records_from_source(
        dataset_source,
        limit=limit,
        secret_env_var=secret_env_var,
        engine=engine,
    )
    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        raise SandboxPostgresDataError(
            f"Dataset materialized as an empty DataFrame for dataset_key={dataset_source.get('dataset_key')}."
        )
    return frame


__all__ = [
    "SandboxPostgresDataError",
    "load_dataset_records_from_source",
    "load_dataset_frame_from_source",
    "make_engine",
]
