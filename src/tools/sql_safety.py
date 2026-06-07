
from __future__ import annotations

import os
import re
from dataclasses import dataclass, asdict
from typing import Any

READONLY_PREFIXES = ('select', 'with', 'show', 'explain')
DEFAULT_BLOCKED_TABLES = {
    'synthetic_dataset_rows',
    'raw_dataset_rows',
    'dataset_rows',
}
DEFAULT_BLOCKED_SCHEMAS = {
    'ml_data',
}
DEFAULT_ALLOWED_TABLES = {
    'real_benchmark_tasks',
    'real_dataset_summaries',
    'dataset_sources',
    'benchmark_run_status',
    'tool_call_log',
    'trajectory_summary',
    'reward_summary',
    'pg_catalog',
}


@dataclass
class SQLSafetyResult:
    original_sql: str
    safe_sql: str
    tables_accessed: list[str]
    max_rows: int
    limit_added: bool
    raw_rows_exposed: bool
    policy_allowed: bool
    reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _csv_env(name: str, default: set[str]) -> set[str]:
    raw = os.getenv(name)
    if raw is None or raw.strip() == '':
        return set(default)
    return {x.strip().lower() for x in raw.split(',') if x.strip()}


def _strip_outer(sql: str) -> str:
    return sql.strip().strip('\ufeff')


def extract_tables(sql: str) -> list[str]:
    """Conservative table-name extractor for common SELECT/WITH/JOIN forms."""
    clean = re.sub(r'\s+', ' ', sql.lower())
    matches = re.findall(r'\b(?:from|join)\s+([a-zA-Z_][\w\.]*)(?:\s|$|,|\))', clean)
    tables: list[str] = []
    for name in matches:
        table = name.split('.')[-1].strip('"')
        if table and table not in tables:
            tables.append(table)
    return tables


def extract_schemas(sql: str) -> list[str]:
    clean = re.sub(r'\s+', ' ', sql.lower())
    matches = re.findall(r'\b(?:from|join)\s+([a-zA-Z_][\w]*)\.([a-zA-Z_][\w]*)(?:\s|$|,|\))', clean)
    schemas: list[str] = []
    for schema, _table in matches:
        schema = schema.strip('"')
        if schema and schema not in schemas:
            schemas.append(schema)
    return schemas


def _has_limit(sql: str) -> bool:
    return bool(re.search(r'\blimit\s+\d+\b', sql, flags=re.I))


def _limit_value(sql: str) -> int | None:
    m = re.search(r'\blimit\s+(\d+)\b', sql, flags=re.I)
    return int(m.group(1)) if m else None


def _enforce_limit(sql: str, max_rows: int, absolute_max_rows: int) -> tuple[str, bool]:
    limit = _limit_value(sql)
    if limit is None:
        return f"{sql.rstrip()} LIMIT {max_rows}", True
    if limit > absolute_max_rows:
        raise ValueError(f'Query LIMIT {limit} exceeds absolute maximum {absolute_max_rows}.')
    return sql, False


def validate_sql(
    sql: str,
    *,
    max_rows: int | None = None,
    absolute_max_rows: int | None = None,
    enforce_table_policy: bool = True,
    allow_information_schema: bool | None = None,
) -> SQLSafetyResult:
    """Validate and normalize agent-facing SQL.

    This validator is intentionally stricter than ordinary database access because
    it protects the agent training context from raw-row leakage and broad table
    dumps. Trusted internal loaders should call SQLAlchemyConnector.execute_internal.
    """
    original = sql
    sql = _strip_outer(sql)
    lowered = sql.lower()
    reasons: list[str] = []
    max_rows = int(max_rows or os.getenv('MCP_SQL_MAX_ROWS', '100'))
    absolute_max_rows = int(absolute_max_rows or os.getenv('MCP_SQL_ABSOLUTE_MAX_ROWS', '1000'))
    allow_information_schema = bool(
        os.getenv('MCP_SQL_ALLOW_INFORMATION_SCHEMA', '0') == '1'
        if allow_information_schema is None
        else allow_information_schema
    )

    if not sql:
        raise ValueError('SQL query is empty.')
    if not lowered.startswith(READONLY_PREFIXES):
        raise ValueError('Only read-style SQL is allowed through the PostgreSQL MCP-style tool layer.')
    if '--' in sql or '/*' in sql or '*/' in sql:
        raise ValueError('SQL comments are blocked to reduce prompt-injection and multi-statement risk.')
    if ';' in sql:
        raise ValueError('Semicolons and multi-statements are blocked in the agent-facing SQL tool.')
    if re.search(r'\bselect\s+\*(?:\s|,|$)', lowered) or re.search(r'\bselect\s+[\w\.]+\.\*(?:\s|,|$)', lowered):
        raise ValueError('SELECT * is blocked. Use safe task-specific PostgreSQL tools or explicit aggregate columns.')

    tables = extract_tables(sql)
    allowed_tables = _csv_env('MCP_SQL_ALLOWED_TABLES', DEFAULT_ALLOWED_TABLES)
    blocked_tables = _csv_env('MCP_SQL_BLOCKED_TABLES', DEFAULT_BLOCKED_TABLES)
    blocked_schemas = _csv_env('MCP_SQL_BLOCKED_SCHEMAS', DEFAULT_BLOCKED_SCHEMAS)
    schemas = extract_schemas(sql)

    if enforce_table_policy:
        for schema in schemas:
            if schema in blocked_schemas:
                raise ValueError(f'Schema {schema} is blocked for agent-facing raw SQL. Use execution-only materialization.')
        for table in tables:
            if table in blocked_tables:
                raise ValueError(f'Table {table} is blocked for agent-facing raw SQL. Use aggregate/safe PostgreSQL tools.')
            if table == 'information_schema' and not allow_information_schema:
                raise ValueError('information_schema access is blocked unless MCP_SQL_ALLOW_INFORMATION_SCHEMA=1.')
            if allowed_tables and table not in allowed_tables and table != 'pg_catalog':
                raise ValueError(f'Table {table} is not in MCP_SQL_ALLOWED_TABLES.')

    raw_rows_exposed = any(t in blocked_tables for t in tables)
    safe_sql = sql
    limit_added = False
    if lowered.startswith(('select', 'with')):
        safe_sql, limit_added = _enforce_limit(sql, max_rows, absolute_max_rows)
        if limit_added:
            reasons.append(f'LIMIT {max_rows} added automatically')

    return SQLSafetyResult(
        original_sql=original,
        safe_sql=safe_sql,
        tables_accessed=tables,
        max_rows=max_rows,
        limit_added=limit_added,
        raw_rows_exposed=raw_rows_exposed,
        policy_allowed=True,
        reasons=reasons,
    )
