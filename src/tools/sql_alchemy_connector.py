
from __future__ import annotations

import os
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from src.tools.sql_safety import validate_sql


def normalize_database_url(database_url: str) -> str:
    """Return a SQLAlchemy-compatible PostgreSQL URL."""
    if database_url.startswith('postgres://'):
        return 'postgresql+psycopg2://' + database_url[len('postgres://'):]
    if database_url.startswith('postgresql://'):
        return 'postgresql+psycopg2://' + database_url[len('postgresql://'):]
    return database_url


def default_database_url() -> str:
    host = os.getenv('POSTGRES_HOST', 'localhost')
    port = os.getenv('POSTGRES_PORT', '5432')
    db = os.getenv('POSTGRES_DB', 'agentic_ml')
    user = os.getenv('POSTGRES_USER', 'agent')
    password = os.getenv('POSTGRES_PASSWORD', 'change_me')
    database_url = (
        os.getenv('MCP_DATABASE_URL')
        or os.getenv('DATABASE_URL')
        or f'postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}'
    )
    return normalize_database_url(database_url)


class SQLAlchemyConnector:
    @classmethod
    def from_env(cls) -> 'SQLAlchemyConnector':
        """Construct a connector using MCP_DATABASE_URL/DATABASE_URL/POSTGRES_* variables."""
        return cls()

    def __init__(self, database_url: str | None = None):
        self.database_url = normalize_database_url(database_url or default_database_url())
        self.engine: Engine = create_engine(
            self.database_url,
            future=True,
            pool_pre_ping=True,
            connect_args={'connect_timeout': int(os.getenv('POSTGRES_CONNECT_TIMEOUT', '30'))},
            use_native_hstore=False,
        )
        self.last_safety_metadata: dict | None = None

    def execute(self, sql: str, params: dict | None = None, *, max_rows: int | None = None) -> list[dict]:
        """Execute agent-facing raw SQL after strict safety validation."""
        validation = validate_sql(sql, max_rows=max_rows, enforce_table_policy=True)
        self.last_safety_metadata = validation.to_dict()
        with self.engine.connect() as conn:
            result = conn.execute(text(validation.safe_sql), params or {})
            return [dict(row._mapping) for row in result]

    def execute_with_metadata(self, sql: str, params: dict | None = None, *, max_rows: int | None = None) -> dict:
        rows = self.execute(sql, params=params, max_rows=max_rows)
        meta = self.last_safety_metadata or {}
        meta.update({'row_count_returned': len(rows)})
        return {'rows': rows, 'row_count': len(rows), 'safety': meta}

    def execute_internal(self, sql: str, params: dict | None = None, *, max_rows: int | None = None) -> list[dict]:
        """Execute trusted internal read SQL.

        This bypasses table allow/block rules for controlled server-side aggregate
        tools while still enforcing read-only, no comments, no multi-statements,
        no SELECT *, and row limits.
        """
        validation = validate_sql(sql, max_rows=max_rows, enforce_table_policy=False)
        self.last_safety_metadata = validation.to_dict()
        with self.engine.connect() as conn:
            result = conn.execute(text(validation.safe_sql), params or {})
            return [dict(row._mapping) for row in result]

    def healthcheck(self) -> bool:
        with self.engine.connect() as conn:
            conn.execute(text('select 1'))
        return True
