from __future__ import annotations

import os
from typing import Any

from mcp.server.fastmcp import FastMCP

from src.mcp_official.logging import log_mcp_tool_call
from src.mcp_official.prompts import register_prompts
from src.mcp_official.resources import register_resources
from src.tools.postgres_tooling import (
    postgres_get_column_profile as _postgres_get_column_profile,
    postgres_get_execution_dataset_source as _postgres_get_execution_dataset_source,
    postgres_get_next_task as _postgres_get_next_task,
    postgres_get_target_profile as _postgres_get_target_profile,
    postgres_get_dataset_schema as _postgres_get_dataset_schema,
    postgres_get_dataset_summary as _postgres_get_dataset_summary,
    postgres_get_reward_history as _postgres_get_reward_history,
    postgres_get_rollout_status as _postgres_get_rollout_status,
    postgres_get_task_metadata as _postgres_get_task_metadata,
    postgres_list_tasks as _postgres_list_tasks,
)
from src.tools.sql_alchemy_connector import SQLAlchemyConnector
from src.tools.sql_safety import validate_sql

mcp = FastMCP("self-improving-ml-agent-tools")


@mcp.tool()
@log_mcp_tool_call("postgres_get_task_metadata")
def postgres_get_task_metadata(task_id: str) -> dict[str, Any]:
    """Return safe task metadata without exposing raw dataset rows."""
    return _postgres_get_task_metadata(task_id)


@mcp.tool()
@log_mcp_tool_call("postgres_get_dataset_schema")
def postgres_get_dataset_schema(task_id: str) -> dict[str, Any]:
    """Return safe dataset schema information without exposing raw row values."""
    return _postgres_get_dataset_schema(task_id)


@mcp.tool()
@log_mcp_tool_call("postgres_get_dataset_summary")
def postgres_get_dataset_summary(task_id: str) -> dict[str, Any]:
    """Return aggregate dataset summary information without exposing raw rows."""
    return _postgres_get_dataset_summary(task_id)


@mcp.tool()
@log_mcp_tool_call("postgres_get_rollout_status")
def postgres_get_rollout_status(task_id: str) -> dict[str, Any]:
    """Return aggregate rollout status counts and trajectory references."""
    return _postgres_get_rollout_status(task_id)


@mcp.tool()
@log_mcp_tool_call("postgres_get_reward_history")
def postgres_get_reward_history(task_id: str) -> dict[str, Any]:
    """Return recent reward summaries for a task."""
    return _postgres_get_reward_history(task_id)


@mcp.tool()
@log_mcp_tool_call("postgres_list_tasks")
def postgres_list_tasks(
    limit: int = 10,
    split: str | None = None,
    status: str | None = None,
) -> dict[str, Any]:
    """Return a bounded list of metadata-only Track A task records.

    This is the safe MCP entry point for the supervisor to enumerate tasks from
    the PostgreSQL registry without exposing raw dataset rows to the LLM.
    """
    return _postgres_list_tasks(limit=limit, split=split, status=status)


@mcp.tool()
@log_mcp_tool_call("postgres_get_next_task")
def postgres_get_next_task(status: str = 'ready', limit: int = 1) -> dict[str, Any]:
    """Return one next-ready task from PostgreSQL registry in task-oriented form."""
    return _postgres_get_next_task(status=status, limit=limit)


@mcp.tool()
@log_mcp_tool_call("postgres_get_column_profile")
def postgres_get_column_profile(task_id: str, column_name: str) -> dict[str, Any]:
    """Return safe aggregate profile for one column without raw values."""
    return _postgres_get_column_profile(task_id=task_id, column_name=column_name)


@mcp.tool()
@log_mcp_tool_call("postgres_get_target_profile")
def postgres_get_target_profile(task_id: str) -> dict[str, Any]:
    """Return safe aggregate target analysis for the task dataset."""
    return _postgres_get_target_profile(task_id=task_id)


@mcp.tool()
@log_mcp_tool_call("postgres_get_execution_dataset_source")
def postgres_get_execution_dataset_source(task_id: str) -> dict[str, Any]:
    """Return execution-only PostgreSQL dataset source contract.

    This tool returns table location metadata for the Docker/materializer
    boundary. It must not return rows or credentials and is not intended for
    agent planning prompts.
    """
    return _postgres_get_execution_dataset_source(task_id=task_id)


@mcp.tool()
@log_mcp_tool_call("postgres_debug_readonly_sql")
def postgres_debug_readonly_sql(sql: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run developer-only read SQL when ALLOW_RAW_SQL_TOOL=1.

    This tool is disabled by default. When enabled, it still uses the strict SQL
    safety validator, which blocks write operations, comments, semicolons,
    SELECT *, and blocked tables such as raw synthetic dataset rows.
    """
    if os.getenv("ALLOW_RAW_SQL_TOOL", "0") != "1":
        return {
            "ok": False,
            "error": "postgres_debug_readonly_sql is disabled unless ALLOW_RAW_SQL_TOOL=1.",
            "raw_sql_debug_mode": False,
        }

    validation = validate_sql(sql, enforce_table_policy=True)
    connector = SQLAlchemyConnector.from_env()
    rows = connector.execute(validation.safe_sql, params=params, max_rows=validation.max_rows)
    return {
        "ok": True,
        "validated_sql": validation.safe_sql,
        "tables_accessed": validation.tables_accessed,
        "row_count": len(rows),
        "rows": rows,
        "raw_sql_debug_mode": True,
        "raw_rows_exposed": bool(validation.raw_rows_exposed),
        "safety": validation.to_dict(),
    }


register_resources(mcp)
register_prompts(mcp)


if __name__ == "__main__":
    mcp.run()
