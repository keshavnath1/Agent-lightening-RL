"""Lightweight strict MCP PostgreSQL tool wrappers.

This module intentionally avoids importing the heavy PostgreSQL connector at
module import time. Each function delegates lazily to the canonical
``src.tools.postgres_tooling`` implementation when called. This keeps tests and
agent-layer imports clean while preserving one source of truth for business
logic.
"""
from __future__ import annotations

from importlib import import_module
from typing import Any

_CANONICAL = "src.tools.postgres_tooling"


def _tool(name: str):
    module = import_module(_CANONICAL)
    return getattr(module, name)


class MCPPostgresToolError(RuntimeError):
    """Import-light proxy error type for validation/tests."""


def postgres_get_task_metadata(task_id: str) -> dict[str, Any]:
    return _tool("postgres_get_task_metadata")(task_id)


def postgres_get_dataset_schema(task_id: str) -> dict[str, Any]:
    return _tool("postgres_get_dataset_schema")(task_id)


def postgres_get_dataset_summary(task_id: str) -> dict[str, Any]:
    return _tool("postgres_get_dataset_summary")(task_id)


def postgres_get_reward_history(task_id: str) -> dict[str, Any]:
    return _tool("postgres_get_reward_history")(task_id)


def postgres_get_rollout_status(task_id: str) -> dict[str, Any]:
    return _tool("postgres_get_rollout_status")(task_id)


def postgres_list_tasks(limit: int = 10, split: str | None = None, status: str | None = "ready") -> dict[str, Any]:
    return _tool("postgres_list_tasks")(limit=limit, split=split, status=status)


def postgres_get_next_task(status: str = "ready", limit: int = 1) -> dict[str, Any]:
    return _tool("postgres_get_next_task")(status=status, limit=limit)


def postgres_get_column_profile(task_id: str, column_name: str) -> dict[str, Any]:
    return _tool("postgres_get_column_profile")(task_id=task_id, column_name=column_name)


def postgres_get_target_profile(task_id: str) -> dict[str, Any]:
    return _tool("postgres_get_target_profile")(task_id=task_id)


def postgres_get_execution_dataset_source(task_id: str) -> dict[str, Any]:
    return _tool("postgres_get_execution_dataset_source")(task_id=task_id)


def get_dataset_rows(*_: Any, **__: Any) -> dict[str, Any]:
    raise RuntimeError("get_dataset_rows is disabled for MCP/agent access in strict mode.")


def get_dataset_sample(*_: Any, **__: Any) -> dict[str, Any]:
    raise RuntimeError("get_dataset_sample is disabled in strict mode.")


def get_dataset_as_frame_spec(*_: Any, **__: Any) -> dict[str, Any]:
    raise RuntimeError("get_dataset_as_frame_spec is disabled for agent-facing MCP in strict mode.")


POSTGRES_SAFE_TOOL_FUNCTIONS = {
    "postgres_get_task_metadata": postgres_get_task_metadata,
    "postgres_get_dataset_schema": postgres_get_dataset_schema,
    "postgres_get_dataset_summary": postgres_get_dataset_summary,
    "postgres_get_reward_history": postgres_get_reward_history,
    "postgres_get_rollout_status": postgres_get_rollout_status,
    "postgres_list_tasks": postgres_list_tasks,
    "postgres_get_next_task": postgres_get_next_task,
    "postgres_get_column_profile": postgres_get_column_profile,
    "postgres_get_target_profile": postgres_get_target_profile,
}

POSTGRES_EXECUTION_TOOL_FUNCTIONS = {
    "postgres_get_execution_dataset_source": postgres_get_execution_dataset_source,
}

__all__ = (
    sorted(POSTGRES_SAFE_TOOL_FUNCTIONS)
    + sorted(POSTGRES_EXECUTION_TOOL_FUNCTIONS)
    + ["MCPPostgresToolError", "POSTGRES_SAFE_TOOL_FUNCTIONS", "POSTGRES_EXECUTION_TOOL_FUNCTIONS"]
)
