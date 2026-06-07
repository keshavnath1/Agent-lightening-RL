from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.tools.sql_alchemy_connector import SQLAlchemyConnector

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOOL_LOG = PROJECT_ROOT / "reports" / "postgres_tool_calls.jsonl"

REGISTRY_TASKS_TABLE = "ml_registry.real_benchmark_tasks"
REGISTRY_SUMMARIES_TABLE = "ml_registry.real_dataset_summaries"
EXECUTION_SOURCES_TABLE = "ml_execution.dataset_sources"

MAX_TASK_LIMIT = 100
MCP_CONTRACT_VERSION = "task_metadata_v1"
EXECUTION_DATASET_SOURCE_CONTRACT_VERSION = "mcp_execution_dataset_source_v1"
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class MCPPostgresToolError(RuntimeError):
    """Raised when MCP PostgreSQL tools cannot safely satisfy a request."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return value
    return value


def _row_to_dict(row: dict[str, Any]) -> dict[str, Any]:
    data = dict(row)
    for key, value in list(data.items()):
        data[key] = _json_safe(value)
    return data


def _write_json(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return str(path)


def _artifact_path(task_id: str, name: str) -> Path:
    safe_task = task_id.replace("/", "_")
    return PROJECT_ROOT / "artifacts" / safe_task / f"{name}.json"


def _log_call(
    tool_name: str,
    arguments: dict[str, Any],
    status: str,
    output_ref: str | None,
    metadata: dict[str, Any],
    error: str | None = None,
) -> None:
    TOOL_LOG.parent.mkdir(parents=True, exist_ok=True)
    rec = {
        "timestamp": _now(),
        "tool_name": tool_name,
        "arguments": arguments,
        "status": status,
        "output_ref": output_ref,
        "error": error,
        "metadata": metadata,
    }
    with TOOL_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, default=str) + "\n")


def _connector() -> SQLAlchemyConnector:
    return SQLAlchemyConnector.from_env()


def _with_contract(payload: dict[str, Any], *, raw_rows_exposed: bool = False) -> dict[str, Any]:
    out = dict(payload)
    out.setdefault("contract_version", MCP_CONTRACT_VERSION)
    out.setdefault("raw_rows_exposed", raw_rows_exposed)
    out.setdefault("raw_rows_exposed_to_llm", False)
    return out


def _validate_safe_identifier(value: str, *, field: str) -> None:
    if not _IDENTIFIER_RE.match(value):
        raise MCPPostgresToolError(
            f"Invalid execution dataset source: {field} must be a safe SQL identifier."
        )


def _validate_task(task: dict[str, Any]) -> None:
    required = ["task_id", "dataset_key", "objective", "target_column", "primary_metric", "problem_type"]
    for field in required:
        if task.get(field) in (None, "", [], {}):
            raise MCPPostgresToolError(
                f"Invalid real_benchmark_tasks record: missing {field} for task_id={task.get('task_id', 'unknown')}."
            )
    if task.get("registry_source") != "postgresql_mcp":
        raise MCPPostgresToolError(
            "Invalid real_benchmark_tasks record: registry_source must be postgresql_mcp "
            f"for task_id={task['task_id']}."
        )
    if task.get("raw_rows_exposed_to_llm") is not False:
        raise MCPPostgresToolError(
            "Invalid real_benchmark_tasks record: raw_rows_exposed_to_llm must be false "
            f"for task_id={task['task_id']}."
        )


def _task_projection_sql() -> str:
    return """
      t.task_id,
      t.dataset_key,
      t.task_name,
      t.problem_statement,
      t.problem_statement as objective,
      t.target_column,
      t.problem_type,
      t.primary_metric,
      t.primary_metric as metric,
      t.secondary_metrics,
      t.status,
      t.priority,
      t.created_at as task_created_at,
      'postgresql_mcp' as registry_source,
      'postgresql_mcp' as registry_source_of_truth,
      false as raw_rows_exposed_to_llm,
      m.dataset_name,
      m.source,
      m.source as source_system,
      m.source_dataset_id,
      m.dataset_name as source_dataset_name,
      m.row_count,
      m.column_count,
      greatest(m.column_count - 1, 0) as feature_count,
      m.schema_json,
      m.dataset_summary,
      m.column_profiles,
      m.target_profile,
      m.raw_rows_exposed_to_llm as summary_raw_rows_exposed_to_llm,
      m.created_at as summary_created_at
    """


def _summary_projection_sql() -> str:
    return """
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
      raw_rows_exposed_to_llm,
      created_at
    """


def _normalize_task_row(row: dict[str, Any]) -> dict[str, Any]:
    task = _row_to_dict(row)
    task["objective"] = task.get("objective") or task.get("problem_statement")
    task["registry_table"] = REGISTRY_TASKS_TABLE
    task["summary_table"] = REGISTRY_SUMMARIES_TABLE
    task["raw_rows_exposed_to_llm"] = bool(task.get("raw_rows_exposed_to_llm")) is True
    task["raw_rows_exposed_to_llm"] = False
    if task.get("summary_raw_rows_exposed_to_llm") is not False:
        raise MCPPostgresToolError(
            "Invalid real_dataset_summaries record: raw_rows_exposed_to_llm must be false "
            f"for dataset_key={task.get('dataset_key')}."
        )
    return task


def list_ready_tasks(limit: int = 10, status: str | None = "ready", split: str | None = None) -> dict[str, Any]:
    """Return validated ready tasks from ml_registry metadata tables."""
    tool = "postgres_list_tasks"
    effective_limit = max(1, min(int(limit), MAX_TASK_LIMIT))
    args: dict[str, Any] = {"limit": effective_limit, "status": status, "split": split}

    clauses: list[str] = []
    params: dict[str, Any] = {"limit_rows": effective_limit}
    if status:
        clauses.append("t.status = :status")
        params["status"] = status
    if split:
        raise MCPPostgresToolError(
            "split filtering is not supported by ml_registry.real_benchmark_tasks yet."
        )
    where_sql = f"where {' and '.join(clauses)}" if clauses else ""

    try:
        rows = _connector().execute_internal(
            f"""
            select {_task_projection_sql()}
            from {REGISTRY_TASKS_TABLE} t
            join {REGISTRY_SUMMARIES_TABLE} m on m.dataset_key = t.dataset_key
            {where_sql}
            order by t.priority asc, t.created_at asc, t.task_id asc
            """,
            params,
            max_rows=effective_limit,
        )
        tasks = [_normalize_task_row(row) for row in rows]
        if not tasks:
            raise MCPPostgresToolError(
                "PostgreSQL ml_registry returned zero ready tasks for the requested limit."
            )
        for task in tasks:
            _validate_task(task)
        payload = _with_contract(
            {
                "tasks": tasks,
                "count": len(tasks),
                "limit": effective_limit,
                "status_filter": status,
                "split_filter": split,
                "registry_source": "postgresql_mcp",
                "raw_rows_exposed_to_llm": False,
            },
            raw_rows_exposed=False,
        )
        _log_call(
            tool,
            args,
            "success",
            None,
            {
                "row_count_returned": len(tasks),
                "raw_rows_exposed": False,
                "tables_accessed": [REGISTRY_TASKS_TABLE, REGISTRY_SUMMARIES_TABLE],
                "policy_allowed": True,
            },
        )
        return {"result": payload}
    except Exception as exc:
        _log_call(
            tool,
            args,
            "error",
            None,
            {
                "row_count_returned": 0,
                "raw_rows_exposed": False,
                "tables_accessed": [REGISTRY_TASKS_TABLE, REGISTRY_SUMMARIES_TABLE],
                "policy_allowed": False,
            },
            str(exc),
        )
        raise


def get_task_context(task_id: str) -> dict[str, Any]:
    """Return one canonical task context from ml_registry metadata tables."""
    tool = "postgres_get_task_metadata"
    args = {"task_id": task_id}
    if not task_id:
        raise MCPPostgresToolError("task_id is required.")

    try:
        rows = _connector().execute_internal(
            f"""
            select {_task_projection_sql()}
            from {REGISTRY_TASKS_TABLE} t
            join {REGISTRY_SUMMARIES_TABLE} m on m.dataset_key = t.dataset_key
            where t.task_id = :task_id
            """,
            {"task_id": task_id},
            max_rows=1,
        )
        if not rows:
            raise MCPPostgresToolError(
                f"No task found in ml_registry.real_benchmark_tasks for task_id={task_id}."
            )
        task = _normalize_task_row(rows[0])
        _validate_task(task)
        _log_call(
            tool,
            args,
            "success",
            None,
            {
                "row_count_returned": 1,
                "raw_rows_exposed": False,
                "tables_accessed": [REGISTRY_TASKS_TABLE, REGISTRY_SUMMARIES_TABLE],
                "policy_allowed": True,
            },
        )
        return {"result": _with_contract(task, raw_rows_exposed=False)}
    except Exception as exc:
        _log_call(
            tool,
            args,
            "error",
            None,
            {
                "row_count_returned": 0,
                "raw_rows_exposed": False,
                "tables_accessed": [REGISTRY_TASKS_TABLE, REGISTRY_SUMMARIES_TABLE],
                "policy_allowed": False,
            },
            str(exc),
        )
        raise


def get_dataset_metadata(dataset_key: str) -> dict[str, Any]:
    """Return one safe ml_registry.real_dataset_summaries record for dataset_key."""
    tool = "postgres_get_dataset_summary"
    args = {"dataset_key": dataset_key}
    if not dataset_key:
        raise MCPPostgresToolError("dataset_key is required.")

    try:
        rows = _connector().execute_internal(
            f"""
            select {_summary_projection_sql()}
            from {REGISTRY_SUMMARIES_TABLE}
            where dataset_key = :dataset_key
            """,
            {"dataset_key": dataset_key},
            max_rows=1,
        )
        if not rows:
            raise MCPPostgresToolError(
                f"No real_dataset_summaries record found for dataset_key={dataset_key}."
            )
        metadata = _row_to_dict(rows[0])
        if metadata.get("raw_rows_exposed_to_llm") is not False:
            raise MCPPostgresToolError(
                "Invalid real_dataset_summaries record: raw_rows_exposed_to_llm must be false "
                f"for dataset_key={dataset_key}."
            )
        _log_call(
            tool,
            args,
            "success",
            None,
            {
                "row_count_returned": 1,
                "raw_rows_exposed": False,
                "tables_accessed": [REGISTRY_SUMMARIES_TABLE],
                "policy_allowed": True,
            },
        )
        return {"result": _with_contract(metadata, raw_rows_exposed=False)}
    except Exception as exc:
        _log_call(
            tool,
            args,
            "error",
            None,
            {
                "row_count_returned": 0,
                "raw_rows_exposed": False,
                "tables_accessed": [REGISTRY_SUMMARIES_TABLE],
                "policy_allowed": False,
            },
            str(exc),
        )
        raise


def get_dataset_rows(dataset_key: str, limit: int | None = None) -> dict[str, Any]:
    """Disabled in strict MCP mode. Full rows are execution/materializer-only."""
    raise MCPPostgresToolError(
        "get_dataset_rows is disabled for MCP/agent access in strict mode. "
        "Use postgres_get_execution_dataset_source inside the Docker execution boundary."
    )


def get_dataset_sample(dataset_key: str, sample_size: int = 5) -> dict[str, Any]:
    """Disabled in strict MCP mode. Planning tools must not return raw sample values."""
    raise MCPPostgresToolError(
        "get_dataset_sample is disabled in strict mode because sample values can leak raw rows. "
        "Use postgres_get_dataset_summary or postgres_get_column_profile."
    )


def get_dataset_as_frame_spec(dataset_key: str, secret_env_var: str = "EXECUTION_DATABASE_URL") -> dict[str, Any]:
    """Disabled for agent-facing MCP. Execution receives a dataset source contract instead."""
    raise MCPPostgresToolError(
        "get_dataset_as_frame_spec is disabled for agent-facing MCP in strict mode. "
        "Execution materializers must use postgres_get_execution_dataset_source."
    )


def postgres_list_tasks(limit: int = 10, split: str | None = None, status: str | None = "ready") -> dict[str, Any]:
    return list_ready_tasks(limit=limit, status=status, split=split)


def postgres_get_next_task(status: str = "ready", limit: int = 1) -> dict[str, Any]:
    """Return the next ready task from ml_registry in task-oriented form."""
    listed = list_ready_tasks(limit=max(1, limit), status=status)
    tasks = listed["result"].get("tasks", [])
    if not tasks:
        raise MCPPostgresToolError("No ready task available from PostgreSQL registry.")
    task = tasks[0]
    payload = _with_contract(
        {
            "task_id": task.get("task_id"),
            "dataset_key": task.get("dataset_key"),
            "problem_statement": task.get("problem_statement") or task.get("objective"),
            "problem_type": task.get("problem_type"),
            "target_column": task.get("target_column"),
            "primary_metric": task.get("primary_metric"),
            "registry_source": "postgresql_mcp",
        },
        raw_rows_exposed=False,
    )
    return {"result": payload}


def postgres_get_task_metadata(task_id: str) -> dict[str, Any]:
    task = get_task_context(task_id)["result"]
    payload = _with_contract(
        {
            "task_id": task_id,
            "found": True,
            "metadata": {
                "task_id": task.get("task_id"),
                "dataset_key": task.get("dataset_key"),
                "task_name": task.get("task_name"),
                "problem_statement": task.get("problem_statement"),
                "objective": task.get("objective"),
                "target_column": task.get("target_column"),
                "problem_type": task.get("problem_type"),
                "primary_metric": task.get("primary_metric"),
                "secondary_metrics": task.get("secondary_metrics") or [],
                "status": task.get("status"),
                "priority": task.get("priority"),
                "registry_source": "postgresql_mcp",
                "raw_rows_exposed_to_llm": False,
            },
        },
        raw_rows_exposed=False,
    )
    output_ref = _write_json(_artifact_path(task_id, "postgres_get_task_metadata"), payload)
    return {"result": payload, "output_ref": output_ref}


def postgres_get_dataset_schema(task_id: str) -> dict[str, Any]:
    task = get_task_context(task_id)["result"]
    metadata = get_dataset_metadata(task["dataset_key"])["result"]
    schema = metadata.get("schema_json") or {}
    columns = schema.get("columns") or []
    payload = _with_contract(
        {
            "task_id": task_id,
            "dataset_key": task["dataset_key"],
            "column_count": metadata.get("column_count") or len(columns),
            "columns": columns,
            "target_column": task.get("target_column"),
            "schema": schema,
        },
        raw_rows_exposed=False,
    )
    output_ref = _write_json(_artifact_path(task_id, "postgres_get_dataset_schema"), payload)
    return {"result": payload, "output_ref": output_ref}


def postgres_get_dataset_summary(task_id: str) -> dict[str, Any]:
    task = get_task_context(task_id)["result"]
    metadata = get_dataset_metadata(task["dataset_key"])["result"]
    payload = _with_contract(
        {
            "task_id": task_id,
            "dataset_key": task["dataset_key"],
            "dataset_name": metadata.get("dataset_name"),
            "source": metadata.get("source"),
            "source_dataset_id": metadata.get("source_dataset_id"),
            "row_count": metadata.get("row_count"),
            "column_count": metadata.get("column_count"),
            "target_column": task.get("target_column"),
            "primary_metric": task.get("primary_metric"),
            "secondary_metrics": task.get("secondary_metrics") or [],
            "problem_type": task.get("problem_type"),
            "summary": metadata.get("dataset_summary") or {},
        },
        raw_rows_exposed=False,
    )
    output_ref = _write_json(_artifact_path(task_id, "postgres_get_dataset_summary"), payload)
    return {"result": payload, "output_ref": output_ref}


def postgres_get_column_profile(task_id: str, column_name: str) -> dict[str, Any]:
    """Return safe aggregate profile for one column without raw values."""
    if not task_id:
        raise MCPPostgresToolError("task_id is required.")
    if not column_name:
        raise MCPPostgresToolError("column_name is required.")

    task = get_task_context(task_id)["result"]
    metadata = get_dataset_metadata(task["dataset_key"])["result"]
    profiles = metadata.get("column_profiles") or []
    if not isinstance(profiles, list):
        profiles = []

    profile = None
    for entry in profiles:
        if isinstance(entry, dict) and str(entry.get("name") or entry.get("column_name") or "") == column_name:
            profile = entry
            break
    profile = dict(profile or {})

    data_type = str(profile.get("dtype") or profile.get("type") or "unknown").lower()
    is_numeric = any(token in data_type for token in ["int", "float", "double", "number", "numeric"])
    if is_numeric:
        payload = {
            "task_id": task_id,
            "dataset_key": task["dataset_key"],
            "column_name": column_name,
            "profile_type": "numeric",
            "stats": {
                "min": profile.get("min"),
                "max": profile.get("max"),
                "mean": profile.get("mean"),
                "median": profile.get("median"),
                "std": profile.get("std") or profile.get("stddev"),
                "missing_pct": profile.get("missing_pct") or profile.get("missing_rate"),
                "zero_pct": profile.get("zero_pct"),
                "outlier_hint": profile.get("outlier_hint"),
            },
            "raw_values_exposed": False,
        }
    else:
        payload = {
            "task_id": task_id,
            "dataset_key": task["dataset_key"],
            "column_name": column_name,
            "profile_type": "categorical",
            "cardinality": profile.get("cardinality"),
            "top_category_frequencies": profile.get("top_category_frequencies") or [],
            "unique_ratio": profile.get("unique_ratio"),
            "id_like": profile.get("id_like", False),
            "raw_values_exposed": False,
        }
    return {"result": _with_contract(payload, raw_rows_exposed=False)}


def postgres_get_target_profile(task_id: str) -> dict[str, Any]:
    """Return aggregate target analysis from metadata and summary only."""
    if not task_id:
        raise MCPPostgresToolError("task_id is required.")

    task = get_task_context(task_id)["result"]
    metadata = get_dataset_metadata(task["dataset_key"])["result"]
    target_profile = metadata.get("target_profile") or {}
    payload = {
        "task_id": task_id,
        "dataset_key": task["dataset_key"],
        "target_column": task.get("target_column"),
        "problem_type": task.get("problem_type"),
        "primary_metric": task.get("primary_metric"),
        "target_profile": target_profile,
        "raw_values_exposed": False,
    }
    return {"result": _with_contract(payload, raw_rows_exposed=False)}


def postgres_get_execution_dataset_source(task_id: str) -> dict[str, Any]:
    """Return execution-only PostgreSQL table source contract for a task."""
    tool = "postgres_get_execution_dataset_source"
    args = {"task_id": task_id}
    task = get_task_context(task_id)["result"]
    dataset_key = task["dataset_key"]
    try:
        rows = _connector().execute_internal(
            f"""
            select
              dataset_key,
              storage_backend,
              source_schema,
              source_table,
              row_count,
              column_count,
              checksum_sha256,
              raw_rows_exposed_to_llm,
              created_at
            from {EXECUTION_SOURCES_TABLE}
            where dataset_key = :dataset_key
            """,
            {"dataset_key": dataset_key},
            max_rows=1,
        )
        if not rows:
            raise MCPPostgresToolError(
                f"No dataset source found in ml_execution.dataset_sources for dataset_key={dataset_key}."
            )
        source = _row_to_dict(rows[0])
        if source.get("storage_backend") != "postgres_table":
            raise MCPPostgresToolError(
                f"Unsupported storage_backend for dataset_key={dataset_key}: {source.get('storage_backend')!r}."
            )
        if source.get("raw_rows_exposed_to_llm") is not False:
            raise MCPPostgresToolError(
                f"Invalid dataset source: raw_rows_exposed_to_llm must be false for dataset_key={dataset_key}."
            )
        _validate_safe_identifier(str(source.get("source_schema") or ""), field="source_schema")
        _validate_safe_identifier(str(source.get("source_table") or ""), field="source_table")
        payload = {
            "contract_version": EXECUTION_DATASET_SOURCE_CONTRACT_VERSION,
            "task_id": task_id,
            "dataset_key": dataset_key,
            "storage_backend": source["storage_backend"],
            "source_schema": source["source_schema"],
            "source_table": source["source_table"],
            "row_count": source.get("row_count"),
            "column_count": source.get("column_count"),
            "checksum_sha256": source.get("checksum_sha256"),
            "raw_rows_exposed_to_llm": False,
            "raw_rows_exposed": False,
            "execution_only": True,
        }
        output_ref = _write_json(_artifact_path(task_id, tool), payload)
        _log_call(
            tool,
            args,
            "success",
            output_ref,
            {
                "row_count_returned": 1,
                "raw_rows_exposed": False,
                "tables_accessed": [EXECUTION_SOURCES_TABLE],
                "policy_allowed": True,
                "execution_only": True,
            },
        )
        return {"result": payload, "output_ref": output_ref}
    except Exception as exc:
        _log_call(
            tool,
            args,
            "error",
            None,
            {
                "row_count_returned": 0,
                "raw_rows_exposed": False,
                "tables_accessed": [EXECUTION_SOURCES_TABLE],
                "policy_allowed": False,
                "execution_only": True,
            },
            str(exc),
        )
        raise


def postgres_get_reward_history(task_id: str) -> dict[str, Any]:
    rewards = []
    for path in sorted((PROJECT_ROOT / "trajectories").glob("**/*.jsonl")):
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("task_id") == task_id and rec.get("reward") is not None:
                rewards.append(
                    {
                        "trajectory_id": rec.get("trajectory_id"),
                        "reward": rec.get("reward"),
                        "source": str(path),
                    }
                )
    payload = _with_contract(
        {"task_id": task_id, "reward_count": len(rewards), "rewards": rewards[-20:]},
        raw_rows_exposed=False,
    )
    output_ref = _write_json(_artifact_path(task_id, "postgres_get_reward_history"), payload)
    return {"result": payload, "output_ref": output_ref}


def postgres_get_rollout_status(task_id: str) -> dict[str, Any]:
    counts = {"completed": 0, "failed": 0, "unknown": 0}
    paths = []
    for path in sorted((PROJECT_ROOT / "trajectories").glob("**/*.jsonl")):
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("task_id") == task_id:
                status = rec.get("final_status") or "unknown"
                counts[status] = counts.get(status, 0) + 1
                paths.append(str(path))
    payload = _with_contract(
        {"task_id": task_id, "status_counts": counts, "trajectory_sources": sorted(set(paths))[-20:]},
        raw_rows_exposed=False,
    )
    output_ref = _write_json(_artifact_path(task_id, "postgres_get_rollout_status"), payload)
    return {"result": payload, "output_ref": output_ref}


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


__all__ = [
    "EXECUTION_DATASET_SOURCE_CONTRACT_VERSION",
    "MCPPostgresToolError",
    "POSTGRES_EXECUTION_TOOL_FUNCTIONS",
    "POSTGRES_SAFE_TOOL_FUNCTIONS",
    "postgres_get_task_metadata",
    "postgres_get_dataset_schema",
    "postgres_get_dataset_summary",
    "postgres_get_reward_history",
    "postgres_get_rollout_status",
    "postgres_list_tasks",
    "postgres_get_next_task",
    "postgres_get_column_profile",
    "postgres_get_target_profile",
    "postgres_get_execution_dataset_source",
]
