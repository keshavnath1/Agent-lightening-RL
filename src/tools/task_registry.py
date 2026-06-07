from __future__ import annotations

import json
import os
from typing import Any



REGISTRY_TASK_TABLE = 'ml_registry.real_benchmark_tasks'
REGISTRY_SUMMARY_TABLE = 'ml_registry.real_dataset_summaries'

TRACKA_DEFAULT_TASK_LIMIT: int = int(os.getenv('TRACKA_DEFAULT_TASK_LIMIT', '1'))
TRACKA_TASK_SOURCE: str = str(os.getenv('TRACKA_TASK_SOURCE', 'postgres_registry')).strip().lower()

STRICT_TRACKA_ALLOWED_SOURCES = {'postgres_registry', 'postgres_mcp'}


_REQUIRED_TASK_FIELDS: tuple[tuple[str, ...], ...] = (
    ('task_id',),
    ('dataset_key',),
    ('objective', 'problem_statement'),
    ('target_column',),
    ('primary_metric', 'metric'),
)


class TrackATaskLoadError(RuntimeError):
    """Raised when the PostgreSQL/MCP task registry cannot supply valid tasks.

    Always contains a sanitized message that never includes database URLs,
    credentials, hostnames, tokens, or secret paths.
    """


def _truthy(value: str | None) -> bool:
    return str(value or '').strip().lower() in {'1', 'true', 'yes', 'on'}


def _json_obj(value: Any, fallback: dict[str, Any] | None = None) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return parsed
    return dict(fallback or {})


def _json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value.strip():
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return parsed
    return []


def _normalize_task(row: dict[str, Any]) -> dict[str, Any]:
    payload = _json_obj(row.get('payload'))
    summary_json = _json_obj(row.get('summary_json'))
    schema_json = _json_obj(row.get('schema_json'))
    secondary_metrics = _json_list(row.get('secondary_metrics'))
    target_profile = _json_obj(row.get('target_profile'))

    task = dict(payload)
    task.update({
        'task_id': row['task_id'],
        'dataset_key': row.get('dataset_key'),
        'source': row.get('source') or 'OpenML',
        'source_task_id': row.get('source_task_id'),
        'data_source': row.get('data_source') or row.get('source') or 'OpenML',
        'openml_data_id': row.get('openml_data_id'),
        'openml_name': row.get('openml_name'),
        'openml_version': row.get('openml_version'),
        'openml_url': row.get('openml_url'),
        'dataset_name': row.get('dataset_name') or row.get('source_dataset_name'),
        'domain': row.get('domain'),
        'problem_statement': row.get('problem_statement'),
        'prediction_goal': row.get('prediction_goal'),
        'problem_type': row.get('problem_type'),
        'objective': row.get('objective') or row.get('problem_statement'),
        'target_column': row.get('target_column'),
        'primary_metric': row.get('primary_metric') or row.get('metric'),
        'secondary_metrics': secondary_metrics,
        'metric': row.get('metric') or row.get('primary_metric'),
        'row_count': row.get('row_count'),
        'column_count': row.get('column_count'),
        'split': row.get('split'),
        'synthetic': False,
        'status': row.get('status'),
        'priority': row.get('priority'),
        'registry_table': row.get('registry_table') or REGISTRY_TASK_TABLE,
        'registry_source': row.get('registry_source') or 'postgresql_mcp',
        'registry_source_of_truth': row.get('registry_source_of_truth') or 'postgresql_mcp',
        'raw_rows_exposed_to_llm': False,
        'schema_summary': schema_json,
        'schema_json': schema_json,
        'dataset_summary': summary_json,
        'column_profiles': _json_list(row.get('column_profiles')),
        'target_profile': target_profile,
    })
    return task



def _validate_task(task: dict[str, Any]) -> None:
    """Raise TrackATaskLoadError if any required field group is entirely absent."""
    task_id = task.get('task_id', '<unknown>')
    for field_group in _REQUIRED_TASK_FIELDS:
        if not any(task.get(f) for f in field_group):
            missing = ' or '.join(field_group)
            raise TrackATaskLoadError(
                f'PostgreSQL registry task metadata is invalid: '
                f'missing required field {missing!r} for task_id={task_id!r}.'
            )
    if task.get('registry_source') != 'postgresql_mcp':
        raise TrackATaskLoadError(
            'PostgreSQL registry task metadata is invalid: '
            f"registry_source must be 'postgresql_mcp' for task_id={task_id!r}."
        )
    if task.get('raw_rows_exposed_to_llm') is not False:
        raise TrackATaskLoadError(
            'PostgreSQL registry task metadata is invalid: '
            f'raw_rows_exposed_to_llm must be false for task_id={task_id!r}.'
        )


def load_tracka_tasks_from_postgres(
    limit: int | None = None,
    split: str | None = None,
    status: str | None = None,
    task_id: str | None = None,
) -> list[dict[str, Any]]:
    """Load metadata-only Track A tasks through the MCP tool layer.

    Delegates entirely to ``postgres_list_tasks`` from ``ml_tools.postgres_tooling``,
    which is the registered MCP tool.  No direct database connection is opened here;
    all PostgreSQL access happens inside the MCP tool, which enforces:
    - metadata-only columns (no raw dataset rows)
    - tool-call logging to reports/postgres_tool_calls.jsonl
    - row-limit cap (max 100 per call)

    If the MCP tool raises, the query returns zero rows, or any task record fails
    field validation, ``TrackATaskLoadError`` is raised immediately with a
    sanitized message.  Secrets are never included in exceptions or logs.
    """
    try:
        from ml_tools.postgres_tooling import postgres_list_tasks as _mcp_list_tasks  # type: ignore[import]
    except ImportError:
        try:
            from src.tools.postgres_tooling import postgres_list_tasks as _mcp_list_tasks
        except ImportError as exc:
            raise TrackATaskLoadError(
                'Track A requires a postgres_list_tasks MCP tool implementation. '
                'Ensure either ml_tools.postgres_tooling or src.tools.postgres_tooling is importable.'
            ) from exc

    effective_limit = int(
        limit
        if limit is not None
        else os.getenv('TRACKA_REGISTRY_LIMIT', str(TRACKA_DEFAULT_TASK_LIMIT))
    )

    resolved_status = status or os.getenv('TRACKA_REGISTRY_STATUS') or None
    resolved_split  = split  or os.getenv('TRACKA_REGISTRY_SPLIT')  or None

    try:
        result = _mcp_list_tasks(
            limit=effective_limit,
            split=resolved_split,
            status=resolved_status or 'ready',
        )
    except TrackATaskLoadError:
        raise
    except Exception as exc:
        raise TrackATaskLoadError(
            'Failed to load Track A tasks via MCP postgres_list_tasks tool. '
            f'Cause: {exc.__class__.__name__}. '
            'Check MCP tool configuration and PostgreSQL registry table availability.'
        ) from exc

    rows: list[dict[str, Any]] = result.get('result', {}).get('tasks', [])

    if not rows:
        raise TrackATaskLoadError(
            'postgres_list_tasks MCP tool returned zero Track A tasks for the requested '
            f'limit={effective_limit} / status={resolved_status!r} / split={resolved_split!r} filters. '
            'Ensure the registry is seeded and the filters match available records.'
        )

    tasks: list[dict[str, Any]] = []
    for raw in rows:
        try:
            task = _normalize_task(raw)
            task['registry_source_of_truth'] = 'postgresql_mcp'
        except Exception as exc:
            raise TrackATaskLoadError(
                'Failed to normalize task record returned by postgres_list_tasks MCP tool. '
                f'Cause: {exc.__class__.__name__}: {exc}'
            ) from exc
        _validate_task(task)
        tasks.append(task)

    return tasks
