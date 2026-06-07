from __future__ import annotations

"""Legacy FastAPI compatibility server.

The official Project MCP server under services/project_mcp_server is the
canonical agent-facing tool interface. This REST server remains only for older
local smoke tooling; it exposes the same task-first safe metadata tools and has
no raw SQL or raw-row endpoints.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.tools.ydata_profiler import YDataProfilingTool
from src.tools.postgres_tooling import (
    postgres_get_column_profile,
    postgres_get_dataset_schema,
    postgres_get_dataset_summary,
    postgres_get_execution_dataset_source,
    postgres_get_next_task,
    postgres_get_reward_history,
    postgres_get_rollout_status,
    postgres_get_target_profile,
    postgres_get_task_metadata,
    postgres_list_tasks,
)

app = FastAPI(title='Agentic ML MCP-Style Compatibility Server')


def _tool(name: str, description: str, endpoint: str, input_schema: dict, output_schema: dict, *, risk_level: str = 'low') -> dict:
    return {
        'name': name,
        'description': description,
        'category': 'postgres_registry' if name.startswith('postgres_') else 'execution',
        'endpoint': endpoint,
        'method': 'POST',
        'input_schema': input_schema,
        'output_schema': output_schema,
        'risk_level': risk_level,
        'success_criteria': ['task-oriented input', 'raw_rows_exposed=false', 'no raw sample values'],
        'failure_modes': ['registry unavailable', 'invalid task_id', 'metadata missing'],
    }


TOOL_SCHEMA = [
    _tool('postgres_get_next_task', 'Return next ready task from PostgreSQL registry.', '/tools/postgres/next_task', {'status': 'string', 'limit': 'integer'}, {'result': 'object'}),
    _tool('postgres_list_tasks', 'Return bounded metadata-only Track A tasks.', '/tools/postgres/list_tasks', {'limit': 'integer', 'split': 'string|null', 'status': 'string|null'}, {'result': 'object'}),
    _tool('postgres_get_task_metadata', 'Fetch one task metadata record without exposing rows.', '/tools/postgres/task_metadata', {'task_id': 'string'}, {'result': 'object'}),
    _tool('postgres_get_dataset_schema', 'Return task dataset schema metadata without raw row values.', '/tools/postgres/dataset_schema', {'task_id': 'string'}, {'result': 'object'}),
    _tool('postgres_get_dataset_summary', 'Return aggregate dataset summary for planning.', '/tools/postgres/dataset_summary', {'task_id': 'string'}, {'result': 'object'}),
    _tool('postgres_get_column_profile', 'Return aggregate profile for one safe column.', '/tools/postgres/column_profile', {'task_id': 'string', 'column_name': 'string'}, {'result': 'object'}),
    _tool('postgres_get_target_profile', 'Return safe aggregate target profile.', '/tools/postgres/target_profile', {'task_id': 'string'}, {'result': 'object'}),
    _tool('postgres_get_execution_dataset_source', 'Return execution-only PostgreSQL dataset source contract.', '/tools/postgres/execution_dataset_source', {'task_id': 'string'}, {'result': 'object'}, risk_level='medium'),
    _tool('postgres_get_rollout_status', 'Return rollout status counts.', '/tools/postgres/rollout_status', {'task_id': 'string'}, {'result': 'object'}),
    _tool('postgres_get_reward_history', 'Return bounded reward history.', '/tools/postgres/reward_history', {'task_id': 'string'}, {'result': 'object'}),
    _tool('YDataProfiler', 'Execution-boundary profiling for approved parquet artifacts only.', '/tools/profile/parquet', {'parquet_path': 'string', 'output_json': 'string', 'output_html': 'string|null'}, {'summary': 'object'}, risk_level='medium'),
]


class TaskRequest(BaseModel):
    task_id: str


class ListTasksRequest(BaseModel):
    limit: int = 10
    split: str | None = None
    status: str | None = 'ready'


class NextTaskRequest(BaseModel):
    status: str = 'ready'
    limit: int = 1


class ColumnProfileRequest(BaseModel):
    task_id: str
    column_name: str


class ProfileRequest(BaseModel):
    parquet_path: str
    output_json: str
    output_html: str | None = None


@app.get('/health')
def health() -> dict:
    return {'status': 'ok', 'strict_mode': True}


@app.get('/tools')
def list_tools() -> dict:
    return {
        'tools': TOOL_SCHEMA,
        'count': len(TOOL_SCHEMA),
        'protocol_note': 'Compatibility REST catalog. Official MCP server is services/project_mcp_server.',
        'raw_sql_available': False,
    }


@app.post('/tools/sql/query')
def sql_query_disabled() -> dict:
    raise HTTPException(status_code=410, detail='Raw SQL REST endpoint removed. Use official MCP safe task tools.')


@app.post('/tools/postgres/next_task')
def next_task(req: NextTaskRequest) -> dict:
    return postgres_get_next_task(status=req.status, limit=req.limit)


@app.post('/tools/postgres/list_tasks')
def tasks(req: ListTasksRequest) -> dict:
    return postgres_list_tasks(limit=req.limit, split=req.split, status=req.status)


@app.post('/tools/postgres/task_metadata')
def task_metadata(req: TaskRequest) -> dict:
    return postgres_get_task_metadata(req.task_id)


@app.post('/tools/postgres/dataset_schema')
def dataset_schema(req: TaskRequest) -> dict:
    return postgres_get_dataset_schema(req.task_id)


@app.post('/tools/postgres/dataset_summary')
def dataset_summary(req: TaskRequest) -> dict:
    return postgres_get_dataset_summary(req.task_id)


@app.post('/tools/postgres/column_profile')
def column_profile(req: ColumnProfileRequest) -> dict:
    return postgres_get_column_profile(req.task_id, req.column_name)


@app.post('/tools/postgres/target_profile')
def target_profile(req: TaskRequest) -> dict:
    return postgres_get_target_profile(req.task_id)


@app.post('/tools/postgres/execution_dataset_source')
def execution_dataset_source(req: TaskRequest) -> dict:
    return postgres_get_execution_dataset_source(req.task_id)


@app.post('/tools/postgres/rollout_status')
def rollout_status(req: TaskRequest) -> dict:
    return postgres_get_rollout_status(req.task_id)


@app.post('/tools/postgres/reward_history')
def reward_history(req: TaskRequest) -> dict:
    return postgres_get_reward_history(req.task_id)


@app.post('/tools/profile/parquet')
def profile_parquet(req: ProfileRequest) -> dict:
    summary = YDataProfilingTool().profile_parquet(req.parquet_path, req.output_json, req.output_html)
    return {'summary': summary, 'output_json': req.output_json, 'output_html': req.output_html, 'raw_rows_exposed_to_llm': False}
