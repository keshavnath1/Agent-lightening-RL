from __future__ import annotations

from typing import Any

SAFE_POSTGRES_TOOLS = {
    'postgres_get_task_metadata',
    'postgres_get_dataset_schema',
    'postgres_get_dataset_summary',
    'postgres_get_reward_history',
    'postgres_get_rollout_status',
    'postgres_get_column_profile',
    'postgres_get_target_profile',
    'postgres_list_tasks',
    'postgres_get_next_task',
}

# Execution-layer tools that are valid evidence of the strict MCP/PostgreSQL
# boundary. They do not return raw rows to the agent; they materialize approved
# artifact references after the registry metadata has been selected as the
# source of truth.
SAFE_REGISTRY_EVIDENCE_TOOLS = {
    'postgres_get_execution_dataset_source',
    'MCPSafeMetadataProfile',
}

RAW_SQL_OR_GENERIC_TOOLS = {'SQLAlchemyConnector', 'postgres_raw_sql_query'}


def _iter_tool_calls(trajectory: dict[str, Any]):
    for step in trajectory.get('steps', []):
        for tc in step.get('tool_calls', []):
            yield tc


def _metadata_raw_rows_exposed(metadata: dict[str, Any], arguments: dict[str, Any]) -> bool:
    return bool(
        metadata.get('raw_rows_exposed', False)
        or metadata.get('raw_rows_exposed_to_llm', False)
        or metadata.get('raw_rows_loaded_into_agent_context', False)
        or arguments.get('raw_rows_exposed', False)
        or arguments.get('raw_rows_exposed_to_llm', False)
        or arguments.get('raw_rows_loaded_into_agent_context', False)
    )


def score_postgres_tool_call(tc: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    name = tc.get('tool_name')
    metadata = tc.get('metadata') or {}
    arguments = tc.get('arguments') or {}
    status = tc.get('status')

    if name not in SAFE_POSTGRES_TOOLS and name not in SAFE_REGISTRY_EVIDENCE_TOOLS and name not in RAW_SQL_OR_GENERIC_TOOLS:
        return 0.0, {'applicable': False}

    score = 0.0
    reasons: list[str] = []
    raw_rows_exposed = _metadata_raw_rows_exposed(metadata, arguments)

    if status == 'success':
        score += 0.20
        reasons.append('success')

    if name in SAFE_POSTGRES_TOOLS:
        score += 0.35
        reasons.append('safe_endpoint')
    elif name in SAFE_REGISTRY_EVIDENCE_TOOLS:
        score += 0.35
        reasons.append('strict_registry_evidence_tool')
    else:
        score -= 0.10
        reasons.append('raw_sql_or_generic_connector')

    if not raw_rows_exposed:
        score += 0.20
        reasons.append('no_raw_rows_exposed')
    else:
        score -= 0.45
        reasons.append('raw_rows_exposed')

    if metadata.get('policy_allowed', True):
        score += 0.10
        reasons.append('policy_allowed')
    else:
        score -= 0.25
        reasons.append('policy_blocked')

    row_count = metadata.get('row_count_returned')
    try:
        row_count_int = int(row_count)
    except Exception:
        row_count_int = 0
    if row_count_int <= 100:
        score += 0.10
        reasons.append('small_result')
    elif row_count_int > 1000:
        score -= 0.20
        reasons.append('too_many_rows')

    if metadata.get('aggregate_only') or arguments.get('execution_only') or arguments.get('materialization_boundary') == 'docker_execution_dataset_source_contract':
        score += 0.05
        reasons.append('aggregate_or_execution_layer_only')

    if name in SAFE_REGISTRY_EVIDENCE_TOOLS:
        score += 0.05
        reasons.append('strict_registry_source_of_truth')

    return max(0.0, min(1.0, round(score, 4))), {
        'applicable': True,
        'tool_name': name,
        'row_count_returned': row_count_int,
        'reasons': reasons,
    }


def score_trajectory_postgres_tool(trajectory: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    postgres_calls = []
    for tc in _iter_tool_calls(trajectory):
        value, detail = score_postgres_tool_call(tc)
        if detail.get('applicable'):
            detail['score'] = value
            postgres_calls.append(detail)
    if not postgres_calls:
        return 0.0, {'postgres_tool_calls': 0, 'details': []}
    score = round(sum(d['score'] for d in postgres_calls) / len(postgres_calls), 4)
    return score, {'postgres_tool_calls': len(postgres_calls), 'details': postgres_calls}
