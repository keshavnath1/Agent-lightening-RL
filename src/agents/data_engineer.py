from __future__ import annotations

from pathlib import Path
import json
from typing import Any

from src.agents.base import BaseAgent, AgentContext
from src.telemetry.schema import AgentStep, ToolCallRecord


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _derive_preprocessing_config(profile: dict, target_column: str) -> dict:
    numeric_columns: list[str] = []
    categorical_columns: list[str] = []
    datetime_columns: list[str] = []
    high_cardinality_columns: list[str] = []
    missing_value_strategy: dict[str, str] = {}

    for col in profile.get('columns', []):
        name = col['name']
        if name == target_column:
            continue
        dtype = str(col.get('dtype', '')).lower()
        missing_rate = float(col.get('missing_rate', col.get('missing_pct', 0.0)) or 0.0)
        cardinality = col.get('cardinality')
        high_cardinality = bool(col.get('is_high_cardinality')) or (
            isinstance(cardinality, int) and cardinality > 50
        )
        if high_cardinality:
            high_cardinality_columns.append(name)
        if 'datetime' in dtype or 'date' in dtype:
            datetime_columns.append(name)
            missing_value_strategy[name] = 'most_frequent' if missing_rate else 'none'
        elif any(token in dtype for token in ['int', 'float', 'double', 'number', 'numeric', 'bool']):
            numeric_columns.append(name)
            missing_value_strategy[name] = 'median' if missing_rate else 'none'
        else:
            categorical_columns.append(name)
            missing_value_strategy[name] = 'most_frequent' if missing_rate else 'none'

    encoding = 'hashing' if high_cardinality_columns else 'one_hot'
    return {
        'target_column': target_column,
        'numeric_columns': numeric_columns,
        'categorical_columns': categorical_columns,
        'datetime_columns': datetime_columns,
        'high_cardinality_columns': high_cardinality_columns,
        'missing_value_strategy': missing_value_strategy,
        'numeric_imputation': 'median',
        'categorical_imputation': 'most_frequent',
        'categorical_encoding': encoding,
        'scaling': 'not_required_for_tree_models',
        'leakage_checks': {
            'target_removed_from_features': True,
            'raw_rows_not_exposed_to_llm_context': True,
        },
        'profile_evidence': {
            'row_count': profile.get('row_count'),
            'column_count': profile.get('column_count'),
            'max_missing_rate': max(
                [float(c.get('missing_rate', c.get('missing_pct', 0.0)) or 0.0) for c in profile.get('columns', [])]
                or [0.0]
            ),
        },
    }


def _column_name(col: Any) -> str | None:
    if isinstance(col, str):
        return col
    if isinstance(col, dict):
        return col.get('name') or col.get('column_name')
    return None


def _column_dtype(col: Any, profile_by_name: dict[str, dict[str, Any]]) -> str:
    if isinstance(col, dict):
        value = col.get('dtype') or col.get('type')
        if value:
            return str(value)
        name = _column_name(col)
        if name and name in profile_by_name:
            return str(profile_by_name[name].get('dtype') or profile_by_name[name].get('type') or 'unknown')
    if isinstance(col, str) and col in profile_by_name:
        return str(profile_by_name[col].get('dtype') or profile_by_name[col].get('type') or 'unknown')
    return 'unknown'


def _build_profile_from_task(task: dict[str, Any]) -> dict[str, Any]:
    schema_json = _as_dict(task.get('schema_json') or task.get('schema_summary'))
    dataset_summary = _as_dict(task.get('dataset_summary'))
    target_profile = _as_dict(task.get('target_profile'))
    column_profiles = _as_list(task.get('column_profiles') or schema_json.get('column_profiles'))
    profile_by_name = {
        str(item.get('name') or item.get('column_name')): item
        for item in column_profiles
        if isinstance(item, dict) and (item.get('name') or item.get('column_name'))
    }

    schema_columns = _as_list(schema_json.get('columns'))
    if not schema_columns:
        schema_columns = column_profiles

    columns: list[dict[str, Any]] = []
    for col in schema_columns:
        name = _column_name(col)
        if not name:
            continue
        safe_profile = dict(profile_by_name.get(name, {}))
        columns.append({
            'name': name,
            'dtype': _column_dtype(col, profile_by_name),
            'missing_rate': safe_profile.get('missing_rate', safe_profile.get('missing_pct', 0.0)),
            'cardinality': safe_profile.get('cardinality'),
            'is_high_cardinality': bool(safe_profile.get('is_high_cardinality', False)),
            'raw_values_exposed': False,
        })

    target_column = str(task.get('target_column') or schema_json.get('target_column') or '')
    row_count = task.get('row_count') or dataset_summary.get('row_count')
    column_count = task.get('column_count') or len(columns)
    return {
        'task_id': task.get('task_id'),
        'dataset_key': task.get('dataset_key'),
        'target_column': target_column,
        'row_count': row_count,
        'column_count': column_count,
        'feature_count': max(int(column_count or 0) - 1, 0),
        'columns': columns,
        'dataset_summary': dataset_summary,
        'target_profile': target_profile,
        'raw_rows_exposed_to_llm': False,
        'raw_rows_loaded_into_agent_context': False,
        'profile_source': 'mcp_safe_metadata',
    }


class DataEngineerAgent(BaseAgent):
    name = 'DataEngineerAgent'

    def step(self, context: AgentContext) -> AgentStep:
        task = context.task
        task_id = task['task_id']
        artifact_dir = Path('artifacts') / task_id
        artifact_dir.mkdir(parents=True, exist_ok=True)

        profile = _build_profile_from_task(task)
        target_column = str(profile.get('target_column') or task['target_column'])
        if not target_column:
            raise RuntimeError(f'Task {task_id} is missing target_column in safe MCP metadata.')

        extraction_path = artifact_dir / 'data_extraction_metadata.json'
        extraction_payload = {
            'task_id': task_id,
            'dataset_key': task.get('dataset_key'),
            'source': 'mcp_safe_metadata_only',
            'registry_table': task.get('registry_table', 'ml_registry.real_benchmark_tasks'),
            'summary_table': task.get('summary_table', 'ml_registry.real_dataset_summaries'),
            'materialization_boundary': 'docker_execution_dataset_source_contract',
            'row_count': profile.get('row_count'),
            'target_column': target_column,
            'raw_rows_exposed_to_llm': False,
            'raw_rows_loaded_into_agent_context': False,
        }
        extraction_path.write_text(json.dumps(extraction_payload, indent=2), encoding='utf-8')

        schema = {
            'target_column': target_column,
            'columns': [{'name': c['name'], 'dtype': str(c.get('dtype', 'unknown'))} for c in profile['columns']],
            'row_count': profile.get('row_count'),
            'feature_count': profile.get('feature_count'),
            'raw_rows_exposed_to_llm': False,
            'raw_rows_loaded_into_agent_context': False,
            'data_source': 'mcp_safe_metadata_only',
            'registry_table': extraction_payload['registry_table'],
        }
        schema_path = artifact_dir / 'schema_metadata.json'
        schema_path.write_text(json.dumps(schema, indent=2), encoding='utf-8')

        profile_path = artifact_dir / 'profile_summary.json'
        profile_path.write_text(json.dumps(profile, indent=2), encoding='utf-8')

        profile_html = artifact_dir / 'profile_report.html'
        profile_html.write_text(
            '<html><body><h1>Safe metadata profile</h1>'
            '<p>Raw rows are not exposed to the agent context.</p></body></html>',
            encoding='utf-8',
        )

        preprocessing_path = artifact_dir / 'preprocessing_config.json'
        preprocessing_config = _derive_preprocessing_config(profile, target_column)
        preprocessing_path.write_text(json.dumps(preprocessing_config, indent=2), encoding='utf-8')

        context.artifacts.update({
            'data_extraction_metadata_json': str(extraction_path),
            'schema_metadata_json': str(schema_path),
            'profile_summary_json': str(profile_path),
            'profile_report_html': str(profile_html),
            'preprocessing_config_json': str(preprocessing_path),
        })

        return AgentStep(
            agent_name=self.name,
            action='prepare_metadata_only_profile_artifacts',
            reasoning_summary='Prepared schema, profile, and preprocessing artifacts from safe MCP metadata only; row materialization is reserved for the Docker execution boundary.',
            tool_calls=[
                ToolCallRecord(
                    tool_name='MCPSafeMetadataProfile',
                    arguments={
                        'task_id': task_id,
                        'dataset_key': task.get('dataset_key'),
                        'raw_rows_exposed_to_llm': False,
                        'raw_rows_loaded_into_agent_context': False,
                    },
                    status='success',
                    output_ref=str(profile_path),
                ),
                self.tool_success(
                    'PreprocessingConfigDeriver',
                    {'profile_summary': str(profile_path), 'raw_rows_exposed_to_llm': False},
                    str(preprocessing_path),
                ),
            ],
        )
