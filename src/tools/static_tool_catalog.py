from __future__ import annotations

from copy import deepcopy
from typing import Any


STATIC_TOOL_CATALOG: list[dict[str, Any]] = [
    {
        "name": "postgres_get_task_metadata",
        "aliases": ["postgres_get_task_metadata", "PostgresTaskMetadataReader"],
        "category": "postgres_state",
        "description": "Safe PostgreSQL task metadata lookup against real_benchmark_tasks with no raw row exposure.",
        "input_schema": {"type": "object", "required": ["task_id"], "properties": {"task_id": {"type": "string"}, "raw_rows_exposed_to_llm": {"type": "boolean"}}},
        "output_schema": {"type": "object", "properties": {"task_id": {"type": "string"}, "found": {"type": "boolean"}, "metadata": {"type": "object"}, "raw_rows_exposed": {"type": "boolean"}}},
        "required_inputs": ["task_id"],
        "expected_outputs": ["task metadata only"],
        "success_criteria": ["returns one metadata record", "raw_rows_exposed=false", "does not query synthetic_dataset_rows"],
        "risks": ["raw_data_leakage", "unsafe_sql"],
        "risk_level": "low",
    },
    {
        "name": "postgres_get_dataset_summary",
        "aliases": ["postgres_get_dataset_summary", "PostgresDatasetSummaryReader"],
        "category": "postgres_state",
        "description": "Safe aggregate PostgreSQL dataset summary for task-specific scenario planning.",
        "input_schema": {"type": "object", "required": ["task_id"], "properties": {"task_id": {"type": "string"}, "raw_rows_exposed_to_llm": {"type": "boolean"}}},
        "output_schema": {"type": "object", "properties": {"row_count": {"type": "integer"}, "column_count": {"type": "integer"}, "target_column": {"type": "string"}, "raw_rows_exposed": {"type": "boolean"}}},
        "required_inputs": ["task_id"],
        "expected_outputs": ["aggregate dataset summary"],
        "success_criteria": ["aggregate-only output", "no raw row leakage"],
        "risks": ["raw_data_leakage"],
        "risk_level": "low",
    },
    {
        "name": "postgres_get_dataset_schema",
        "aliases": ["postgres_get_dataset_schema", "PostgresDatasetSchemaReader"],
        "category": "postgres_state",
        "description": "Safe PostgreSQL schema extraction returning column names/types only.",
        "input_schema": {"type": "object", "required": ["task_id"], "properties": {"task_id": {"type": "string"}, "raw_rows_exposed_to_llm": {"type": "boolean"}}},
        "output_schema": {"type": "object", "properties": {"column_count": {"type": "integer"}, "columns": {"type": "array"}, "raw_rows_exposed": {"type": "boolean"}}},
        "required_inputs": ["task_id"],
        "expected_outputs": ["schema_metadata.json"],
        "success_criteria": ["schema only", "raw_rows_exposed=false"],
        "risks": ["raw_data_leakage"],
        "risk_level": "low",
    },
    {
        "name": "postgres_get_execution_dataset_source",
        "aliases": ["postgres_get_execution_dataset_source", "ExecutionDatasetSourceContract"],
        "category": "data_access",
        "description": "Return the execution-only PostgreSQL dataset source contract for Docker materialization.",
        "input_schema": {"type": "object", "required": ["task_id"], "properties": {"task_id": {"type": "string"}, "execution_only": {"type": "boolean"}, "raw_rows_exposed_to_llm": {"type": "boolean"}}},
        "output_schema": {"type": "object", "properties": {"dataset_key": {"type": "string"}, "storage_backend": {"type": "string"}, "source_schema": {"type": "string"}, "source_table": {"type": "string"}, "execution_only": {"type": "boolean"}, "raw_rows_exposed_to_llm": {"type": "boolean"}}},
        "required_inputs": ["task_id"],
        "expected_outputs": ["execution_dataset_source.json"],
        "success_criteria": ["execution_only=true", "raw_rows_exposed_to_llm=false", "no credentials returned"],
        "risks": ["raw_data_leakage"],
        "risk_level": "medium",
    },
    {
        "name": "SQLAlchemyConnector",
        "aliases": ["SQLAlchemyConnector", "LocalParquetExtractor"],
        "category": "data_access",
        "description": "Connect to SQL databases and run metadata-safe profiling queries without exposing raw rows.",
        "input_schema": {"type": "object", "required": ["connection_ref", "query"], "properties": {"connection_ref": {"type": "string"}, "query": {"type": "string"}, "dataset_ref": {"type": "string"}, "raw_rows_exposed_to_llm": {"type": "boolean"}}},
        "output_schema": {"type": "object", "properties": {"schema_metadata": {"type": "object"}, "safe_preview_ref": {"type": "string"}}},
        "required_inputs": ["connection_ref", "query"],
        "expected_outputs": ["schema_metadata.json"],
        "risks": ["raw_data_leakage", "unsafe_sql"],
        "risk_level": "medium",
    },
    {
        "name": "YDataProfiler",
        "aliases": ["YDataProfiler", "YDataProfiling", "ProfileAndSchemaMetadataReader"],
        "category": "data_quality",
        "description": "Generate profile summaries and schema metadata for tabular data without leaking raw rows.",
        "input_schema": {"type": "object", "required": ["input"], "properties": {"input": {"type": "string"}}},
        "output_schema": {"type": "object", "properties": {"profile_summary": {"type": "object"}, "schema_metadata": {"type": "object"}}},
        "required_inputs": ["input"],
        "expected_outputs": ["profile_summary.json", "schema_metadata.json"],
        "risks": ["raw_data_leakage"],
        "risk_level": "low",
    },
    {
        "name": "GBMBenchmark",
        "aliases": ["GBMBenchmark"],
        "category": "modeling",
        "description": "Train and compare gradient-boosting candidates from an execution dataset source contract, then write benchmark metrics and champion metadata.",
        "input_schema": {"type": "object", "required": ["dataset_source", "target_column"], "properties": {"dataset_source": {"type": "object"}, "target_column": {"type": "string"}}},
        "output_schema": {"type": "object", "properties": {"benchmark_metrics": {"type": "object"}, "champion_model": {"type": "object"}}},
        "required_inputs": ["dataset_source", "target_column"],
        "expected_outputs": ["benchmark_metrics.json", "champion_model.json"],
        "risks": ["fake_metrics", "missing_model_artifact"],
        "risk_level": "medium",
    },
    {
        "name": "DockerizedCodeInterpreter",
        "aliases": ["DockerizedCodeInterpreter"],
        "category": "sandbox",
        "description": "Execute generated analysis or model code through an isolated, reproducible sandbox path.",
        "input_schema": {"type": "object", "required": ["script_path"], "properties": {"script_path": {"type": "string"}, "execution_mode": {"type": "string"}}},
        "output_schema": {"type": "object", "properties": {"stdout": {"type": "string"}, "artifact_ref": {"type": "string"}}},
        "required_inputs": ["script_path"],
        "expected_outputs": ["execution_log.txt", "benchmark_metrics.json"],
        "risks": ["unsafe_code_execution"],
        "risk_level": "high",
    },
    {
        "name": "MLflowTracking",
        "aliases": ["MLflowTracking", "MLflowDVCTracker", "DVCArtifactTracking"],
        "category": "tracking",
        "description": "Record parameters, metrics, artifacts, and fallback JSON tracking metadata for experiment reproducibility.",
        "input_schema": {"type": "object", "required": ["run_id"], "properties": {"run_id": {"type": "string"}, "backend": {"type": "string"}}},
        "output_schema": {"type": "object", "properties": {"tracking_uri": {"type": "string"}, "run_ref": {"type": "string"}}},
        "required_inputs": ["run_id"],
        "expected_outputs": ["mlflow_run", "reports/mlflow_runs/*.json"],
        "risks": ["missing_reproducibility_metadata"],
        "risk_level": "low",
    },
    {
        "name": "DVCArtifactTracking",
        "aliases": ["DVCArtifactTracking", "MLflowDVCTracker"],
        "category": "artifacts",
        "description": "Track datasets, model artifacts, and reproducibility metadata with DVC-compatible records.",
        "input_schema": {"type": "object", "required": ["artifact_path"], "properties": {"artifact_path": {"type": "string"}, "stage_name": {"type": "string"}}},
        "output_schema": {"type": "object", "properties": {"dvc_ref": {"type": "string"}}},
        "required_inputs": ["artifact_path"],
        "expected_outputs": ["dvc.lock", "artifact_manifest.json"],
        "risks": ["untracked_artifact"],
        "risk_level": "low",
    },
    {
        "name": "ReviewerCritic",
        "aliases": ["ReviewerCritic", "GuardrailValidator"],
        "category": "review",
        "description": "Review artifact completeness, guardrail compliance, raw-data leakage, and workflow violations.",
        "input_schema": {"type": "object", "required": ["artifact_dir"], "properties": {"artifact_dir": {"type": "string"}, "guardrails": {"type": "array", "items": {"type": "string"}}}},
        "output_schema": {"type": "object", "properties": {"review_report": {"type": "object"}, "violations": {"type": "array"}}},
        "required_inputs": ["artifact_dir"],
        "expected_outputs": ["review_report.json"],
        "risks": ["missed_policy_violation", "raw_data_leakage"],
        "risk_level": "medium",
    },
    {
        "name": "PreprocessingConfigDeriver",
        "aliases": ["PreprocessingConfigDeriver"],
        "category": "preprocessing",
        "description": "Derive reproducible preprocessing configuration from schema/profile metadata.",
        "input_schema": {"type": "object", "required": [], "properties": {"profile_ref": {"type": "string"}, "schema_ref": {"type": "string"}, "target_column": {"type": "string"}, "profile_summary": {"type": "string"}}},
        "output_schema": {"type": "object", "properties": {"preprocessing_config": {"type": "object"}}},
        "required_inputs": [],
        "expected_outputs": ["preprocessing_config.json"],
        "risks": ["non_reproducible_transform"],
        "risk_level": "low",
    },
]


def get_static_tool_catalog() -> list[dict[str, Any]]:
    """Return a deep copy of the project MCP-style fallback tool catalog."""
    return deepcopy(STATIC_TOOL_CATALOG)


def as_lookup(catalog: list[dict[str, Any]] | None = None) -> dict[str, dict[str, Any]]:
    rows = catalog or STATIC_TOOL_CATALOG
    lookup: dict[str, dict[str, Any]] = {}
    for tool in rows:
        lookup[tool["name"]] = tool
        for alias in tool.get("aliases", []):
            lookup[alias] = tool
    return lookup
