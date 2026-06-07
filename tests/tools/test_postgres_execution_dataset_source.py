from __future__ import annotations

import pytest

from src.tools import postgres_tooling


class FakeConnector:
    def execute_internal(self, sql: str, params: dict | None = None, *, max_rows: int | None = None):
        params = params or {}
        if "ml_registry.real_benchmark_tasks" in sql:
            return [
                {
                    "task_id": params["task_id"],
                    "dataset_key": "openml_31_german_credit",
                    "task_name": "German credit baseline",
                    "problem_statement": "Train a credit model",
                    "objective": "Train a credit model",
                    "target_column": "target",
                    "problem_type": "binary_classification",
                    "primary_metric": "roc_auc",
                    "metric": "roc_auc",
                    "secondary_metrics": [],
                    "status": "ready",
                    "priority": 1,
                    "registry_source": "postgresql_mcp",
                    "registry_source_of_truth": "postgresql_mcp",
                    "raw_rows_exposed_to_llm": False,
                    "dataset_name": "german_credit",
                    "source": "openml",
                    "row_count": 1000,
                    "column_count": 21,
                    "schema_json": {"columns": []},
                    "dataset_summary": {},
                    "column_profiles": [],
                    "target_profile": {},
                    "summary_raw_rows_exposed_to_llm": False,
                }
            ]
        if "ml_execution.dataset_sources" in sql:
            return [
                {
                    "dataset_key": params["dataset_key"],
                    "storage_backend": "postgres_table",
                    "source_schema": "ml_data",
                    "source_table": "openml_31_german_credit",
                    "row_count": 1000,
                    "column_count": 21,
                    "checksum_sha256": "abc123",
                    "raw_rows_exposed_to_llm": False,
                }
            ]
        raise AssertionError(sql)


def test_postgres_get_execution_dataset_source_contract(monkeypatch, tmp_path):
    monkeypatch.setattr(postgres_tooling, "_connector", lambda: FakeConnector())
    monkeypatch.setattr(postgres_tooling, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(postgres_tooling, "TOOL_LOG", tmp_path / "reports" / "postgres_tool_calls.jsonl")

    result = postgres_tooling.postgres_get_execution_dataset_source("mltask_openml_31_german_credit_baseline")

    payload = result["result"]
    assert payload["contract_version"] == "mcp_execution_dataset_source_v1"
    assert payload["dataset_key"] == "openml_31_german_credit"
    assert payload["storage_backend"] == "postgres_table"
    assert payload["source_schema"] == "ml_data"
    assert payload["source_table"] == "openml_31_german_credit"
    assert payload["execution_only"] is True
    assert payload["raw_rows_exposed_to_llm"] is False


def test_postgres_get_execution_dataset_source_rejects_unsafe_identifier(monkeypatch, tmp_path):
    class UnsafeConnector(FakeConnector):
        def execute_internal(self, sql: str, params: dict | None = None, *, max_rows: int | None = None):
            rows = super().execute_internal(sql, params=params, max_rows=max_rows)
            if "ml_execution.dataset_sources" in sql:
                rows[0]["source_table"] = "bad;drop"
            return rows

    monkeypatch.setattr(postgres_tooling, "_connector", lambda: UnsafeConnector())
    monkeypatch.setattr(postgres_tooling, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(postgres_tooling, "TOOL_LOG", tmp_path / "reports" / "postgres_tool_calls.jsonl")

    with pytest.raises(postgres_tooling.MCPPostgresToolError, match="safe SQL identifier"):
        postgres_tooling.postgres_get_execution_dataset_source("mltask_openml_31_german_credit_baseline")
