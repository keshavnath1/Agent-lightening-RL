from __future__ import annotations

import importlib
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from src.tools.task_registry import (
    TRACKA_DEFAULT_TASK_LIMIT,
    TrackATaskLoadError,
    _validate_task,
    load_tracka_tasks_from_postgres,
)


class TestValidateTask:
    def test_valid_task_passes(self) -> None:
        _validate_task(
            {
                "task_id": "task_001",
                "dataset_key": "openml_31_german_credit",
                "objective": "Train baseline",
                "target_column": "class",
                "primary_metric": "roc_auc",
                "registry_source": "postgresql_mcp",
                "raw_rows_exposed_to_llm": False,
            }
        )

    def test_missing_dataset_key_raises(self) -> None:
        with pytest.raises(TrackATaskLoadError, match="dataset_key"):
            _validate_task(
                {
                    "task_id": "task_001",
                    "objective": "Train baseline",
                    "target_column": "class",
                    "primary_metric": "roc_auc",
                    "registry_source": "postgresql_mcp",
                    "raw_rows_exposed_to_llm": False,
                }
            )

    def test_registry_source_must_be_postgresql_mcp(self) -> None:
        with pytest.raises(TrackATaskLoadError, match="registry_source"):
            _validate_task(
                {
                    "task_id": "task_001",
                    "dataset_key": "openml_31_german_credit",
                    "objective": "Train baseline",
                    "target_column": "class",
                    "primary_metric": "roc_auc",
                    "registry_source": "postgresql",
                    "raw_rows_exposed_to_llm": False,
                }
            )

    def test_raw_rows_exposed_must_be_false(self) -> None:
        with pytest.raises(TrackATaskLoadError, match="raw_rows_exposed_to_llm"):
            _validate_task(
                {
                    "task_id": "task_001",
                    "dataset_key": "openml_31_german_credit",
                    "objective": "Train baseline",
                    "target_column": "class",
                    "primary_metric": "roc_auc",
                    "registry_source": "postgresql_mcp",
                    "raw_rows_exposed_to_llm": True,
                }
            )


class TestLoadTrackaTasksFromPostgres:
    @patch("ml_tools.postgres_tooling.postgres_list_tasks")
    def test_returns_normalized_tasks(self, mock_list_tasks: MagicMock) -> None:
        mock_list_tasks.return_value = {
            "result": {
                "tasks": [
                    {
                        "task_id": "task_001",
                        "dataset_key": "openml_31_german_credit",
                        "objective": "Train baseline",
                        "problem_statement": "Train baseline",
                        "target_column": "class",
                        "primary_metric": "roc_auc",
                        "secondary_metrics": ["accuracy"],
                        "status": "ready",
                        "priority": 1,
                        "registry_source": "postgresql_mcp",
                        "raw_rows_exposed_to_llm": False,
                        "schema_json": {},
                        "dataset_summary": {},
                        "column_profiles": [],
                        "target_profile": {},
                    }
                ]
            }
        }

        tasks = load_tracka_tasks_from_postgres(limit=4)
        assert len(tasks) == 1
        assert tasks[0]["task_id"] == "task_001"
        assert tasks[0]["dataset_key"] == "openml_31_german_credit"
        assert tasks[0]["registry_source"] == "postgresql_mcp"
        assert tasks[0]["raw_rows_exposed_to_llm"] is False
        mock_list_tasks.assert_called_once_with(limit=4, split=None, status="ready")

    @patch("ml_tools.postgres_tooling.postgres_list_tasks")
    def test_default_limit_is_used_when_not_specified(self, mock_list_tasks: MagicMock) -> None:
        mock_list_tasks.return_value = {"result": {"tasks": []}}
        with pytest.raises(TrackATaskLoadError, match="zero Track A tasks"):
            load_tracka_tasks_from_postgres()
        mock_list_tasks.assert_called_once_with(
            limit=TRACKA_DEFAULT_TASK_LIMIT,
            split=None,
            status="ready",
        )

    @patch("ml_tools.postgres_tooling.postgres_list_tasks")
    def test_zero_tasks_fail_closed(self, mock_list_tasks: MagicMock) -> None:
        mock_list_tasks.return_value = {"result": {"tasks": []}}
        with pytest.raises(TrackATaskLoadError, match="zero Track A tasks"):
            load_tracka_tasks_from_postgres(limit=1)

    @patch("ml_tools.postgres_tooling.postgres_list_tasks")
    def test_mcp_failure_raises_sanitized_typed_error(self, mock_list_tasks: MagicMock) -> None:
        mock_list_tasks.side_effect = RuntimeError("database unreachable")
        with pytest.raises(TrackATaskLoadError, match="Failed to load Track A tasks via MCP"):
            load_tracka_tasks_from_postgres(limit=1)

    @patch("ml_tools.postgres_tooling.postgres_list_tasks")
    def test_invalid_task_metadata_raises(self, mock_list_tasks: MagicMock) -> None:
        mock_list_tasks.return_value = {
            "result": {
                "tasks": [
                    {
                        "task_id": "task_bad",
                        "dataset_key": "openml_31_german_credit",
                        "objective": "Train baseline",
                        "target_column": None,
                        "primary_metric": "roc_auc",
                        "registry_source": "postgresql_mcp",
                        "raw_rows_exposed_to_llm": False,
                    }
                ]
            }
        }
        with pytest.raises(TrackATaskLoadError, match="target_column"):
            load_tracka_tasks_from_postgres(limit=1)

    def test_import_fallback_to_src_tools_module(self) -> None:
        fake_module = types.ModuleType("src.tools.postgres_tooling")

        def _fake_postgres_list_tasks(limit: int, split: str | None, status: str | None) -> dict:
            return {
                "result": {
                    "tasks": [
                        {
                            "task_id": "task_001",
                            "dataset_key": "openml_31_german_credit",
                            "objective": "Train baseline",
                            "problem_statement": "Train baseline",
                            "target_column": "class",
                            "primary_metric": "roc_auc",
                            "registry_source": "postgresql_mcp",
                            "raw_rows_exposed_to_llm": False,
                        }
                    ]
                }
            }

        fake_module.postgres_list_tasks = _fake_postgres_list_tasks  # type: ignore[attr-defined]

        original_ml_tools = sys.modules.get("ml_tools")
        original_ml_tools_postgres = sys.modules.get("ml_tools.postgres_tooling")
        original_src_tools_postgres = sys.modules.get("src.tools.postgres_tooling")

        try:
            if "ml_tools" in sys.modules:
                del sys.modules["ml_tools"]
            if "ml_tools.postgres_tooling" in sys.modules:
                del sys.modules["ml_tools.postgres_tooling"]
            sys.modules["src.tools.postgres_tooling"] = fake_module

            tasks = load_tracka_tasks_from_postgres(limit=1)
            assert len(tasks) == 1
            assert tasks[0]["task_id"] == "task_001"
        finally:
            if original_ml_tools is not None:
                sys.modules["ml_tools"] = original_ml_tools
            elif "ml_tools" in sys.modules:
                del sys.modules["ml_tools"]

            if original_ml_tools_postgres is not None:
                sys.modules["ml_tools.postgres_tooling"] = original_ml_tools_postgres
            elif "ml_tools.postgres_tooling" in sys.modules:
                del sys.modules["ml_tools.postgres_tooling"]

            if original_src_tools_postgres is not None:
                sys.modules["src.tools.postgres_tooling"] = original_src_tools_postgres
            elif "src.tools.postgres_tooling" in sys.modules:
                del sys.modules["src.tools.postgres_tooling"]


import importlib as _importlib

try:
    _supervisor_mod = _importlib.import_module("src.agents.supervisor")
except ModuleNotFoundError:
    _supervisor_mod = None


@pytest.mark.skipif(_supervisor_mod is None, reason="langgraph/supervisor runtime deps not installed")
class TestSupervisorRunTasksPostgresPath:
    @patch.object(_supervisor_mod, "load_tracka_tasks_from_postgres")
    @patch.object(_supervisor_mod, "SupervisorAgent")
    @patch.object(_supervisor_mod, "TrajectoryLogger")
    def test_supervisor_calls_postgres_loader(
        self,
        mock_logger_cls: MagicMock,
        mock_agent_cls: MagicMock,
        mock_loader: MagicMock,
    ) -> None:
        mock_loader.return_value = [
            {
                "task_id": "task_001",
                "dataset_key": "openml_31_german_credit",
                "objective": "Train baseline",
                "target_column": "class",
                "primary_metric": "roc_auc",
                "registry_source": "postgresql_mcp",
                "raw_rows_exposed_to_llm": False,
            }
        ]
        mock_agent = MagicMock()
        mock_agent_cls.return_value = mock_agent
        mock_logger_cls.return_value = MagicMock()
        mock_agent.run_task.return_value = MagicMock()

        _supervisor_mod.run_tasks(
            output_dir="trajectories/test",
            policy_version="baseline",
            task_limit=1,
            lightning_server_url="http://localhost:19123",
        )

        mock_loader.assert_called_once_with(limit=1)
        mock_agent.run_task.assert_called_once()

    @patch.object(_supervisor_mod, "load_tracka_tasks_from_postgres")
    def test_missing_lightning_server_raises(self, mock_loader: MagicMock) -> None:
        with pytest.raises(TrackATaskLoadError, match="lightning-server-url"):
            _supervisor_mod.run_tasks(
                output_dir="out",
                policy_version="v1",
                lightning_server_url=None,
            )
        mock_loader.assert_not_called()

    @patch.object(_supervisor_mod, "load_tracka_tasks_from_postgres")
    def test_zero_tasks_raises_and_no_fallback(self, mock_loader: MagicMock) -> None:
        mock_loader.side_effect = TrackATaskLoadError("zero tasks")
        with pytest.raises(TrackATaskLoadError, match="zero tasks"):
            _supervisor_mod.run_tasks(
                output_dir="out",
                policy_version="v1",
                lightning_server_url="http://localhost:19123",
            )
