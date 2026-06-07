from project_mcp_server.server import mcp
from mcp_client_bridge.schema_from_function import function_to_json_schema


def test_project_mcp_server_wrapper_imports_mcp():
    assert mcp is not None


def test_schema_bridge_wrapper_imports():
    def sample(task_id: str) -> dict:
        return {"task_id": task_id}
    schema = function_to_json_schema(sample)
    assert schema["name"] == "sample"
