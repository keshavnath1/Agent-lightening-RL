from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def ok(name: str) -> None:
    print(f"OK {name}")


def fail(name: str, err: Exception | str) -> None:
    print(f"FAIL {name}: {err}")
    sys.exit(1)


def check(name: str, fn) -> None:
    try:
        fn()
        ok(name)
    except Exception as exc:
        fail(name, exc)


def check_imports() -> None:
    importlib.import_module("mcp.server.fastmcp")
    importlib.import_module("src.mcp_official.server")
    importlib.import_module("src.mcp_official.logging")
    importlib.import_module("src.mcp_official.resources")
    importlib.import_module("src.mcp_official.prompts")
    importlib.import_module("src.tools.official_mcp_client")
    importlib.import_module("src.tools.schema_from_function")
    importlib.import_module("src.tools.langchain_mcp_bridge")


def check_start_script() -> None:
    script = Path("scripts/cpu/start_official_mcp_server.sh")
    assert script.exists(), "start_official_mcp_server.sh missing"
    result = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def check_cursor_example_no_real_secret() -> None:
    path = Path(".cursor/mcp.example.json")
    assert path.exists(), ".cursor/mcp.example.json missing"
    text = path.read_text(encoding="utf-8")
    assert "USER:PASSWORD" in text, "example should use placeholder credentials"
    assert "HOST:PORT" in text, "example should use placeholder host and port"
    forbidden = ["cv.keshavnath", "38.80.152.249", "OPENAI_API_KEY", "sk-"]
    for marker in forbidden:
        assert marker not in text, f"real secret or host marker found: {marker}"


def check_raw_sql_disabled_by_default() -> None:
    os.environ.pop("ALLOW_RAW_SQL_TOOL", None)
    from src.mcp_official.server import postgres_debug_readonly_sql

    result = postgres_debug_readonly_sql("select 1")
    assert result["ok"] is False
    assert "disabled" in result["error"].lower()
    assert result["raw_sql_debug_mode"] is False


def check_sql_safety_blocks_select_star() -> None:
    from src.tools.sql_safety import validate_sql

    try:
        validate_sql("select * from synthetic_tasks")
    except Exception:
        return
    raise AssertionError("SELECT * was not blocked")


def check_sql_safety_blocks_write() -> None:
    from src.tools.sql_safety import validate_sql

    try:
        validate_sql("delete from synthetic_tasks")
    except Exception:
        return
    raise AssertionError("write SQL was not blocked")


def check_schema_extraction() -> None:
    from src.tools.schema_from_function import function_to_json_schema
    from src.mcp_official.server import postgres_get_dataset_summary

    schema = function_to_json_schema(postgres_get_dataset_summary)
    assert schema["name"] == "postgres_get_dataset_summary"
    assert "task_id" in schema["input_schema"]["properties"]
    assert "task_id" in schema["input_schema"]["required"]


def check_log_file_created() -> None:
    from src.mcp_official.logging import MCP_TOOL_CALL_LOG

    assert MCP_TOOL_CALL_LOG.parent.exists(), "reports directory missing"
    # check_raw_sql_disabled_by_default calls the decorated debug tool and should create this file.
    assert MCP_TOOL_CALL_LOG.exists(), "official MCP JSONL log was not created"
    last = MCP_TOOL_CALL_LOG.read_text(encoding="utf-8").strip().splitlines()[-1]
    assert '"official_mcp": true' in last


def main() -> None:
    check("imports", check_imports)
    check("start script bash syntax", check_start_script)
    check("cursor example has no real secret", check_cursor_example_no_real_secret)
    check("raw SQL disabled by default", check_raw_sql_disabled_by_default)
    check("SQL safety blocks SELECT *", check_sql_safety_blocks_select_star)
    check("SQL safety blocks write SQL", check_sql_safety_blocks_write)
    check("schema extraction works", check_schema_extraction)
    check("official MCP call logging works", check_log_file_created)

    print("\nOfficial MCP server validation passed.")


if __name__ == "__main__":
    main()
