from __future__ import annotations

from pathlib import Path
import stat

ROOT = Path('/workspace/self-improving-ml-agent')


def write(path: str, text: str, executable: bool = False) -> None:
    p = ROOT / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding='utf-8')
    if executable:
        mode = p.stat().st_mode
        p.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


write('.cursor/mcp.example.json', r'''{
  "mcpServers": {
    "self-improving-ml-agent": {
      "command": "python",
      "args": ["-m", "src.mcp_official.server"],
      "env": {
        "DATABASE_URL": "postgresql+psycopg2://USER:PASSWORD@HOST:PORT/DBNAME",
        "ALLOW_RAW_SQL_TOOL": "0"
      }
    }
  }
}
''')

write('scripts/dev_validate_official_mcp_server.py', r'''from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path


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
''')

# Refine prompt registration to preserve the public prompt names.
write('src/mcp_official/prompts.py', r'''from __future__ import annotations

from typing import Any


def safe_ml_workflow_prompt(task_id: str) -> str:
    """Prompt an agent to solve an ML task through safe project metadata tools."""
    return f"""You are operating inside the self-improving ML-agent project.

Task ID: {task_id}

Use the official Project MCP safe tools before reasoning about this task. Prefer `postgres_get_task_metadata`, `postgres_get_dataset_schema`, `postgres_get_dataset_summary`, `postgres_get_artifact_manifest`, `postgres_get_rollout_status`, and `postgres_get_reward_history`. Do not request or expose raw dataset rows. Summarize what each tool establishes, then propose the next ML workflow step.
"""


def postgres_safe_query_policy() -> str:
    """Describe the PostgreSQL safety policy for project agents."""
    return """Official PostgreSQL safety policy for project agents:

Use the safe task-specific tools for normal agent operation. Raw SQL is a developer diagnostic path only and is disabled unless `ALLOW_RAW_SQL_TOOL=1`. Even when enabled, SQL must be read-only, must not include comments or semicolons, must not use `SELECT *`, and must not access blocked raw-row tables such as `synthetic_dataset_rows`. Never store `DATABASE_URL`, API keys, or other credentials in source code or prompts.
"""


def tool_learning_rollout_prompt(task_id: str) -> str:
    """Prompt a rollout that can be evaluated for tool-selection quality."""
    return f"""Run a tool-learning rollout for task `{task_id}`.

First discover what is known about the task through safe official MCP tools. Next, choose only the minimum necessary tools to answer the user question. Record the rationale for each tool choice, explicitly note that raw rows were not exposed, and produce a concise final answer. RULER should be able to judge whether the correct tools were selected, whether unsafe raw data access was avoided, and whether the answer used the tool outputs faithfully.
"""


def register_prompts(mcp: Any) -> None:
    """Register reusable official MCP prompts on a FastMCP server."""
    mcp.prompt()(safe_ml_workflow_prompt)
    mcp.prompt()(postgres_safe_query_policy)
    mcp.prompt()(tool_learning_rollout_prompt)
''')

# Streamlit Tool Learning page update.
tool_page = ROOT / 'src/ui/pages/tool_learning.py'
text = tool_page.read_text(encoding='utf-8')
if 'import os' not in text.split('\n')[:12]:
    text = text.replace('import json\n', 'import json\nimport os\n', 1)
section = r'''

    st.subheader("F. Official Project MCP Server")
    raw_sql_enabled = os.getenv("ALLOW_RAW_SQL_TOOL", "0") == "1"
    official_log = ROOT / "reports" / "mcp_official_tool_calls.jsonl"
    st.markdown(
        """
The **Official Project MCP Server** is the ML-agent tool-learning interface. It is separate from any RunPod MCP integration: RunPod MCP should be treated as infrastructure control, while Project MCP exposes safe, schema-driven project tools for rollouts, RULER judgment, and TRL GRPO policy improvement.

| Server | Role | Data policy |
|---|---|---|
| RunPod MCP | Infrastructure lifecycle and pod operations only | Do not use for the core learning loop |
| Official Project MCP | Safe task metadata, dataset summaries, artifacts, rollout status, and reward history | No raw dataset rows through ordinary tools |

The official server wraps `postgres_get_task_metadata`, `postgres_get_dataset_schema`, `postgres_get_dataset_summary`, `postgres_get_artifact_manifest`, `postgres_get_rollout_status`, and `postgres_get_reward_history`. The developer-only `postgres_debug_readonly_sql` tool remains disabled unless `ALLOW_RAW_SQL_TOOL=1`, and SQL safety still blocks `SELECT *`, comments, semicolons, blocked raw-row tables, and write operations.
"""
    )
    st.code("bash scripts/cpu/start_official_mcp_server.sh", language="bash")
    st.code(
        '{\n  "mcpServers": {\n    "self-improving-ml-agent": {\n      "command": "python",\n      "args": ["-m", "src.mcp_official.server"],\n      "env": {"DATABASE_URL": "postgresql+psycopg2://USER:PASSWORD@HOST:PORT/DBNAME", "ALLOW_RAW_SQL_TOOL": "0"}\n    }\n  }\n}',
        language="json",
    )
    st.markdown(f"**Official MCP log:** `reports/mcp_official_tool_calls.jsonl` — present: `{official_log.exists()}`")
    st.markdown(f"**Raw SQL status:** `{'enabled' if raw_sql_enabled else 'disabled'}`")
'''
if 'Official Project MCP Server' not in text:
    text = text.rstrip() + section + '\n'
tool_page.write_text(text, encoding='utf-8')

# README update.
readme = ROOT / 'README.md'
readme_text = readme.read_text(encoding='utf-8')
readme_section = r'''

## Official Project MCP Server

This repository now includes an **Official Project MCP Server** in `src/mcp_official/`. The server uses the official Python MCP SDK to expose safe project tools for the ML-agent tool layer, while leaving the older REST-style `src/mcp_server/` implementation in place for compatibility.

| Capability | Implementation | Safety posture |
|---|---|---|
| Safe project tools | `postgres_get_task_metadata`, `postgres_get_dataset_schema`, `postgres_get_dataset_summary`, `postgres_get_artifact_manifest`, `postgres_get_rollout_status`, and `postgres_get_reward_history` | Designed for metadata, summaries, manifests, and trajectory/reward summaries; ordinary tools do not expose raw dataset rows |
| Developer SQL diagnostics | `postgres_debug_readonly_sql` | Disabled unless `ALLOW_RAW_SQL_TOOL=1`; guarded by `validate_sql`, which blocks `SELECT *`, write operations, comments, semicolons, and blocked raw-row tables |
| Tool-call telemetry | `reports/mcp_official_tool_calls.jsonl` | Every official MCP tool call records timestamp, arguments, status, error, duration, `official_mcp=true`, and a compact result summary |
| Agent integration | `src/tools/official_mcp_client.py`, `src/tools/langchain_mcp_bridge.py`, and `src/tools/schema_from_function.py` | Agents can discover tools and schemas dynamically without embedding credentials or raw data |

Start the server from the project root with:

```bash
bash scripts/cpu/start_official_mcp_server.sh
```

A placeholder Cursor/Claude configuration is available at `.cursor/mcp.example.json`. It intentionally uses `postgresql+psycopg2://USER:PASSWORD@HOST:PORT/DBNAME` and must not be replaced with real credentials in source control.

The Project MCP server is separate from RunPod MCP. RunPod MCP is infrastructure-only; Project MCP is the ML-agent tool-learning interface. Its logged tool interactions can become trajectory data, RULER can judge whether the agent selected and used tools correctly, and TRL GRPO can train the policy to improve future tool choices while keeping the existing RULER/vLLM, TRL, QLoRA, and Agent-Lightning-style pipeline intact.
'''
if '## Official Project MCP Server' not in readme_text:
    readme.write_text(readme_text.rstrip() + readme_section + '\n', encoding='utf-8')

print('Track 3 official MCP documentation/configuration patch applied.')
