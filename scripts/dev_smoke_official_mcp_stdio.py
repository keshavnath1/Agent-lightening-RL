from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tools.official_mcp_client import call_mcp_tool, list_mcp_tools


async def main() -> None:
    command = sys.executable
    args = ["-m", "src.mcp_official.server"]
    tools = await list_mcp_tools(command, args)
    names = sorted(tool.get("name", "") for tool in tools)
    expected = {
        "postgres_get_task_metadata",
        "postgres_get_dataset_schema",
        "postgres_get_dataset_summary",
        "postgres_get_execution_dataset_source",
        "postgres_get_rollout_status",
        "postgres_get_reward_history",
        "postgres_debug_readonly_sql",
    }
    missing = sorted(expected.difference(names))
    if missing:
        raise AssertionError(f"Missing official MCP tools: {missing}")

    result = await call_mcp_tool(command, args, "postgres_debug_readonly_sql", {"sql": "select 1"})
    dumped = result.model_dump() if hasattr(result, "model_dump") else result
    text = json.dumps(dumped, default=str)
    if "disabled unless ALLOW_RAW_SQL_TOOL=1" not in text:
        raise AssertionError("Raw SQL debug call did not report disabled state")

    reports_dir = ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)
    out = reports_dir / "official_mcp_stdio_smoke.json"
    out.write_text(
        json.dumps(
            {
                "tool_count": len(tools),
                "tool_names": names,
                "raw_sql_debug_result": dumped,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"official_mcp_stdio_smoke_ok {out}")


if __name__ == "__main__":
    asyncio.run(main())
