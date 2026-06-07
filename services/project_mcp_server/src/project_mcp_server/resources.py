from __future__ import annotations

from pathlib import Path
from typing import Any

from ml_tools.postgres_tooling import POSTGRES_SAFE_TOOL_FUNCTIONS

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def tool_catalog() -> str:
    """Return a markdown catalog of safe official MCP project tools."""
    rows: list[str] = []
    descriptions = {
        "postgres_get_task_metadata": "Returns task-level metadata such as objective, target column, and metric without raw rows.",
        "postgres_get_dataset_schema": "Returns safe column names and types from ml_registry metadata without exposing row values.",
        "postgres_get_dataset_summary": "Returns aggregate dataset metadata including row count, column count, source, and target metadata.",
        "postgres_get_column_profile": "Returns one safe aggregate column profile without raw values.",
        "postgres_get_target_profile": "Returns safe aggregate target profile metadata.",
        "postgres_get_rollout_status": "Returns aggregate rollout status counts and trajectory source references.",
        "postgres_get_reward_history": "Returns recent reward records for the task from trajectory summaries.",
        "postgres_debug_readonly_sql": "Developer-only read SQL tool. Disabled unless ALLOW_RAW_SQL_TOOL=1 and guarded by strict SQL safety validation.",
    }
    for name in sorted(POSTGRES_SAFE_TOOL_FUNCTIONS):
        rows.append(f"| `{name}` | {descriptions.get(name, 'Safe project tool.')} | No raw dataset rows |")
    rows.append(f"| `postgres_debug_readonly_sql` | {descriptions['postgres_debug_readonly_sql']} | Gated debug only |")
    table = "\n".join(rows)
    return f"""# Official Project MCP Tool Catalog

The official project MCP server exposes safe ML-agent tools for task metadata, dataset summaries, column/target profiles, rollouts, and reward history. Execution-only dataset source contracts are kept separate from ordinary planning tools.

| Tool | Purpose | Data policy |
|---|---|---|
{table}

All ordinary tools are designed to avoid raw dataset row exposure. The raw SQL debug tool is disabled by default and is intended for developer diagnostics only.
"""


def reward_schema() -> str:
    """Return a markdown description of tool-use reward records."""
    return """# Reward and Tool-Use Schema

Official MCP calls are logged to `reports/mcp_official_tool_calls.jsonl`. Each record is suitable for trajectory analysis and includes `timestamp`, `tool_name`, `arguments`, `status`, `error`, `duration_ms`, `official_mcp`, and `result_summary`.

RULER can evaluate whether a rollout selected a relevant safe tool, avoided raw-row access, and produced a useful task answer. TRL GRPO can then consume grouped rewards to improve future policy tool selection while preserving the existing RULER, GRPO, and Agent-Lightning-style pipeline.
"""


def workflow_architecture() -> str:
    """Return the official MCP workflow architecture as markdown."""
    return """# Official MCP Workflow Architecture

```text
LangGraph / Agent / ART rollout
  -> Official MCP client
  -> Official Project MCP Server
  -> Safe project tools
  -> ToolCallRecord logs
  -> RULER evaluates tool usage
  -> TRL GRPO trains policy
  -> Agent Lightning registers/reloads checkpoint
```

RunPod MCP, when available, should be treated as infrastructure control only. The Project MCP server is the ML-agent tool-learning interface and should not expose raw dataset rows or credentials.
"""


def latest_rollout_summary() -> str:
    """Return a compact markdown view of recent rollout/evaluation report files."""
    reports_dir = PROJECT_ROOT / "reports"
    if not reports_dir.exists():
        return "# Latest Rollout Summary\n\nNo `reports/` directory is present yet."
    candidates = sorted(
        [p for p in reports_dir.glob("*") if p.is_file() and p.suffix.lower() in {".md", ".json", ".txt"}],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )[:12]
    if not candidates:
        return "# Latest Rollout Summary\n\nNo rollout report files were found in `reports/`."
    rows = ["| File | Bytes |", "|---|---:|"]
    for path in candidates:
        rows.append(f"| `{path.relative_to(PROJECT_ROOT)}` | {path.stat().st_size} |")
    return "# Latest Rollout Summary\n\n" + "\n".join(rows)


def register_resources(mcp: Any) -> None:
    """Register read-only project resources on a FastMCP server."""

    @mcp.resource("project://tool-catalog")
    def _tool_catalog() -> str:
        return tool_catalog()

    @mcp.resource("project://reward-schema")
    def _reward_schema() -> str:
        return reward_schema()

    @mcp.resource("project://workflow-architecture")
    def _workflow_architecture() -> str:
        return workflow_architecture()

    @mcp.resource("project://latest-rollout-summary")
    def _latest_rollout_summary() -> str:
        return latest_rollout_summary()
