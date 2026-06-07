from __future__ import annotations

from typing import Any


def safe_ml_workflow_prompt(task_id: str) -> str:
    """Prompt an agent to solve an ML task through safe project metadata tools."""
    return f"""You are operating inside the self-improving ML-agent project.

Task ID: {task_id}

Use the official Project MCP safe tools before reasoning about this task. Prefer `postgres_get_task_metadata`, `postgres_get_dataset_schema`, `postgres_get_dataset_summary`, `postgres_get_column_profile`, `postgres_get_target_profile`, `postgres_get_rollout_status`, and `postgres_get_reward_history`. Do not request or expose raw dataset rows. The execution-only `postgres_get_execution_dataset_source` contract is for the Docker/materializer boundary, not planning prompts. Summarize what each safe tool establishes, then propose the next ML workflow step.
"""


def postgres_safe_query_policy() -> str:
    """Describe the PostgreSQL safety policy for project agents."""
    return """Official PostgreSQL safety policy for project agents:

Use the safe task-specific tools for normal agent operation. Raw SQL is a developer diagnostic path only and is disabled unless `ALLOW_RAW_SQL_TOOL=1`. Even when enabled, SQL must be read-only, must not include comments or semicolons, must not use `SELECT *`, and must not access blocked raw-row schemas such as `ml_data`. Never store database URLs, API keys, or other credentials in source code or prompts.
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
