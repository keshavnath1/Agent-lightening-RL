from __future__ import annotations

from typing import Any, List


async def load_tools_from_mcp_session(session: Any) -> List[Any]:
    """Load MCP tools into LangChain-compatible tools.

    Requires `langchain-mcp-adapters>=0.1.0`.
    """
    try:
        from langchain_mcp_adapters.tools import load_mcp_tools
    except ImportError as exc:
        raise RuntimeError(
            "langchain-mcp-adapters is not installed. "
            "Install with: pip install langchain-mcp-adapters"
        ) from exc

    return await load_mcp_tools(session)
