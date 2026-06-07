from __future__ import annotations

from typing import Any, Dict, List, Optional


async def list_mcp_tools(command: str, args: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """List tools exposed by an official MCP stdio server."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    server_params = StdioServerParameters(
        command=command,
        args=args or [],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.list_tools()
            return [tool.model_dump() for tool in result.tools]


async def call_mcp_tool(
    command: str,
    args: Optional[List[str]],
    tool_name: str,
    arguments: Dict[str, Any],
) -> Any:
    """Call a named tool on an official MCP stdio server."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    server_params = StdioServerParameters(
        command=command,
        args=args or [],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return await session.call_tool(tool_name, arguments)
