"""Official Project MCP Server service package.

The package initializer intentionally avoids importing ``mcp`` because that
requires the optional MCP SDK and database-tool dependencies. Use
``from project_mcp_server.server import mcp`` when starting the service.
"""

from __future__ import annotations

__all__: list[str] = []
