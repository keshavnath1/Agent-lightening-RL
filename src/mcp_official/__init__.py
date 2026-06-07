"""Official MCP server package for the self-improving ML agent project.

This package intentionally coexists with the older REST-style ``src/mcp_server``
implementation. The official server exposes safe, schema-driven project tools
through the Python MCP SDK and records tool-call telemetry for policy learning.
"""

__all__ = ["server", "logging", "resources", "prompts"]
