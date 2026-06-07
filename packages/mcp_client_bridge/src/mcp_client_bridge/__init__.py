"""MCP client bridge package boundary.

Import concrete helpers from submodules to avoid pulling optional MCP or
LangChain adapter dependencies at package import time.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["function_to_json_schema"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        module = import_module("mcp_client_bridge.schema_from_function")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
