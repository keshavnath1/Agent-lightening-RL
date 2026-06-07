"""Safe ML tools package.

Phase 2 keeps package imports lightweight. Import concrete tools from their
submodules, for example ``from ml_tools.sql_safety import validate_sql``.
A tiny lazy compatibility surface is provided for the lightweight SQL-safety
symbols only; PostgreSQL and profiling helpers are intentionally not imported
from package import time.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["SQLSafetyResult", "validate_sql"]


def __getattr__(name: str) -> Any:
    if name == "postgres_tooling":
        return import_module("ml_tools.postgres_tooling")
    if name in __all__:
        module = import_module("ml_tools.sql_safety")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
