"""Rollout worker app boundary.

The package initializer is intentionally lazy so importing ``rollout_worker``
does not load LangGraph, policy-client, or endpoint runtime dependencies. Use
``python -m src.training.agent_lightning_official_runner`` as the canonical
Track A entrypoint. ``rollout_worker.run_track_a`` is retained as a thin
compatibility wrapper.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["run_tasks"]


def __getattr__(name: str) -> Any:
    if name == "run_tasks":
        module = import_module("rollout_worker.run_track_a")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
