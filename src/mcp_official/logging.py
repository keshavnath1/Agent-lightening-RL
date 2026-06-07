from __future__ import annotations

import functools
import inspect
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, TypeVar, cast

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MCP_TOOL_CALL_LOG = PROJECT_ROOT / "reports" / "mcp_official_tool_calls.jsonl"

F = TypeVar("F", bound=Callable[..., Any])


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_jsonable(value: Any, *, max_chars: int = 2000) -> Any:
    """Return a compact JSON-serializable representation for logs only."""
    try:
        json.dumps(value, default=str)
        rendered = value
    except Exception:
        rendered = repr(value)
    if isinstance(rendered, str) and len(rendered) > max_chars:
        return rendered[:max_chars] + "...<truncated>"
    if isinstance(rendered, dict):
        return {str(k): _safe_jsonable(v, max_chars=max_chars) for k, v in list(rendered.items())[:50]}
    if isinstance(rendered, (list, tuple)):
        return [_safe_jsonable(v, max_chars=max_chars) for v in list(rendered)[:50]]
    return rendered


def _summarize_result(result: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {"type": type(result).__name__}
    if isinstance(result, dict):
        summary["keys"] = sorted(str(k) for k in result.keys())[:25]
        if "ok" in result:
            summary["ok"] = bool(result.get("ok"))
        if "result" in result and isinstance(result["result"], dict):
            inner = result["result"]
            summary["result_keys"] = sorted(str(k) for k in inner.keys())[:25]
            for key in ("task_id", "found", "row_count", "column_count", "raw_rows_exposed"):
                if key in inner:
                    summary[key] = inner[key]
        if "row_count" in result:
            summary["row_count"] = result.get("row_count")
        if "raw_rows_exposed" in result:
            summary["raw_rows_exposed"] = result.get("raw_rows_exposed")
    elif isinstance(result, str):
        summary["chars"] = len(result)
        summary["preview"] = result[:240]
    elif isinstance(result, list):
        summary["items"] = len(result)
    return summary


def _append_log(record: dict[str, Any]) -> None:
    MCP_TOOL_CALL_LOG.parent.mkdir(parents=True, exist_ok=True)
    with MCP_TOOL_CALL_LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")


def log_mcp_tool_call(tool_name: str) -> Callable[[F], F]:
    """Decorate an official MCP tool and append a JSONL ToolCallRecord."""

    def decorator(func: F) -> F:
        signature = inspect.signature(func)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            started = time.perf_counter()
            try:
                bound = signature.bind_partial(*args, **kwargs)
                bound.apply_defaults()
                arguments = dict(bound.arguments)
            except Exception:
                arguments = {"args": list(args), "kwargs": kwargs}

            record: dict[str, Any] = {
                "timestamp": _now(),
                "tool_name": tool_name,
                "arguments": _safe_jsonable(arguments),
                "status": "started",
                "error": None,
                "duration_ms": None,
                "official_mcp": True,
                "result_summary": None,
            }
            try:
                result = func(*args, **kwargs)
                record["status"] = "success"
                record["result_summary"] = _summarize_result(result)
                return result
            except Exception as exc:
                record["status"] = "error"
                record["error"] = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                record["duration_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
                _append_log(record)

        return cast(F, wrapper)

    return decorator
