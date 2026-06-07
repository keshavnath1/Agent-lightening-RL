from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class ToolCallRecord:
    """Stable record for one agent/tool interaction."""

    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    status: str = "unknown"
    output_ref: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolCallRecord":
        return cls(
            tool_name=str(data.get("tool_name") or data.get("name") or "unknown"),
            arguments=dict(data.get("arguments") or {}),
            status=str(data.get("status") or "unknown"),
            output_ref=data.get("output_ref"),
            error=data.get("error"),
            metadata=dict(data.get("metadata") or {}),
        )
