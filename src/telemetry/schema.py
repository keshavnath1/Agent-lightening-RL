from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any
import uuid


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ToolCallRecord:
    tool_name: str
    arguments: dict[str, Any]
    status: str
    output_ref: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    started_at: str = field(default_factory=now_iso)
    completed_at: str = field(default_factory=now_iso)


@dataclass
class AgentStep:
    agent_name: str
    action: str
    reasoning_summary: str
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    timestamp: str = field(default_factory=now_iso)


@dataclass
class Trajectory:
    task_id: str
    policy_version: str
    trajectory_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    steps: list[AgentStep] = field(default_factory=list)
    final_status: str = 'unknown'
    reward: float | None = None
    reward_metadata: dict[str, Any] = field(default_factory=dict)
    additional_histories: list[dict[str, Any]] = field(default_factory=list)
    created_at: str = field(default_factory=now_iso)

    def add_history(
        self,
        name: str,
        messages_and_choices: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Attach a compact auxiliary history without exposing chain-of-thought."""
        self.additional_histories.append({
            'name': name,
            'messages_and_choices': messages_and_choices,
            'tools': tools or [],
            'metadata': metadata or {},
        })

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
