from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from src.telemetry.schema import AgentStep, ToolCallRecord


@dataclass
class AgentContext:
    task: dict[str, Any]
    policy_version: str
    artifacts: dict[str, str]
    tool_outputs: dict[str, Any]


class BaseAgent:
    name = 'BaseAgent'

    def step(self, context: AgentContext) -> AgentStep:
        raise NotImplementedError

    def tool_success(self, tool_name: str, arguments: dict[str, Any], output_ref: str | None = None, metadata: dict[str, Any] | None = None) -> ToolCallRecord:
        return ToolCallRecord(tool_name=tool_name, arguments=arguments, status='success', output_ref=output_ref, metadata=metadata or {})
