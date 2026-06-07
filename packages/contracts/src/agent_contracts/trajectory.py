from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .rewards import RewardBreakdown
from .tool_calls import ToolCallRecord


@dataclass(slots=True)
class Trajectory:
    """Stable rollout trajectory exchanged between rollout, scoring, and training."""

    task_id: str
    policy: str
    prompt: str = ""
    completion: str = ""
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    reward: RewardBreakdown | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["tool_calls"] = [call.to_dict() for call in self.tool_calls]
        payload["reward"] = None if self.reward is None else self.reward.to_dict()
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Trajectory":
        return cls(
            task_id=str(data.get("task_id") or data.get("id") or "unknown"),
            policy=str(data.get("policy") or data.get("policy_label") or "unknown"),
            prompt=str(data.get("prompt") or ""),
            completion=str(data.get("completion") or data.get("response") or ""),
            tool_calls=[ToolCallRecord.from_dict(x) for x in data.get("tool_calls", [])],
            reward=(None if data.get("reward") is None else RewardBreakdown.from_dict(data["reward"])),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass(slots=True)
class TrajectoryGroup:
    """Group of candidate trajectories for the same task, used by RULER scoring."""

    task_id: str
    trajectories: list[Trajectory] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "trajectories": [traj.to_dict() for traj in self.trajectories],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrajectoryGroup":
        return cls(
            task_id=str(data.get("task_id") or "unknown"),
            trajectories=[Trajectory.from_dict(x) for x in data.get("trajectories", [])],
            metadata=dict(data.get("metadata") or {}),
        )
