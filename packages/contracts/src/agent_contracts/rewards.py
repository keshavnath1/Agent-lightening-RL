from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


class RewardContractError(ValueError):
    """Raised when reward metadata does not satisfy the shared contract."""


@dataclass(slots=True)
class RewardBreakdown:
    """Shared reward payload consumed by evaluation, RULER scoring, and training."""

    final_reward: float
    valid_json: float = 0.0
    workflow_score: float = 0.0
    tool_score: float = 0.0
    ruler_relative_score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RewardBreakdown":
        return cls(
            final_reward=float(data.get("final_reward", data.get("reward", 0.0))),
            valid_json=float(data.get("valid_json", 0.0)),
            workflow_score=float(data.get("workflow_score", 0.0)),
            tool_score=float(data.get("tool_score", 0.0)),
            ruler_relative_score=(
                None if data.get("ruler_relative_score") is None else float(data["ruler_relative_score"])
            ),
            metadata=dict(data.get("metadata") or {}),
        )


def validate_reward_breakdown(data: dict[str, Any] | RewardBreakdown) -> RewardBreakdown:
    reward = data if isinstance(data, RewardBreakdown) else RewardBreakdown.from_dict(data)
    for name in ("final_reward", "valid_json", "workflow_score", "tool_score"):
        value = getattr(reward, name)
        if not 0.0 <= float(value) <= 1.0:
            raise RewardContractError(f"{name} must be in [0, 1], got {value!r}")
    if reward.ruler_relative_score is not None and not 0.0 <= reward.ruler_relative_score <= 1.0:
        raise RewardContractError(
            f"ruler_relative_score must be in [0, 1], got {reward.ruler_relative_score!r}"
        )
    return reward
