from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .trajectory import TrajectoryGroup


@dataclass(slots=True)
class RULERScore:
    """Relative judge score for one trajectory inside a group."""

    trajectory_id: str
    rank: int
    ruler_relative_score: float
    rationale: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RULERScoredGroup:
    """RULER-scored group consumed by TRL/GRPO dataset preparation."""

    task_id: str
    group: TrajectoryGroup
    scores: list[RULERScore] = field(default_factory=list)
    judge_model: str | None = None
    judge_mode: str = "vllm_judge"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "group": self.group.to_dict(),
            "scores": [score.to_dict() for score in self.scores],
            "judge_model": self.judge_model,
            "judge_mode": self.judge_mode,
            "metadata": self.metadata,
        }
