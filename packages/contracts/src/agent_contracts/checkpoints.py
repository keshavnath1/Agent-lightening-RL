from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class CheckpointManifest:
    """Stable checkpoint metadata shared by training, serving, and dashboard code."""

    checkpoint_id: str
    path: str
    base_model: str
    training_method: str
    reward_mode: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CheckpointManifest":
        return cls(
            checkpoint_id=str(data.get("checkpoint_id") or data.get("id") or "unknown"),
            path=str(data.get("path") or ""),
            base_model=str(data.get("base_model") or data.get("model_name") or "unknown"),
            training_method=str(data.get("training_method") or data.get("method") or "unknown"),
            reward_mode=data.get("reward_mode"),
            metrics={k: float(v) for k, v in dict(data.get("metrics") or {}).items()},
            metadata=dict(data.get("metadata") or {}),
        )
