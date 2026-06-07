"""Reward and RULER scoring package boundary.

Concrete reward implementations remain importable from submodules, for example
``agent_rewards.policy_reward`` and ``agent_rewards.ruler_vllm_judge``. The
initializer avoids eager imports so optional judge/training dependencies are
not pulled into lightweight environments.
"""

from __future__ import annotations

__all__: list[str] = []
