"""
MLAgentLitAgent — ``agentlightning.LitAgent`` wrapper for the ML-agent pipeline.

Architecture position
---------------------

  agentlightning.Runner / LightningTrainer
           │  calls rollout(task, resources, rollout_obj)
           ▼
   MLAgentLitAgent.rollout()
           │  wraps SupervisorAgent.run_task()
           ▼
  LangGraph pipeline (policy_router → agents → reviewer_critic)
           │  returns Trajectory
           ▼
  emit_reward(final_reward) → spans captured in store via sidecar
           │
           ▼
  GRPOAlgorithm reads spans → SpanTraceAdapter → Triplet list → training

Official API compatibility
--------------------------
When ``agentlightning >= 0.3.1`` is installed this class extends
``agentlightning.LitAgent``.  When absent it uses a plain stub with the same
``rollout(task, resources, rollout)`` interface.

Both sync (``rollout()``) and async (``rollout_async()``) variants are
implemented so the agent works with both blocking and async runners.

Usage
-----
With official Trainer::

    from agentlightning import Trainer
    from src.agents.lit_agent import MLAgentLitAgent

    agent   = MLAgentLitAgent(policy_version='v1')
    trainer = Trainer(...)
    trainer.fit(agent)

Standalone (e.g. integration test)::

    from src.agents.lit_agent import MLAgentLitAgent
    agent = MLAgentLitAgent()
    reward = agent.rollout(
        task={'task_id': 't1', 'task_description': 'Predict churn'},
        resources={'policy_version': 'v1', 'checkpoint_path': 'checkpoints/...'},
        rollout=None,
    )
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ── Optional official base class ───────────────────────────────────────────────

try:
    from agentlightning import LitAgent as _AglLitAgent  # type: ignore[import]
    _LITAGENT_BASE = _AglLitAgent
    _HAS_AGL = True
except ImportError:
    class _LITAGENT_BASE:  # type: ignore[no-redef]
        """Minimal stub mirroring agentlightning.LitAgent[dict]."""

        def __init__(self, **kwargs: Any) -> None:
            pass

        def rollout(
            self,
            task: Any,
            resources: Any,
            rollout: Any,
        ) -> Any:
            raise NotImplementedError('Agents must implement rollout().')

        async def rollout_async(
            self,
            task: Any,
            resources: Any,
            rollout: Any,
        ) -> Any:
            raise NotImplementedError('Agents must implement rollout_async().')

        # Stubs for optional hooks the official Runner calls
        def on_rollout_start(self, *args: Any, **kwargs: Any) -> None: ...
        def on_rollout_end(self, *args: Any, **kwargs: Any) -> None: ...
    _HAS_AGL = False


class MLAgentLitAgent(_LITAGENT_BASE):
    """
    Agent Lightning ``LitAgent`` implementation wrapping the ML-agent pipeline.

    Each ``rollout()`` call runs the full LangGraph workflow via
    ``SupervisorAgent.run_task()``, then emits the final reward via
    ``agentlightning.emit_reward()`` (when the package is available) so the
    official Algorithm/Trainer loop can attribute credit across attempts.

    Parameters
    ----------
    policy_version:
        String label for the policy version used in this rollout batch.
        Overridden by ``resources.get('policy_version')`` when provided.
    policy_decision_dir:
        Directory containing baseline policy decision JSON files.
    require_live_policy:
        When True, the supervisor creates a live ``PolicyClient`` for the
        rollout (requires a running vLLM endpoint).
    lightning_server_url:
        Override URL for the Lightning Server (default: ``$LIGHTNING_SERVER_URL``).
    """

    def __init__(
        self,
        policy_version: str = 'v1',
        policy_decision_dir: str | Path = 'artifacts/policy_decisions',
        require_live_policy: bool = False,
        lightning_server_url: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.policy_version       = policy_version
        self.policy_decision_dir  = Path(policy_decision_dir)
        self.require_live_policy  = require_live_policy
        self.lightning_server_url = lightning_server_url or os.getenv('LIGHTNING_SERVER_URL', '')

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _make_supervisor(self) -> Any:
        """Lazily import and construct ``SupervisorAgent`` to avoid circular deps."""
        from src.agents.supervisor import SupervisorAgent  # noqa: PLC0415
        return SupervisorAgent(
            require_live_policy  = self.require_live_policy,
            policy_decision_dir  = self.policy_decision_dir,
            lightning_server_url = self.lightning_server_url,
        )

    def _resolve_policy_version(self, resources: Any) -> str:
        """Extract policy_version from the resources snapshot, if available."""
        if isinstance(resources, dict):
            return str(resources.get('policy_version') or self.policy_version)
        return self.policy_version

    def _emit_reward(self, reward: float) -> None:
        """
        Emit the final reward via the official ``agentlightning.emit_reward()``
        API so the Trainer/Algorithm loop can perform credit assignment.

        Silently ignored when the package is not installed.
        """
        if _HAS_AGL:
            try:
                import agentlightning as agl  # noqa: PLC0415
                agl.emit_reward(reward)
            except Exception:
                pass  # not inside a Trainer tracing context — safe to ignore

    # ── LitAgent interface ─────────────────────────────────────────────────────

    def rollout(
        self,
        task: Any,
        resources: Any,
        rollout: Any,
    ) -> Any:
        """
        Execute one rollout synchronously.

        Mirrors ``agentlightning.LitAgent.rollout(task, resources, rollout)``.

        Args:
            task:
                Task payload dict (must contain at least ``task_id`` and
                ``task_description``).
            resources:
                Named resource snapshot from the store (e.g.
                ``{'checkpoint_path': '...', 'policy_version': 'v2'}``).
            rollout:
                Rollout metadata object (or ``None`` in standalone mode).

        Returns:
            Final scalar reward (``float``).  The official ``LitAgent``
            interface accepts ``float | List[Span] | None``; returning a float
            is the simplest and most compatible form.
        """
        if not isinstance(task, dict):
            task = {'task_id': str(task), 'task_description': str(task)}

        policy_version = self._resolve_policy_version(resources)

        try:
            supervisor  = self._make_supervisor()
            trajectory  = supervisor.run_task(task, policy_version=policy_version)
            reward      = float(trajectory.reward or 0.0)
            self._emit_reward(reward)
            return reward
        except Exception as exc:
            logger.error('MLAgentLitAgent.rollout() failed: %s', exc, exc_info=True)
            raise RuntimeError(
                'MLAgentLitAgent rollout failed in strict mode and cannot fall back to a synthetic reward.'
            ) from exc

    async def rollout_async(
        self,
        task: Any,
        resources: Any,
        rollout: Any,
    ) -> Any:
        """
        Async variant of ``rollout()``.

        Delegates to the sync implementation via ``asyncio.to_thread`` so the
        blocking LangGraph pipeline does not freeze the event loop.

        Mirrors ``agentlightning.LitAgent.rollout_async(task, resources, rollout)``.
        """
        import asyncio  # noqa: PLC0415
        return await asyncio.to_thread(self.rollout, task, resources, rollout)

    # ── Optional lifecycle hooks ───────────────────────────────────────────────

    def on_rollout_start(self, task: Any, *args: Any, **kwargs: Any) -> None:
        logger.debug('MLAgentLitAgent: rollout starting for task %s', task.get('task_id') if isinstance(task, dict) else task)

    def on_rollout_end(self, task: Any, rollout: Any, *args: Any, **kwargs: Any) -> None:
        logger.debug('MLAgentLitAgent: rollout complete')
