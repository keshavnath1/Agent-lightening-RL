"""
Lightning Client Sidecar  – CPU agent plane.

Mirrors the Agent Lightning "non-intrusive sidecar" design:

  - Wraps any object that exposes .chat() / .decision() (e.g. PolicyClient)
    WITHOUT touching agent business logic.
  - Intercepts every LLM call: records full LLMSpan
    (messages, response, error, latency_ms).
  - After a rollout completes, reports a structured RolloutReport
    to the GPU Lightning Server via its /api/rollouts/report endpoint.
  - Can also pull the next task from /api/tasks/pull when
    the supervisor is running in Lightning-server-driven mode.

Enabled automatically when LIGHTNING_SERVER_URL is set in the environment.
If the env var is absent or the server is unreachable, all reporting calls
are silent no-ops so the normal CPU workflow is never broken.
"""
from __future__ import annotations

import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

import requests


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class LLMSpan:
    """One intercepted LLM call: prompt-in, completion-out, error, latency."""

    span_id: str
    messages: list[dict[str, Any]]
    response: str | None
    error: str | None
    latency_ms: float
    model_name: str
    started_at: str
    completed_at: str
    # Rollout attribution — populated by proxy_chat() when routing through
    # the Lightning LLMProxy so SpanTraceAdapter can group spans by rollout.
    rollout_id:  str | None = None
    attempt_id:  str | None = None
    sequence_id: int | None = None


@dataclass
class RolloutReport:
    """
    Structured report sent to the GPU Lightning Server after one agent rollout.

    Maps to the Agent Lightning transition tuple format:
        state_t, action_t, reward_t, state_t+1
    surfaced through the transitions list, with LLM spans and error monitoring
    included for the server's credit-assignment algorithms.

    trajectory_dict carries the full scored Trajectory.to_dict() payload so
    the Lightning Server can reconstruct grouped-rollouts format (task_id +
    ranked_trajectories with full step dicts) for QLoRA/GRPO training without
    needing a separate prepare_grpo_dataset step.
    """

    task_id: str
    trajectory_id: str
    policy_version: str
    llm_spans: list[dict[str, Any]]        # serialised LLMSpan dicts
    transitions: list[dict[str, Any]]      # (state_t, action_t, reward_t, state_t+1)
    final_reward: float
    reward_metadata: dict[str, Any]
    error_types: list[str]                 # tool/execution errors for error monitoring
    trajectory_dict: dict[str, Any] = field(default_factory=dict)  # full Trajectory.to_dict()
    completed_at: str = field(default_factory=_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── Sidecar ────────────────────────────────────────────────────────────────────

class LightningClientSidecar:
    """
    Non-intrusive Lightning Client Sidecar.

    Drop-in wrapper around a PolicyClient (or any object with .chat()
    and .decision()).  Agent code calls sidecar.chat() identically to
    client.chat() — zero changes to agent logic required.

    Usage
    -----
    client  = PolicyClient(policy_version=...)
    sidecar = LightningClientSidecar(client, server_url=LIGHTNING_SERVER_URL)
    # pass sidecar wherever client would normally be used

    When LIGHTNING_SERVER_URL is not configured, enabled=False and all
    reporting methods are silent no-ops.
    """

    def __init__(
        self,
        policy_client: Any | None = None,
        server_url: str | None = None,
    ) -> None:
        self._client = policy_client
        self.server_url = (server_url or os.getenv('LIGHTNING_SERVER_URL', '')).rstrip('/')
        self.enabled = bool(self.server_url)
        self._spans: list[LLMSpan] = []
        self._original_chat: Any | None = None

        # Non-intrusive monkey-patch: replace client.chat at the instance level so
        # ALL LLM calls — including those made indirectly via client.decision() —
        # are routed through this sidecar's interception logic.
        # This implements the Agent Lightning "sidecar design" where no modification
        # to agent business-logic code is required.
        if policy_client is not None and self.enabled and hasattr(policy_client, 'chat'):
            self._original_chat = policy_client.chat   # save unpatched bound method
            _sidecar = self
            def _patched_chat(messages: list, temperature: float = 0.2) -> str:  # noqa: E306
                return _sidecar.chat(messages, temperature)
            policy_client.chat = _patched_chat         # instance-level shadow

    # ── passthrough properties ──────────────────────────────────────────────

    @property
    def model_name(self) -> str:
        return self._client.model_name if self._client else 'unknown'

    @property
    def base_url(self) -> str:
        return self._client.base_url if self._client else ''

    # ── intercepted LLM call ────────────────────────────────────────────────

    def chat(self, messages: list[dict[str, Any]], temperature: float = 0.2) -> str:
        """
        Drop-in replacement for PolicyClient.chat().

        When the sidecar is enabled and has monkey-patched the client, this method
        is also called by client.decision() → client.chat() (the patched version),
        so every LLM call — direct or indirect — is intercepted without touching
        any agent code.
        """
        if self._client is None:
            raise RuntimeError('LightningClientSidecar: no underlying PolicyClient configured.')

        # Use the stored original (unpatched) bound method to avoid infinite recursion
        original = self._original_chat or self._client.chat

        span_id = str(uuid.uuid4())
        started_at = _now_iso()
        t0 = time.monotonic()
        response: str | None = None
        error: str | None = None
        try:
            response = original(messages, temperature)
            return response
        except Exception as exc:
            error = f'{exc.__class__.__name__}: {exc}'
            raise
        finally:
            latency_ms = round((time.monotonic() - t0) * 1000, 2)
            if self.enabled:
                self._spans.append(LLMSpan(
                    span_id=span_id,
                    messages=messages,
                    response=response,
                    error=error,
                    latency_ms=latency_ms,
                    model_name=self.model_name,
                    started_at=started_at,
                    completed_at=_now_iso(),
                ))

    def decision(self, task: dict[str, Any], policy_version: str | None = None) -> dict[str, Any]:
        """Delegates to the underlying client (spans are captured via chat() internally)."""
        if self._client is None:
            raise RuntimeError('LightningClientSidecar: no underlying PolicyClient configured.')
        return self._client.decision(task, policy_version=policy_version)

    # ── span access ────────────────────────────────────────────────────────

    def get_spans(self) -> list[LLMSpan]:
        return list(self._spans)

    def clear_spans(self) -> None:
        self._spans.clear()

    # ── Lightning Server API ────────────────────────────────────────────────

    def report_rollout(self, report: RolloutReport) -> bool:
        """
        POST the finished rollout to the Lightning Server's reporting API.

        Mirrors Agent Lightning's server reporting API:
            POST /api/rollouts/report
        Raises a RuntimeError when reporting cannot be completed. Strict
        integration mode should never silently drop rollout reports.
        """
        if not self.enabled:
            raise RuntimeError(
                'LightningClientSidecar.report_rollout() failed: LIGHTNING_SERVER_URL is not configured.'
            )
        try:
            resp = requests.post(
                f'{self.server_url}/api/rollouts/report',
                json=report.to_dict(),
                timeout=30,
            )
            resp.raise_for_status()
            return True
        except Exception as exc:
            raise RuntimeError(
                'LightningClientSidecar.report_rollout() failed while POSTing '
                f'{self.server_url}/api/rollouts/report: {exc.__class__.__name__}: {exc}'
            ) from exc

    def pull_task(self) -> dict[str, Any] | None:
        """
        Pull the next task from the Lightning Server task pool.

        Returns None when the pool is empty. Raises a RuntimeError for
        configuration or connectivity failures.
        """
        if not self.server_url:
            raise RuntimeError(
                'LightningClientSidecar.pull_task() failed: LIGHTNING_SERVER_URL is not configured.'
            )
        try:
            resp = requests.get(f'{self.server_url}/api/tasks/pull', timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                return data if data else None
            if resp.status_code in (204, 404):
                return None
            raise RuntimeError(
                f'LightningClientSidecar.pull_task() failed with HTTP {resp.status_code}: {resp.text}'
            )
        except Exception as exc:
            raise RuntimeError(
                'LightningClientSidecar.pull_task() failed while GET '
                f'{self.server_url}/api/tasks/pull: {exc.__class__.__name__}: {exc}'
            ) from exc

    # ── LLMProxy-aware chat (rollout-attributed routing) ──────────────────────

    def proxy_chat(
        self,
        messages: list[dict[str, Any]],
        rollout_id: str,
        attempt_id: str | None = None,
        sequence_id: int = 0,
        temperature: float = 0.2,
        model: str | None = None,
    ) -> str:
        """
        Route an LLM call through the Lightning LLMProxy URL format:

            ``{llm_proxy_url}/rollout/{rollout_id}/attempt/{attempt_id}/v1/chat/completions``

        This mirrors the Agent Lightning ``LLMProxy.get_llm(rollout_id, attempt_id)``
        pattern:  The ``RolloutAttemptMiddleware`` on the proxy server rewrites the
        path to the base ``/v1/chat/completions`` endpoint and injects the rollout
        attribution headers::

            x-rollout-id:   <rollout_id>
            x-attempt-id:   <attempt_id>
            x-sequence-id:  <seq>

        The span captured during this call is enriched with ``rollout_id`` and
        ``attempt_id`` so ``SpanTraceAdapter`` can group it back into the
        rollout's trace tree.

        Falls back to the standard ``chat()`` method when no LLMProxy URL is
        configured (``LIGHTNING_LLM_PROXY_URL`` env var absent).
        """
        llm_proxy_url = os.getenv('LIGHTNING_LLM_PROXY_URL', '').rstrip('/')
        if not llm_proxy_url:
            # No proxy configured — fall through to normal sidecar-intercepted chat
            return self.chat(messages, temperature)

        import json as _json

        aid = attempt_id or 'a0'
        proxy_endpoint = (
            f'{llm_proxy_url}/rollout/{rollout_id}/attempt/{aid}/v1/chat/completions'
        )
        headers = {
            'Content-Type':    'application/json',
            'x-rollout-id':    rollout_id,
            'x-attempt-id':    aid,
            'x-sequence-id':   str(sequence_id),
        }
        body = {
            'model':       model or (self._client.model_name if self._client else 'default'),
            'messages':    messages,
            'temperature': temperature,
        }

        span_id    = str(uuid.uuid4())
        started_at = _now_iso()
        t0         = time.monotonic()
        response: str | None = None
        error: str | None    = None
        try:
            resp = requests.post(proxy_endpoint, headers=headers, json=body, timeout=120)
            resp.raise_for_status()
            data     = resp.json()
            response = data['choices'][0]['message']['content']
            return response
        except Exception as exc:
            error = f'{exc.__class__.__name__}: {exc}'
            raise
        finally:
            latency_ms = round((time.monotonic() - t0) * 1000, 2)
            if self.enabled:
                # Enrich span with full rollout attribution for SpanTraceAdapter
                self._spans.append(LLMSpan(
                    span_id=span_id,
                    messages=messages,
                    response=response,
                    error=error,
                    latency_ms=latency_ms,
                    model_name=body['model'],
                    started_at=started_at,
                    completed_at=_now_iso(),
                    rollout_id=rollout_id,
                    attempt_id=aid,
                    sequence_id=sequence_id,
                ))

