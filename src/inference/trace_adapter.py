"""
SpanTraceAdapter — converts LLMSpan dicts / trajectory steps into the
``agentlightning.Triplet`` format consumed by the GRPO training backend.

Architecture position
---------------------

  LightningStoreAdapter.query_spans(rollout_id)
           │  list[dict]  (LLMSpan dicts)
           ▼
   SpanTraceAdapter.adapt(spans)
           │  list[Triplet]
           ▼
  GRPOAlgorithm / veRL training script

Official API compatibility
--------------------------
``SpanTraceAdapter`` extends ``agentlightning.TraceAdapter[List[Triplet]]``
when the package is available, otherwise it subclasses a plain stub with the
same ``adapt(source) -> List[Triplet]`` contract.

``Triplet`` schema (mirrors agentlightning.Triplet)::

    {
      "prompt":   {"token_ids": List[int], "raw_content": Any},
      "response": {"token_ids": List[int], "raw_content": str},
      "reward":   float | None,
      "metadata": {"agent_name": str, "response_id": str, ...},
    }

Because our LLMSpans are text-level (no token IDs from the raw vLLM response),
``token_ids`` is always ``[]``.  The GRPO training script in this repository
uses ``raw_content`` for the prompt/completion text, so this is sufficient for
local training.  When the LLMProxy is active and ``return_token_ids=True`` is
enabled, token IDs arrive in the OTEL span attributes and can be extracted;
the adapter handles both cases transparently.
"""
from __future__ import annotations

from typing import Any, List, Optional, Sequence
from agentlightning import TraceAdapter as _ADAPTER_BASE  # type: ignore[import]
from agentlightning.types import Triplet as Triplet  # type: ignore[import]


def _make_triplet(
    prompt:   dict[str, Any],
    response: dict[str, Any],
    reward:   Optional[float],
    metadata: dict[str, Any],
) -> Any:
    """Return an official ``agentlightning.Triplet``."""
    return Triplet(
        prompt=prompt,
        response=response,
        reward=reward,
        metadata=metadata,
    )


class SpanTraceAdapter(_ADAPTER_BASE):
    """
    Convert LLMSpan dicts (from ``LightningStoreAdapter.query_spans()``) into
    ``Triplet`` trajectories for GRPO/veRL training.

    This adapter covers two span formats:

    1. **LLMSpan dicts** (produced by our sidecar):
       Keys: ``messages``, ``response``, ``error``, ``latency_ms``,
       ``model_name``, ``span_id``, ``rollout_id``.

    2. **OTEL-enriched span dicts** (produced when LLMProxy is active):
       Additional keys: ``prompt_token_ids``, ``response_token_ids``,
       ``gen_ai.response.id``, ``attributes.gen_ai.prompt.*``.

    The ``final_reward`` parameter overrides reward on the last triplet when
    per-step rewards are not available in the spans.

    Parameters
    ----------
    final_reward:
        If provided, sets the reward on the last triplet when the spans
        themselves carry no reward annotations.
    agent_name:
        Human-readable agent label injected into every ``Triplet.metadata``.
    """

    def __init__(
        self,
        final_reward: Optional[float] = None,
        agent_name: str = 'ml_agent',
    ) -> None:
        self.final_reward = final_reward
        self.agent_name   = agent_name

    # ── Official adapt() interface ─────────────────────────────────────────

    def adapt(self, source: Sequence[dict[str, Any]], /) -> List[Any]:
        """
        Convert a sequence of LLMSpan dicts into ``Triplet`` objects.

        Mirrors ``agentlightning.TraceAdapter.adapt()``.

        Args:
            source: Sequence of span dicts (LLMSpan or OTEL-enriched).

        Returns:
            Ordered list of ``Triplet`` objects (one per LLM call).
        """
        spans = list(source)
        if not spans:
            return []

        triplets: List[Any] = []
        for i, span in enumerate(spans):
            is_last = i == len(spans) - 1

            prompt  = self._extract_prompt(span)
            resp    = self._extract_response(span)
            reward  = self._extract_reward(span, is_last)
            meta    = self._extract_metadata(span)

            triplets.append(_make_triplet(prompt, resp, reward, meta))

        return triplets

    # ── Span field extractors ──────────────────────────────────────────────

    def _extract_prompt(self, span: dict[str, Any]) -> dict[str, Any]:
        """Extract prompt/messages from a span dict."""
        # OTEL-enriched: try gen_ai.prompt attributes
        attrs = span.get('attributes') or {}

        # Token IDs (present when LLMProxy + vLLM return_token_ids=True)
        token_ids: list[int] = []
        raw = attrs.get('prompt_token_ids') or span.get('prompt_token_ids')
        if isinstance(raw, (list, tuple)) and all(isinstance(x, int) for x in raw):
            token_ids = list(raw)

        # Raw content: our sidecar stores messages list
        raw_content = span.get('messages') or attrs.get('gen_ai.prompt') or []

        return {'token_ids': token_ids, 'raw_content': raw_content}

    def _extract_response(self, span: dict[str, Any]) -> dict[str, Any]:
        """Extract completion text / token IDs from a span dict."""
        attrs = span.get('attributes') or {}

        token_ids: list[int] = []
        raw = attrs.get('response_token_ids') or span.get('response_token_ids')
        if isinstance(raw, (list, tuple)) and all(isinstance(x, int) for x in raw):
            token_ids = list(raw)

        # Prefer the full response string stored by the sidecar
        raw_content = span.get('response') or attrs.get('gen_ai.completion') or ''

        return {'token_ids': token_ids, 'raw_content': raw_content}

    def _extract_reward(self, span: dict[str, Any], is_last: bool) -> Optional[float]:
        """
        Extract per-step reward from the span, falling back to ``final_reward``
        on the last step.
        """
        # Span may carry an explicit reward annotation
        r = span.get('reward') or span.get('attributes', {}).get('reward')
        if r is not None:
            try:
                return float(r)
            except (TypeError, ValueError):
                pass

        if is_last and self.final_reward is not None:
            return self.final_reward

        return None

    def _extract_metadata(self, span: dict[str, Any]) -> dict[str, Any]:
        """Build the Triplet metadata dict from span fields."""
        attrs = span.get('attributes') or {}
        return {
            'agent_name':   self.agent_name,
            'response_id':  span.get('span_id') or attrs.get('gen_ai.response.id'),
            'model_name':   span.get('model_name') or attrs.get('gen_ai.system'),
            'latency_ms':   span.get('latency_ms'),
            'rollout_id':   span.get('rollout_id'),
            'span_type':    span.get('span_type', 'llm_call'),
            'error':        span.get('error'),
        }


# ── Convenience: adapt a full rollout from the store ──────────────────────────

def adapt_rollout_spans(
    store: Any,
    rollout_id: str,
    final_reward: Optional[float] = None,
    agent_name: str = 'ml_agent',
) -> List[Any]:
    """
    Pull all spans for ``rollout_id`` from the store and return ``Triplet`` list.

    Convenience wrapper around ``SpanTraceAdapter.adapt()`` for single-rollout use::

        triplets = adapt_rollout_spans(store, 'r_abc123', final_reward=0.87)

    Parameters
    ----------
    store:
        A ``LightningStoreAdapter`` (or any object with ``query_spans(rollout_id)``).
    rollout_id:
        Rollout identifier to retrieve spans for.
    final_reward:
        Reward to assign to the last triplet when spans carry no reward.
    agent_name:
        Human-readable agent label for metadata.

    Returns
    -------
    List of ``Triplet`` objects, empty when no spans are available.
    """
    spans = store.query_spans(rollout_id)
    adapter = SpanTraceAdapter(final_reward=final_reward, agent_name=agent_name)
    return adapter.adapt(spans)
