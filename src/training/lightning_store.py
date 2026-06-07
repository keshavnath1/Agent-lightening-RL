"""
LightningStoreAdapter — thread-safe in-memory implementation of the
Agent Lightning LightningStore interface.

Mirrors the core methods of ``agentlightning.InMemoryLightningStore`` so
our Lightning Server can slot in the official package when it is available
while running standalone today.  The official interface is fully async and
comprehensive; this adapter implements the subset we actually exercise:

  Rollout lifecycle
    enqueue_rollout / dequeue_rollout / mark_rollout_succeeded /
    mark_rollout_failed / query_rollouts / get_rollout_by_id /
    wait_for_rollouts

  Span ingest
    add_span / query_spans

  Resource versioning
    add_resources / get_latest_resources / query_resources

  Rollout-report ingestion  (our non-standard, pragmatic entry point)
    accept_rollout_report

  Observability
    statistics

All mutating methods are synchronous and protected by a single re-entrant
lock so they are safe to call from multiple FastAPI handler threads.

Parity note
-----------
The official LightningStore uses an async interface throughout (``async def``
methods).  We keep the adapter synchronous to avoid injecting asyncio into
the FastAPI handlers; the trade-off is accepted because we run on a single
GPU pod where contention is low.  If the official package becomes available
the server can swap ``LightningStoreAdapter`` for
``agentlightning.InMemoryLightningStore`` with minimal handler changes.
"""
from __future__ import annotations

import threading
import uuid
from collections import deque
from datetime import datetime, timezone
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id(prefix: str = '') -> str:
    return f'{prefix}{uuid.uuid4().hex[:12]}'


# ── Rollout status constants (mirrors agentlightning.RolloutStatus) ────────────

ROLLOUT_QUEUING    = 'queuing'
ROLLOUT_PREPARING  = 'preparing'
ROLLOUT_RUNNING    = 'running'
ROLLOUT_SUCCEEDED  = 'succeeded'
ROLLOUT_FAILED     = 'failed'
ROLLOUT_CANCELLED  = 'cancelled'
ROLLOUT_REQUEUING  = 'requeuing'

_TERMINAL_STATUSES = {ROLLOUT_SUCCEEDED, ROLLOUT_FAILED, ROLLOUT_CANCELLED}


class LightningStoreAdapter:
    """
    Thread-safe in-memory store compatible with the Agent Lightning
    LightningStore interface.

    The adapter is the single source of truth for task pool, rollout records,
    spans, and resource snapshots inside the Lightning Server.  FastAPI
    handlers access it exclusively through this class; the raw
    ``_task_pool`` / ``_rollouts`` module-level state in the legacy server
    has been replaced by a singleton instance of this class.

    Usage::

        store = LightningStoreAdapter()

        # Task side
        store.enqueue_rollout({'task_description': 'Predict churn'})
        rollout = store.dequeue_rollout()

        # After agent execution
        store.accept_rollout_report(report_dict)
        store.mark_rollout_succeeded(rollout_id, reward=0.87)

        # Resource versioning
        store.add_resources({'checkpoint_path': '/checkpoints/v2'})
        latest = store.get_latest_resources()

        # Algorithm reads
        all_done = store.query_rollouts(status_in=['succeeded'])
    """

    def __init__(self, max_rollout_history: int = 500) -> None:
        self._lock               = threading.Lock()
        self._max_history        = max_rollout_history

        # Task queue: FIFO deque of raw task dicts
        self._task_queue: deque[dict[str, Any]] = deque()

        # Rollout records keyed by rollout_id
        self._rollouts: dict[str, dict[str, Any]] = {}
        # Ordered list of rollout_ids for history management
        self._rollout_order: list[str] = []

        # Spans per rollout_id: list[dict]
        self._spans: dict[str, list[dict[str, Any]]] = {}

        # Resource snapshots: list[{resources_id, resources, created_at}]
        self._resources: list[dict[str, Any]] = []
        self._latest_resources_id: str | None  = None

        # Counters exposed via statistics()
        self._transitions_written: int = 0

    # ── Task queue ─────────────────────────────────────────────────────────────

    def enqueue_rollout(
        self,
        input: dict[str, Any],
        mode: str | None = None,
        resources_id: str | None = None,
        config: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Persist a rollout in ``queuing`` state.

        Mirrors ``LightningStore.enqueue_rollout()``.  The returned dict
        has the same shape as ``agentlightning.Rollout``:
        ``rollout_id``, ``input``, ``status``, ``mode``, ``start_time``.
        """
        rollout_id = _new_id('r_')
        rollout = {
            'rollout_id':    rollout_id,
            'input':         input,
            'status':        ROLLOUT_QUEUING,
            'mode':          mode,
            'resources_id':  resources_id or self._latest_resources_id,
            'config':        config or {},
            'metadata':      metadata or {},
            'start_time':    _now(),
            'end_time':      None,
            'final_reward':  None,
            'reward_metadata': {},
            'trajectory_dict': {},
            'error_types':   [],
            'llm_span_count': 0,
            'transitions_count': 0,
            'policy_version': None,
        }
        with self._lock:
            self._task_queue.append({'rollout_id': rollout_id, 'input': input})
            self._rollouts[rollout_id] = rollout
            self._rollout_order.append(rollout_id)
            self._spans[rollout_id] = []
            self._trim_history()
        return rollout

    def dequeue_rollout(self, worker_id: str | None = None) -> dict[str, Any] | None:
        """
        Claim the oldest queued rollout and transition it to ``preparing``.

        Returns ``None`` when the queue is empty (mirrors the official API).
        """
        with self._lock:
            if not self._task_queue:
                return None
            item = self._task_queue.popleft()
            rollout_id = item['rollout_id']
            if rollout_id in self._rollouts:
                self._rollouts[rollout_id]['status'] = ROLLOUT_PREPARING
            return item

    def pull_task(self) -> dict[str, Any]:
        """
        Convenience wrapper used by the FastAPI ``GET /api/tasks/pull``
        endpoint.

        Returns the task input dict enriched with ``lightning_rollout_id`` so
        CPU agents can echo it back in ``RolloutReport`` and
        ``accept_rollout_report()`` can update the *same* rollout record
        instead of creating a second one.  Returns ``{}`` when the queue is
        empty so the normal CPU fallback path is never broken.
        """
        item = self.dequeue_rollout()
        if item is None:
            return {}
        task_input = item.get('input', {})
        rollout_id = item.get('rollout_id')
        if rollout_id and isinstance(task_input, dict):
            # Inject so the sidecar / supervisor can correlate the report back
            # to this store record and avoid stale 'preparing' rollouts.
            return {**task_input, 'lightning_rollout_id': rollout_id}
        return task_input

    def load_tasks_from_jsonl(self, path: str) -> int:
        """Bulk-enqueue tasks from a JSONL file.  Returns number loaded."""
        from pathlib import Path
        count = 0
        for line in Path(path).read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if line:
                import json
                self.enqueue_rollout(json.loads(line))
                count += 1
        return count

    # ── Rollout state transitions ──────────────────────────────────────────────

    def mark_rollout_running(self, rollout_id: str) -> None:
        """Transition a rollout from ``preparing`` to ``running``."""
        with self._lock:
            r = self._rollouts.get(rollout_id)
            if r and r['status'] == ROLLOUT_PREPARING:
                r['status'] = ROLLOUT_RUNNING

    def mark_rollout_succeeded(
        self,
        rollout_id: str,
        reward: float = 0.0,
        reward_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Transition a rollout to ``succeeded`` and record its final reward."""
        with self._lock:
            r = self._rollouts.get(rollout_id)
            if r:
                r['status']          = ROLLOUT_SUCCEEDED
                r['end_time']        = _now()
                r['final_reward']    = reward
                r['reward_metadata'] = reward_metadata or {}

    def mark_rollout_failed(self, rollout_id: str, error: str = '') -> None:
        """Transition a rollout to ``failed``."""
        with self._lock:
            r = self._rollouts.get(rollout_id)
            if r:
                r['status']   = ROLLOUT_FAILED
                r['end_time'] = _now()
                if error:
                    r['error_types'].append(error)

    # ── Span ingest ────────────────────────────────────────────────────────────

    def add_span(self, span: dict[str, Any]) -> dict[str, Any]:
        """
        Persist a span (LLM call record or tool-call record) for a rollout.

        Mirrors ``LightningStore.add_span()``.  ``span`` must contain
        ``rollout_id``; ``span_id`` is auto-assigned when absent.
        Transitions the rollout to ``running`` if it was still ``preparing``.
        """
        rollout_id = span.get('rollout_id', '')
        span = {**span, 'span_id': span.get('span_id') or _new_id('sp_')}
        with self._lock:
            if rollout_id not in self._spans:
                self._spans[rollout_id] = []
            self._spans[rollout_id].append(span)
            # heartbeat: move to running
            r = self._rollouts.get(rollout_id)
            if r and r['status'] in (ROLLOUT_PREPARING, ROLLOUT_QUEUING):
                r['status'] = ROLLOUT_RUNNING
        return span

    def query_spans(
        self,
        rollout_id: str,
        limit: int = -1,
    ) -> list[dict[str, Any]]:
        """Return all stored spans for ``rollout_id``."""
        with self._lock:
            spans = list(self._spans.get(rollout_id, []))
        if limit > 0:
            spans = spans[-limit:]
        return spans

    # ── Rollout-report ingestion (our extension, not in official interface) ────

    def accept_rollout_report(self, report: dict[str, Any]) -> None:
        """
        Ingest a full ``RolloutReport`` dict from the CPU-side sidecar.

        Stores the report as a pseudo-rollout record so ``query_rollouts``
        and the algorithm can access the full payload.  This is the primary
        write path for the ``POST /api/rollouts/report`` endpoint.

        Also persists each LLMSpan as a native span record and updates the
        rollout's status to ``succeeded`` / ``failed`` based on ``error_types``.

        Rollout-ID correlation
        ----------------------
        When the task was pulled via ``pull_task()`` the returned dict contains
        ``lightning_rollout_id``.  If the report echoes that field back, this
        method updates the *existing* preparing record rather than creating a
        second one, preventing stale ``preparing`` entries from accumulating in
        ``statistics()``.
        """
        task_id       = report.get('task_id', _new_id('task_'))
        trajectory_id = report.get('trajectory_id', _new_id('traj_'))

        # Prefer the rollout_id the store issued when the task was pulled so we
        # update the existing record instead of creating an orphaned duplicate.
        preexisting_id = report.get('lightning_rollout_id')
        rollout_id = (
            preexisting_id
            if preexisting_id and preexisting_id in self._rollouts
            else trajectory_id
        )

        # Upsert rollout record
        with self._lock:
            if rollout_id not in self._rollouts:
                self._rollouts[rollout_id] = {
                    'rollout_id':    rollout_id,
                    'input':         {'task_id': task_id},
                    'status':        ROLLOUT_RUNNING,
                    'mode':          'train',
                    'resources_id':  self._latest_resources_id,
                    'config':        {},
                    'metadata':      {},
                    'start_time':    report.get('completed_at') or _now(),
                    'end_time':      None,
                    'final_reward':  None,
                    'reward_metadata': {},
                    'trajectory_dict': {},
                    'error_types':   [],
                    'llm_span_count': 0,
                    'transitions_count': 0,
                    'policy_version': None,
                }
                self._rollout_order.append(rollout_id)
                self._spans[rollout_id] = []

            r = self._rollouts[rollout_id]
            r['task_id']          = task_id
            r['trajectory_id']    = trajectory_id
            r['policy_version']   = report.get('policy_version')
            r['final_reward']     = float(report.get('final_reward') or 0.0)
            r['reward_metadata']  = report.get('reward_metadata') or {}
            r['error_types']      = report.get('error_types') or []
            r['trajectory_dict']  = report.get('trajectory_dict') or {}
            r['llm_span_count']   = len(report.get('llm_spans') or [])
            r['transitions_count'] = len(report.get('transitions') or [])
            r['end_time']         = report.get('completed_at') or _now()
            r['status']           = ROLLOUT_FAILED if r['error_types'] else ROLLOUT_SUCCEEDED

            # Store LLMSpans as native span records
            for llm_span in report.get('llm_spans') or []:
                self._spans[rollout_id].append({
                    **llm_span,
                    'rollout_id': rollout_id,
                    'span_type':  'llm_call',
                })

            self._transitions_written += r['transitions_count']

            # Cancel any stale 'preparing' record for the same task_id that
            # was created when the task was first pulled but never correlated.
            # This keeps statistics() accurate when lightning_rollout_id was
            # not echoed back (e.g. older sidecar versions).
            for rid, rec in self._rollouts.items():
                if (
                    rid != rollout_id
                    and rec.get('status') == ROLLOUT_PREPARING
                    and rec.get('input', {}).get('task_id') == task_id
                ):
                    rec['status']   = ROLLOUT_CANCELLED
                    rec['end_time'] = _now()

            self._trim_history()

    def _trim_history(self) -> None:
        """Evict oldest rollout records when over capacity (call under lock)."""
        while len(self._rollout_order) > self._max_history:
            old_id = self._rollout_order.pop(0)
            self._rollouts.pop(old_id, None)
            self._spans.pop(old_id, None)

    # ── Rollout queries ────────────────────────────────────────────────────────

    def query_rollouts(
        self,
        status_in: list[str] | None = None,
        limit: int = -1,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """
        Return rollout records filtered by status.

        Mirrors ``LightningStore.query_rollouts()``.  When ``status_in`` is
        ``None`` all statuses are returned.
        """
        with self._lock:
            all_rollouts = [self._rollouts[rid] for rid in self._rollout_order
                            if rid in self._rollouts]
        if status_in:
            all_rollouts = [r for r in all_rollouts if r['status'] in status_in]
        all_rollouts = all_rollouts[offset:]
        if limit > 0:
            all_rollouts = all_rollouts[:limit]
        return all_rollouts

    def get_rollout_by_id(self, rollout_id: str) -> dict[str, Any] | None:
        """Fetch a rollout by identifier."""
        with self._lock:
            return dict(self._rollouts[rollout_id]) if rollout_id in self._rollouts else None

    def wait_for_rollouts(
        self,
        rollout_ids: list[str],
        timeout: float | None = None,
    ) -> list[dict[str, Any]]:
        """
        Return the subset of ``rollout_ids`` that are already in a terminal
        state.  Non-blocking (mirrors poll semantics; the caller should loop).
        """
        with self._lock:
            return [
                dict(self._rollouts[rid])
                for rid in rollout_ids
                if rid in self._rollouts
                and self._rollouts[rid]['status'] in _TERMINAL_STATUSES
            ]

    # ── Resource versioning ────────────────────────────────────────────────────

    def add_resources(self, resources: dict[str, Any]) -> dict[str, Any]:
        """
        Persist a new immutable resource snapshot and mark it as latest.

        Mirrors ``LightningStore.add_resources()``.  ``resources`` is a
        free-form mapping, e.g.::

            {'checkpoint_path': '/checkpoints/qwen25-3b-agent-lora',
             'policy_version': 'v2_finetuned', 'adapter_name': 'agent_adapter'}

        Returns the stored snapshot including its auto-generated
        ``resources_id``.
        """
        resources_id = _new_id('res_')
        snapshot = {
            'resources_id': resources_id,
            'resources':    dict(resources),
            'created_at':   _now(),
        }
        with self._lock:
            self._resources.append(snapshot)
            self._latest_resources_id = resources_id
        return snapshot

    def get_latest_resources(self) -> dict[str, Any] | None:
        """Fetch the most recently added resource snapshot."""
        with self._lock:
            if not self._resources:
                return None
            return dict(self._resources[-1])

    def get_resources_by_id(self, resources_id: str) -> dict[str, Any] | None:
        """Return a specific resource snapshot by identifier."""
        with self._lock:
            for r in self._resources:
                if r['resources_id'] == resources_id:
                    return dict(r)
        return None

    def query_resources(self, limit: int = -1) -> list[dict[str, Any]]:
        """Return all resource snapshots in insertion order."""
        with self._lock:
            snapshots = list(self._resources)
        if limit > 0:
            snapshots = snapshots[-limit:]
        return snapshots

    # ── Observability ──────────────────────────────────────────────────────────

    def statistics(self) -> dict[str, Any]:
        """
        Return store statistics compatible with
        ``LightningStore.statistics()``.
        """
        with self._lock:
            total          = len(self._rollouts)
            by_status: dict[str, int] = {}
            for r in self._rollouts.values():
                s = r.get('status', 'unknown')
                by_status[s] = by_status.get(s, 0) + 1
            queue_size     = len(self._task_queue)
            resource_count = len(self._resources)
            tw             = self._transitions_written
        return {
            'name':                  'LightningStoreAdapter',
            'total_rollouts':        total,
            'rollouts_by_status':    by_status,
            'queue_size':            queue_size,
            'resource_snapshots':    resource_count,
            'transitions_written':   tw,
        }

    # ── Legacy accessors (used by existing server endpoints) ──────────────────

    def rollout_count(self) -> int:
        """Total number of rollout records (all statuses)."""
        with self._lock:
            return len(self._rollouts)

    def succeeded_rollouts(self, limit: int = -1) -> list[dict[str, Any]]:
        """Convenience: return succeeded rollouts (for training dataset assembly)."""
        return self.query_rollouts(status_in=[ROLLOUT_SUCCEEDED], limit=limit)

    @property
    def transitions_written(self) -> int:
        with self._lock:
            return self._transitions_written

    @transitions_written.setter
    def transitions_written(self, value: int) -> None:
        with self._lock:
            self._transitions_written = value

    # ── Async interface (mirrors agentlightning.LightningStore async API) ──────
    #
    # The official LightningStore defines all primary methods as ``async def``.
    # These wrappers delegate to the synchronous implementations via
    # ``asyncio.to_thread`` so callers that rely on ``await store.method()``
    # work without modification while the thread-safe core remains synchronous.
    #
    # Only the methods used by the official Runner/Algorithm/Trainer are
    # provided here.  Extend as needed when additional async paths are required.

    async def async_enqueue_rollout(self, **kwargs: Any) -> dict[str, Any]:
        """Async wrapper for :meth:`enqueue_rollout`."""
        import asyncio
        return await asyncio.to_thread(lambda: self.enqueue_rollout(**kwargs))

    async def async_dequeue_rollout(self, worker_id: str | None = None) -> dict[str, Any] | None:
        """Async wrapper for :meth:`dequeue_rollout`."""
        import asyncio
        return await asyncio.to_thread(lambda: self.dequeue_rollout(worker_id))

    async def async_accept_rollout_report(self, report_dict: dict[str, Any]) -> None:
        """Async wrapper for :meth:`accept_rollout_report`."""
        import asyncio
        await asyncio.to_thread(self.accept_rollout_report, report_dict)

    async def async_query_rollouts(
        self,
        status_in: list[str] | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Async wrapper for :meth:`query_rollouts`."""
        import asyncio
        return await asyncio.to_thread(
            lambda: self.query_rollouts(status_in=status_in, limit=limit, offset=offset)
        )

    async def async_add_span(self, span: dict[str, Any]) -> dict[str, Any]:
        """Async wrapper for :meth:`add_span`."""
        import asyncio
        return await asyncio.to_thread(self.add_span, span)

    async def async_query_spans(
        self,
        rollout_id: str,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        """Async wrapper for :meth:`query_spans`."""
        import asyncio
        return await asyncio.to_thread(lambda: self.query_spans(rollout_id, limit=limit))

    async def async_add_resources(self, resources: dict[str, Any]) -> dict[str, Any]:
        """Async wrapper for :meth:`add_resources`."""
        import asyncio
        return await asyncio.to_thread(self.add_resources, resources)

    async def async_get_latest_resources(self) -> dict[str, Any] | None:
        """Async wrapper for :meth:`get_latest_resources`."""
        import asyncio
        return await asyncio.to_thread(self.get_latest_resources)

    async def async_statistics(self) -> dict[str, Any]:
        """Async wrapper for :meth:`statistics`."""
        import asyncio
        return await asyncio.to_thread(self.statistics)

    # ── capabilities property (mirrors agentlightning.LightningStore) ─────────

    @property
    def capabilities(self) -> dict[str, bool]:
        """
        Declare store capabilities to callers (mirrors official interface).

        The official ``LightningStore.capabilities`` property signals which
        optional features the store supports so Runners/Algorithms can
        adapt their behaviour.
        """
        return {
            'async_interface':    True,
            'span_storage':       True,
            'resource_versioning': True,
            'task_queue':         True,
            'rollout_reports':    True,
            'wait_for_rollouts':  True,
        }
