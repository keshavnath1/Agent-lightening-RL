"""
LightningTrainer — ties LightningStoreAdapter + GRPOAlgorithm together.

Mirrors ``agentlightning.Trainer``: manages the continuous execution loop,
routes tasks to runners (via the store's task queue), feeds completed
rollouts to the algorithm, and checks the training threshold.

Architecture
------------

  ┌──────────────┐     enqueue_rollout     ┌──────────────────────┐
  │   Algorithm  │ ─────────────────────►  │  LightningStoreAdapter │
  │  (GRPO/veRL) │                         │                        │
  └──────┬───────┘                         │  task_queue            │
         │                                 │  rollout_records       │
         │  succeeded_rollouts()           │  span_store            │
         ◄─────────────────────────────────│  resource_snapshots    │
         │                                 └──────────────────────┘
         │  run(train_dataset)                         ▲
         ▼                                             │
  training subprocess ─► checkpoint ─► add_resources() ─► vLLM reload

Usage
-----

Standalone (e.g. test harness, notebook)::

    from src.training.lightning_store   import LightningStoreAdapter
    from src.training.grpo_algorithm    import GRPOAlgorithm
    from src.training.trainer_loop      import LightningTrainer

    store   = LightningStoreAdapter()
    algo    = GRPOAlgorithm(store)
    trainer = LightningTrainer(store, algo, min_rollouts=4)

    # Pre-load some tasks
    trainer.load_tasks('data/synthetic/tasks.jsonl')

    # After agents complete rollouts and call store.accept_rollout_report()
    # from the FastAPI server, trigger training manually:
    trainer.maybe_trigger_training()

    # Or run the continuous loop (blocks):
    trainer.run(max_rounds=10)

FastAPI integration
-------------------
The Lightning Server instantiates a global ``LightningTrainer`` on startup
and delegates the ``/api/training/trigger`` endpoint to
``trainer.maybe_trigger_training(force=True)``.

Official API compatibility
--------------------------
When ``agentlightning >= 0.3.1`` is installed this class can extend
``agentlightning.Trainer``.  The import is guarded so the class works
as a plain Python object when the package is absent.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

from agentlightning import Trainer as _TRAINER_BASE  # type: ignore[import]
from src.training.lightning_store  import LightningStoreAdapter
from src.training.grpo_algorithm   import GRPOAlgorithm

logger = logging.getLogger(__name__)


class LightningTrainer(_TRAINER_BASE):
    """
    Ties a ``LightningStoreAdapter`` and a ``GRPOAlgorithm`` together into
    the Agent Lightning training loop.

    Responsibilities
    ~~~~~~~~~~~~~~~~
    * Load tasks from JSONL into the store's task queue.
    * Expose ``maybe_trigger_training()`` which the FastAPI server calls from
      ``POST /api/training/trigger`` — checks the threshold and delegates to
      ``GRPOAlgorithm.run()``.
    * Expose ``run(max_rounds)`` for a self-contained blocking loop that
      keeps triggering training every time ``min_rollouts`` new succeeded
      rollouts have accumulated since the last round.

    Parameters
    ----------
    store:
        Shared ``LightningStoreAdapter`` instance.
    algorithm:
        ``GRPOAlgorithm`` (or any compatible Algorithm) instance.
    min_rollouts:
        Minimum succeeded rollouts before a training round is triggered.
    poll_interval:
        Seconds to sleep between threshold checks in ``run()``.
    """

    def __init__(
        self,
        store: LightningStoreAdapter,
        algorithm: GRPOAlgorithm,
        min_rollouts: int | None = None,
        poll_interval: float = 5.0,
    ) -> None:
        self.store          = store
        self.algorithm      = algorithm
        self.min_rollouts   = min_rollouts or int(os.getenv('LIGHTNING_MIN_ROLLOUTS', '4'))
        self.poll_interval  = poll_interval
        self._rounds_fired: int = 0
        self._rollouts_at_last_round: int = 0

    # ── Task loading ───────────────────────────────────────────────────────────

    def load_tasks(self, tasks_jsonl_path: str | Path) -> int:
        """
        Bulk-enqueue tasks from a JSONL file into the store's task queue.

        Returns the number of tasks loaded.  Mirrors the pattern used by
        the Lightning Server's ``POST /api/tasks/load`` endpoint.
        """
        count = self.store.load_tasks_from_jsonl(str(tasks_jsonl_path))
        logger.info('LightningTrainer: loaded %d tasks from %s', count, tasks_jsonl_path)
        return count

    # ── Training trigger ───────────────────────────────────────────────────────

    def maybe_trigger_training(
        self,
        force: bool = False,
        trainer: str | None = None,
        model_name: str | None = None,
        min_rollouts: int | None = None,
        reward_mode: str | None = None,
        num_generations: int | None = None,
        checkpoint_path: str | None = None,
        reload_after: bool = True,
    ) -> dict[str, Any]:
        """
        Trigger a GRPO training round if the threshold is met.

        Parameters
        ----------
        force:
            Bypass the minimum-rollouts check (used by manual trigger).
        trainer:
            Override the algorithm's trainer backend for this round only.
        model_name:
            Override the algorithm's model_name for this round only.
        min_rollouts:
            Override the trainer's default threshold for this check only.
        reward_mode:
            Override the algorithm's reward_mode for this round only.
        num_generations:
            Override the algorithm's num_generations for this round only.
        checkpoint_path:
            Override the checkpoint directory for this round only.
        reload_after:
            If False, skip vLLM hot-reload after training.

        Returns a status dict compatible with the ``/api/training/trigger``
        response schema.
        """
        threshold = min_rollouts if min_rollouts is not None else self.min_rollouts

        if self.algorithm.is_running():
            return {
                'status':             'already_running',
                'rollouts_collected': self.store.rollout_count(),
            }

        succeeded = self.store.succeeded_rollouts()
        n_succeeded = len(succeeded)

        if not force and n_succeeded < threshold:
            return {
                'status':             'insufficient_rollouts',
                'rollouts_collected': n_succeeded,
                'required':           threshold,
            }

        # Override algorithm settings for this round if requested
        orig_trainer         = self.algorithm.trainer
        orig_model_name      = self.algorithm.model_name
        orig_reward_mode     = self.algorithm.reward_mode
        orig_num_generations = self.algorithm.num_generations
        orig_reload_after    = self.algorithm.reload_after
        if trainer:
            self.algorithm.trainer         = trainer
        if model_name:
            self.algorithm.model_name      = model_name
        if reward_mode is not None:
            self.algorithm.reward_mode     = reward_mode
        if num_generations is not None:
            self.algorithm.num_generations = num_generations
        self.algorithm.reload_after = reload_after

        ckpt_override = Path(checkpoint_path) if checkpoint_path else None

        try:
            result = self.algorithm.run(train_dataset=succeeded, checkpoint_dir=ckpt_override)
        finally:
            self.algorithm.trainer         = orig_trainer
            self.algorithm.model_name      = orig_model_name
            self.algorithm.reward_mode     = orig_reward_mode
            self.algorithm.num_generations = orig_num_generations
            self.algorithm.reload_after    = orig_reload_after

        self._rounds_fired += 1
        self._rollouts_at_last_round = n_succeeded
        logger.info(
            'LightningTrainer: training round %d started — %d rollouts, %d tasks',
            self._rounds_fired,
            result.get('rollouts_used', n_succeeded),
            result.get('grouped_tasks', 0),
        )
        return result

    # ── Continuous loop ────────────────────────────────────────────────────────

    def run(self, max_rounds: int = 0) -> None:
        """
        Blocking training loop.  Polls the store and triggers training
        every time ``min_rollouts`` new succeeded rollouts have accumulated
        since the previous round.

        Parameters
        ----------
        max_rounds:
            Stop after this many training rounds.  ``0`` means run forever
            (until KeyboardInterrupt).
        """
        logger.info(
            'LightningTrainer.run() started — min_rollouts=%d, poll_interval=%.1fs',
            self.min_rollouts,
            self.poll_interval,
        )
        rounds = 0
        try:
            while True:
                n = len(self.store.succeeded_rollouts())
                new_since_last = n - self._rollouts_at_last_round

                if new_since_last >= self.min_rollouts and not self.algorithm.is_running():
                    self.maybe_trigger_training()
                    rounds += 1
                    if max_rounds and rounds >= max_rounds:
                        logger.info('LightningTrainer: reached max_rounds=%d, stopping.', max_rounds)
                        break

                time.sleep(self.poll_interval)
        except KeyboardInterrupt:
            logger.info('LightningTrainer: interrupted.')

    # ── agentlightning.Trainer compatibility ──────────────────────────────────

    def fit(
        self,
        algorithm: Any = None,
        train_dataset: Any = None,
        val_dataset: Any = None,
    ) -> None:
        """
        Mirrors ``agentlightning.Trainer.fit()``.

        When called without a dataset, uses the store's succeeded rollouts.
        Useful when the official Trainer is substituted in tests.
        """
        alg = algorithm or self.algorithm
        if train_dataset is None:
            train_dataset = self.store.succeeded_rollouts()
        alg.run(train_dataset=train_dataset, val_dataset=val_dataset)

    # ── Observability ──────────────────────────────────────────────────────────

    def status(self) -> dict[str, Any]:
        """Return combined trainer + algorithm + store status."""
        algo_status  = self.algorithm.status()
        store_stats  = self.store.statistics()
        return {
            'trainer_rounds_fired':      self._rounds_fired,
            'rollouts_at_last_round':    self._rollouts_at_last_round,
            'min_rollouts':              self.min_rollouts,
            **algo_status,
            'store': store_stats,
        }
