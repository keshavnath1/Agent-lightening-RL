"""
GRPOAlgorithm — Agent Lightning Algorithm implementation for TRL GRPO.

Implements the ``Algorithm.run()`` interface from ``agentlightning.Algorithm``
so the training loop is a first-class citizen in the Agent Lightning
architecture, independent of the FastAPI server layer.

Architecture position
---------------------

  LightningStoreAdapter   ←──  CPU agent reports rollouts via sidecar
           │
           ▼
   GRPOAlgorithm.run()    ←──  called by LightningTrainer when
           │                    MIN_ROLLOUTS threshold is met
           ▼
  write grouped_rollouts.jsonl
           │
           ▼
  train_policy_qlora_grpo   (subprocess: TRL GRPO / veRL / ART-RULER)
           │
           ▼
  checkpoint saved → store.add_resources({'checkpoint_path': ...})
           │
           ▼
  vLLM hot-reload  (POST /v1/load_lora_adapter)

Official API compatibility
--------------------------
When ``agentlightning >= 0.3.1`` is installed this class can be made to
extend ``agentlightning.Algorithm``.  The import is guarded so the class
works as a plain Python object when the package is absent (CPU-only pods).

Example::

    store = LightningStoreAdapter()
    algo  = GRPOAlgorithm(store, checkpoint_dir='checkpoints/qwen25-3b-agent-lora')
    algo.run()          # trains on all succeeded rollouts in the store
"""
from __future__ import annotations

import json
import os
import subprocess
import threading
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agentlightning import Algorithm as _BASE  # type: ignore[import]
from lightning_bridge.store_adapter import LightningStoreAdapter


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class GRPOAlgorithm(_BASE):
    """
    TRL GRPO Algorithm for the Agent Lightning Optimization Framework.

    Reads succeeded rollouts from a ``LightningStoreAdapter``, groups them
    by task_id ranked by reward, writes the canonical grouped-rollouts JSONL
    format, and launches the training subprocess.  After a successful training
    run it calls ``store.add_resources()`` to register the new checkpoint and
    hot-reloads the vLLM adapter so the next rollout cycle uses updated weights.

    Parameters
    ----------
    store:
        The shared ``LightningStoreAdapter`` instance that holds rollout data.
    checkpoint_dir:
        Directory where the LoRA adapter checkpoint is written by training.
    model_name:
        Base model identifier forwarded to the training script.
    trainer:
        Training backend: ``trl_grpo``, ``agent_lightning_official``, ``verl``, or ``official_art_ruler``.
    transitions_dir:
        Directory where training data files are written.
    vllm_base_url:
        vLLM OpenAI-compatible server URL for hot-reload.
    policy_version:
        Label written into the resource snapshot after training.
    on_training_complete:
        Optional callback ``(checkpoint_path: str) -> None`` invoked in the
        watcher thread after successful training.
    """

    def __init__(
        self,
        store: LightningStoreAdapter,
        checkpoint_dir: str | Path = 'checkpoints/qwen25-3b-agent-lora',
        model_name: str = 'Qwen/Qwen2.5-3B-Instruct',
        trainer: str = 'trl_grpo',
        transitions_dir: str | Path = 'data/grpo',
        vllm_base_url: str = 'http://localhost:8000',
        policy_version: str = 'v2_finetuned',
        on_training_complete: Any = None,
    ) -> None:
        self.store               = store
        self.checkpoint_dir      = Path(checkpoint_dir)
        self.model_name          = model_name
        self.trainer             = trainer
        self.transitions_dir     = Path(transitions_dir)
        self.vllm_base_url       = vllm_base_url
        self.policy_version      = policy_version
        self._on_complete_cb     = on_training_complete

        self._lock               = threading.Lock()
        self._proc: subprocess.Popen | None    = None
        self._started_at: str | None           = None
        self._last_reload_at: str | None       = None
        self._last_reload_status: str          = 'never'
        self._last_return_code: int | None     = None

    # ── Public Algorithm API ───────────────────────────────────────────────────

    def run(
        self,
        train_dataset: Any = None,
        val_dataset: Any = None,
    ) -> dict[str, Any]:
        """
        Execute one training round.

        Mirrors ``agentlightning.Algorithm.run()``.  If ``train_dataset``
        is provided it is used directly (list of rollout dicts); otherwise
        the succeeded rollouts from the store are used.

        Returns a status dict with ``status``, ``rollouts_used``,
        ``grouped_tasks``, ``grouped_file``, and ``pid``.
        """
        rollouts: list[dict[str, Any]] = train_dataset or self.store.succeeded_rollouts()
        if not rollouts:
            return {'status': 'no_rollouts'}

        grouped_file = self._write_grouped_rollouts(rollouts)
        pid          = self._launch_training(grouped_file)

        return {
            'status':        'training_started',
            'rollouts_used': len(rollouts),
            'grouped_tasks': self._count_tasks(rollouts),
            'grouped_file':  str(grouped_file),
            'pid':           pid,
            'started_at':    self._started_at,
        }

    def is_running(self) -> bool:
        """True while the training subprocess is alive."""
        with self._lock:
            return self._proc is not None and self._proc.poll() is None

    def status(self) -> dict[str, Any]:
        """Return training status compatible with ``/api/training/status``."""
        with self._lock:
            active   = self._proc is not None and self._proc.poll() is None
            rc       = self._proc.poll() if self._proc else None
            started  = self._started_at
        return {
            'training_active':     active,
            'training_started_at': started,
            'return_code':         rc,
            'last_reload_at':      self._last_reload_at,
            'last_reload_status':  self._last_reload_status,
        }

    # ── Grouped-rollouts JSONL writer ──────────────────────────────────────────

    def _write_grouped_rollouts(
        self,
        rollouts: list[dict[str, Any]],
    ) -> Path:
        """
        Assemble ``lightning_grouped_rollouts.jsonl`` in the format expected
        by ``train_policy_qlora_grpo.build_training_examples()``:

        .. code-block:: json

            {
              "task_id": "...",
              "group_size": 4,
              "ranked_trajectories": [
                {"trajectory_id": "...", "reward": 0.9, "policy_version": "v1",
                 "steps": [...]}
              ]
            }

        Trajectories within each task group are sorted by reward descending
        so the training script can apply the GRPO advantage calculation
        directly without a second sort pass.
        """
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in rollouts:
            traj = r.get('trajectory_dict') or {}
            if traj.get('steps'):
                task_id = r.get('task_id') or traj.get('task_id') or 'unknown'
                grouped[task_id].append(traj)

        self.transitions_dir.mkdir(parents=True, exist_ok=True)
        out_file = self.transitions_dir / 'lightning_grouped_rollouts.jsonl'

        with out_file.open('w', encoding='utf-8') as fh:
            for task_id, trajs in grouped.items():
                sorted_trajs = sorted(
                    trajs,
                    key=lambda t: float(t.get('reward') or 0.0),
                    reverse=True,
                )
                fh.write(json.dumps({
                    'task_id':    task_id,
                    'group_size': len(sorted_trajs),
                    'ranked_trajectories': [
                        {
                            'trajectory_id': t.get('trajectory_id'),
                            'reward':        float(t.get('reward') or 0.0),
                            'policy_version': t.get('policy_version'),
                            'steps':         t.get('steps', []),
                        }
                        for t in sorted_trajs
                    ],
                }, default=str) + '\n')

        return out_file

    @staticmethod
    def _count_tasks(rollouts: list[dict[str, Any]]) -> int:
        return len({r.get('task_id') or r.get('trajectory_dict', {}).get('task_id') for r in rollouts})

    # ── Training subprocess ────────────────────────────────────────────────────

    def _launch_training(self, grouped_file: Path) -> int:
        """Launch the TRL GRPO training subprocess and start the watcher thread."""
        cmd = [
            'python', '-m', 'src.training.train_policy_qlora_grpo',
            '--dataset',    str(grouped_file),
            '--output-dir', str(self.checkpoint_dir),
            '--model-name', self.model_name,
            '--trainer',    self.trainer,
        ]

        with self._lock:
            self._proc       = subprocess.Popen(cmd)  # noqa: S603
            self._started_at = _now()
            proc_ref         = self._proc

        threading.Thread(
            target=self._watcher,
            args=(proc_ref,),
            daemon=True,
            name='grpo-training-watcher',
        ).start()

        return proc_ref.pid

    def _watcher(self, proc: subprocess.Popen) -> None:
        """
        Background daemon thread: blocks until training finishes, then:
        1. Calls ``store.add_resources()`` with the new checkpoint path.
        2. Hot-reloads the vLLM adapter.
        3. Fires the optional ``on_training_complete`` callback.
        """
        proc.wait()
        with self._lock:
            self._last_return_code = proc.returncode

        if proc.returncode == 0:
            # Register new checkpoint as latest resource snapshot
            checkpoint_path = str(self.checkpoint_dir.resolve())
            snapshot = self.store.add_resources({
                'checkpoint_path': checkpoint_path,
                'policy_version':  self.policy_version,
                'adapter_name':    'agent_adapter',
                'trained_at':      _now(),
            })

            # Hot-reload vLLM
            reload_status = self._reload_vllm(checkpoint_path)

            # Optional caller callback (e.g. update server globals)
            if self._on_complete_cb:
                try:
                    self._on_complete_cb(checkpoint_path)
                except Exception:
                    pass
        else:
            reload_status = f'skipped_training_failed_rc{proc.returncode}'

        with self._lock:
            self._last_reload_at     = _now()
            self._last_reload_status = reload_status

    def _reload_vllm(self, checkpoint_path: str) -> str:
        """POST the new adapter path to vLLM's LoRA hot-swap endpoint."""
        try:
            import requests as _req  # noqa: PLC0415
            resp = _req.post(
                f'{self.vllm_base_url}/v1/load_lora_adapter',
                json={'lora_name': 'agent_adapter', 'lora_path': checkpoint_path},
                timeout=30,
            )
            return 'reloaded' if resp.status_code in (200, 201) else f'http_{resp.status_code}'
        except Exception as exc:
            return f'error:{exc.__class__.__name__}'

    # ── agentlightning.Algorithm compatibility stubs ───────────────────────────

    def get_client(self) -> None:
        """
        Stub satisfying ``agentlightning.Algorithm.get_client()``.

        The GRPO algorithm communicates through the store directly; no
        separate client is needed.
        """
        return None

    # ── SpanTraceAdapter integration ──────────────────────────────────────────

    def extract_triplets_for_rollout(
        self,
        rollout_id: str,
        final_reward: float | None = None,
    ) -> list[Any]:
        """
        Pull spans for ``rollout_id`` from the store and convert them to
        ``Triplet`` objects via ``SpanTraceAdapter``.

        This enables token-level GRPO training when the LLMProxy is active
        and emits prompt/response token IDs into the span attributes.  When
        token IDs are absent (text-level spans from the sidecar), the
        resulting ``Triplet.prompt/response.token_ids`` are empty lists and
        the training script falls back to ``raw_content``.

        Parameters
        ----------
        rollout_id:
            Rollout identifier to retrieve spans for.
        final_reward:
            Reward applied to the last triplet when spans carry no per-step
            reward annotation.

        Returns
        -------
        List of ``Triplet`` objects (may be empty when no spans are stored).
        """
        from src.inference.trace_adapter import adapt_rollout_spans  # noqa: PLC0415
        return adapt_rollout_spans(
            self.store,
            rollout_id=rollout_id,
            final_reward=final_reward,
        )

    def build_triplet_dataset(
        self,
        rollouts: list[dict[str, Any]] | None = None,
    ) -> list[Any]:
        """
        Build the full ``Triplet`` dataset across all succeeded rollouts.

        Iterates every rollout in ``rollouts`` (defaulting to store's
        succeeded rollouts), extracts span-level triplets via
        ``SpanTraceAdapter``, and returns a flat list ordered by rollout.

        This dataset can be passed directly to veRL / TRL's GRPO collator
        for token-level advantage computation.

        Parameters
        ----------
        rollouts:
            Optional pre-fetched rollout dicts.  When ``None``, fetches all
            succeeded rollouts from the store.

        Returns
        -------
        Flat list of ``Triplet`` objects across all rollouts.
        """
        from src.inference.trace_adapter import SpanTraceAdapter  # noqa: PLC0415

        rollouts = rollouts or self.store.succeeded_rollouts()
        triplets: list[Any] = []

        for r in rollouts:
            rid    = r.get('rollout_id') or r.get('trajectory_dict', {}).get('trajectory_id')
            reward = float(r.get('final_reward') or r.get('reward') or 0.0)

            if rid:
                batch = self.extract_triplets_for_rollout(rid, final_reward=reward)
                triplets.extend(batch)

        return triplets
