"""
Lightning Server – GPU training-plane bridge.

FastAPI application that connects the CPU agent execution side to the GPU
Optimization Framework (veRL/GRPO), implementing the three stages described
in the Microsoft Agent Lightning architecture:

  Stage 1 – Task Pulling & Agent Execution
      Lightning Server maintains a task pool.  CPU agents call GET /api/tasks/pull
      to receive one task at a time, execute their native LangGraph workflow,
      then report results.

  Stage 2 – Non-Intrusive Trace Collection (Sidecar Design)
      CPU-side LightningClientSidecar intercepts all LLM calls and POSTs a
      structured RolloutReport to POST /api/rollouts/report.  The server
      converts reports into standard (state_t, action_t, reward_t, state_t+1)
      transition tuples and persists them to JSONL.

  Stage 3 – Trajectory Organisation & Training Loop (Optimization Framework)
      When rollout count ≥ MIN_ROLLOUTS, POST /api/training/trigger launches
      the GRPO/QLoRA training script.  The updated LoRA adapter is loaded by
      vLLM for the next rollout cycle, closing the feedback loop between
      agent behaviour and learning.

Run on the GPU pod:
    python -m src.training.lightning_server_app
or via:
    scripts/gpu/start_lightning_server.sh
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel

from src.training.lightning_store  import LightningStoreAdapter
from src.training.grpo_algorithm   import GRPOAlgorithm
from src.training.trainer_loop     import LightningTrainer

# ── Configuration ──────────────────────────────────────────────────────────────

PORT                = int(os.getenv('LIGHTNING_SERVER_PORT', '19123'))
TRANSITIONS_DIR      = Path(os.getenv('LIGHTNING_TRANSITIONS_DIR', 'data/grpo'))
TRANSITIONS_FILE     = TRANSITIONS_DIR / 'lightning_server_transitions.jsonl'
MIN_ROLLOUTS         = int(os.getenv('LIGHTNING_MIN_ROLLOUTS', '4'))
MAX_ROLLOUT_HISTORY  = int(os.getenv('LIGHTNING_MAX_ROLLOUT_HISTORY', '500'))
VLLM_BASE_URL        = os.getenv('VLLM_BASE_URL', 'http://localhost:8000')
CHECKPOINT_DIR       = Path(os.getenv('LIGHTNING_CHECKPOINT_DIR', 'checkpoints/qwen25-3b-agent-lora'))
DEFAULT_TRAINER      = os.getenv('LIGHTNING_DEFAULT_TRAINER', 'trl_grpo')
DEFAULT_MODEL        = os.getenv('DEFAULT_MODEL', 'Qwen/Qwen2.5-3B-Instruct')

# ── Optimization Framework singletons ─────────────────────────────────────────
#
# These three objects replace all the raw module-level state that used to
# live in this file (_task_pool, _rollouts, _training_process, etc.).
# The FastAPI handlers below are now thin wrappers that delegate to them.

_store   = LightningStoreAdapter(max_rollout_history=MAX_ROLLOUT_HISTORY)
_algo    = GRPOAlgorithm(
    store           = _store,
    checkpoint_dir  = CHECKPOINT_DIR,
    model_name      = DEFAULT_MODEL,
    trainer         = DEFAULT_TRAINER,
    transitions_dir = TRANSITIONS_DIR,
    vllm_base_url   = VLLM_BASE_URL,
)
_trainer = LightningTrainer(
    store        = _store,
    algorithm    = _algo,
    min_rollouts = MIN_ROLLOUTS,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Pydantic models ────────────────────────────────────────────────────────────

class RolloutReportRequest(BaseModel):
    """
    Mirrors LightningClientSidecar.RolloutReport on the CPU side.

    transitions contains the Agent Lightning-style tuples:
        state_t, action_t, reward_t, state_t+1
    produced by agent_lightning_export.trajectory_to_transitions().

    trajectory_dict carries the full scored Trajectory.to_dict() payload so
    the server can write grouped_rollouts.jsonl (task_id + ranked_trajectories
    with full step dicts) for GRPO training via build_training_examples().
    """
    task_id:          str
    trajectory_id:    str
    policy_version:   str
    llm_spans:        list[dict[str, Any]]
    transitions:      list[dict[str, Any]]
    final_reward:     float
    reward_metadata:  dict[str, Any]
    error_types:      list[str]
    trajectory_dict:  dict[str, Any] = {}
    completed_at:     str | None = None


class TasksLoadRequest(BaseModel):
    tasks_jsonl_path: str


class TrainingTriggerRequest(BaseModel):
    min_rollouts:    int | None = None
    trainer:         str | None = None    # trl_grpo | verl | agent_lightning_official
    model_name:      str | None = None    # overrides DEFAULT_MODEL env var
    reward_mode:     str | None = 'hybrid'       # json_validity | workflow_policy | trajectory_reward | hybrid
    num_generations: int | None = 4              # GRPO group size
    checkpoint_path: str | None = None           # overrides default checkpoint_dir for this round
    reload_after:    bool       = True           # hot-reload vLLM after training


# ── FastAPI app ────────────────────────────────────────────────────────────────

app = FastAPI(
    title='Lightning Server',
    description=(
        'GPU-side bridge between CPU agent execution and the veRL/GRPO '
        'Optimization Framework (Microsoft Agent Lightning architecture).'
    ),
    version='1.0.0',
)


# ── Health / observability ─────────────────────────────────────────────────────

@app.get('/health')
def health() -> dict[str, Any]:
    stats        = _store.statistics()
    algo_status  = _algo.status()
    return {
        'status':              'ok',
        'queue_size':          stats['queue_size'],
        'rollouts_collected':  stats['total_rollouts'],
        'transitions_written': stats['transitions_written'],
        'training_active':     algo_status['training_active'],
        'last_reload_at':      algo_status['last_reload_at'],
        'last_reload_status':  algo_status['last_reload_status'],
        'timestamp':           _now(),
    }


# ── vLLM adapter reload (closes the Agent Lightning feedback loop) ────────────

class ReloadRequest(BaseModel):
    adapter_path: str | None = None   # override checkpoint dir for one-shot reload


@app.post('/api/inference/reload', status_code=status.HTTP_200_OK)
def reload_inference(req: ReloadRequest = ReloadRequest()) -> dict[str, Any]:
    """
    Manually trigger vLLM to reload the LoRA adapter.

    Delegates to GRPOAlgorithm._reload_vllm() and mirrors the
    last_reload_at / last_reload_status fields exposed by /health.
    """
    adapter_path = req.adapter_path or str(CHECKPOINT_DIR.resolve())
    reload_status = _algo._reload_vllm(adapter_path)  # noqa: SLF001
    import threading as _t
    with _algo._lock:  # noqa: SLF001
        _algo._last_reload_at     = _now()
        _algo._last_reload_status = reload_status
    return {
        'reload_status':  reload_status,
        'adapter_path':   adapter_path,
        'vllm_base_url':  VLLM_BASE_URL,
        'reloaded_at':    _algo._last_reload_at,  # noqa: SLF001
    }




@app.post('/api/tasks/load', status_code=status.HTTP_200_OK)
def load_tasks(req: TasksLoadRequest) -> dict[str, Any]:
    """Load tasks from a JSONL file into the store's task queue."""
    path = Path(req.tasks_jsonl_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f'Tasks file not found: {path}')
    loaded = _trainer.load_tasks(path)
    stats  = _store.statistics()
    return {'loaded': loaded, 'queue_size': stats['queue_size']}


@app.get('/api/tasks/pull')
def pull_task() -> dict[str, Any]:
    """
    Pop the next task from the store's task queue.

    Returns {} when the queue is empty so CPU agents fall back to local
    JSONL task loading without disrupting their workflow.
    """
    return _store.pull_task()


# ── Stage 2: Rollout reporting & transition storage ───────────────────────────

@app.post('/api/rollouts/report', status_code=status.HTTP_201_CREATED)
def report_rollout(req: RolloutReportRequest) -> dict[str, Any]:
    """
    Accept a completed RolloutReport from the CPU-side sidecar.

    Persists each (state_t, action_t, reward_t, state_t+1) transition tuple
    to the transitions JSONL file and ingests the rollout into the
    LightningStoreAdapter so the GRPOAlgorithm can consume it at training time.
    """
    new_transitions = len(req.transitions)

    # ── Persist transitions to JSONL (durable, file-based) ────────────────
    if new_transitions:
        TRANSITIONS_DIR.mkdir(parents=True, exist_ok=True)
        with TRANSITIONS_FILE.open('a', encoding='utf-8') as fh:
            for t in req.transitions:
                fh.write(json.dumps(t, default=str) + '\n')

    # ── Ingest into LightningStoreAdapter ─────────────────────────────────
    _store.accept_rollout_report(req.model_dump())

    stats = _store.statistics()
    return {
        'received':           True,
        'transitions_stored': new_transitions,
        'rollouts_collected': stats['total_rollouts'],
    }


@app.get('/api/rollouts')
def list_rollouts(limit: int = 50) -> dict[str, Any]:
    rollouts = _store.query_rollouts(limit=limit)
    return {'rollouts': rollouts, 'total': _store.rollout_count()}


@app.get('/api/transitions')
def list_transitions(limit: int = 100) -> dict[str, Any]:
    """Return the last `limit` transition records from the JSONL file."""
    if not TRANSITIONS_FILE.exists():
        return {'transitions': [], 'total': 0}
    lines = TRANSITIONS_FILE.read_text(encoding='utf-8').splitlines()
    parsed = []
    for line in lines[-limit:]:
        if line.strip():
            try:
                parsed.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return {'transitions': parsed, 'total': _store.statistics()['transitions_written']}


# ── Stage 3: Optimization Framework (GRPO/veRL training trigger) ──────────────

@app.get('/api/training/status')
def training_status() -> dict[str, Any]:
    """
    Reflects the state of the Optimization Framework (LightningTrainer) loop:
      - ready_for_training: enough rollouts collected and no training running
      - training_active: GRPOAlgorithm subprocess is live
      - return_code: exit code of last training run (None if still running)
    """
    t_status = _trainer.status()
    stats    = _store.statistics()
    rollouts = stats['total_rollouts']
    active   = t_status['training_active']
    return {
        'training_active':     active,
        'training_started_at': t_status.get('training_started_at'),
        'return_code':         t_status.get('return_code'),
        'rollouts_collected':  rollouts,
        'transitions_written': stats['transitions_written'],
        'ready_for_training':  rollouts >= MIN_ROLLOUTS and not active,
        'store':               stats['rollouts_by_status'],
    }


@app.get('/api/store/statistics')
def store_statistics() -> dict[str, Any]:
    """
    Full LightningStoreAdapter statistics.

    Exposes the agent-lightning-compatible store interface diagnostics:
    rollout counts by status, resource snapshots, queue depth, and
    transitions written.  Useful for debugging the Agent Lightning pipeline.
    """
    return _store.statistics()


@app.get('/api/store/resources')
def list_resources(limit: int = 20) -> dict[str, Any]:
    """
    List resource snapshots (checkpoint paths) registered by the algorithm
    after each successful training round.

    Mirrors agentlightning.LightningStore.query_resources().
    """
    resources = _store.query_resources(limit=limit)
    latest    = _store.get_latest_resources()
    return {
        'resources':         resources,
        'latest':            latest,
        'total_snapshots':   len(_store.query_resources()),
    }


@app.post('/api/training/trigger', status_code=status.HTTP_202_ACCEPTED)
def trigger_training(req: TrainingTriggerRequest = TrainingTriggerRequest()) -> dict[str, Any]:
    """
    Delegate to LightningTrainer.maybe_trigger_training().

    This is the Optimization Framework entry point in the Agent Lightning loop:
      collected rollouts → GRPOAlgorithm.run() → write grouped_rollouts.jsonl
      → training subprocess → store.add_resources() → vLLM hot-reload
      → next rollout cycle uses updated weights.
    """
    return _trainer.maybe_trigger_training(
        force            = True,
        trainer          = req.trainer,
        model_name       = req.model_name,
        min_rollouts     = req.min_rollouts,
        reward_mode      = req.reward_mode,
        num_generations  = req.num_generations,
        checkpoint_path  = req.checkpoint_path,
        reload_after     = req.reload_after,
    )


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    uvicorn.run(
        'src.training.lightning_server_app:app',
        host='0.0.0.0',
        port=PORT,
        log_level='info',
    )
