"""
Service Manager — safe bridge between the Streamlit dashboard and shell scripts.

SECURITY: Only commands registered in ALLOWED_COMMANDS may be launched.
No arbitrary user-supplied shell input is ever executed.

Usage::

    from src.ui.service_manager import launch, job_status, all_jobs, kill_job

    job_id = launch('lightning_server')          # start in background
    status = job_status(job_id)                  # {'status': 'running', ...}
    logs   = read_job_log(job_id, lines=50)      # last N lines of output

Environment variables can be injected via ``env_overrides`` but only keys
listed in ENV_OVERRIDE_ALLOWLIST are accepted to avoid injection attacks.
"""
from __future__ import annotations

import os
import signal
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]

# ── Allowlisted commands ───────────────────────────────────────────────────────
# Maps a logical service key → the EXACT shell command that will be run.
# Add new entries here; never build commands from user input.

ALLOWED_COMMANDS: dict[str, str] = {
    # ── CPU services ──────────────────────────────────────────────────────────
    'start_postgres':          'bash scripts/cpu/start_postgres.sh',
    'start_mcp_server':        'bash scripts/cpu/start_mcp_server.sh',
    'start_mlflow':            'bash scripts/cpu/start_mlflow.sh',
    'validate_architecture':   'bash scripts/cpu/validate_architecture.sh',
    # ── CPU data / workflow steps ─────────────────────────────────────────────
    'generate_synthetic':      'bash scripts/cpu/run_01_generate_synthetic.sh',
    'load_postgres':           'bash scripts/cpu/run_02_load_synthetic_to_postgres.sh',
    'run_baseline_workflow':   'bash scripts/cpu/run_03_run_baseline_workflow.sh',
    'score_trajectories':      'bash scripts/cpu/run_04_score_trajectories.sh',
    'prepare_grpo_dataset':    'bash scripts/cpu/run_05_prepare_grpo_dataset.sh',
    'compare_policies':        'bash scripts/cpu/run_06_compare_baseline_vs_tuned.sh',
    'benchmark_policy':        'bash scripts/cpu/run_08_benchmark_policy_endpoints.sh',
    # ── GPU services ──────────────────────────────────────────────────────────
    'start_lightning_server':  'bash scripts/gpu/start_lightning_server.sh',
    'start_vllm_baseline':     'bash scripts/gpu/start_baseline_inference.sh',
    'start_vllm_tuned':        'bash scripts/gpu/start_tuned_inference.sh',
    'train_policy':            'bash scripts/gpu/run_02_train_policy_qlora_grpo.sh',
}

# Only these env-var keys may be injected by the UI to avoid shell injection.
ENV_OVERRIDE_ALLOWLIST: frozenset[str] = frozenset({
    'TRAINER',
    'MODEL_NAME',
    'LORA_R',
    'LORA_ALPHA',
    'TRAIN_EPOCHS',
    'MIN_ROLLOUTS',
    'ALLOW_VERL_FALLBACK',
    'POLICY_OUTPUT_DIR',
    'GRPO_DATASET_PATH',
    'LIGHTNING_SERVER_URL',
    'VLLM_BASE_URL',
    'MLFLOW_TRACKING_URI',
    'WORKSPACE_DIR',
})


# ── In-process job registry ───────────────────────────────────────────────────

_running_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock    = threading.Lock()


def launch(
    service_key:   str,
    env_overrides: dict[str, str] | None = None,
) -> str:
    """
    Launch an allowlisted command in the background.

    Parameters
    ----------
    service_key:
        Must be a key from ``ALLOWED_COMMANDS``.
    env_overrides:
        Additional env vars to set.  Only keys in ``ENV_OVERRIDE_ALLOWLIST``
        are accepted; others are silently dropped.

    Returns
    -------
    job_id: str
        Use with ``job_status()`` / ``read_job_log()``.
    """
    if service_key not in ALLOWED_COMMANDS:
        raise ValueError(
            f'Unknown service key: {service_key!r}.  '
            f'Allowed: {sorted(ALLOWED_COMMANDS)}'
        )

    cmd = ALLOWED_COMMANDS[service_key]

    # Build environment — filter to allowlist only
    env = os.environ.copy()
    if env_overrides:
        for k, v in env_overrides.items():
            if k in ENV_OVERRIDE_ALLOWLIST:
                env[k] = str(v)

    job_id   = f'{service_key}_{datetime.now(timezone.utc).strftime("%H%M%S")}'
    log_path = _ROOT / 'logs' / f'{job_id}.log'
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open('w', encoding='utf-8') as log_fh:
        proc = subprocess.Popen(
            cmd,
            shell=True,
            cwd=str(_ROOT),
            env=env,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            text=True,
        )

    with _jobs_lock:
        _running_jobs[job_id] = {
            'service':     service_key,
            'cmd':         cmd,
            'pid':         proc.pid,
            '_proc':       proc,           # private — not serialised
            'log_path':    str(log_path),
            'started_at':  datetime.now(timezone.utc).isoformat(),
            'status':      'running',
            'return_code': None,
        }

    # Reap exit code in background thread
    def _reap() -> None:
        proc.wait()
        with _jobs_lock:
            job = _running_jobs.get(job_id)
            if job:
                job['return_code'] = proc.returncode
                job['status']      = 'succeeded' if proc.returncode == 0 else 'failed'

    threading.Thread(target=_reap, daemon=True).start()
    return job_id


# ── Job introspection ──────────────────────────────────────────────────────────

def _safe_copy(job: dict[str, Any]) -> dict[str, Any]:
    """Return a serialisable copy (no subprocess.Popen)."""
    return {k: v for k, v in job.items() if k != '_proc'}


def job_status(job_id: str) -> dict[str, Any] | None:
    with _jobs_lock:
        job = _running_jobs.get(job_id)
        if not job:
            return None
        if job['status'] == 'running':
            rc = job['_proc'].poll()
            if rc is not None:
                job['return_code'] = rc
                job['status']      = 'succeeded' if rc == 0 else 'failed'
        return _safe_copy(job)


def all_jobs() -> list[dict[str, Any]]:
    with _jobs_lock:
        # Refresh status for running jobs
        for job in _running_jobs.values():
            if job['status'] == 'running':
                rc = job['_proc'].poll()
                if rc is not None:
                    job['return_code'] = rc
                    job['status']      = 'succeeded' if rc == 0 else 'failed'
        return [_safe_copy(j) for j in _running_jobs.values()]


def kill_job(job_id: str) -> bool:
    with _jobs_lock:
        job = _running_jobs.get(job_id)
        if not job:
            return False
        proc: subprocess.Popen = job['_proc']
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:
            try:
                proc.terminate()
            except Exception:
                pass
        job['status'] = 'stopped'
        return True


# ── Log access ────────────────────────────────────────────────────────────────

def read_job_log(job_id: str, lines: int = 60) -> str:
    with _jobs_lock:
        job = _running_jobs.get(job_id)
        if not job:
            return ''
        path = Path(job['log_path'])
    return _tail(path, lines)


def read_log_file(path: str | Path, lines: int = 100) -> str:
    return _tail(Path(path), lines)


def _tail(path: Path, lines: int) -> str:
    if not path.exists():
        return ''
    all_lines = path.read_text(encoding='utf-8', errors='replace').splitlines()
    return '\n'.join(all_lines[-lines:])


# ── Convenience: list all log files in logs/ ──────────────────────────────────

def list_log_files() -> list[Path]:
    log_dir = _ROOT / 'logs'
    if not log_dir.exists():
        return []
    return sorted(log_dir.glob('*.log'), key=lambda p: p.stat().st_mtime, reverse=True)
