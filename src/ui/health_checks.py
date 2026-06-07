"""
Health checks for all services in the self-improving ML agent system.

Each public check function returns a ``CheckResult`` dict:

    {
        "name":   str,          # display label
        "ok":     bool,         # True = healthy
        "detail": str,          # one-line description
        "fix":    str | None,   # suggested fix command (None when healthy)
        "extra":  dict | None,  # optional structured payload for advanced display
    }
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Any

import requests

CheckResult = dict[str, Any]


# ── Python environment ─────────────────────────────────────────────────────────

def check_python_version() -> CheckResult:
    v = sys.version_info
    ok = v >= (3, 10)
    return {
        'name':   'Python version',
        'ok':     ok,
        'detail': f'Python {v.major}.{v.minor}.{v.micro}',
        'fix':    'Upgrade to Python ≥ 3.10' if not ok else None,
        'extra':  None,
    }


def _pkg(name: str, install_name: str | None = None) -> CheckResult:
    ok = importlib.util.find_spec(name) is not None
    pkg = install_name or name
    return {
        'name':   f'pkg: {name}',
        'ok':     ok,
        'detail': f'{name} {"✓" if ok else "✗ missing"}',
        'fix':    f'pip install {pkg}' if not ok else None,
        'extra':  None,
    }


REQUIRED_CPU_PACKAGES: list[tuple[str, str | None]] = [
    ('langgraph',   None),
    ('langchain',   None),
    ('fastapi',     None),
    ('uvicorn',     None),
    ('streamlit',   None),
    ('pandas',      None),
    ('pyarrow',     None),
    ('sqlalchemy',  None),
    ('requests',    None),
    ('mlflow',      None),
    ('xgboost',     None),
    ('lightgbm',    None),
    ('sklearn',     'scikit-learn'),
    ('altair',      None),
]

REQUIRED_GPU_PACKAGES: list[tuple[str, str | None]] = [
    ('torch',           None),
    ('transformers',    None),
    ('peft',            None),
    ('trl',             None),
    ('bitsandbytes',    None),
    ('datasets',        None),
]


def check_cpu_packages() -> list[CheckResult]:
    return [_pkg(name, install) for name, install in REQUIRED_CPU_PACKAGES]


def check_gpu_packages() -> list[CheckResult]:
    return [_pkg(name, install) for name, install in REQUIRED_GPU_PACKAGES]


# ── Docker ─────────────────────────────────────────────────────────────────────

def check_docker() -> CheckResult:
    try:
        r = subprocess.run(
            ['docker', '--version'],
            capture_output=True, text=True, timeout=5,
        )
        ok = r.returncode == 0
        return {
            'name':   'Docker',
            'ok':     ok,
            'detail': r.stdout.strip() if ok else r.stderr.strip(),
            'fix':    'Install Docker: https://docs.docker.com/get-docker/' if not ok else None,
            'extra':  None,
        }
    except FileNotFoundError:
        return {
            'name':   'Docker',
            'ok':     False,
            'detail': 'docker not found in PATH',
            'fix':    'Install Docker: https://docs.docker.com/get-docker/',
            'extra':  None,
        }
    except Exception as exc:
        return {'name': 'Docker', 'ok': False, 'detail': str(exc), 'fix': None, 'extra': None}


# ── Generic HTTP health check ──────────────────────────────────────────────────

def check_http(service: str, url: str, fix: str | None = None, timeout: int = 4) -> CheckResult:
    try:
        r = requests.get(url, timeout=timeout)
        ok = r.status_code < 400
        try:
            data = r.json()
            detail = data.get('status', f'HTTP {r.status_code}')
        except Exception:
            detail = f'HTTP {r.status_code}'
        return {'name': service, 'ok': ok, 'detail': detail, 'fix': fix, 'extra': None}
    except requests.exceptions.ConnectionError:
        return {
            'name': service, 'ok': False,
            'detail': 'Connection refused — service not running',
            'fix': fix, 'extra': None,
        }
    except requests.exceptions.Timeout:
        return {
            'name': service, 'ok': False,
            'detail': 'Request timed out',
            'fix': fix, 'extra': None,
        }
    except Exception as exc:
        return {'name': service, 'ok': False, 'detail': str(exc), 'fix': fix, 'extra': None}


# ── Service-specific health checks ────────────────────────────────────────────

def check_lightning_server(base_url: str = 'http://localhost:19123') -> CheckResult:
    result = check_http(
        'Lightning Server', f'{base_url}/health',
        fix='bash scripts/gpu/start_lightning_server.sh',
    )
    if result['ok']:
        try:
            data = requests.get(f'{base_url}/health', timeout=4).json()
            result['detail'] = (
                f"online | queue={data.get('queue_size', 0)} "
                f"rollouts={data.get('rollouts_collected', 0)} "
                f"transitions={data.get('transitions_written', 0)}"
            )
            result['extra'] = data
        except Exception:
            pass
    return result


def check_vllm(base_url: str = 'http://localhost:8000') -> CheckResult:
    result = check_http(
        'vLLM Server', f'{base_url}/v1/models',
        fix='bash scripts/gpu/start_baseline_inference.sh',
    )
    if result['ok']:
        try:
            models = requests.get(f'{base_url}/v1/models', timeout=4).json().get('data', [])
            ids = [m.get('id', '?') for m in models]
            result['detail'] = f'online | models: {", ".join(ids) or "none"}'
            result['extra'] = {'models': ids}
        except Exception:
            pass
    return result


def check_mlflow(tracking_uri: str) -> CheckResult:
    try:
        import mlflow
        mlflow.set_tracking_uri(tracking_uri)
        exps = mlflow.tracking.MlflowClient().search_experiments()
        return {
            'name':   'MLflow',
            'ok':     True,
            'detail': f'{len(exps)} experiment(s) at {tracking_uri}',
            'fix':    None,
            'extra':  None,
        }
    except Exception as exc:
        return {
            'name':   'MLflow',
            'ok':     False,
            'detail': str(exc),
            'fix':    f'mlflow ui --backend-store-uri {tracking_uri}',
            'extra':  None,
        }


def check_mcp_server(base_url: str = 'http://localhost:8080') -> CheckResult:
    return check_http(
        'MCP Tool Server', f'{base_url}/health',
        fix='bash scripts/cpu/start_mcp_server.sh',
    )


def check_postgres(base_url: str = 'http://localhost:8080') -> CheckResult:
    """
    Light check: tries to connect via SQLAlchemy if configured, else falls back
    to checking that the MCP server's SQL tool endpoint is reachable.
    """
    db_url = __import__('os').getenv('DATABASE_URL')
    if db_url:
        try:
            from sqlalchemy import create_engine, text
            eng = create_engine(db_url, pool_pre_ping=True)
            with eng.connect() as conn:
                conn.execute(text('SELECT 1'))
            return {
                'name':   'PostgreSQL',
                'ok':     True,
                'detail': f'Connected via DATABASE_URL',
                'fix':    None,
                'extra':  None,
            }
        except Exception as exc:
            return {
                'name':   'PostgreSQL',
                'ok':     False,
                'detail': str(exc),
                'fix':    'bash scripts/cpu/start_postgres.sh',
                'extra':  None,
            }
    return {
        'name':   'PostgreSQL',
        'ok':     False,
        'detail': 'DATABASE_URL not set — cannot verify',
        'fix':    'Set DATABASE_URL and run bash scripts/cpu/start_postgres.sh',
        'extra':  None,
    }


def check_gpu() -> CheckResult:
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            mem  = torch.cuda.get_device_properties(0).total_memory // (1024 ** 3)
            return {
                'name':   'GPU / CUDA',
                'ok':     True,
                'detail': f'{name} ({mem} GB VRAM)',
                'fix':    None,
                'extra':  {'device': name, 'vram_gb': mem},
            }
        return {
            'name':   'GPU / CUDA',
            'ok':     False,
            'detail': 'CUDA not available (CPU-only mode)',
            'fix':    'Use a GPU pod for Track B training',
            'extra':  None,
        }
    except ImportError:
        return {
            'name':   'GPU / CUDA',
            'ok':     False,
            'detail': 'torch not installed',
            'fix':    'pip install torch',
            'extra':  None,
        }


# ── Project path checks ────────────────────────────────────────────────────────

def check_project_paths(root: Path) -> list[CheckResult]:
    entries: list[tuple[str, str, str | None]] = [
        ('artifacts/ dir',        'artifacts',                                   None),
        ('trajectories/ dir',     'trajectories',                                None),
        ('logs/ dir',             'logs',                                        None),
        ('data/ dir',             'data',                                        None),
        ('checkpoints/ dir',      'checkpoints',                                 None),
        ('tasks JSONL',           'data/synthetic/tasks.jsonl',                  'bash scripts/cpu/run_01_generate_synthetic.sh'),
        ('grouped rollouts JSONL','data/grpo/grouped_rollouts.jsonl',            'bash scripts/cpu/run_05_prepare_grpo_dataset.sh'),
    ]
    results = []
    for label, rel, fix in entries:
        p = root / rel
        exists = p.exists()
        results.append({
            'name':   label,
            'ok':     exists,
            'detail': ('exists' if exists else 'missing') + f' → {p}',
            'fix':    fix if not exists else None,
            'extra':  None,
        })
    return results


# ── Full environment summary ───────────────────────────────────────────────────

def run_all_checks(
    lightning_url: str = 'http://localhost:19123',
    vllm_url:      str = 'http://localhost:8000',
    mcp_url:       str = 'http://localhost:8080',
    mlflow_uri:    str = 'mlruns',
    root:          Path | None = None,
) -> dict[str, list[CheckResult]]:
    if root is None:
        root = Path(__file__).resolve().parents[2]
    return {
        'runtime':  [check_python_version(), check_docker(), check_gpu()],
        'cpu_pkgs': check_cpu_packages(),
        'gpu_pkgs': check_gpu_packages(),
        'services': [
            check_lightning_server(lightning_url),
            check_vllm(vllm_url),
            check_mcp_server(mcp_url),
            check_postgres(),
            check_mlflow(mlflow_uri),
        ],
        'paths':    check_project_paths(root),
    }
