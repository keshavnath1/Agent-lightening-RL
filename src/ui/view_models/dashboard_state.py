"""
dashboard_state.py — shared state + data loading for the Streamlit dashboard.

All pages import DashboardState to avoid duplicated data-loading logic.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]


@dataclass
class ServiceHealth:
    lightning_ok: bool = False
    vllm_ok: bool = False
    mlflow_ok: bool = False
    rollouts_collected: int = 0
    training_active: bool = False
    queue_size: int = 0


@dataclass
class DataStatus:
    grouped_rollouts: int = 0
    scored_trajectories: int = 0
    tasks_loaded: int = 0
    artifacts: list[Path] = field(default_factory=list)

    @property
    def ready_for_training(self) -> bool:
        return self.grouped_rollouts >= 1


@dataclass
class GPUStatus:
    available: bool = False
    name: str = "—"
    vram_gb: float = 0.0
    count: int = 0


@dataclass
class TrainingConfig:
    trainer: str = "qlora_sft"
    reward_mode: str = "hybrid"
    num_generations: int = 4
    checkpoint_path: str = "checkpoints/trackb_qlora_sft"
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    min_rollouts: int = 4
    reload_after: bool = True


# ── Data loaders ──────────────────────────────────────────────────────────────

def load_data_status() -> DataStatus:
    ds = DataStatus()

    # Grouped rollouts
    for candidate in [
        ROOT / "data" / "grpo" / "grouped_rollouts.jsonl",
        ROOT / "data" / "grpo" / "lightning_grouped_rollouts.jsonl",
    ]:
        if candidate.exists():
            lines = [l for l in candidate.read_text().splitlines() if l.strip()]
            ds.grouped_rollouts = max(ds.grouped_rollouts, len(lines))

    # Scored trajectories
    trajectory_root = ROOT / "trajectories"
    scored_dirs = [trajectory_root / "scored"]
    if trajectory_root.exists():
        scored_dirs.extend(
            p for p in trajectory_root.iterdir()
            if p.is_dir() and p.name.endswith("_scored")
        )
    seen_scored_dirs = []
    for scored_dir in scored_dirs:
        if scored_dir.exists() and scored_dir not in seen_scored_dirs:
            seen_scored_dirs.append(scored_dir)
            ds.scored_trajectories += sum(
                1 for f in scored_dir.glob("*.jsonl")
                for line in f.read_text().splitlines() if line.strip()
            )
            ds.scored_trajectories += sum(1 for _ in scored_dir.glob("*.json"))

    # Tasks
    for task_path in [
        ROOT / "data" / "real_openml" / "tasks.jsonl",
        ROOT / "data" / "synthetic" / "tasks.jsonl",
        ROOT / "data" / "synthetic" / "tasks_dataset.jsonl",
    ]:
        if task_path.exists():
            ds.tasks_loaded = sum(1 for l in task_path.read_text().splitlines() if l.strip())
            break

    # Artifacts
    artifacts_dir = ROOT / "artifacts"
    if artifacts_dir.exists():
        ds.artifacts = sorted(artifacts_dir.glob("*/champion_model.json"))
        if ds.tasks_loaded == 0:
            ds.tasks_loaded = len(ds.artifacts)

    return ds


def load_gpu_status() -> GPUStatus:
    gs = GPUStatus()
    try:
        import torch
        if torch.cuda.is_available():
            gs.available = True
            gs.count = torch.cuda.device_count()
            gs.name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            gs.vram_gb = round(props.total_memory / 1e9, 1)
    except Exception:
        pass
    return gs


def load_service_health(server_url: str, vllm_url: str, mlflow_uri: str) -> ServiceHealth:
    import requests as _req
    sh = ServiceHealth()

    try:
        r = _req.get(f"{server_url}/health", timeout=3)
        if r.status_code == 200:
            d = r.json()
            sh.lightning_ok = True
            sh.rollouts_collected = d.get("rollouts_collected", 0)
            sh.training_active = d.get("training_active", False)
            sh.queue_size = d.get("queue_size", 0)
    except Exception:
        pass

    try:
        r = _req.get(f"{vllm_url}/v1/models", timeout=3)
        sh.vllm_ok = r.status_code == 200
    except Exception:
        pass

    try:
        import mlflow
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.tracking.MlflowClient().search_experiments()
        sh.mlflow_ok = True
    except Exception:
        pass

    return sh


def load_task_results() -> list[dict[str, Any]]:
    """Load champion_model.json artifacts for the Track A results table."""
    rows = []
    artifacts_dir = ROOT / "artifacts"
    if not artifacts_dir.exists():
        return rows
    for task_dir in sorted(p for p in artifacts_dir.iterdir() if p.is_dir()):
        cm = task_dir / "champion_model.json"
        bm = task_dir / "benchmark_metrics.json"
        if not cm.exists():
            continue
        try:
            data = json.loads(cm.read_text())
            metrics = json.loads(bm.read_text()) if bm.exists() else {}
            champion = metrics.get("champion", {}) if isinstance(metrics, dict) else {}
            champion_metrics = champion.get("metrics", {}) if isinstance(champion, dict) else {}
            metric_name = (
                metrics.get("metric")
                or data.get("primary_metric")
                or "roc_auc"
            )
            metric_score = (
                champion_metrics.get(metric_name)
                or champion_metrics.get("roc_auc")
                or champion_metrics.get("accuracy")
                or data.get("score")
                or data.get("champion_score")
                or 0
            )
            reward = (
                metrics.get("final_reward")
                or data.get("reward")
                or _load_latest_task_reward(task_dir.name)
                or 0
            )
            rows.append({
                "task_id":       task_dir.name,
                "status":        "✅ Complete",
                "champion":      champion.get("model_name") or data.get("model_name") or data.get("trainer") or "—",
                "metric":        metric_name,
                "score":         round(float(metric_score), 4),
                "reward":        round(float(reward), 4),
            })
        except Exception:
            rows.append({"task_id": task_dir.name, "status": "⚠️ Parse error",
                         "champion": "—", "metric": "—", "score": 0, "reward": 0})
    return rows


def _load_latest_task_reward(task_id: str) -> float | None:
    """Read the newest scored trajectory reward for a task."""
    trajectory_root = ROOT / "trajectories"
    if not trajectory_root.exists():
        return None
    scored_files = sorted(
        [
            p for scored_dir in trajectory_root.glob("*_scored")
            for p in scored_dir.glob("*.jsonl")
            if p.name.startswith(task_id)
        ],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for path in scored_files:
        try:
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                reward = json.loads(line).get("reward")
                if reward is not None:
                    return float(reward)
        except Exception:
            continue
    return None


def load_grouped_rollouts_for_grpo() -> list[dict[str, Any]]:
    """Load grouped rollouts for the GRPO advantage table."""
    for candidate in [
        ROOT / "data" / "grpo" / "grouped_rollouts.jsonl",
        ROOT / "data" / "grpo" / "lightning_grouped_rollouts.jsonl",
    ]:
        if candidate.exists():
            rows = []
            for line in candidate.read_text().splitlines():
                if line.strip():
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        pass
            return rows
    return []


def get_checkpoint_path(trainer: str, reward_mode: str | None = None) -> str:
    CHECKPOINT_MAP = {
        "qlora_sft":                "checkpoints/trackb_qlora_sft",
        "trl_grpo":                 "checkpoints/trackb_trl_grpo_hybrid",
        "official_art_ruler":      "checkpoints/trackb_official_art_ruler",
        "verl":                     "checkpoints/trackb_verl",
        "agent_lightning_official": "checkpoints/trackb_agent_lightning",
    }
    if trainer == "trl_grpo" and reward_mode == "ruler_relative":
        return "checkpoints/trackb_trl_grpo_ruler"
    return CHECKPOINT_MAP.get(trainer, f"checkpoints/trackb_{trainer}")


def get_shell_command(cfg: TrainingConfig) -> str:
    return (
        f"TRAINER={cfg.trainer} "
        f"REWARD_MODE={cfg.reward_mode} "
        f"NUM_GENERATIONS={cfg.num_generations} "
        f"POLICY_OUTPUT_DIR={cfg.checkpoint_path} "
        f"bash scripts/gpu/run_02_train_policy_qlora_grpo.sh"
    )
