from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from src.agents.supervisor import SupervisorAgent
from src.rewards.scorer import score_trajectory
from src.telemetry.logger import TrajectoryLogger
from src.tools.task_registry import (
    TrackATaskLoadError,
    load_tracka_tasks_from_postgres,
    TRACKA_DEFAULT_TASK_LIMIT,
)


class AgentLightningDependencyError(RuntimeError):
    """Raised when the official Agent Lightning runtime is not installed."""


def _import_agentlightning():
    try:
        import agentlightning as agl  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on pod runtime
        raise AgentLightningDependencyError(
            'Official Agent Lightning runtime is required for this mode. '
            'Install it in the CPU pod with `pip install agentlightning` and rerun; '
            'this script intentionally does not fall back to local-only training.'
        ) from exc
    return agl


def load_tasks(
    path: str | Path,
    limit: int | None = None,
    task_source: str = 'postgres_mcp',
) -> list[dict[str, Any]]:
    """Load Track A tasks.

    When ``task_source`` is ``postgres_mcp`` (default and production), tasks
    are fetched from PostgreSQL via MCP using the strict fail-closed loader.
    Any non-postgres_mcp task source fails closed.
    """
    if task_source == 'postgres_mcp':
        return load_tracka_tasks_from_postgres(limit=limit)
    raise TrackATaskLoadError(
        f'Unsupported --task-source {task_source!r}. Only postgres_mcp is allowed in strict mode.'
    )


def _resource_llm_metadata(resources: Any) -> dict[str, str]:
    """Extract documented Agent Lightning LLM resource fields when present."""
    llm = None
    if isinstance(resources, dict):
        llm = resources.get('llm') or resources.get('policy_llm') or resources.get('model')
    metadata: dict[str, str] = {}
    for attr, env_name in [
        ('endpoint', 'ACTIVE_POLICY_URL'),
        ('model', 'ACTIVE_POLICY_MODEL'),
        ('api_key', 'ACTIVE_POLICY_API_KEY'),
    ]:
        value = getattr(llm, attr, None) if llm is not None else None
        if value:
            metadata[attr] = str(value)
            os.environ[env_name] = str(value)
    return metadata


def build_lightning_agent(policy_version: str, output_dir: str | Path):
    agl = _import_agentlightning()

    class SelfImprovingMLLightningAgent(agl.LitAgent[dict[str, Any]]):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.supervisor = SupervisorAgent()
            self.logger = TrajectoryLogger(output_dir)

        def rollout(self, task: dict[str, Any], resources: Any, rollout: Any) -> float:
            llm_metadata = _resource_llm_metadata(resources)
            task_variant = dict(task)
            if getattr(rollout, 'rollout_id', None):
                task_variant['agent_lightning_rollout_id'] = str(rollout.rollout_id)
            if getattr(rollout, 'mode', None):
                task_variant['agent_lightning_mode'] = str(rollout.mode)
            if llm_metadata:
                task_variant['agent_lightning_llm_resource'] = {
                    key: ('***' if key == 'api_key' else value)
                    for key, value in llm_metadata.items()
                }
            trajectory = self.supervisor.run_task(task_variant, policy_version=policy_version)
            reward, metadata = score_trajectory(trajectory.to_dict())
            trajectory.reward = reward
            trajectory.reward_metadata = metadata
            self.logger.write(trajectory)
            return float(reward)

    return SelfImprovingMLLightningAgent()


def run_official_agent_lightning(
    tasks_path: str | Path,
    output_dir: str | Path,
    policy_version: str,
    limit: int | None = None,
    task_source: str = 'postgres_mcp',
) -> dict[str, Any]:
    agl = _import_agentlightning()
    tasks = load_tasks(tasks_path, limit=limit, task_source=task_source)
    agent = build_lightning_agent(policy_version=policy_version, output_dir=output_dir)
    trainer = agl.Trainer()
    try:
        trainer.fit(agent=agent, train_dataset=tasks)
    except Exception as exc:
        raise RuntimeError(
            'Official Agent Lightning Trainer.fit(agent=..., train_dataset=...) failed. '
            f'error={exc.__class__.__name__}: {exc}'
        ) from exc
    return {
        'runtime': 'official_agent_lightning',
        'tasks_path': str(tasks_path),
        'output_dir': str(output_dir),
        'policy_version': policy_version,
        'num_tasks': len(tasks),
        'status': 'completed',
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Run the repository CPU workflow through the official Agent Lightning Trainer/LitAgent API.'
    )
    parser.add_argument('--tasks', default='data/synthetic/tasks.jsonl')
    parser.add_argument(
        '--task-source',
        dest='task_source',
        default='postgres_mcp',
        choices=['postgres_mcp'],
        help='Task source. postgres_mcp is required for production Track A runs.'
    )
    parser.add_argument(
        '--task-limit',
        dest='task_limit',
        type=int,
        default=None,
        help=f'Maximum tasks to load (default: TRACKA_DEFAULT_TASK_LIMIT={TRACKA_DEFAULT_TASK_LIMIT}).'
    )
    parser.add_argument('--limit', type=int, default=None, help='Alias for --task-limit (deprecated).')
    parser.add_argument('--output-dir', default='trajectories/agent_lightning_official')
    parser.add_argument('--policy-version', default='agent_lightning_policy')
    parser.add_argument('--metadata-output', default='reports/agent_lightning_official_run.json')
    args = parser.parse_args()

    effective_limit = args.task_limit if args.task_limit is not None else args.limit

    metadata = run_official_agent_lightning(
        tasks_path=args.tasks,
        output_dir=args.output_dir,
        policy_version=args.policy_version,
        limit=effective_limit,
        task_source=args.task_source,
    )
    out = Path(args.metadata_output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
