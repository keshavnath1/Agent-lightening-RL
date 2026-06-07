from __future__ import annotations

import argparse
import os
import warnings

from src.training.agent_lightning_official_runner import run_official_agent_lightning


def run_tasks(
    tasks_path: str,
    output_dir: str,
    policy_version: str,
    limit: int | None = None,
    rollouts: int = 1,
    require_live_policy: bool = False,
    lightning_server_url: str | None = None,
) -> None:
    """Compatibility wrapper for legacy Track A entrypoint.

    Canonical execution path is now
    ``python -m src.training.agent_lightning_official_runner``.
    """
    warnings.warn(
        'rollout_worker.run_track_a.run_tasks is deprecated; use '
        'src.training.agent_lightning_official_runner instead.',
        DeprecationWarning,
        stacklevel=2,
    )

    if rollouts != 1:
        raise RuntimeError(
            'Strict official mode does not support --rollouts via rollout_worker wrapper. '
            'Use the official runner directly.'
        )
    if require_live_policy:
        raise RuntimeError(
            'Strict official mode does not support --require-live-policy via rollout_worker wrapper. '
            'Use the official runner directly.'
        )

    if lightning_server_url:
        os.environ['LIGHTNING_SERVER_URL'] = lightning_server_url

    if tasks_path and 'data/synthetic/' in str(tasks_path).replace('\\\\', '/'):
        raise RuntimeError(
            'Strict official mode does not accept synthetic task paths via rollout_worker wrapper. '
            'Use PostgreSQL/MCP-backed task loading only.'
        )

    run_official_agent_lightning(
        tasks_path=tasks_path,
        output_dir=output_dir,
        policy_version=policy_version,
        limit=limit,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--tasks',
        default='',
        help='Deprecated compatibility argument. Strict mode ignores local task files and uses PostgreSQL/MCP.',
    )
    parser.add_argument('--output-dir', default='trajectories/agent_lightning_official')
    parser.add_argument('--policy-version', default='agent_lightning_policy')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--rollouts', type=int, default=1)
    parser.add_argument('--require-live-policy', action='store_true')
    parser.add_argument(
        '--lightning-server-url',
        default=None,
        help='GPU Lightning Server URL (overrides LIGHTNING_SERVER_URL env var)',
    )
    args = parser.parse_args()
    run_tasks(
        args.tasks,
        args.output_dir,
        args.policy_version,
        args.limit,
        args.rollouts,
        require_live_policy=args.require_live_policy,
        lightning_server_url=args.lightning_server_url,
    )

