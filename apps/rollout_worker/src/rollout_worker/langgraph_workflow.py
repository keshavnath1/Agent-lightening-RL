from __future__ import annotations

import argparse

from rollout_worker.run_track_a import run_tasks


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

