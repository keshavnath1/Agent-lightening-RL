from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from src.telemetry.logger import TrajectoryLogger


def prepare(input_dir: str, output_path: str) -> None:
    records = TrajectoryLogger.read_many(input_dir)
    grouped = defaultdict(list)
    for r in records:
        grouped[r['task_id']].append(r)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', encoding='utf-8') as f:
        for task_id, group in grouped.items():
            sorted_group = sorted(group, key=lambda x: x.get('reward') or 0.0, reverse=True)
            payload = {
                'task_id': task_id,
                'group_size': len(sorted_group),
                'ranked_trajectories': [
                    {
                        'trajectory_id': g['trajectory_id'],
                        'reward': g.get('reward', 0.0),
                        'policy_version': g.get('policy_version'),
                        'steps': g.get('steps', []),
                    }
                    for g in sorted_group
                ],
            }
            f.write(json.dumps(payload) + '\n')
    print(f'Wrote grouped rollout dataset to {output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', default='trajectories/scored')
    parser.add_argument('--output', default='data/grpo/grouped_rollouts.jsonl')
    args = parser.parse_args()
    prepare(args.input_dir, args.output)
