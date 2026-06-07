from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.telemetry.logger import TrajectoryLogger


def _compact_state(trajectory: dict[str, Any], step_index: int) -> dict[str, Any]:
    steps = trajectory.get('steps', [])
    prior_steps = steps[:step_index]
    return {
        'task_id': trajectory.get('task_id'),
        'policy_version': trajectory.get('policy_version'),
        'trajectory_id': trajectory.get('trajectory_id'),
        'step_index': step_index,
        'prior_agents': [s.get('agent_name') for s in prior_steps],
        'prior_actions': [s.get('action') for s in prior_steps],
        'final_status': trajectory.get('final_status'),
    }


def trajectory_to_transitions(trajectory: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert one internal trajectory into Agent Lightning-style transitions.

    This module is intentionally dependency-free. It does not run Microsoft's
    Lightning Server or Client. Instead, it creates a portable transition export
    with the fields needed to bridge this repository's trajectory logger into an
    Agent Lightning or veRL training job later: state_t, action_t, reward_t,
    state_t+1, tool calls, and error metadata.
    """
    steps = trajectory.get('steps', [])
    final_reward = float(trajectory.get('reward') or 0.0)
    transitions: list[dict[str, Any]] = []
    for idx, step in enumerate(steps):
        is_terminal_step = idx == len(steps) - 1
        reward = final_reward if is_terminal_step else 0.0
        transition = {
            'task_id': trajectory.get('task_id'),
            'trajectory_id': trajectory.get('trajectory_id'),
            'policy_version': trajectory.get('policy_version'),
            'transition_id': f"{trajectory.get('trajectory_id')}:step_{idx:03d}",
            'state_t': _compact_state(trajectory, idx),
            'action_t': {
                'agent_name': step.get('agent_name'),
                'action': step.get('action'),
                'reasoning_summary': step.get('reasoning_summary'),
                'tool_calls': step.get('tool_calls', []),
            },
            'reward_t': reward,
            'state_t_plus_1': _compact_state(trajectory, idx + 1),
            'terminal': is_terminal_step,
            'final_reward': final_reward,
            'reward_metadata': trajectory.get('reward_metadata', {}),
            'error_types': [
                tc.get('error')
                for tc in step.get('tool_calls', [])
                if tc.get('status') != 'success' and tc.get('error')
            ],
        }
        transitions.append(transition)
    return transitions


def export_transitions(input_dir: str, output_path: str) -> None:
    records = TrajectoryLogger.read_many(input_dir)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out.open('w', encoding='utf-8') as f:
        for trajectory in records:
            for transition in trajectory_to_transitions(trajectory):
                f.write(json.dumps(transition) + '\n')
                count += 1
    print(f'Wrote {count} Agent Lightning-style transitions to {output_path}')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', default='trajectories/scored')
    parser.add_argument('--output', default='data/grpo/agent_lightning_transitions.jsonl')
    args = parser.parse_args()
    export_transitions(args.input_dir, args.output)


if __name__ == '__main__':
    main()
