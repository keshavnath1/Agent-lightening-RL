from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.telemetry.logger import TrajectoryLogger


def summarize(input_dir: str, policy_label: str | None = None) -> pd.DataFrame:
    records = TrajectoryLogger.read_many(input_dir)
    rows = []
    for r in records:
        tool_calls = [tc for s in r.get('steps', []) for tc in s.get('tool_calls', [])]
        live_policy_calls = [tc for tc in tool_calls if tc.get('tool_name') == 'PolicyClient.chat']
        rows.append({
            'task_id': r['task_id'],
            'policy': policy_label or r.get('policy_version'),
            'policy_version': r.get('policy_version'),
            'reward': float(r.get('reward') or 0.0),
            'task_success': 1 if r.get('final_status') == 'completed' else 0,
            'tool_calls': len(tool_calls),
            'valid_tool_rate': sum(1 for tc in tool_calls if tc.get('status') == 'success') / max(len(tool_calls), 1),
            'live_policy_endpoint_used': bool(live_policy_calls),
        })
    return pd.DataFrame(rows)


def _summary_row(policy: str, df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            'policy': policy,
            'tasks': 0,
            'avg_reward': 0.0,
            'task_success_rate': 0.0,
            'valid_tool_rate': 0.0,
            'live_policy_endpoint_rate': 0.0,
        }
    return {
        'policy': policy,
        'tasks': len(df),
        'avg_reward': round(float(df['reward'].mean()), 4),
        'task_success_rate': round(float(df['task_success'].mean()), 4),
        'valid_tool_rate': round(float(df['valid_tool_rate'].mean()), 4),
        'live_policy_endpoint_rate': round(float(df['live_policy_endpoint_used'].mean()), 4),
    }


def compare_policy_dirs(policy_dirs: list[str], output: str) -> None:
    frames = []
    raw_frames = []
    for item in policy_dirs:
        if '=' in item:
            label, directory = item.split('=', 1)
        else:
            directory = item
            label = Path(directory).name
        df = summarize(directory, policy_label=label)
        raw_frames.append(df)
        frames.append(_summary_row(label, df))
    summary = pd.DataFrame(frames)
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    markdown = '# Multi-Policy Benchmark Evaluation\n\n'
    markdown += 'This report compares named policy trajectory directories, for example baseline, V2, and RL-tuned GPU endpoint runs. The `live_policy_endpoint_rate` column verifies whether each trajectory used the strict live OpenAI-compatible policy endpoint wrapper.\n\n'
    markdown += summary.to_markdown(index=False) + '\n'
    if raw_frames:
        raw = pd.concat(raw_frames, ignore_index=True)
        if not raw.empty:
            pivot = raw.pivot_table(index='task_id', columns='policy', values='reward', aggfunc='mean').reset_index()
            markdown += '\n## Per-Task Reward Matrix\n\n' + pivot.to_markdown(index=False) + '\n'
    out.write_text(markdown, encoding='utf-8')
    print(f'Wrote comparison report to {output}')


def compare(baseline_dir: str, tuned_dir: str, output: str) -> None:
    compare_policy_dirs([f'baseline={baseline_dir}', f'rl_tuned={tuned_dir}'], output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--policy-dir',
        action='append',
        default=None,
        help='Policy directory in label=path form. Repeat for baseline, v2, tuned, etc.',
    )
    parser.add_argument('--baseline-dir', default='trajectories/scored')
    parser.add_argument('--v2-dir', default=None)
    parser.add_argument('--tuned-dir', default='trajectories/tuned_scored')
    parser.add_argument('--output', default='reports/baseline_v2_tuned_benchmark.md')
    args = parser.parse_args()
    if args.policy_dir:
        policy_dirs = args.policy_dir
    else:
        policy_dirs = [f'baseline={args.baseline_dir}', f'rl_tuned={args.tuned_dir}']
        if args.v2_dir:
            policy_dirs.insert(1, f'v2={args.v2_dir}')
    compare_policy_dirs(policy_dirs, args.output)
