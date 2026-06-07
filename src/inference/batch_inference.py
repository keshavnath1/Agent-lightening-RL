from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.inference.policy_client import PolicyClient, resolve_policy_endpoint


def _iter_tasks(tasks_path: str, limit: int | None = None) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for line in Path(tasks_path).read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        tasks.append(json.loads(line))
        if limit is not None and len(tasks) >= limit:
            break
    if not tasks:
        raise ValueError(f'No tasks found in {tasks_path}')
    return tasks


def create_policy_decision_jobs(tasks_path: str, output_path: str, policy_version: str, limit: int | None = None) -> None:
    """Write policy prompts and resolved endpoint metadata without calling the endpoint."""
    config = resolve_policy_endpoint(policy_version)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', encoding='utf-8') as f:
        for task in _iter_tasks(tasks_path, limit=limit):
            prompt = {
                'task_id': task['task_id'],
                'policy_version': config.policy_version,
                'endpoint_url_redacted': config.base_url.split('?')[0],
                'model_name': config.model_name,
                'messages': [
                    {
                        'role': 'system',
                        'content': 'You are the policy for a governed multi-agent tabular ML workflow. Do not request raw rows.',
                    },
                    {
                        'role': 'user',
                        'content': f"Plan workflow for task {task['task_id']} using only schema/profile artifacts.",
                    },
                ],
            }
            f.write(json.dumps(prompt) + '\n')
    print(f'Wrote strict policy decision jobs to {output_path}')


def run_policy_decisions(tasks_path: str, output_path: str, policy_version: str, limit: int | None = None) -> None:
    """Call the configured live GPU endpoint for each task and persist decisions.

    This path is intentionally strict: missing BASELINE/V2/TUNED endpoint variables
    or endpoint failures raise errors and make the benchmark fail.
    """
    client = PolicyClient(policy_version=policy_version)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', encoding='utf-8') as f:
        for task in _iter_tasks(tasks_path, limit=limit):
            decision = client.decision(task, policy_version=policy_version)
            decision['task_id'] = task['task_id']
            f.write(json.dumps(decision) + '\n')
    print(f'Wrote live policy decisions to {output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks', default='data/synthetic/tasks.jsonl')
    parser.add_argument('--output', default='data/synthetic/policy_jobs.jsonl')
    parser.add_argument('--policy-version', default='baseline')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--execute', action='store_true', help='Call the live endpoint instead of only writing strict job prompts.')
    args = parser.parse_args()
    if args.execute:
        run_policy_decisions(args.tasks, args.output, args.policy_version, args.limit)
    else:
        create_policy_decision_jobs(args.tasks, args.output, args.policy_version, args.limit)
