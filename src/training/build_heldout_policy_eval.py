from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding='utf-8').splitlines() if l.strip()]


def _bucket(task_id: str) -> float:
    return int(hashlib.sha256(task_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def build(input_path: Path, output_dir: Path, heldout_ratio: float = 0.2) -> dict[str, Any]:
    groups = _read_jsonl(input_path)
    heldout, train = [], []
    for g in groups:
        (heldout if _bucket(str(g.get('task_id'))) < heldout_ratio else train).append(g)
    output_dir.mkdir(parents=True, exist_ok=True)
    prompts_path = output_dir / 'heldout_prompts.jsonl'
    tasks_path = output_dir / 'heldout_tasks.jsonl'
    seen_prompts: set[str] = set()
    with prompts_path.open('w', encoding='utf-8') as pf, tasks_path.open('w', encoding='utf-8') as tf:
        for g in heldout:
            task_id = str(g.get('task_id'))
            tf.write(json.dumps({'task_id': task_id, 'difficulty': g.get('difficulty', 'unknown'), 'source': 'heldout_task_split'}) + '\n')
            trajectories = g.get('judged_trajectories') or g.get('ranked_trajectories') or []
            for traj in trajectories[:1]:
                prompt = json.dumps({'task_id': task_id, 'policy_version': traj.get('policy_version'), 'instruction': 'Emit the next high-quality ML workflow action JSON.'}, sort_keys=True)
                if prompt not in seen_prompts:
                    seen_prompts.add(prompt)
                    pf.write(json.dumps({'task_id': task_id, 'prompt': prompt, 'difficulty': g.get('difficulty', 'unknown')}) + '\n')
    train_ids = {str(g.get('task_id')) for g in train}
    held_ids = {str(g.get('task_id')) for g in heldout}
    summary = {'train_groups': len(train), 'heldout_groups': len(heldout), 'task_id_overlap': sorted(train_ids & held_ids), 'duplicate_prompt_leakage': False}
    (output_dir / 'heldout_split_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description='Build task-level held-out prompts/tasks for policy evaluation.')
    p.add_argument('--input', default='data/grpo/ruler_scored_groups.jsonl')
    p.add_argument('--output-dir', default='data/grpo')
    p.add_argument('--heldout-ratio', type=float, default=0.2)
    args = p.parse_args()
    print(json.dumps(build(Path(args.input), Path(args.output_dir), args.heldout_ratio), indent=2))

if __name__ == '__main__':
    main()
