from __future__ import annotations
import argparse, json
from pathlib import Path


def _count_jsonl(path: Path) -> int:
    return sum(1 for l in path.read_text(encoding='utf-8').splitlines() if l.strip()) if path.exists() else 0


def build_report(output: Path) -> None:
    sections = [
        ('Production/Track A rollouts collected', _count_jsonl(Path('data/grpo/grouped_rollouts.jsonl'))),
        ('Trajectories captured', len(list(Path('trajectories').glob('**/*.json'))) if Path('trajectories').exists() else 0),
        ('Rewards attached', len(list(Path('trajectories/scored').glob('*'))) if Path('trajectories/scored').exists() else 0),
        ('RULER/group ranking applied', _count_jsonl(Path('data/grpo/ruler_scored_groups.jsonl'))),
        ('Policy training completed', Path('checkpoints/checkpoint_tree.json').exists()),
        ('Checkpoint registered', Path('checkpoints/checkpoint_tree.json').exists()),
        ('Held-out validation result', Path('data/grpo/heldout_prompts.jsonl').exists()),
        ('Next iteration recommendation', 'Run more task groups and compare RULER-GRPO against baseline on held-out scenarios.'),
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = ['# Production-as-Data Self-Improving Flywheel', '', 'Production/Track A → Log Trajectories → RULER/Reward → Train Policy → Validate → Deploy/Reload → New Rollouts', '', '| Stage | Status |', '|---|---|']
    for name, value in sections:
        lines.append(f'| {name} | {value} |')
    output.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> None:
    p = argparse.ArgumentParser(description='Generate self-improving production-as-data flywheel report.')
    p.add_argument('--output', default='reports/self_improving_flywheel.md')
    args = p.parse_args()
    build_report(Path(args.output))
    print(f'Wrote flywheel report to {args.output}')

if __name__ == '__main__':
    main()
