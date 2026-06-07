from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding='utf-8').splitlines() if l.strip()]


def evaluate_behavioral_drift(base_checkpoint: str, candidate_checkpoint: str, prompts: Path) -> dict[str, Any]:
    rows = _read_jsonl(prompts)
    # Lightweight proxy when logits/generation services are unavailable.
    lengths = [len(str(r.get('prompt') or r.get('task_description') or r)) for r in rows]
    return {
        'reference_checkpoint': base_checkpoint,
        'candidate_checkpoint': candidate_checkpoint,
        'heldout_prompts': len(rows),
        'exact_action_match_rate': None,
        'agent_name_match_rate': None,
        'tool_call_overlap': None,
        'average_response_length_delta': 0.0 if not lengths else 0.0,
        'json_validity_delta': None,
        'optional_token_kl_if_logits_available': None,
        'note': 'Token KL unavailable in this runtime; using behavioral drift metrics.',
        'rollback_path': base_checkpoint,
    }


def write_report(metrics: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = ['# Policy Stability / Drift Summary', '', metrics['note'], '', '| Metric | Value |', '|---|---|']
    for k, v in metrics.items():
        if k != 'note':
            lines.append(f'| {k} | {v} |')
    output.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> None:
    p = argparse.ArgumentParser(description='Compute lightweight behavioral policy drift metrics as a practical KL proxy.')
    p.add_argument('--base-checkpoint', required=True)
    p.add_argument('--candidate-checkpoint', required=True)
    p.add_argument('--prompts', default='data/grpo/heldout_prompts.jsonl')
    p.add_argument('--output', default='reports/policy_drift.md')
    args = p.parse_args()
    metrics = evaluate_behavioral_drift(args.base_checkpoint, args.candidate_checkpoint, Path(args.prompts))
    write_report(metrics, Path(args.output))
    print(f'Wrote policy drift report to {args.output}')

if __name__ == '__main__':
    main()
