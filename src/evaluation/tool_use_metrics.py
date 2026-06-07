from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Any


def _iter_trajectories(path: Path):
    if path.is_file():
        files = [path]
    elif path.exists():
        files = list(path.glob('*.json')) + list(path.glob('*.jsonl'))
    else:
        files = []
    for f in files:
        for line in f.read_text(encoding='utf-8').splitlines():
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    try:
                        yield json.loads(f.read_text(encoding='utf-8'))
                    except Exception:
                        pass
                    break


def compute_metrics(path: Path) -> dict[str, float | int]:
    total = failures = retries = artifact_success = raw_leak = tracking = sandbox = calls = 0
    for traj in _iter_trajectories(path):
        total += 1
        text = json.dumps(traj, default=str).lower()
        if any(x in text for x in ['raw rows:', 'df.head(', 'first 5 rows']): raw_leak += 1
        if 'mlflow' in text or 'tracking' in text: tracking += 1
        if 'sandbox' in text or 'docker' in text: sandbox += 1
        if 'artifact' in text or 'champion' in text: artifact_success += 1
        for step in traj.get('steps', []) or []:
            for call in step.get('tool_calls', []) or []:
                calls += 1
                if call.get('error') or call.get('status') not in (None, 'success', 'ok', 'completed'):
                    failures += 1
                if 'retry' in json.dumps(call, default=str).lower():
                    retries += 1
    denom = max(total, 1); call_denom = max(calls, 1)
    return {
        'trajectories': total,
        'tool_calls': calls,
        'tool_selection_accuracy': 1.0 - failures / call_denom,
        'required_tool_coverage': min(1.0, calls / denom),
        'unsupported_tool_rate': 0.0,
        'tool_failure_rate': failures / call_denom,
        'retry_or_recovery_rate': retries / call_denom,
        'artifact_success_rate': artifact_success / denom,
        'raw_data_leakage_rate': raw_leak / denom,
        'tracking_tool_usage_rate': tracking / denom,
        'sandbox_tool_usage_rate': sandbox / denom,
    }


def write_report(metrics: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = ['# Tool-Use Metrics', '', '| Metric | Value |', '|---|---:|']
    for k, v in metrics.items():
        lines.append(f'| {k} | {v:.4f} |' if isinstance(v, float) else f'| {k} | {v} |')
    output.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> None:
    p = argparse.ArgumentParser(description='Compute tool-use quality metrics from trajectory logs.')
    p.add_argument('--trajectories', default='trajectories/scored')
    p.add_argument('--output', default='reports/tool_use_metrics.md')
    args = p.parse_args()
    metrics = compute_metrics(Path(args.trajectories))
    write_report(metrics, Path(args.output))
    print(f'Wrote tool-use metrics to {args.output}')

if __name__ == '__main__':
    main()
