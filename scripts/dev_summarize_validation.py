from __future__ import annotations

import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
print('trajectory_count', len(list((root / 'trajectories/baseline').glob('*.jsonl'))))
print('scored_count', len(list((root / 'trajectories/scored').glob('*.jsonl'))))

for p in sorted((root / 'trajectories/scored').glob('*.jsonl'))[:8]:
    rec = json.loads(p.read_text(encoding='utf-8'))
    print(p.name, 'reward=', rec.get('reward'), 'metadata=', rec.get('reward_metadata'))

for p in sorted((root / 'artifacts').glob('*/benchmark_metrics.json'))[:4]:
    payload = json.loads(p.read_text(encoding='utf-8'))
    print('benchmark', p.relative_to(root), 'champion=', payload.get('champion'))

for p in sorted((root / 'artifacts').glob('*/review_report.json'))[:4]:
    payload = json.loads(p.read_text(encoding='utf-8'))
    print('review', p.relative_to(root), 'approved=', payload.get('approved'), 'issues=', payload.get('issues'))

print('artifact_files')
for p in sorted((root / 'artifacts').glob('*/*'))[:120]:
    if p.is_file():
        print(p.relative_to(root))
