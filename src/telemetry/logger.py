from __future__ import annotations

import json
from pathlib import Path
from .schema import Trajectory


class TrajectoryLogger:
    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write(self, trajectory: Trajectory) -> Path:
        path = self.output_dir / f'{trajectory.task_id}_{trajectory.policy_version}_{trajectory.trajectory_id}.jsonl'
        with path.open('w', encoding='utf-8') as f:
            f.write(json.dumps(trajectory.to_dict()) + '\n')
        return path

    @staticmethod
    def read_many(input_dir: str | Path) -> list[dict]:
        records: list[dict] = []
        for path in Path(input_dir).glob('*.jsonl'):
            with path.open('r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line))
        return records
