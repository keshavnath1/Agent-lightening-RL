from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json


@dataclass
class CheckpointNode:
    checkpoint_id: str
    path: str
    parent_id: str | None
    trainer: str
    reward_mode: str
    model_name: str
    created_at: str
    metrics: dict[str, Any]
    notes: str | None = None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_checkpoint_tree(path: str | Path = 'checkpoints/checkpoint_tree.json') -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except json.JSONDecodeError:
        return {}


def write_checkpoint_tree(tree: dict[str, Any], path: str | Path = 'checkpoints/checkpoint_tree.json') -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(tree, indent=2, sort_keys=True), encoding='utf-8')


def register_checkpoint(
    checkpoint_id: str,
    path: str,
    parent_id: str | None,
    trainer: str,
    reward_mode: str,
    model_name: str,
    metrics: dict[str, Any] | None = None,
    notes: str | None = None,
    tree_path: str | Path = 'checkpoints/checkpoint_tree.json',
) -> dict[str, Any]:
    tree = load_checkpoint_tree(tree_path)
    if 'base' not in tree:
        tree['base'] = asdict(CheckpointNode(
            checkpoint_id='base', path=model_name, parent_id=None, trainer='base_model',
            reward_mode='none', model_name=model_name, created_at=_now(), metrics={}, notes='Base model root.'
        ))
    node = CheckpointNode(
        checkpoint_id=checkpoint_id,
        path=path,
        parent_id=parent_id,
        trainer=trainer,
        reward_mode=reward_mode,
        model_name=model_name,
        created_at=_now(),
        metrics=metrics or {},
        notes=notes,
    )
    tree[checkpoint_id] = asdict(node)
    write_checkpoint_tree(tree, tree_path)
    return tree


def write_checkpoint_report(
    tree_path: str | Path = 'checkpoints/checkpoint_tree.json',
    output: str | Path = 'reports/checkpoint_forking_summary.md',
) -> Path:
    tree = load_checkpoint_tree(tree_path)
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = ['# Checkpoint Forking Summary', '', '| Checkpoint | Parent | Trainer | Reward Mode | Path |', '|---|---|---|---|---|']
    for cid, node in sorted(tree.items()):
        lines.append(f"| {cid} | {node.get('parent_id')} | {node.get('trainer')} | {node.get('reward_mode')} | `{node.get('path')}` |")
    out.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return out
