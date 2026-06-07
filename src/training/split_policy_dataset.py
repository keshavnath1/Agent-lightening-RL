"""
Deterministic train / validation / test split for the GRPO grouped rollout dataset.

Splitting is done at the **task_id level**, not at the row level.  This prevents
the same task from appearing in both train and test sets and avoids prompt-level
data leakage.

Usage
-----
    from src.training.split_policy_dataset import split_by_task_id, check_leakage

    groups = [...]  # list of dicts, each with a "task_id" key
    train, val, test = split_by_task_id(groups, train_ratio=0.8, val_ratio=0.1, seed=42)
    report = check_leakage(train, val, test)
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


# ── public API ────────────────────────────────────────────────────────────────

def split_by_task_id(
    groups: list[dict[str, Any]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Split *groups* into train / val / test by hashing task_id.

    Parameters
    ----------
    groups:
        List of grouped rollout records.  Each record must have a ``task_id`` key.
    train_ratio, val_ratio, test_ratio:
        Must sum to 1.0 (within floating-point tolerance).
    seed:
        Integer mixed into the hash to allow reproducible but varied splits.

    Returns
    -------
    (train_groups, val_groups, test_groups)
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError('train_ratio + val_ratio + test_ratio must equal 1.0')

    # Deduplicate by task_id, preserving first-seen record
    seen: dict[str, dict[str, Any]] = {}
    for rec in groups:
        tid = str(rec.get('task_id', ''))
        if tid and tid not in seen:
            seen[tid] = rec

    train_groups: list[dict[str, Any]] = []
    val_groups:   list[dict[str, Any]] = []
    test_groups:  list[dict[str, Any]] = []

    val_boundary  = train_ratio
    test_boundary = train_ratio + val_ratio

    for task_id, rec in seen.items():
        bucket = _hash_fraction(task_id, seed)
        if bucket < val_boundary:
            train_groups.append(rec)
        elif bucket < test_boundary:
            val_groups.append(rec)
        else:
            test_groups.append(rec)

    return train_groups, val_groups, test_groups


def check_leakage(
    train: list[dict[str, Any]],
    val:   list[dict[str, Any]],
    test:  list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Return a leakage report dict.

    Checks
    ------
    - Same task_id in train and test
    - Same prompt appearing in train and test
    """
    def _task_ids(groups: list[dict]) -> set[str]:
        ids: set[str] = set()
        for g in groups:
            ids.add(str(g.get('task_id', '')))
            for traj in g.get('ranked_trajectories', []):
                ids.add(str(traj.get('task_id', g.get('task_id', ''))))
        return ids

    def _prompts(groups: list[dict]) -> set[str]:
        ps: set[str] = set()
        for g in groups:
            for traj in g.get('ranked_trajectories', []):
                for step in traj.get('steps', []):
                    p = step.get('prompt') or step.get('reasoning_summary') or ''
                    if p:
                        ps.add(p[:200])  # truncate to first 200 chars for comparison
        return ps

    train_ids   = _task_ids(train)
    val_ids     = _task_ids(val)
    test_ids    = _task_ids(test)
    train_prompts = _prompts(train)
    test_prompts  = _prompts(test)

    task_leakage_train_test  = train_ids & test_ids
    task_leakage_train_val   = train_ids & val_ids
    prompt_leakage_train_test = train_prompts & test_prompts

    return {
        'total_task_groups':        len(train) + len(val) + len(test),
        'train_task_groups':        len(train),
        'val_task_groups':          len(val),
        'test_task_groups':         len(test),
        'task_id_leakage_train_test':  sorted(task_leakage_train_test),
        'task_id_leakage_train_val':   sorted(task_leakage_train_val),
        'prompt_leakage_count':     len(prompt_leakage_train_test),
        'heldout_evaluation_available': len(test) > 0,
        'leakage_detected': bool(task_leakage_train_test or prompt_leakage_train_test),
    }


def load_grouped_rollouts(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file of grouped rollouts."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    rows: list[dict[str, Any]] = []
    for line in p.read_text(encoding='utf-8').splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_split(
    split: list[dict[str, Any]],
    path: str | Path,
) -> None:
    """Write a list of grouped rollout records to a JSONL file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w', encoding='utf-8') as f:
        for rec in split:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')


# ── internals ─────────────────────────────────────────────────────────────────

def _hash_fraction(task_id: str, seed: int) -> float:
    """Map task_id to a stable float in [0, 1) using SHA-256."""
    raw = f'{seed}:{task_id}'.encode('utf-8')
    digest = hashlib.sha256(raw).hexdigest()
    # Use first 8 hex chars → 32-bit integer → fraction
    return int(digest[:8], 16) / 0xFFFFFFFF


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    import sys

    parser = argparse.ArgumentParser(description='Split a grouped rollout JSONL by task_id.')
    parser.add_argument('--input',       required=True, help='Path to grouped_rollouts.jsonl')
    parser.add_argument('--output-dir',  default='data/grpo', help='Directory for split files')
    parser.add_argument('--train-ratio', type=float, default=0.8)
    parser.add_argument('--val-ratio',   type=float, default=0.1)
    parser.add_argument('--test-ratio',  type=float, default=0.1)
    parser.add_argument('--seed',        type=int,   default=42)
    args = parser.parse_args()

    groups = load_grouped_rollouts(args.input)
    train, val, test = split_by_task_id(
        groups,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    report = check_leakage(train, val, test)

    out = Path(args.output_dir)
    write_split(train, out / 'train_rollouts.jsonl')
    write_split(val,   out / 'val_rollouts.jsonl')
    write_split(test,  out / 'test_rollouts.jsonl')

    print(json.dumps(report, indent=2))
    if report['leakage_detected']:
        print('WARNING: leakage detected — review report above.', file=sys.stderr)
    else:
        print('No leakage detected.')
