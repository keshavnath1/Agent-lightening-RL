#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export WORKSPACE_DIR="${WORKSPACE_DIR:-$REPO_DIR}"

: "${TASK_COUNT:=20}"
: "${SYNTHETIC_TASKS_PATH:=$WORKSPACE_DIR/data/synthetic/tasks.jsonl}"
: "${SYNTHETIC_DATASET_DIR:=$WORKSPACE_DIR/data/synthetic/datasets}"
mkdir -p "$(dirname "$SYNTHETIC_TASKS_PATH")" "$SYNTHETIC_DATASET_DIR"
python -m src.synthetic_data.generate_tasks --count "$TASK_COUNT" --output "$SYNTHETIC_TASKS_PATH" --dataset-dir "$SYNTHETIC_DATASET_DIR"
echo "Synthetic task catalog: $SYNTHETIC_TASKS_PATH"
