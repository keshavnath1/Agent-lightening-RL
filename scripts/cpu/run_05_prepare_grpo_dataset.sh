#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export WORKSPACE_DIR="${WORKSPACE_DIR:-$REPO_DIR}"

: "${SCORED_TRAJECTORY_DIR:=$WORKSPACE_DIR/trajectories/scored}"
: "${GRPO_DATASET_PATH:=$WORKSPACE_DIR/data/grpo/grouped_rollouts.jsonl}"
mkdir -p "$(dirname "$GRPO_DATASET_PATH")"
python -m src.training.prepare_grpo_dataset --input-dir "$SCORED_TRAJECTORY_DIR" --output "$GRPO_DATASET_PATH"
