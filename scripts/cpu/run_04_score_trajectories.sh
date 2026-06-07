#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export WORKSPACE_DIR="${WORKSPACE_DIR:-$REPO_DIR}"

: "${TRAJECTORY_INPUT_DIR:=$WORKSPACE_DIR/trajectories/baseline}"
: "${SCORED_TRAJECTORY_DIR:=$WORKSPACE_DIR/trajectories/scored}"
mkdir -p "$SCORED_TRAJECTORY_DIR"
python -m src.rewards.scorer --input-dir "$TRAJECTORY_INPUT_DIR" --output-dir "$SCORED_TRAJECTORY_DIR"
