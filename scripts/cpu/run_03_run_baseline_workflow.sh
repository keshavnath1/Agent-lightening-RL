#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export WORKSPACE_DIR="${WORKSPACE_DIR:-$REPO_DIR}"

: "${REAL_TASKS_PATH:=$WORKSPACE_DIR/data/real_openml/tasks.jsonl}"
: "${SYNTHETIC_TASKS_PATH:=$REAL_TASKS_PATH}"  # backwards-compatible alias
: "${BASELINE_TRAJECTORY_DIR:=$WORKSPACE_DIR/trajectories/baseline}"
: "${POLICY_VERSION:=baseline}"
: "${TASK_LIMIT:=10}"
: "${ROLLOUTS_PER_TASK:=1}"
mkdir -p "$BASELINE_TRAJECTORY_DIR"
python -m src.agents.supervisor --tasks "$SYNTHETIC_TASKS_PATH" --output-dir "$BASELINE_TRAJECTORY_DIR" --policy-version "$POLICY_VERSION" --limit "$TASK_LIMIT" --rollouts "$ROLLOUTS_PER_TASK"
