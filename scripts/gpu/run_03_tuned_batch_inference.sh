#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export WORKSPACE_DIR="${WORKSPACE_DIR:-$REPO_DIR}"

: "${REAL_TASKS_PATH:=$WORKSPACE_DIR/data/real_openml/tasks.jsonl}"
: "${SYNTHETIC_TASKS_PATH:=$REAL_TASKS_PATH}"  # backwards-compatible alias
: "${TUNED_POLICY_JOBS:=$WORKSPACE_DIR/data/real_openml/tuned_policy_jobs.jsonl}"
python -m src.inference.batch_inference --tasks "$SYNTHETIC_TASKS_PATH" --output "$TUNED_POLICY_JOBS" --policy-version rl_tuned
echo "Tuned policy decision jobs written to $TUNED_POLICY_JOBS"
