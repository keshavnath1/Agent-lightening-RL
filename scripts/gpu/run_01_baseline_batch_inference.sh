#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export WORKSPACE_DIR="${WORKSPACE_DIR:-$REPO_DIR}"

: "${SYNTHETIC_TASKS_PATH:=$WORKSPACE_DIR/data/synthetic/tasks.jsonl}"
: "${BASELINE_POLICY_JOBS:=$WORKSPACE_DIR/data/synthetic/baseline_policy_jobs.jsonl}"
python -m src.inference.batch_inference --tasks "$SYNTHETIC_TASKS_PATH" --output "$BASELINE_POLICY_JOBS" --policy-version baseline
echo "Baseline policy decision jobs written to $BASELINE_POLICY_JOBS"
