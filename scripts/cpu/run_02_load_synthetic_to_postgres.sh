#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export WORKSPACE_DIR="${WORKSPACE_DIR:-$REPO_DIR}"

: "${SYNTHETIC_TASKS_PATH:=$WORKSPACE_DIR/data/synthetic/tasks.jsonl}"
python -m src.synthetic_data.load_to_postgres --tasks "$SYNTHETIC_TASKS_PATH"
