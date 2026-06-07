#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export WORKSPACE_DIR="${WORKSPACE_DIR:-$REPO_DIR}"

: "${AGENT_LIGHTNING_TASKS:=$WORKSPACE_DIR/data/synthetic/tasks.jsonl}"
: "${AGENT_LIGHTNING_OUTPUT_DIR:=$WORKSPACE_DIR/trajectories/agent_lightning_official}"
: "${AGENT_LIGHTNING_POLICY_VERSION:=agent_lightning_policy}"
: "${AGENT_LIGHTNING_METADATA_OUTPUT:=$WORKSPACE_DIR/reports/agent_lightning_official_run.json}"
: "${AGENT_LIGHTNING_LIMIT:=}"

cmd=(
  python -m src.training.agent_lightning_official_runner
  --tasks "$AGENT_LIGHTNING_TASKS"
  --output-dir "$AGENT_LIGHTNING_OUTPUT_DIR"
  --policy-version "$AGENT_LIGHTNING_POLICY_VERSION"
  --metadata-output "$AGENT_LIGHTNING_METADATA_OUTPUT"
)
if [[ -n "$AGENT_LIGHTNING_LIMIT" ]]; then
  cmd+=(--limit "$AGENT_LIGHTNING_LIMIT")
fi

"${cmd[@]}"
echo "Official Agent Lightning output: $AGENT_LIGHTNING_OUTPUT_DIR"
