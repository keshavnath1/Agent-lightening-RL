#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/workspace/self-improving-ml-agent}"
if [[ ! -d "$REPO_DIR" ]]; then
  REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"

INPUT_DIR="${INPUT_DIR:-trajectories/scored}"
OUTPUT_PATH="${OUTPUT_PATH:-data/grpo/agent_lightning_transitions.jsonl}"

python -m src.training.agent_lightning_export \
  --input-dir "$INPUT_DIR" \
  --output "$OUTPUT_PATH"
