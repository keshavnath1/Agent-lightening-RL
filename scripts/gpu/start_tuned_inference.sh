#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export WORKSPACE_DIR="${WORKSPACE_DIR:-$REPO_DIR}"

: "${MODEL_NAME:=Qwen/Qwen2.5-3B-Instruct}"
: "${TUNED_PORT:=8001}"
: "${CHECKPOINT_DIR:=$WORKSPACE_DIR/checkpoints/qwen25-3b-agent-lora}"
python -m src.inference.serve_vllm --model "$MODEL_NAME" --port "$TUNED_PORT" --adapter "$CHECKPOINT_DIR"
