#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export WORKSPACE_DIR="${WORKSPACE_DIR:-$REPO_DIR}"

: "${VLLM_MODEL_ID:=Qwen/Qwen2.5-Coder-32B-Instruct}"
: "${BASELINE_PORT:=8000}"
python -m src.inference.serve_vllm --model "$VLLM_MODEL_ID" --port "$BASELINE_PORT" --served-model-name "${MODEL_NAME:-qwen2.5-coder-32b-instruct}"
