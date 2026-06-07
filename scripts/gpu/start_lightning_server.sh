#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Start the Lightning Server on the GPU pod.
#
# The Lightning Server is the GPU-side bridge between the CPU agent execution
# plane and the GRPO/veRL Optimization Framework (Agent Lightning Stage 2-3).
#
# Environment variables (all optional – defaults shown):
#   LIGHTNING_SERVER_PORT       default: 19123  (matches RunPod proxy URL)
#   LIGHTNING_TRANSITIONS_DIR   default: data/grpo
#   LIGHTNING_MIN_ROLLOUTS      default: 4
#   WORKSPACE_DIR               default: /workspace/self-improving-ml-agent
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace/self-improving-ml-agent}"
cd "$WORKSPACE_DIR"

export LIGHTNING_SERVER_PORT="${LIGHTNING_SERVER_PORT:-19123}"
export LIGHTNING_TRANSITIONS_DIR="${LIGHTNING_TRANSITIONS_DIR:-${WORKSPACE_DIR}/data/grpo}"
export LIGHTNING_MIN_ROLLOUTS="${LIGHTNING_MIN_ROLLOUTS:-4}"
export LIGHTNING_CHECKPOINT_DIR="${LIGHTNING_CHECKPOINT_DIR:-${WORKSPACE_DIR}/checkpoints/qwen25-3b-agent-lora}"
export VLLM_BASE_URL="${VLLM_BASE_URL:-http://localhost:8000}"
# Set VERL_TRAIN_CMD so /api/training/trigger (trainer=verl) has a real command.
export VERL_TRAIN_CMD="${VERL_TRAIN_CMD:-bash ${WORKSPACE_DIR}/scripts/gpu/start_verl_training.sh}"

mkdir -p "$LIGHTNING_TRANSITIONS_DIR"

echo "[lightning-server] Starting on port ${LIGHTNING_SERVER_PORT}"
echo "[lightning-server] Transitions dir: ${LIGHTNING_TRANSITIONS_DIR}"
echo "[lightning-server] Min rollouts for training: ${LIGHTNING_MIN_ROLLOUTS}"
echo "[lightning-server] Checkpoint dir: ${LIGHTNING_CHECKPOINT_DIR}"
echo "[lightning-server] vLLM URL: ${VLLM_BASE_URL}"
echo "[lightning-server] veRL command: ${VERL_TRAIN_CMD}"

python -m src.training.lightning_server_app
