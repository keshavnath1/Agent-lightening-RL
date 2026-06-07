#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# veRL training launcher – Optimization Framework backend (Agent Lightning).
#
# Runs the official veRL GRPO training job against the grouped-rollouts JSONL
# produced by the Lightning Server's /api/training/trigger endpoint.
#
# Usage
# ─────
#   # Triggered automatically by Lightning Server when MIN_ROLLOUTS is reached
#   # and VERL_TRAIN_CMD is set to this script.  Can also be run manually:
#   WORKSPACE_DIR=/workspace/self-improving-ml-agent \
#     bash scripts/gpu/start_verl_training.sh
#
# Environment variables (all optional – defaults shown)
# ─────────────────────────────────────────────────────
#   WORKSPACE_DIR            default: /workspace/self-improving-ml-agent
#   GRPO_DATASET_PATH        default: data/grpo/lightning_grouped_rollouts.jsonl
#   POLICY_TRAINING_EXAMPLES_PATH  (auto-derived from GRPO_DATASET_PATH)
#   POLICY_OUTPUT_DIR        default: checkpoints/qwen25-3b-agent-lora
#   MODEL_NAME               default: Qwen/Qwen2.5-3B-Instruct
#   VLLM_BASE_URL            default: http://localhost:8000
#   LIGHTNING_SERVER_URL     optional – notify server to reload vLLM when done
#   VERL_N_GPUS              default: 1
#   VERL_ROLLOUT_N           default: 4   (rollouts per task in veRL loop)
#   VERL_TRAIN_EPOCHS        default: 1
#   VERL_MAX_SEQ_LEN         default: 2048
#   LORA_R                   default: 16
#   LORA_ALPHA               default: 32
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace/self-improving-ml-agent}"
cd "$WORKSPACE_DIR"

# ── Paths ──────────────────────────────────────────────────────────────────────
GRPO_DATASET_PATH="${GRPO_DATASET_PATH:-${WORKSPACE_DIR}/data/grpo/lightning_grouped_rollouts.jsonl}"
POLICY_OUTPUT_DIR="${POLICY_OUTPUT_DIR:-${WORKSPACE_DIR}/checkpoints/qwen25-3b-agent-lora}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-3B-Instruct}"
VLLM_BASE_URL="${VLLM_BASE_URL:-http://localhost:8000}"
LIGHTNING_SERVER_URL="${LIGHTNING_SERVER_URL:-}"

# ── Hyper-params ───────────────────────────────────────────────────────────────
VERL_N_GPUS="${VERL_N_GPUS:-1}"
VERL_ROLLOUT_N="${VERL_ROLLOUT_N:-4}"
VERL_TRAIN_EPOCHS="${VERL_TRAIN_EPOCHS:-1}"
VERL_MAX_SEQ_LEN="${VERL_MAX_SEQ_LEN:-2048}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"

mkdir -p "$POLICY_OUTPUT_DIR"

echo "[verl-train] Dataset    : $GRPO_DATASET_PATH"
echo "[verl-train] Output dir : $POLICY_OUTPUT_DIR"
echo "[verl-train] Model      : $MODEL_NAME"
echo "[verl-train] GPUs       : $VERL_N_GPUS"

# ── Check dataset exists ───────────────────────────────────────────────────────
if [[ ! -f "$GRPO_DATASET_PATH" ]]; then
  echo "[verl-train] ERROR: grouped-rollouts file not found: $GRPO_DATASET_PATH"
  echo "[verl-train] Run the Lightning Server and collect rollouts first, or"
  echo "[verl-train] set GRPO_DATASET_PATH to an existing grouped-rollouts JSONL."
  exit 1
fi

# ── Try official veRL GRPO training ───────────────────────────────────────────
# veRL exposes a Python entry-point: python -m verl.trainer.main_grpo
# We pass a minimal config using its Hydra-style overrides.
#
# Strict mode (default): if veRL is not installed the script exits with a
# non-zero code so the Lightning Server surfaces the error rather than
# silently substituting a different trainer.  Set ALLOW_VERL_FALLBACK=1
# to re-enable the TRL GRPO fallback (useful for local CPU-only testing).
if python -c "import verl" 2>/dev/null; then
  echo "[verl-train] veRL found – launching verl.trainer.main_grpo"

  VERL_CONFIG_OVERRIDES=(
    "data.train_files=${GRPO_DATASET_PATH}"
    "data.val_files=${GRPO_DATASET_PATH}"
    "actor_rollout_ref.model.path=${MODEL_NAME}"
    "actor_rollout_ref.model.lora_rank=${LORA_R}"
    "actor_rollout_ref.model.lora_alpha=${LORA_ALPHA}"
    "actor_rollout_ref.rollout.n=${VERL_ROLLOUT_N}"
    "actor_rollout_ref.rollout.temperature=0.8"
    "actor_rollout_ref.actor.optim.lr=1e-6"
    "critic.optim.lr=1e-5"
    "algorithm.kl_ctrl.kl_coef=0.001"
    "trainer.total_epochs=${VERL_TRAIN_EPOCHS}"
    "trainer.n_gpus_per_node=${VERL_N_GPUS}"
    "trainer.save_freq=1"
    "trainer.default_local_dir=${POLICY_OUTPUT_DIR}"
    "trainer.project_name=self-improving-ml-agent"
    "trainer.experiment_name=grpo-$(date +%Y%m%d-%H%M%S)"
    "rollout_manager.max_prompt_length=${VERL_MAX_SEQ_LEN}"
    "rollout_manager.max_response_length=${VERL_MAX_SEQ_LEN}"
    "reward_model.reward_manager=self_improving_ml"
  )

  python -m verl.trainer.main_grpo "${VERL_CONFIG_OVERRIDES[@]}"
  TRAIN_EXIT=$?

else
  if [[ "${ALLOW_VERL_FALLBACK:-0}" == "1" ]]; then
    echo "[verl-train] WARNING: veRL not installed – falling back to TRL GRPOTrainer (ALLOW_VERL_FALLBACK=1)"
    # Invoke the existing train_policy_qlora_grpo.py TRL GRPO fallback
    # on the grouped-rollouts format.
    python -m src.training.train_policy_qlora_grpo \
      --dataset    "$GRPO_DATASET_PATH" \
      --output-dir "$POLICY_OUTPUT_DIR" \
      --model-name "$MODEL_NAME" \
      --trainer    trl_grpo \
      --lora-r     "$LORA_R" \
      --lora-alpha "$LORA_ALPHA" \
      --epochs     "$VERL_TRAIN_EPOCHS"
    TRAIN_EXIT=$?
  else
    echo "[verl-train] ERROR: veRL is not installed and ALLOW_VERL_FALLBACK is not set to 1."
    echo "[verl-train] Install veRL on the GPU pod:  pip install verl"
    echo "[verl-train] Or set ALLOW_VERL_FALLBACK=1 to use TRL GRPO as a substitute."
    exit 1
  fi
fi

# ── Hot-reload vLLM adapter after successful training ─────────────────────────
if [[ $TRAIN_EXIT -eq 0 ]]; then
  echo "[verl-train] Training complete – reloading vLLM adapter at $VLLM_BASE_URL"
  curl -s -X POST "${VLLM_BASE_URL}/v1/load_lora_adapter" \
    -H 'Content-Type: application/json' \
    -d "{\"lora_name\": \"agent_adapter\", \"lora_path\": \"${POLICY_OUTPUT_DIR}\"}" \
    && echo "[verl-train] vLLM reload request sent" \
    || echo "[verl-train] WARNING: vLLM reload request failed (non-fatal)"

  # Also notify Lightning Server so its /health endpoint shows updated status
  if [[ -n "$LIGHTNING_SERVER_URL" ]]; then
    curl -s -X POST "${LIGHTNING_SERVER_URL}/api/inference/reload" \
      -H 'Content-Type: application/json' \
      -d "{\"adapter_path\": \"${POLICY_OUTPUT_DIR}\"}" \
      && echo "[verl-train] Lightning Server reload acknowledged" \
      || echo "[verl-train] WARNING: Lightning Server notify failed (non-fatal)"
  fi
else
  echo "[verl-train] ERROR: training exited with code $TRAIN_EXIT – skipping reload"
  exit $TRAIN_EXIT
fi

echo "[verl-train] Done. New LoRA adapter: $POLICY_OUTPUT_DIR"
