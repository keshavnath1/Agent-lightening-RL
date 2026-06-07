#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export WORKSPACE_DIR="${WORKSPACE_DIR:-$REPO_DIR}"

# ── Core configuration ─────────────────────────────────────────────────────
: "${MODEL_NAME:=Qwen/Qwen2.5-3B-Instruct}"
: "${GRPO_DATASET_PATH:=$WORKSPACE_DIR/data/grpo/grouped_rollouts.jsonl}"
: "${POLICY_OUTPUT_DIR:=$WORKSPACE_DIR/checkpoints/qwen25-3b-agent-lora}"
# Alias: keep old CHECKPOINT_DIR env-var working
: "${CHECKPOINT_DIR:=$POLICY_OUTPUT_DIR}"

# ── Trainer selection ──────────────────────────────────────────────────────
# Supported trainers: trl_grpo | verl | agent_lightning_official | official_art_ruler
: "${TRAINER:=trl_grpo}"

# ── Reward mode (json_validity | workflow_policy | trajectory_reward | hybrid | ruler_relative)
: "${REWARD_MODE:=hybrid}"

# ── GRPO-specific ──────────────────────────────────────────────────────────
: "${NUM_GENERATIONS:=4}"

# ── Official ART/RULER-specific ──────────────────────────────────────────
: "${JUDGE_MODEL:=openai/o4-mini}"
: "${ROLLOUTS_PER_GROUP:=4}"
: "${GROUPS_PER_STEP:=2}"
: "${MAX_STEPS:=5}"

# ── Training hyperparameters ───────────────────────────────────────────────
: "${TRAIN_EPOCHS:=1}"
: "${BATCH_SIZE:=4}"
: "${GRADIENT_ACCUMULATION_STEPS:=2}"
: "${LEARNING_RATE:=0.0002}"
: "${LORA_R:=16}"
: "${LORA_ALPHA:=32}"

# ── Agent Lightning official mode ──────────────────────────────────────────
: "${AGENT_LIGHTNING_TASKS:=$WORKSPACE_DIR/data/synthetic/tasks.jsonl}"
: "${AGENT_LIGHTNING_POLICY_VERSION:=agent_lightning_policy}"
: "${AGENT_LIGHTNING_LIMIT:=}"

mkdir -p "$CHECKPOINT_DIR"

echo "=== Track B: Policy Optimization ==="
echo "  Trainer      : $TRAINER"
echo "  Model        : $MODEL_NAME"
echo "  Reward mode  : $REWARD_MODE"
echo "  Dataset      : $GRPO_DATASET_PATH"
echo "  Output dir   : $CHECKPOINT_DIR"
echo "  Generations  : $NUM_GENERATIONS"
if [[ "$TRAINER" == "official_art_ruler" ]]; then
  echo "  Official ART : requested via TRAINER=official_art_ruler"
  echo "  Judge model  : $JUDGE_MODEL"
  echo "  Rollouts/group: $ROLLOUTS_PER_GROUP"
  echo "  Groups/step  : $GROUPS_PER_STEP"
  echo "  RULER mode  : official RULER only; no heuristic fallback"
  if ! python -m src.training.art_availability >/tmp/official_art_ruler_status.json 2>/tmp/official_art_ruler_status.err; then
    echo "Official ART/RULER unavailable. Install: uv pip install -U \"openpipe-art[backend,langgraph]>=0.4.9\"" >&2
    echo "Status:" >&2
    cat /tmp/official_art_ruler_status.json >&2 || true
    cat /tmp/official_art_ruler_status.err >&2 || true
    exit 78
  fi
fi
echo ""

cmd=(
  python -m src.training.train_policy_qlora_grpo
  --dataset                    "${GRPO_DATASET_PATH}"
  --output-dir                 "${CHECKPOINT_DIR}"
  --model-name                 "${MODEL_NAME}"
  --trainer                    "${TRAINER}"
  --reward-mode                "${REWARD_MODE}"
  --num-generations            "${NUM_GENERATIONS}"
  --epochs                     "${TRAIN_EPOCHS}"
  --batch-size                 "${BATCH_SIZE}"
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}"
  --learning-rate              "${LEARNING_RATE}"
  --lora-r                     "${LORA_R}"
  --lora-alpha                 "${LORA_ALPHA}"
  --agent-lightning-tasks      "${AGENT_LIGHTNING_TASKS}"
  --agent-lightning-policy-version "${AGENT_LIGHTNING_POLICY_VERSION}"
  --judge-model               "${JUDGE_MODEL}"
  --rollouts-per-group        "${ROLLOUTS_PER_GROUP}"
  --groups-per-step           "${GROUPS_PER_STEP}"
  --max-steps                 "${MAX_STEPS}"
)

if [[ -n "${AGENT_LIGHTNING_LIMIT:-}" ]]; then
  cmd+=(--agent-lightning-limit "$AGENT_LIGHTNING_LIMIT")
fi
if [[ -n "${VERL_TRAIN_CMD:-}" ]]; then
  cmd+=(--verl-command "$VERL_TRAIN_CMD")
fi

"${cmd[@]}"
echo ""
echo "Policy training complete. Output: $CHECKPOINT_DIR"
