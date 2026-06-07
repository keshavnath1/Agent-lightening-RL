#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export WORKSPACE_DIR="${WORKSPACE_DIR:-$REPO_DIR}"

: "${GROUPED_ROLLOUTS:=$WORKSPACE_DIR/data/grpo/grouped_rollouts.jsonl}"
: "${RULER_SCORED_OUTPUT:=$WORKSPACE_DIR/data/grpo/ruler_scored_groups.jsonl}"
: "${RULER_SCORING_REPORT:=$WORKSPACE_DIR/reports/ruler_vllm_scoring_summary.md}"
: "${RULER_MODE:=vllm_judge}"
: "${RULER_ALPHA:=0.7}"
: "${RULER_BETA:=0.3}"
: "${RULER_JUDGE_MODEL:=Qwen/Qwen2.5-3B-Instruct}"
: "${RULER_JUDGE_BASE_URL:=http://127.0.0.1:8001/v1}"
: "${RULER_JUDGE_API_KEY:=EMPTY}"
: "${TRAINER:=trl_grpo}"
: "${REWARD_MODE:=ruler_relative}"
: "${POLICY_OUTPUT_DIR:=$WORKSPACE_DIR/checkpoints/qwen25-3b-agent-lora-ruler}"
: "${MODEL_NAME:=Qwen/Qwen2.5-3B-Instruct}"
: "${NUM_GENERATIONS:=4}"
: "${RUN_TRAINING:=0}"

usage() {
  cat <<USAGE
Usage: bash scripts/gpu/run_ruler_trl_handoff.sh [--dry-run|--train]

One-step operator handoff for the RULER -> TRL path:
  1. Score grouped rollouts with vLLM-backed RULER judging only.
  2. Validate that RULER reward fields survive into TRL examples.
  3. Print the exact GPU training command, or execute it when --train is used.

Default behavior is safe dry-run only. Use RUN_TRAINING=1 or --train to launch
GPU training.
USAGE
}

case "${1:---dry-run}" in
  --help|-h)
    usage
    exit 0
    ;;
  --dry-run)
    RUN_TRAINING=0
    ;;
  --train)
    RUN_TRAINING=1
    ;;
  *)
    echo "Unknown option: $1" >&2
    usage >&2
    exit 2
    ;;
esac

mkdir -p "$(dirname "$RULER_SCORED_OUTPUT")" "$(dirname "$RULER_SCORING_REPORT")" "$POLICY_OUTPUT_DIR"

echo "=== Track 3: RULER -> TRL handoff ==="
echo "  Input grouped rollouts : $GROUPED_ROLLOUTS"
echo "  RULER output           : $RULER_SCORED_OUTPUT"
echo "  RULER mode             : $RULER_MODE"
echo "  Judge model            : $RULER_JUDGE_MODEL"
echo "  Judge base URL         : $RULER_JUDGE_BASE_URL"
echo "  Trainer                : $TRAINER"
echo "  Reward mode            : $REWARD_MODE"
echo "  Output dir             : $POLICY_OUTPUT_DIR"
echo "  RUN_TRAINING           : $RUN_TRAINING"
echo ""

if [[ ! -s "$GROUPED_ROLLOUTS" ]]; then
  echo "Grouped rollout input is missing or empty: $GROUPED_ROLLOUTS" >&2
  exit 66
fi

if [[ "$RULER_MODE" != "vllm_judge" ]]; then
  echo "RULER_MODE must be vllm_judge; no other RULER scoring mode is supported." >&2
  exit 64
fi

: "${RULER_JUDGE_BASE_URL:?Set RULER_JUDGE_BASE_URL to the local OpenAI-compatible vLLM judge endpoint}"
echo "Checking local vLLM judge health..."
if bash scripts/gpu/start_vllm_ruler_judge.sh --health; then
  echo "Judge endpoint is reachable."
else
  echo "Judge endpoint is not reachable; fail-closed vLLM RULER scoring cannot proceed." >&2
  exit 78
fi
echo ""

python -m src.rewards.apply_ruler_scores \
  --input "$GROUPED_ROLLOUTS" \
  --output "$RULER_SCORED_OUTPUT" \
  --mode "$RULER_MODE" \
  --judge-model "$RULER_JUDGE_MODEL" \
  --judge-base-url "$RULER_JUDGE_BASE_URL" \
  --judge-api-key "$RULER_JUDGE_API_KEY" \
  --alpha "$RULER_ALPHA" \
  --beta "$RULER_BETA" \
  --report "$RULER_SCORING_REPORT"

echo ""
echo "Running TRL dataset/reward dry-run against scored RULER data..."
python -m src.training.train_policy_qlora_grpo \
  --dataset "$RULER_SCORED_OUTPUT" \
  --output-dir "$POLICY_OUTPUT_DIR" \
  --model-name "$MODEL_NAME" \
  --trainer "$TRAINER" \
  --reward-mode "$REWARD_MODE" \
  --num-generations "$NUM_GENERATIONS" \
  --dry-run

echo ""
echo "Exact training command:"
cat <<CMD
GRPO_DATASET_PATH="$RULER_SCORED_OUTPUT" \\
POLICY_OUTPUT_DIR="$POLICY_OUTPUT_DIR" \\
MODEL_NAME="$MODEL_NAME" \\
TRAINER="$TRAINER" \\
REWARD_MODE="$REWARD_MODE" \\
NUM_GENERATIONS="$NUM_GENERATIONS" \\
bash scripts/gpu/run_02_train_policy_qlora_grpo.sh
CMD

if [[ "$RUN_TRAINING" == "1" ]]; then
  echo ""
  echo "RUN_TRAINING=1 requested; launching GPU training now."
  GRPO_DATASET_PATH="$RULER_SCORED_OUTPUT" \
  POLICY_OUTPUT_DIR="$POLICY_OUTPUT_DIR" \
  MODEL_NAME="$MODEL_NAME" \
  TRAINER="$TRAINER" \
  REWARD_MODE="$REWARD_MODE" \
  NUM_GENERATIONS="$NUM_GENERATIONS" \
  bash scripts/gpu/run_02_train_policy_qlora_grpo.sh
else
  echo ""
  echo "Safe gate: no GPU training was launched. Re-run with --train after reviewing the dry-run output."
fi
