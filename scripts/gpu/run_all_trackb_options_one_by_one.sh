#!/usr/bin/env bash
# ============================================================================
# run_all_trackb_options_one_by_one.sh
#
# Runs every Track-B Policy Optimisation option sequentially:
#   Option 1 – QLoRA SFT                (trl_grpo variant without RL)
#   Option 2 – TRL GRPO  (hybrid)       reward_mode=hybrid
#   Option 3 – TRL GRPO  (workflow)     reward_mode=workflow_policy
#   Option 4 – veRL handoff             (skipped if verl not installed)
#   Option 5 – Agent Lightning Official (skipped if agentlightning not installed)
#   Option 6 – TRL GRPO + RULER relative reward
#   Option 7 – Official ART + RULER (skipped if openpipe-art/RULER unavailable)
#
# Each option writes to its own checkpoint directory so results are
# independent and comparable.
#
# Usage:
#   bash scripts/gpu/run_all_trackb_options_one_by_one.sh [--dry-run] [--skip-preflight]
#
# Environment overrides:
#   DATASET_PATH           grouped rollouts JSONL (default: data/grpo/grouped_rollouts.jsonl)
#   MODEL_NAME             base model HF id        (default: Qwen/Qwen2.5-3B-Instruct)
#   NUM_GENERATIONS        GRPO group size          (default: 4)
#   BATCH_SIZE             per-device train batch   (default: 1)
#   GRAD_ACCUM             gradient accumulation    (default: 8)
#   LOG_DIR                log root                 (default: logs/trackb_option_runs)
#   VERL_TRAIN_CMD         operator-supplied veRL command  (required for option 4)
#   TASKS_PATH             tasks JSONL for agent-lightning  (default: data/synthetic/tasks.jsonl)
#   RULER_DATASET_PATH     RULER-scored groups JSONL (default: data/grpo/ruler_scored_groups.jsonl)
# ============================================================================
set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────────────
DATASET_PATH="${DATASET_PATH:-data/grpo/grouped_rollouts.jsonl}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-3B-Instruct}"
NUM_GENERATIONS="${NUM_GENERATIONS:-4}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LOG_DIR="${LOG_DIR:-logs/trackb_option_runs}"
TASKS_PATH="${TASKS_PATH:-data/synthetic/tasks.jsonl}"
RULER_DATASET_PATH="${RULER_DATASET_PATH:-data/grpo/ruler_scored_groups.jsonl}"

DRY_RUN=false
SKIP_PREFLIGHT=false
for arg in "$@"; do
  case "$arg" in
    --dry-run)        DRY_RUN=true ;;
    --skip-preflight) SKIP_PREFLIGHT=true ;;
  esac
done

# ── Helpers ───────────────────────────────────────────────────────────────────
BOLD="\033[1m"; GREEN="\033[32m"; YELLOW="\033[33m"; RED="\033[31m"; RESET="\033[0m"

ts() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
header() { echo -e "\n${BOLD}══════════════════════════════════════════${RESET}"; echo -e "${BOLD}  $1${RESET}"; echo -e "${BOLD}══════════════════════════════════════════${RESET}"; }
ok()   { echo -e "$(ts) ${GREEN}[OK]${RESET}    $*"; }
warn() { echo -e "$(ts) ${YELLOW}[WARN]${RESET}  $*"; }
fail() { echo -e "$(ts) ${RED}[FAIL]${RESET}  $*"; }
skip() { echo -e "$(ts) ${YELLOW}[SKIP]${RESET}  $*"; }

# Track per-option results for the summary table
declare -A OPTION_STATUS
declare -A OPTION_CKPT
declare -A OPTION_ELAPSED

WORKSPACE_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$WORKSPACE_ROOT"

mkdir -p "$LOG_DIR"
SUMMARY_LOG="$LOG_DIR/run_summary_$(date -u '+%Y%m%dT%H%M%SZ').txt"

# ── Preflight checks ──────────────────────────────────────────────────────────
if [[ "$SKIP_PREFLIGHT" == "false" ]]; then
  header "Preflight checks"

  if [[ ! -f "$DATASET_PATH" ]]; then
    fail "Dataset not found: $DATASET_PATH"
    fail "Run scripts/gpu/run_01_collect_rollouts.sh first, or set DATASET_PATH."
    exit 1
  fi
  ok "Dataset found: $DATASET_PATH"

  NLINES=$(wc -l < "$DATASET_PATH")
  ok "Dataset has $NLINES task group(s)"

  # Dry-run sanity check with all backends that don't need GPU
  echo "Running dry-run sanity check …"
  python -m src.training.train_policy_qlora_grpo \
    --dataset "$DATASET_PATH" \
    --trainer trl_grpo \
    --reward-mode hybrid \
    --num-generations "$NUM_GENERATIONS" \
    --dry-run \
    || { fail "Dry-run failed — check Python path / dependencies"; exit 1; }
  ok "Dry-run passed"
fi

if [[ "$DRY_RUN" == "true" ]]; then
  ok "Dry-run mode — skipping all training runs."
  exit 0
fi

# ── Option runner helper ──────────────────────────────────────────────────────
run_option() {
  local OPT_NUM="$1"
  local OPT_NAME="$2"
  local CKPT_DIR="$3"
  shift 3
  local CMD=("$@")

  header "Option $OPT_NUM — $OPT_NAME"
  echo "  Checkpoint → $CKPT_DIR"
  echo "  Command    → ${CMD[*]}"
  echo ""

  local LOG_FILE="$LOG_DIR/option${OPT_NUM}_$(date -u '+%H%M%SZ').log"
  local T_START=$SECONDS

  if "${CMD[@]}" 2>&1 | tee "$LOG_FILE"; then
    local ELAPSED=$(( SECONDS - T_START ))
    ok "Option $OPT_NUM finished in ${ELAPSED}s"
    OPTION_STATUS[$OPT_NUM]="PASS"
    OPTION_CKPT[$OPT_NUM]="$CKPT_DIR"
    OPTION_ELAPSED[$OPT_NUM]="${ELAPSED}s"

    # Print adapter metadata if it exists
    local META="$CKPT_DIR/adapter_metadata.json"
    if [[ -f "$META" ]]; then
      echo ""
      echo "  adapter_metadata.json:"
      python -c "import json,sys; d=json.load(open('$META')); [print('    '+k+': '+str(v)) for k,v in d.items()]" 2>/dev/null || cat "$META"
    fi
  else
    local ELAPSED=$(( SECONDS - T_START ))
    fail "Option $OPT_NUM failed after ${ELAPSED}s (log: $LOG_FILE)"
    OPTION_STATUS[$OPT_NUM]="FAIL"
    OPTION_CKPT[$OPT_NUM]="$CKPT_DIR"
    OPTION_ELAPSED[$OPT_NUM]="${ELAPSED}s (FAILED)"
  fi
}

# ── Option 1: QLoRA SFT ───────────────────────────────────────────────────────
CKPT_1="checkpoints/trackb_qlora_sft"
run_option 1 "QLoRA SFT" "$CKPT_1" \
  python -m src.training.train_policy_qlora_grpo \
    --dataset "$DATASET_PATH" \
    --output-dir "$CKPT_1" \
    --model-name "$MODEL_NAME" \
    --trainer qlora_sft \
    --batch-size "$BATCH_SIZE" \
    --gradient-accumulation-steps "$GRAD_ACCUM"

# ── Option 2: TRL GRPO – hybrid reward ───────────────────────────────────────
CKPT_2="checkpoints/trackb_trl_grpo_hybrid"
run_option 2 "TRL GRPO (hybrid reward)" "$CKPT_2" \
  python -m src.training.train_policy_qlora_grpo \
    --dataset "$DATASET_PATH" \
    --output-dir "$CKPT_2" \
    --model-name "$MODEL_NAME" \
    --trainer trl_grpo \
    --reward-mode hybrid \
    --num-generations "$NUM_GENERATIONS" \
    --batch-size "$BATCH_SIZE" \
    --gradient-accumulation-steps "$GRAD_ACCUM"

# ── Option 3: TRL GRPO – workflow_policy reward ───────────────────────────────
CKPT_3="checkpoints/trackb_trl_grpo_workflow"
run_option 3 "TRL GRPO (workflow_policy reward)" "$CKPT_3" \
  python -m src.training.train_policy_qlora_grpo \
    --dataset "$DATASET_PATH" \
    --output-dir "$CKPT_3" \
    --model-name "$MODEL_NAME" \
    --trainer trl_grpo \
    --reward-mode workflow_policy \
    --num-generations "$NUM_GENERATIONS" \
    --batch-size "$BATCH_SIZE" \
    --gradient-accumulation-steps "$GRAD_ACCUM"

# ── Option 4: veRL ────────────────────────────────────────────────────────────
header "Option 4 — veRL"
if python -c "import verl" 2>/dev/null; then
  if [[ -z "${VERL_TRAIN_CMD:-}" ]]; then
    warn "verl Python package found but VERL_TRAIN_CMD is not set."
    warn "Set VERL_TRAIN_CMD to your official veRL launch command, then re-run."
    OPTION_STATUS[4]="SKIP (VERL_TRAIN_CMD not set)"
    OPTION_CKPT[4]="n/a"
    OPTION_ELAPSED[4]="0s"
  else
    CKPT_4="checkpoints/trackb_verl"
    run_option 4 "veRL" "$CKPT_4" \
      python -m src.training.train_policy_qlora_grpo \
        --dataset "$DATASET_PATH" \
        --output-dir "$CKPT_4" \
        --model-name "$MODEL_NAME" \
        --trainer verl \
        --verl-command "$VERL_TRAIN_CMD"
  fi
else
  skip "verl not installed — skipping Option 4."
  skip "To enable: pip install verl (GPU pod) and set VERL_TRAIN_CMD."
  OPTION_STATUS[4]="SKIP (verl not installed)"
  OPTION_CKPT[4]="n/a"
  OPTION_ELAPSED[4]="0s"
fi

# ── Option 5: Agent Lightning Official ───────────────────────────────────────
header "Option 5 — Agent Lightning Official"
if python -c "import agentlightning" 2>/dev/null; then
  CKPT_5="checkpoints/trackb_agent_lightning"
  run_option 5 "Agent Lightning Official" "$CKPT_5" \
    python -m src.training.train_policy_qlora_grpo \
      --trainer agent_lightning_official \
      --output-dir "$CKPT_5" \
      --agent-lightning-tasks "$TASKS_PATH"
else
  skip "agentlightning not installed — skipping Option 5."
  skip "To enable: pip install agentlightning"
  OPTION_STATUS[5]="SKIP (agentlightning not installed)"
  OPTION_CKPT[5]="n/a"
  OPTION_ELAPSED[5]="0s"
fi

# ── Option 6: TRL GRPO + RULER relative reward ───────────────────────────────
header "Option 6 — TRL GRPO + RULER relative reward"
if [[ -f "$RULER_DATASET_PATH" ]]; then
  CKPT_6="checkpoints/trackb_trl_grpo_ruler"
  run_option 6 "TRL GRPO (RULER relative reward)" "$CKPT_6" \
    python -m src.training.train_policy_qlora_grpo \
      --dataset "$RULER_DATASET_PATH" \
      --output-dir "$CKPT_6" \
      --model-name "$MODEL_NAME" \
      --trainer trl_grpo \
      --reward-mode ruler_relative \
      --num-generations "$NUM_GENERATIONS" \
      --batch-size "$BATCH_SIZE" \
      --gradient-accumulation-steps "$GRAD_ACCUM"
else
  skip "RULER dataset not found — skipping Option 6: $RULER_DATASET_PATH"
  skip "Run: python -m src.rewards.apply_ruler_scores --input data/grpo/grouped_rollouts.jsonl --output $RULER_DATASET_PATH"
  OPTION_STATUS[6]="SKIP (RULER dataset not found)"
  OPTION_CKPT[6]="n/a"
  OPTION_ELAPSED[6]="0s"
fi

# ── Option 7: Official ART + RULER ────────────────────────────────────────────
header "Option 7 — Official ART + RULER"
if python - <<'PY'
from src.training.art_availability import check_art_available
s = check_art_available()
raise SystemExit(0 if s.get('art') and s.get('art_langgraph') and s.get('ruler') else 1)
PY
then
  CKPT_7="checkpoints/trackb_official_art_ruler"
  run_option 7 "Official ART + RULER" "$CKPT_7" \
    python -m src.training.train_policy_qlora_grpo \
      --trainer official_art_ruler \
      --output-dir "$CKPT_7" \
      --agent-lightning-tasks "$TASKS_PATH"
else
  skip "openpipe-art[backend,langgraph] or official RULER unavailable — skipping Option 7."
  skip "To enable: uv pip install -U \"openpipe-art[backend,langgraph]>=0.4.9\""
  OPTION_STATUS[7]="SKIP (official ART/RULER unavailable)"
  OPTION_CKPT[7]="n/a"
  OPTION_ELAPSED[7]="0s"
fi

# ── Summary table ─────────────────────────────────────────────────────────────
header "Run Summary"
printf "%-8s %-42s %-12s %s\n" "OPTION" "CHECKPOINT" "STATUS" "ELAPSED"
printf "%-8s %-42s %-12s %s\n" "------" "----------" "------" "-------"
for i in 1 2 3 4 5 6 7; do
  printf "%-8s %-42s %-12s %s\n" \
    "$i" \
    "${OPTION_CKPT[$i]:-n/a}" \
    "${OPTION_STATUS[$i]:-UNKNOWN}" \
    "${OPTION_ELAPSED[$i]:-}"
done

# Write summary to file
{
  echo "Track-B all-options run — $(ts)"
  printf "%-8s %-42s %-12s %s\n" "OPTION" "CHECKPOINT" "STATUS" "ELAPSED"
  for i in 1 2 3 4 5 6 7; do
    printf "%-8s %-42s %-12s %s\n" \
      "$i" \
      "${OPTION_CKPT[$i]:-n/a}" \
      "${OPTION_STATUS[$i]:-UNKNOWN}" \
      "${OPTION_ELAPSED[$i]:-}"
  done
} > "$SUMMARY_LOG"
ok "Summary written to $SUMMARY_LOG"

# ── Exit code: fail if any option failed ──────────────────────────────────────
FAILED=0
for i in 1 2 3 4 5 6 7; do
  [[ "${OPTION_STATUS[$i]:-}" == "FAIL" ]] && FAILED=1
done
exit $FAILED
