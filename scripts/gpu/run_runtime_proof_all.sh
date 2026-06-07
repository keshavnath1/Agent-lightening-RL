#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export WORKSPACE_DIR="${WORKSPACE_DIR:-$REPO_DIR}"

: "${RUNTIME_PROOF_ID:=runtime_proof_$(date -u +%Y%m%d_%H%M%S)}"
: "${RUNTIME_PROOF_DIR:=$WORKSPACE_DIR/reports/runtime_proof/$RUNTIME_PROOF_ID}"
: "${RUN_RUNTIME_PROOF_RULER:=1}"
: "${RUN_RUNTIME_PROOF_GRPO_SMOKE:=1}"
: "${RUN_RUNTIME_PROOF_COMPARE:=1}"
: "${RULER_SCORED_OUTPUT:=$WORKSPACE_DIR/data/grpo/runtime_proof/${RUNTIME_PROOF_ID}_ruler_scored_groups_vllm_no_fallback.jsonl}"
: "${POLICY_OUTPUT_DIR:=$WORKSPACE_DIR/checkpoints/qwen25-3b-agent-lora-ruler-smoke-$RUNTIME_PROOF_ID}"
: "${BENCHMARK_OUTPUT_DIR:=$WORKSPACE_DIR/trajectories/runtime_proof/${RUNTIME_PROOF_ID}_endpoint_benchmark}"
: "${BENCHMARK_REPORT:=$RUNTIME_PROOF_DIR/baseline_vs_tuned_checkpoint_reload_heldout.md}"

mkdir -p "$RUNTIME_PROOF_DIR"
MASTER_LOG="$RUNTIME_PROOF_DIR/runtime_proof_all.log"
INDEX_REPORT="$RUNTIME_PROOF_DIR/README.md"

run_step() {
  local name="$1"
  shift
  echo "===== BEGIN $name $(date -u +%Y-%m-%dT%H:%M:%SZ) =====" | tee -a "$MASTER_LOG"
  set +e
  "$@" 2>&1 | tee -a "$MASTER_LOG"
  local rc=${PIPESTATUS[0]}
  set -e
  echo "===== END $name exit_code=$rc $(date -u +%Y-%m-%dT%H:%M:%SZ) =====" | tee -a "$MASTER_LOG"
  if [[ "$rc" -ne 0 ]]; then
    exit "$rc"
  fi
}

if [[ "$RUN_RUNTIME_PROOF_RULER" == "1" ]]; then
  run_step "live_vllm_ruler_no_fallback" env \
    RUNTIME_PROOF_ID="$RUNTIME_PROOF_ID" \
    RUNTIME_PROOF_DIR="$RUNTIME_PROOF_DIR" \
    RULER_SCORED_OUTPUT="$RULER_SCORED_OUTPUT" \
    bash scripts/gpu/run_runtime_proof_vllm_judge_ruler.sh
fi

if [[ "$RUN_RUNTIME_PROOF_GRPO_SMOKE" == "1" ]]; then
  run_step "bounded_trl_grpo_smoke_train" env \
    RUNTIME_PROOF_ID="$RUNTIME_PROOF_ID" \
    RUNTIME_PROOF_DIR="$RUNTIME_PROOF_DIR" \
    GRPO_DATASET_PATH="$RULER_SCORED_OUTPUT" \
    POLICY_OUTPUT_DIR="$POLICY_OUTPUT_DIR" \
    bash scripts/gpu/run_runtime_proof_grpo_smoke_train.sh
fi

if [[ "$RUN_RUNTIME_PROOF_COMPARE" == "1" ]]; then
  run_step "checkpoint_reload_heldout_compare" env \
    RUNTIME_PROOF_ID="$RUNTIME_PROOF_ID" \
    RUNTIME_PROOF_DIR="$RUNTIME_PROOF_DIR" \
    RULER_SCORED_OUTPUT="$RULER_SCORED_OUTPUT" \
    POLICY_OUTPUT_DIR="$POLICY_OUTPUT_DIR" \
    BENCHMARK_OUTPUT_DIR="$BENCHMARK_OUTPUT_DIR" \
    BENCHMARK_REPORT="$BENCHMARK_REPORT" \
    bash scripts/gpu/run_runtime_proof_checkpoint_reload_compare.sh
fi

cat > "$INDEX_REPORT" <<EOF
# Runtime Proof Bundle: $RUNTIME_PROOF_ID

This directory is the evidence bundle for the runtime-proof path. The orchestrator keeps the run bounded by default while proving the three required gaps: live vLLM RULER judging without fallback, real TRL GRPO smoke training that writes a reloadable checkpoint, and checkpoint reload through live policy endpoints followed by a held-out baseline-vs-tuned comparison.

| Proof Step | Primary Evidence |
|---|---|
| Live vLLM RULER no-fallback scoring | \`runtime_proof_vllm_judge_ruler.md\`, \`ruler_no_fallback_assertion.json\` |
| Bounded TRL GRPO smoke training | \`runtime_proof_grpo_smoke_train.md\`, \`grpo_smoke_checkpoint_assertion.json\` |
| Checkpoint reload plus held-out comparison | \`runtime_proof_checkpoint_reload_compare.md\`, \`checkpoint_reload_compare_assertion.json\` |
| Combined console log | `runtime_proof_all.log` |

The generated checkpoint path is \`$POLICY_OUTPUT_DIR\`, the RULER scored JSONL path is \`$RULER_SCORED_OUTPUT\`, and the benchmark report path is \`$BENCHMARK_REPORT\`.
EOF

echo "Runtime proof bundle complete: $INDEX_REPORT"
