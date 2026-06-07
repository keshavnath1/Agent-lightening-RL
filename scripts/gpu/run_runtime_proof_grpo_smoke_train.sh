#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export WORKSPACE_DIR="${WORKSPACE_DIR:-$REPO_DIR}"

: "${RUNTIME_PROOF_ID:=runtime_proof_$(date -u +%Y%m%d_%H%M%S)}"
: "${RUNTIME_PROOF_DIR:=$WORKSPACE_DIR/reports/runtime_proof/$RUNTIME_PROOF_ID}"
: "${GRPO_DATASET_PATH:=$WORKSPACE_DIR/data/grpo/runtime_proof/${RUNTIME_PROOF_ID}_ruler_scored_groups_vllm_no_fallback.jsonl}"
: "${RUNTIME_PROOF_TRAIN_GROUP_LIMIT:=2}"
: "${POLICY_OUTPUT_DIR:=$WORKSPACE_DIR/checkpoints/qwen25-3b-agent-lora-ruler-smoke-$RUNTIME_PROOF_ID}"
: "${MODEL_NAME:=Qwen/Qwen2.5-3B-Instruct}"
: "${TRAINER:=trl_grpo}"
: "${REWARD_MODE:=ruler_relative}"
: "${NUM_GENERATIONS:=2}"
: "${TRAIN_EPOCHS:=1}"
: "${BATCH_SIZE:=1}"
: "${GRADIENT_ACCUMULATION_STEPS:=1}"
: "${LEARNING_RATE:=0.00005}"
: "${LORA_R:=8}"
: "${LORA_ALPHA:=16}"
: "${REQUIRE_CUDA:=1}"
: "${GRPO_SMOKE_TIMEOUT_SECONDS:=3600}"

mkdir -p "$RUNTIME_PROOF_DIR" "$POLICY_OUTPUT_DIR"
LIMITED_DATASET="$RUNTIME_PROOF_DIR/grpo_smoke_train_dataset.jsonl"
TRAIN_LOG="$RUNTIME_PROOF_DIR/grpo_smoke_train.log"
ASSERTION_JSON="$RUNTIME_PROOF_DIR/grpo_smoke_checkpoint_assertion.json"
PROOF_REPORT="$RUNTIME_PROOF_DIR/runtime_proof_grpo_smoke_train.md"

if [[ "$TRAINER" != "trl_grpo" ]]; then
  echo "This proof wrapper requires TRAINER=trl_grpo; got '$TRAINER'." >&2
  exit 64
fi
if [[ "$REWARD_MODE" != "ruler_relative" ]]; then
  echo "This proof wrapper requires REWARD_MODE=ruler_relative; got '$REWARD_MODE'." >&2
  exit 64
fi
if [[ ! -s "$GRPO_DATASET_PATH" ]]; then
  echo "GRPO dataset is missing or empty: $GRPO_DATASET_PATH" >&2
  echo "Run scripts/gpu/run_runtime_proof_vllm_judge_ruler.sh first or set GRPO_DATASET_PATH to a scored RULER JSONL." >&2
  exit 66
fi

if [[ "$REQUIRE_CUDA" == "1" ]]; then
  python - <<'PY'
try:
    import torch
except Exception as exc:
    raise SystemExit(f'PyTorch import failed; GPU training environment is not ready: {exc}')
if not torch.cuda.is_available():
    raise SystemExit('CUDA is not available. Refusing to run real GRPO training in this environment.')
print(f'CUDA available: {torch.cuda.get_device_name(0)}')
PY
fi

python - "$GRPO_DATASET_PATH" "$LIMITED_DATASET" "$RUNTIME_PROOF_TRAIN_GROUP_LIMIT" <<'PY'
import json
import sys
from pathlib import Path
src = Path(sys.argv[1])
dst = Path(sys.argv[2])
limit = int(sys.argv[3])
if limit <= 0:
    raise SystemExit('RUNTIME_PROOF_TRAIN_GROUP_LIMIT must be positive')
dst.parent.mkdir(parents=True, exist_ok=True)
count = 0
fallback_count = 0
with src.open('r', encoding='utf-8') as f_in, dst.open('w', encoding='utf-8') as f_out:
    for line in f_in:
        if not line.strip():
            continue
        row = json.loads(line)
        for traj in row.get('judged_trajectories') or []:
            if traj.get('ruler_fallback_used'):
                fallback_count += 1
        f_out.write(json.dumps(row, ensure_ascii=False) + '\n')
        count += 1
        if count >= limit:
            break
if count == 0:
    raise SystemExit(f'No groups copied from {src}')
if fallback_count:
    raise SystemExit(f'Refusing GRPO smoke train because the bounded dataset contains {fallback_count} fallback-ranked trajectories')
print(f'Prepared bounded GRPO smoke dataset with {count} group(s): {dst}')
PY

set +e
timeout "$GRPO_SMOKE_TIMEOUT_SECONDS" env \
  GRPO_DATASET_PATH="$LIMITED_DATASET" \
  POLICY_OUTPUT_DIR="$POLICY_OUTPUT_DIR" \
  CHECKPOINT_DIR="$POLICY_OUTPUT_DIR" \
  MODEL_NAME="$MODEL_NAME" \
  TRAINER="trl_grpo" \
  REWARD_MODE="ruler_relative" \
  NUM_GENERATIONS="$NUM_GENERATIONS" \
  TRAIN_EPOCHS="$TRAIN_EPOCHS" \
  BATCH_SIZE="$BATCH_SIZE" \
  GRADIENT_ACCUMULATION_STEPS="$GRADIENT_ACCUMULATION_STEPS" \
  LEARNING_RATE="$LEARNING_RATE" \
  LORA_R="$LORA_R" \
  LORA_ALPHA="$LORA_ALPHA" \
  bash scripts/gpu/run_02_train_policy_qlora_grpo.sh >"$TRAIN_LOG" 2>&1
train_rc=$?
set -e
if [[ "$train_rc" -ne 0 ]]; then
  echo "Bounded TRL GRPO smoke training failed with exit code $train_rc. See $TRAIN_LOG" >&2
  exit "$train_rc"
fi

python - "$POLICY_OUTPUT_DIR" "$ASSERTION_JSON" "$LIMITED_DATASET" <<'PY'
import json
import sys
from pathlib import Path
ckpt = Path(sys.argv[1])
out = Path(sys.argv[2])
dataset = Path(sys.argv[3])
metadata_path = ckpt / 'adapter_metadata.json'
examples_path = ckpt / 'policy_training_examples.jsonl'
artifact_candidates = [
    ckpt / 'adapter_model.safetensors',
    ckpt / 'adapter_model.bin',
    ckpt / 'pytorch_model.bin',
    ckpt / 'model.safetensors',
]
summary = {
    'checkpoint_dir': str(ckpt),
    'dataset': str(dataset),
    'metadata_path': str(metadata_path),
    'examples_path': str(examples_path),
    'metadata_present': metadata_path.exists(),
    'examples_present': examples_path.exists(),
    'adapter_config_present': (ckpt / 'adapter_config.json').exists(),
    'adapter_weight_files': [str(p) for p in artifact_candidates if p.exists()],
    'assertion': 'pending',
}
metadata = {}
if metadata_path.exists():
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
summary['trainer'] = metadata.get('trainer')
summary['adapter_status'] = metadata.get('adapter_status')
summary['training_method'] = metadata.get('training_method')
summary['fallback_used'] = metadata.get('fallback_used')
summary['num_examples'] = metadata.get('num_examples')
if (
    summary['metadata_present']
    and summary['adapter_status'] == 'trained'
    and summary['trainer'] == 'trl_grpo'
    and summary['fallback_used'] is False
    and summary['examples_present']
    and summary['adapter_config_present']
    and summary['adapter_weight_files']
):
    summary['assertion'] = 'passed_reloadable_trl_grpo_checkpoint_present'
else:
    summary['assertion'] = 'failed_checkpoint_artifact_gate'
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
if summary['assertion'] != 'passed_reloadable_trl_grpo_checkpoint_present':
    raise SystemExit(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
PY

cat > "$PROOF_REPORT" <<EOF
# Runtime Proof: Bounded TRL GRPO Smoke Training

This proof run executed a bounded **TRL GRPO** training pass with **ruler_relative** reward on a small no-fallback RULER-scored dataset. The wrapper refuses to train if CUDA is unavailable, if the input contains RULER fallback rows, or if checkpoint artifacts are missing after training.

| Evidence Item | Path |
|---|---|
| Bounded GRPO dataset | \`$LIMITED_DATASET\` |
| Training log | \`$TRAIN_LOG\` |
| Checkpoint directory | \`$POLICY_OUTPUT_DIR\` |
| Checkpoint assertion JSON | \`$ASSERTION_JSON\` |

The assertion passed only if the checkpoint metadata reported **trainer=trl_grpo**, **adapter_status=trained**, **fallback_used=false**, and both adapter configuration and adapter weight files were present.
EOF

echo "Bounded TRL GRPO smoke-training proof complete: $PROOF_REPORT"
