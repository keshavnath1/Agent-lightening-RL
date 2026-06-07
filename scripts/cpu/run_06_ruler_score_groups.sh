#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"

: "${RULER_MODE:=vllm_judge}"
if [[ "$RULER_MODE" != "vllm_judge" ]]; then
  echo "RULER_MODE must be vllm_judge; no other RULER scoring mode is supported." >&2
  exit 64
fi
: "${RULER_JUDGE_BASE_URL:?Set RULER_JUDGE_BASE_URL to the local OpenAI-compatible vLLM judge endpoint, for example http://127.0.0.1:8001/v1}"

python -m src.rewards.apply_ruler_scores \
  --input "${RULER_INPUT:-data/grpo/grouped_rollouts.jsonl}" \
  --output "${RULER_OUTPUT:-data/grpo/ruler_scored_groups.jsonl}" \
  --mode vllm_judge \
  --judge-model "${RULER_JUDGE_MODEL:-Qwen/Qwen2.5-3B-Instruct}" \
  --judge-base-url "$RULER_JUDGE_BASE_URL" \
  --judge-api-key "${RULER_JUDGE_API_KEY:-EMPTY}" \
  --alpha "${RULER_ALPHA:-0.7}" \
  --beta "${RULER_BETA:-0.3}" \
  --report "${RULER_REPORT:-reports/ruler_scoring_summary.md}"
