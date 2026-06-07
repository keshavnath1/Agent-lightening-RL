#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export WORKSPACE_DIR="${WORKSPACE_DIR:-$REPO_DIR}"

: "${RUNTIME_PROOF_ID:=runtime_proof_$(date -u +%Y%m%d_%H%M%S)}"
: "${RUNTIME_PROOF_DIR:=$WORKSPACE_DIR/reports/runtime_proof/$RUNTIME_PROOF_ID}"
: "${GROUPED_ROLLOUTS:=$WORKSPACE_DIR/data/grpo/grouped_rollouts.jsonl}"
: "${RUNTIME_PROOF_GROUP_LIMIT:=}"
: "${RULER_SCORED_OUTPUT:=$WORKSPACE_DIR/data/grpo/runtime_proof/${RUNTIME_PROOF_ID}_ruler_scored_groups_vllm_no_fallback.jsonl}"
: "${RULER_SCORING_REPORT:=$RUNTIME_PROOF_DIR/ruler_vllm_no_fallback_summary.md}"
: "${RULER_MODE:=vllm_judge}"
: "${RULER_ALPHA:=0.7}"
: "${RULER_BETA:=0.3}"
: "${RULER_JUDGE_MODEL:=Qwen/Qwen2.5-3B-Instruct}"
: "${RULER_JUDGE_BASE_URL:=http://127.0.0.1:8001/v1}"
: "${RULER_JUDGE_API_KEY:=EMPTY}"
: "${START_RULER_JUDGE:=1}"
: "${KEEP_RULER_JUDGE_SERVER:=0}"
: "${RULER_JUDGE_STARTUP_TIMEOUT_SECONDS:=900}"
: "${RULER_JUDGE_HEALTH_INTERVAL_SECONDS:=10}"

mkdir -p "$RUNTIME_PROOF_DIR" "$(dirname "$RULER_SCORED_OUTPUT")"
JUDGE_LOG="$RUNTIME_PROOF_DIR/vllm_ruler_judge_server.log"
ASSERTION_JSON="$RUNTIME_PROOF_DIR/ruler_no_fallback_assertion.json"
PROOF_REPORT="$RUNTIME_PROOF_DIR/runtime_proof_vllm_judge_ruler.md"
LIMITED_INPUT="$RUNTIME_PROOF_DIR/grouped_rollouts_runtime_limit.jsonl"
JUDGE_PID=""

health_check() {
  RULER_JUDGE_BASE_URL="$RULER_JUDGE_BASE_URL" \
  RULER_JUDGE_API_KEY="$RULER_JUDGE_API_KEY" \
  bash scripts/gpu/start_vllm_ruler_judge.sh --health >/dev/null 2>&1
}

cleanup() {
  if [[ -n "${JUDGE_PID:-}" && "$KEEP_RULER_JUDGE_SERVER" != "1" ]]; then
    if kill -0 "$JUDGE_PID" >/dev/null 2>&1; then
      echo "Stopping runtime-proof vLLM judge process $JUDGE_PID"
      kill "$JUDGE_PID" >/dev/null 2>&1 || true
      wait "$JUDGE_PID" >/dev/null 2>&1 || true
    fi
  fi
}
trap cleanup EXIT

if [[ ! -s "$GROUPED_ROLLOUTS" ]]; then
  echo "Grouped rollout input is missing or empty: $GROUPED_ROLLOUTS" >&2
  exit 66
fi

if [[ "$RULER_MODE" != "vllm_judge" ]]; then
  echo "This proof wrapper requires RULER_MODE=vllm_judge; got '$RULER_MODE'." >&2
  exit 64
fi

INPUT_FOR_SCORING="$GROUPED_ROLLOUTS"
if [[ -n "$RUNTIME_PROOF_GROUP_LIMIT" ]]; then
  python - "$GROUPED_ROLLOUTS" "$LIMITED_INPUT" "$RUNTIME_PROOF_GROUP_LIMIT" <<'PY'
import json
import sys
from pathlib import Path
src = Path(sys.argv[1])
dst = Path(sys.argv[2])
limit = int(sys.argv[3])
if limit <= 0:
    raise SystemExit('RUNTIME_PROOF_GROUP_LIMIT must be positive when set')
dst.parent.mkdir(parents=True, exist_ok=True)
count = 0
with src.open('r', encoding='utf-8') as f_in, dst.open('w', encoding='utf-8') as f_out:
    for line in f_in:
        if not line.strip():
            continue
        json.loads(line)
        f_out.write(line)
        count += 1
        if count >= limit:
            break
if count == 0:
    raise SystemExit(f'No groups copied from {src}')
print(f'Prepared bounded RULER input with {count} group(s): {dst}')
PY
  INPUT_FOR_SCORING="$LIMITED_INPUT"
fi

if health_check; then
  echo "Existing vLLM RULER judge is healthy at $RULER_JUDGE_BASE_URL"
elif [[ "$START_RULER_JUDGE" == "1" ]]; then
  echo "Starting vLLM RULER judge at $RULER_JUDGE_BASE_URL; log: $JUDGE_LOG"
  RULER_JUDGE_MODEL="$RULER_JUDGE_MODEL" \
  RULER_JUDGE_BASE_URL="$RULER_JUDGE_BASE_URL" \
  RULER_JUDGE_API_KEY="$RULER_JUDGE_API_KEY" \
  nohup bash scripts/gpu/start_vllm_ruler_judge.sh --serve >"$JUDGE_LOG" 2>&1 &
  JUDGE_PID="$!"
  deadline=$((SECONDS + RULER_JUDGE_STARTUP_TIMEOUT_SECONDS))
  until health_check; do
    if (( SECONDS >= deadline )); then
      echo "Timed out waiting for vLLM RULER judge. See $JUDGE_LOG" >&2
      exit 75
    fi
    sleep "$RULER_JUDGE_HEALTH_INTERVAL_SECONDS"
  done
  echo "vLLM RULER judge became healthy."
else
  echo "vLLM RULER judge is not healthy and START_RULER_JUDGE is not enabled." >&2
  exit 75
fi

python -m src.rewards.apply_ruler_scores \
  --input "$INPUT_FOR_SCORING" \
  --output "$RULER_SCORED_OUTPUT" \
  --mode vllm_judge \
  --judge-model "$RULER_JUDGE_MODEL" \
  --judge-base-url "$RULER_JUDGE_BASE_URL" \
  --judge-api-key "$RULER_JUDGE_API_KEY" \
  --alpha "$RULER_ALPHA" \
  --beta "$RULER_BETA" \
  --report "$RULER_SCORING_REPORT"

python - "$RULER_SCORED_OUTPUT" "$ASSERTION_JSON" "$RULER_JUDGE_BASE_URL" "$RULER_JUDGE_MODEL" <<'PY'
import json
import sys
from pathlib import Path
scored = Path(sys.argv[1])
out = Path(sys.argv[2])
base_url = sys.argv[3]
model = sys.argv[4]
if not scored.exists() or scored.stat().st_size == 0:
    raise SystemExit(f'RULER scored output missing or empty: {scored}')
summary = {
    'scored_output': str(scored),
    'judge_base_url': base_url,
    'judge_model': model,
    'groups': 0,
    'trajectories': 0,
    'fallback_ranked_trajectories': 0,
    'non_vllm_groups': 0,
    'empty_groups': 0,
    'assertion': 'pending',
}
with scored.open('r', encoding='utf-8') as f:
    for line in f:
        if not line.strip():
            continue
        row = json.loads(line)
        summary['groups'] += 1
        if row.get('ruler_mode') != 'vllm_judge':
            summary['non_vllm_groups'] += 1
        judged = row.get('judged_trajectories') or []
        if not judged:
            summary['empty_groups'] += 1
        for traj in judged:
            summary['trajectories'] += 1
            if bool(traj.get('ruler_fallback_used')):
                summary['fallback_ranked_trajectories'] += 1
if summary['groups'] <= 0 or summary['trajectories'] <= 0:
    summary['assertion'] = 'failed_no_scored_rows'
elif summary['non_vllm_groups'] or summary['empty_groups'] or summary['fallback_ranked_trajectories']:
    summary['assertion'] = 'failed_no_fallback_gate'
else:
    summary['assertion'] = 'passed_live_vllm_no_fallback'
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
if summary['assertion'] != 'passed_live_vllm_no_fallback':
    raise SystemExit(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
PY

cat > "$PROOF_REPORT" <<EOF
# Runtime Proof: Live vLLM RULER Judge With No Fallback

This proof run used the project RULER scoring path in **vllm_judge** mode and then parsed the scored JSONL to hard-fail if any trajectory was marked with **ruler_fallback_used=true**.

| Evidence Item | Path |
|---|---|
| RULER scored output | \`$RULER_SCORED_OUTPUT\` |
| RULER scoring report | \`$RULER_SCORING_REPORT\` |
| No-fallback assertion JSON | \`$ASSERTION_JSON\` |
| Judge server log | \`$JUDGE_LOG\` |

The assertion passed only if at least one group and one trajectory were scored, every group reported **ruler_mode=vllm_judge**, and the fallback-ranked trajectory count was zero.
EOF

echo "Live vLLM RULER no-fallback proof complete: $PROOF_REPORT"
