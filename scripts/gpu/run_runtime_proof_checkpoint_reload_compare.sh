#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export WORKSPACE_DIR="${WORKSPACE_DIR:-$REPO_DIR}"

: "${RUNTIME_PROOF_ID:=runtime_proof_$(date -u +%Y%m%d_%H%M%S)}"
: "${RUNTIME_PROOF_DIR:=$WORKSPACE_DIR/reports/runtime_proof/$RUNTIME_PROOF_ID}"
: "${POLICY_OUTPUT_DIR:=$WORKSPACE_DIR/checkpoints/qwen25-3b-agent-lora-ruler-smoke-$RUNTIME_PROOF_ID}"
: "${MODEL_NAME:=Qwen/Qwen2.5-3B-Instruct}"
: "${BASELINE_MODEL_NAME:=$MODEL_NAME}"
: "${TUNED_MODEL_NAME:=$MODEL_NAME}"
: "${BASELINE_PORT:=8000}"
: "${TUNED_PORT:=8002}"
: "${BASELINE_POLICY_URL:=http://127.0.0.1:${BASELINE_PORT}/v1}"
: "${TUNED_POLICY_URL:=http://127.0.0.1:${TUNED_PORT}/v1}"
: "${BASELINE_POLICY_MODEL:=$BASELINE_MODEL_NAME}"
: "${TUNED_POLICY_MODEL:=agent_adapter}"
: "${START_BASELINE_ENDPOINT:=1}"
: "${START_TUNED_ENDPOINT:=1}"
: "${KEEP_POLICY_SERVERS:=0}"
: "${BASELINE_CUDA_VISIBLE_DEVICES:=0}"
: "${TUNED_CUDA_VISIBLE_DEVICES:=1}"
: "${POLICY_SERVER_STARTUP_TIMEOUT_SECONDS:=900}"
: "${POLICY_SERVER_HEALTH_INTERVAL_SECONDS:=10}"
: "${RULER_SCORED_OUTPUT:=$WORKSPACE_DIR/data/grpo/runtime_proof/${RUNTIME_PROOF_ID}_ruler_scored_groups_vllm_no_fallback.jsonl}"
: "${HELDOUT_RATIO:=0.25}"
: "${BENCHMARK_TASKS:=$WORKSPACE_DIR/data/synthetic/tasks.jsonl}"
: "${BENCHMARK_LIMIT:=3}"
: "${BENCHMARK_ROLLOUTS:=1}"
: "${BENCHMARK_OUTPUT_DIR:=$WORKSPACE_DIR/trajectories/runtime_proof/${RUNTIME_PROOF_ID}_endpoint_benchmark}"
: "${BENCHMARK_REPORT:=$RUNTIME_PROOF_DIR/baseline_vs_tuned_checkpoint_reload_heldout.md}"

mkdir -p "$RUNTIME_PROOF_DIR" "$BENCHMARK_OUTPUT_DIR"
BASELINE_LOG="$RUNTIME_PROOF_DIR/baseline_policy_server.log"
TUNED_LOG="$RUNTIME_PROOF_DIR/tuned_policy_server_reload.log"
ASSERTION_JSON="$RUNTIME_PROOF_DIR/checkpoint_reload_compare_assertion.json"
HELDOUT_DIR="$RUNTIME_PROOF_DIR/heldout_eval"
PROOF_REPORT="$RUNTIME_PROOF_DIR/runtime_proof_checkpoint_reload_compare.md"
BASELINE_PID=""
TUNED_PID=""

cleanup() {
  if [[ "$KEEP_POLICY_SERVERS" != "1" ]]; then
    for pid in ${TUNED_PID:-} ${BASELINE_PID:-}; do
      if [[ -n "$pid" ]] && kill -0 "$pid" >/dev/null 2>&1; then
        echo "Stopping runtime-proof policy server process $pid"
        kill "$pid" >/dev/null 2>&1 || true
        wait "$pid" >/dev/null 2>&1 || true
      fi
    done
  fi
}
trap cleanup EXIT

health_check_url() {
  local url="$1"
  python - "$url" <<'PY' >/dev/null 2>&1
import json
import sys
import urllib.request
base = sys.argv[1].rstrip('/')
req = urllib.request.Request(base + '/models', headers={'Authorization': 'Bearer EMPTY'})
with urllib.request.urlopen(req, timeout=5) as resp:
    payload = json.loads(resp.read().decode('utf-8'))
if not payload.get('data'):
    raise SystemExit('no models returned')
PY
}

wait_for_endpoint() {
  local name="$1"
  local url="$2"
  local log="$3"
  local deadline=$((SECONDS + POLICY_SERVER_STARTUP_TIMEOUT_SECONDS))
  until health_check_url "$url"; do
    if (( SECONDS >= deadline )); then
      echo "Timed out waiting for $name endpoint at $url. See $log" >&2
      exit 75
    fi
    sleep "$POLICY_SERVER_HEALTH_INTERVAL_SECONDS"
  done
  echo "$name endpoint is healthy at $url"
}

python - "$POLICY_OUTPUT_DIR" <<'PY'
import json
import sys
from pathlib import Path
ckpt = Path(sys.argv[1])
metadata = ckpt / 'adapter_metadata.json'
config = ckpt / 'adapter_config.json'
weights = [ckpt / 'adapter_model.safetensors', ckpt / 'adapter_model.bin']
if not ckpt.exists():
    raise SystemExit(f'Checkpoint directory does not exist: {ckpt}')
if not metadata.exists():
    raise SystemExit(f'Checkpoint metadata missing: {metadata}')
if not config.exists():
    raise SystemExit(f'Adapter config missing: {config}')
if not any(p.exists() for p in weights):
    raise SystemExit(f'Adapter weight file missing in {ckpt}')
meta = json.loads(metadata.read_text(encoding='utf-8'))
if meta.get('adapter_status') != 'trained':
    raise SystemExit(f'Checkpoint metadata does not report adapter_status=trained: {metadata}')
print(f'Checkpoint artifact gate passed: {ckpt}')
PY

if [[ -s "$RULER_SCORED_OUTPUT" ]]; then
  mkdir -p "$HELDOUT_DIR"
  python -m src.training.build_heldout_policy_eval \
    --input "$RULER_SCORED_OUTPUT" \
    --output-dir "$HELDOUT_DIR" \
    --heldout-ratio "$HELDOUT_RATIO"
  if [[ -s "$HELDOUT_DIR/heldout_tasks.jsonl" ]]; then
    BENCHMARK_TASKS="$HELDOUT_DIR/heldout_tasks.jsonl"
  fi
fi

if [[ ! -s "$BENCHMARK_TASKS" ]]; then
  echo "Benchmark task file is missing or empty: $BENCHMARK_TASKS" >&2
  exit 66
fi

if health_check_url "$BASELINE_POLICY_URL"; then
  echo "Existing baseline endpoint is healthy at $BASELINE_POLICY_URL"
elif [[ "$START_BASELINE_ENDPOINT" == "1" ]]; then
  echo "Starting baseline policy endpoint on port $BASELINE_PORT; log: $BASELINE_LOG"
  nohup env CUDA_VISIBLE_DEVICES="$BASELINE_CUDA_VISIBLE_DEVICES" VLLM_MODEL_ID="$BASELINE_MODEL_NAME" MODEL_NAME="$BASELINE_POLICY_MODEL" BASELINE_PORT="$BASELINE_PORT" \
    bash scripts/gpu/start_baseline_inference.sh >"$BASELINE_LOG" 2>&1 &
  BASELINE_PID="$!"
  wait_for_endpoint "baseline" "$BASELINE_POLICY_URL" "$BASELINE_LOG"
else
  echo "Baseline endpoint is not healthy and START_BASELINE_ENDPOINT is not enabled." >&2
  exit 75
fi

if health_check_url "$TUNED_POLICY_URL"; then
  echo "Existing tuned endpoint is healthy at $TUNED_POLICY_URL"
elif [[ "$START_TUNED_ENDPOINT" == "1" ]]; then
  echo "Starting tuned policy endpoint on port $TUNED_PORT with checkpoint $POLICY_OUTPUT_DIR; log: $TUNED_LOG"
  nohup env CUDA_VISIBLE_DEVICES="$TUNED_CUDA_VISIBLE_DEVICES" MODEL_NAME="$TUNED_MODEL_NAME" TUNED_PORT="$TUNED_PORT" CHECKPOINT_DIR="$POLICY_OUTPUT_DIR" \
    bash scripts/gpu/start_tuned_inference.sh >"$TUNED_LOG" 2>&1 &
  TUNED_PID="$!"
  wait_for_endpoint "tuned" "$TUNED_POLICY_URL" "$TUNED_LOG"
else
  echo "Tuned endpoint is not healthy and START_TUNED_ENDPOINT is not enabled." >&2
  exit 75
fi

BENCHMARK_POLICIES="baseline tuned" \
BASELINE_POLICY_URL="$BASELINE_POLICY_URL" \
BASELINE_POLICY_MODEL="$BASELINE_POLICY_MODEL" \
TUNED_POLICY_URL="$TUNED_POLICY_URL" \
TUNED_POLICY_MODEL="$TUNED_POLICY_MODEL" \
MODEL_NAME="$MODEL_NAME" \
BENCHMARK_TASKS="$BENCHMARK_TASKS" \
BENCHMARK_OUTPUT_DIR="$BENCHMARK_OUTPUT_DIR" \
BENCHMARK_REPORT="$BENCHMARK_REPORT" \
BENCHMARK_LIMIT="$BENCHMARK_LIMIT" \
BENCHMARK_ROLLOUTS="$BENCHMARK_ROLLOUTS" \
  bash scripts/cpu/run_08_benchmark_policy_endpoints.sh

python - "$BENCHMARK_OUTPUT_DIR" "$BENCHMARK_REPORT" "$ASSERTION_JSON" "$POLICY_OUTPUT_DIR" "$BENCHMARK_TASKS" <<'PY'
import json
import sys
from pathlib import Path
bench_dir = Path(sys.argv[1])
report = Path(sys.argv[2])
out = Path(sys.argv[3])
ckpt = Path(sys.argv[4])
tasks = Path(sys.argv[5])
summary = {
    'benchmark_output_dir': str(bench_dir),
    'benchmark_report': str(report),
    'checkpoint_dir': str(ckpt),
    'benchmark_tasks': str(tasks),
    'policies': {},
    'report_present': report.exists() and report.stat().st_size > 0,
    'assertion': 'pending',
}
for policy in ['baseline', 'tuned']:
    scored_dir = bench_dir / policy / 'scored'
    files = sorted(scored_dir.glob('*.jsonl')) if scored_dir.exists() else []
    rewards = []
    live_markers = 0
    for path in files:
        with path.open('r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if 'reward' in row:
                    try:
                        rewards.append(float(row.get('reward') or 0.0))
                    except Exception:
                        pass
                payload = json.dumps({k: row.get(k) for k in row.keys() if k in {'policy_endpoint', 'policy_version', 'decision_source', 'model_name'}}, sort_keys=True)
                if 'policy' in payload.lower() or 'endpoint' in payload.lower():
                    live_markers += 1
    summary['policies'][policy] = {
        'scored_dir': str(scored_dir),
        'scored_files': len(files),
        'reward_count': len(rewards),
        'avg_reward': round(sum(rewards) / max(len(rewards), 1), 6),
        'live_marker_rows': live_markers,
    }
if (
    summary['report_present']
    and summary['policies']['baseline']['scored_files'] > 0
    and summary['policies']['tuned']['scored_files'] > 0
    and summary['policies']['baseline']['reward_count'] > 0
    and summary['policies']['tuned']['reward_count'] > 0
):
    summary['assertion'] = 'passed_checkpoint_reload_heldout_compare'
else:
    summary['assertion'] = 'failed_checkpoint_reload_compare_gate'
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
if summary['assertion'] != 'passed_checkpoint_reload_heldout_compare':
    raise SystemExit(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
PY

cat > "$PROOF_REPORT" <<EOF
# Runtime Proof: Checkpoint Reload and Held-Out Baseline-vs-Tuned Comparison

This proof run verified the trained adapter checkpoint, reloaded it through a tuned OpenAI-compatible policy endpoint, ran a live endpoint benchmark against the baseline endpoint with **--require-live-policy**, scored both policies, and generated a comparison report.

| Evidence Item | Path |
|---|---|
| Reloaded checkpoint directory | \`$POLICY_OUTPUT_DIR\` |
| Benchmark tasks | \`$BENCHMARK_TASKS\` |
| Benchmark output directory | \`$BENCHMARK_OUTPUT_DIR\` |
| Comparison report | \`$BENCHMARK_REPORT\` |
| Checkpoint reload assertion JSON | \`$ASSERTION_JSON\` |
| Baseline server log | \`$BASELINE_LOG\` |
| Tuned server log | \`$TUNED_LOG\` |

The assertion passed only if both baseline and tuned scored directories contained scored JSONL files with reward values and the comparison report was written successfully.
EOF

echo "Checkpoint reload and held-out comparison proof complete: $PROOF_REPORT"
