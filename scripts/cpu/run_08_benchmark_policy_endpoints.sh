#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export WORKSPACE_DIR="${WORKSPACE_DIR:-$REPO_DIR}"

: "${BENCHMARK_TASKS:=$WORKSPACE_DIR/data/synthetic/tasks.jsonl}"
: "${BENCHMARK_OUTPUT_DIR:=$WORKSPACE_DIR/trajectories/endpoint_benchmark}"
: "${BENCHMARK_REPORT:=$WORKSPACE_DIR/reports/baseline_v2_tuned_benchmark.md}"
: "${BENCHMARK_LIMIT:=}"
: "${BENCHMARK_ROLLOUTS:=1}"
: "${BENCHMARK_POLICIES:=baseline v2 tuned}"

require_profile() {
  local policy="$1"
  local prefix
  case "$policy" in
    baseline) prefix="BASELINE_POLICY" ;;
    v2) prefix="V2_POLICY" ;;
    tuned|rl_tuned|agent_lightning_policy) prefix="TUNED_POLICY" ;;
    *) prefix="$(printf '%s' "$policy" | tr '[:lower:]-' '[:upper:]_')_POLICY" ;;
  esac
  local url_var="${prefix}_URL"
  local model_var="${prefix}_MODEL"
  if [[ -z "${!url_var:-}" ]]; then
    echo "Missing $url_var for policy $policy" >&2
    exit 2
  fi
  if [[ -z "${!model_var:-}" && -z "${MODEL_NAME:-}" ]]; then
    echo "Missing $model_var or MODEL_NAME for policy $policy" >&2
    exit 2
  fi
}

policy_dir_args=()
for policy in $BENCHMARK_POLICIES; do
  require_profile "$policy"
  out_dir="$BENCHMARK_OUTPUT_DIR/$policy"
  rm -rf "$out_dir"
  cmd=(
    python -m src.agents.supervisor
    --tasks "$BENCHMARK_TASKS"
    --output-dir "$out_dir"
    --policy-version "$policy"
    --rollouts "$BENCHMARK_ROLLOUTS"
    --require-live-policy
  )
  if [[ -n "$BENCHMARK_LIMIT" ]]; then
    cmd+=(--limit "$BENCHMARK_LIMIT")
  fi
  "${cmd[@]}"
  scored_dir="$out_dir/scored"
  python -m src.rewards.scorer --input-dir "$out_dir" --output-dir "$scored_dir"
  policy_dir_args+=(--policy-dir "$policy=$scored_dir")
done

python -m src.evaluation.compare_policies "${policy_dir_args[@]}" --output "$BENCHMARK_REPORT"
echo "Endpoint benchmark report: $BENCHMARK_REPORT"
