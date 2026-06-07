#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export WORKSPACE_DIR="${WORKSPACE_DIR:-$REPO_DIR}"

: "${BASELINE_SCORED_DIR:=$WORKSPACE_DIR/trajectories/scored}"
: "${TUNED_SCORED_DIR:=$WORKSPACE_DIR/trajectories/tuned_scored}"
: "${COMPARISON_REPORT:=$WORKSPACE_DIR/reports/baseline_vs_rl_tuned.md}"
mkdir -p "$(dirname "$COMPARISON_REPORT")"
python -m src.evaluation.compare_policies --baseline-dir "$BASELINE_SCORED_DIR" --tuned-dir "$TUNED_SCORED_DIR" --output "$COMPARISON_REPORT"
echo "Comparison report: $COMPARISON_REPORT"
