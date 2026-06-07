#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Start the Agent Lightning Demo Dashboard
#
# Launches the Streamlit UI and (optionally) the Lightning Server together.
#
# Usage:
#   bash scripts/start_demo.sh
#
# Options (env vars):
#   LIGHTNING_SERVER_URL   default http://localhost:19123
#   DASHBOARD_PORT         default 8501
#   START_LIGHTNING_SERVER  set to "1" to also start Lightning Server locally
#   MLFLOW_PORT            default 5000
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

WORKSPACE_DIR="${WORKSPACE_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$WORKSPACE_DIR"

DASHBOARD_PORT="${DASHBOARD_PORT:-8501}"
LIGHTNING_SERVER_URL="${LIGHTNING_SERVER_URL:-http://localhost:19123}"
START_LIGHTNING_SERVER="${START_LIGHTNING_SERVER:-0}"
MLFLOW_PORT="${MLFLOW_PORT:-5000}"

export LIGHTNING_SERVER_URL
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-${WORKSPACE_DIR}/mlruns}"

# ── Optionally start the Lightning Server in the background ───────────────────
if [[ "$START_LIGHTNING_SERVER" == "1" ]]; then
  echo "[demo] Starting Lightning Server on port 19123..."
  bash scripts/gpu/start_lightning_server.sh &
  LIGHTNING_PID=$!
  echo "[demo] Lightning Server PID: $LIGHTNING_PID"
  sleep 2
fi

# ── Optionally start MLflow UI ─────────────────────────────────────────────────
echo "[demo] Starting MLflow UI on port $MLFLOW_PORT (background)..."
mlflow ui \
  --backend-store-uri "$MLFLOW_TRACKING_URI" \
  --host 0.0.0.0 \
  --port "$MLFLOW_PORT" \
  &>/dev/null &
MLFLOW_PID=$!
echo "[demo] MLflow UI PID: $MLFLOW_PID  →  http://localhost:${MLFLOW_PORT}"

# ── Start Streamlit Dashboard ──────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║     Agent Lightning Demo Dashboard                   ║"
echo "╠══════════════════════════════════════════════════════╣"
echo "║  Dashboard  →  http://localhost:${DASHBOARD_PORT}          ║"
echo "║  MLflow     →  http://localhost:${MLFLOW_PORT}           ║"
echo "║  API Docs   →  ${LIGHTNING_SERVER_URL}/docs      ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

streamlit run scripts/demo_dashboard.py \
  --server.port "$DASHBOARD_PORT" \
  --server.address 0.0.0.0 \
  --server.headless true \
  --browser.gatherUsageStats false
