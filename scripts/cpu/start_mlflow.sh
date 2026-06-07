#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export WORKSPACE_DIR="${WORKSPACE_DIR:-$REPO_DIR}"

: "${MLFLOW_TRACKING_URI:=file:${WORKSPACE_DIR}/mlruns}"
: "${MLFLOW_HOST:=0.0.0.0}"
: "${MLFLOW_PORT:=5000}"
mkdir -p "${WORKSPACE_DIR}/mlruns"
echo "Starting MLflow UI on ${MLFLOW_HOST}:${MLFLOW_PORT} with backend ${MLFLOW_TRACKING_URI}"
exec mlflow server --backend-store-uri "$MLFLOW_TRACKING_URI" --default-artifact-root "${WORKSPACE_DIR}/mlruns" --host "$MLFLOW_HOST" --port "$MLFLOW_PORT"
