#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export WORKSPACE_DIR="${WORKSPACE_DIR:-$REPO_DIR}"

: "${MCP_HOST:=0.0.0.0}"
: "${MCP_PORT:=8090}"
echo "Starting MCP-style tool server on ${MCP_HOST}:${MCP_PORT}"
exec uvicorn src.mcp_server.server:app --host "$MCP_HOST" --port "$MCP_PORT"
