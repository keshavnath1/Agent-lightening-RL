#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

echo "Starting official Project MCP server..."
echo "Tools: safe PostgreSQL metadata, artifact, rollout, reward tools"
echo "Raw SQL enabled: ${ALLOW_RAW_SQL_TOOL:-0}"

python -m src.mcp_official.server
