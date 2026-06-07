#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export WORKSPACE_DIR="${WORKSPACE_DIR:-$REPO_DIR}"

: "${POSTGRES_DB:=agentic_ml}"
: "${POSTGRES_USER:=agent}"
: "${POSTGRES_PASSWORD:=change_me}"
: "${POSTGRES_PORT:=5432}"

if command -v pg_isready >/dev/null 2>&1 && pg_isready -h "${POSTGRES_HOST:-localhost}" -p "$POSTGRES_PORT" >/dev/null 2>&1; then
  echo "PostgreSQL is already reachable on ${POSTGRES_HOST:-localhost}:$POSTGRES_PORT"
  exit 0
fi

if command -v docker >/dev/null 2>&1; then
  echo "Starting PostgreSQL with docker run. RunPod does not require Docker Compose for this demo."
  docker rm -f agentic-ml-postgres >/dev/null 2>&1 || true
  docker run -d --name agentic-ml-postgres \
    -e POSTGRES_DB="$POSTGRES_DB" \
    -e POSTGRES_USER="$POSTGRES_USER" \
    -e POSTGRES_PASSWORD="$POSTGRES_PASSWORD" \
    -p "$POSTGRES_PORT:5432" \
    -v "${WORKSPACE_DIR}/postgres_data:/var/lib/postgresql/data" \
    postgres:16
  echo "Waiting for PostgreSQL readiness..."
  for i in {1..30}; do
    if docker exec agentic-ml-postgres pg_isready -U "$POSTGRES_USER" -d "$POSTGRES_DB" >/dev/null 2>&1; then
      echo "PostgreSQL ready."
      exit 0
    fi
    sleep 2
  done
  echo "PostgreSQL did not become ready in time." >&2
  exit 1
fi

echo "Docker is unavailable and PostgreSQL is not already running. Start PostgreSQL on the CPU Pod or set DATABASE_URL." >&2
exit 1
