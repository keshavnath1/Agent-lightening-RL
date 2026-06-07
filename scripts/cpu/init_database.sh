#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export WORKSPACE_DIR="${WORKSPACE_DIR:-$REPO_DIR}"

python - <<'PY'
from sqlalchemy import create_engine, text
from src.tools.sql_alchemy_connector import default_database_url
engine = create_engine(default_database_url(), future=True)
with engine.begin() as conn:
    conn.execute(text('select 1'))
print('Database connectivity ok.')
PY
