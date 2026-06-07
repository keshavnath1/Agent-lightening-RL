#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export WORKSPACE_DIR="${WORKSPACE_DIR:-$REPO_DIR}"

REPORT_DIR="${REPORT_DIR:-$REPO_DIR/reports}"
REPORT_PATH="${REPORT_PATH:-$REPORT_DIR/runpod_architecture_validation_summary.txt}"
mkdir -p "$REPORT_DIR"

MCP_URL="${MCP_URL:-http://127.0.0.1:${MCP_PORT:-8090}}"
MLFLOW_URL="${MLFLOW_URL:-http://127.0.0.1:${MLFLOW_PORT:-5000}}"
GPU_POLICY_BASE_URL="${GPU_POLICY_BASE_URL:-}"
BASELINE_POLICY_URL="${BASELINE_POLICY_URL:-}"
MODEL_NAME="${MODEL_NAME:-qwen2.5-coder-32b-instruct}"
POSTGRES_HOST="${POSTGRES_HOST:-localhost}"
POSTGRES_PORT="${POSTGRES_PORT:-5432}"

status_line() {
  local name="$1"
  local status="$2"
  local detail="$3"
  printf '%-32s %-8s %s\n' "$name" "$status" "$detail"
}

write_header() {
  {
    echo "RunPod architecture validation summary"
    echo "Generated: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo "Repository: $REPO_DIR"
    echo "Workspace: $WORKSPACE_DIR"
    echo
  } > "$REPORT_PATH"
}

append_line() {
  tee -a "$REPORT_PATH" >/dev/null
}

check_cmd() {
  local name="$1"
  local cmd="$2"
  if command -v "$cmd" >/dev/null 2>&1; then
    status_line "$name" "PASS" "$(command -v "$cmd")"
  else
    status_line "$name" "WARN" "$cmd not found"
  fi
}

check_python_imports() {
  python3 - <<'PY'
imports = [
    'fastapi', 'uvicorn', 'sqlalchemy', 'psycopg2', 'mlflow', 'dvc',
    'pandas', 'pyarrow', 'sklearn', 'xgboost', 'lightgbm', 'requests'
]
missing = []
for name in imports:
    try:
        __import__(name)
    except Exception as exc:
        missing.append(f"{name}: {exc.__class__.__name__}: {exc}")
if missing:
    print('WARN|' + '; '.join(missing))
else:
    print('PASS|all required CPU imports succeeded')
PY
}

check_http() {
  local name="$1"
  local url="$2"
  python3 - "$url" <<'PY'
import sys
import requests
url = sys.argv[1]
try:
    r = requests.get(url, timeout=10)
    body = r.text[:200].replace('\n', ' ')
    if 200 <= r.status_code < 300:
        print(f"PASS|HTTP {r.status_code} {body}")
    else:
        print(f"WARN|HTTP {r.status_code} {body}")
except Exception as exc:
    print(f"WARN|{exc.__class__.__name__}: {exc}")
PY
}

check_postgres() {
  python3 - <<'PY'
try:
    from src.tools.sql_alchemy_connector import SQLAlchemyConnector, default_database_url
    connector = SQLAlchemyConnector(default_database_url())
    connector.healthcheck()
    print('PASS|PostgreSQL SQLAlchemy healthcheck passed')
except Exception as exc:
    print(f'WARN|PostgreSQL not reachable: {exc.__class__.__name__}: {exc}')
PY
}

check_postgres_synthetic_data() {
  python3 - <<'PY'
import os
from src.tools.sql_alchemy_connector import SQLAlchemyConnector
schema = os.getenv('POSTGRES_SCHEMA', os.getenv('PG_SCHEMA', 'agentic_ml')).replace('"', '""')
try:
    connector = SQLAlchemyConnector()
    rows = connector.execute(f'''
        select
            (select count(*) from "{schema}".synthetic_tasks) as task_count,
            (select count(*) from "{schema}".synthetic_dataset_rows) as dataset_row_count,
            (select count(distinct task_id) from "{schema}".synthetic_dataset_rows) as dataset_task_count
    ''')
    summary = rows[0]
    if summary['task_count'] >= 20 and summary['dataset_row_count'] >= 20000 and summary['dataset_task_count'] >= 20:
        print(f"PASS|tasks={summary['task_count']} dataset_rows={summary['dataset_row_count']} dataset_tasks={summary['dataset_task_count']}")
    else:
        print(f"WARN|unexpected synthetic table counts: {summary}")
except Exception as exc:
    print(f'WARN|synthetic tables not validated: {exc.__class__.__name__}: {exc}')
PY
}

check_mcp_profile() {
  local parquet="${PROFILE_PARQUET:-$REPO_DIR/data/synthetic/datasets/task_0000.parquet}"
  if [[ ! -f "$parquet" ]]; then
    echo "WARN|sample parquet not found at $parquet"
    return 0
  fi
  python3 - "$MCP_URL" "$parquet" <<'PY'
import sys
import requests
mcp_url, parquet_path = sys.argv[1], sys.argv[2]
try:
    r = requests.post(
        f'{mcp_url}/tools/profile/parquet',
        json={
            'parquet_path': parquet_path,
            'output_json': 'reports/mcp_profile_validation/profile_summary.json',
            'output_html': 'reports/mcp_profile_validation/profile_report.html',
        },
        timeout=30,
    )
    text = r.text[:300].replace('\n', ' ')
    if r.status_code == 200:
        data = r.json()
        summary = data.get('summary', {})
        print(f"PASS|profile endpoint ok rows={summary.get('row_count')} cols={summary.get('column_count')}")
    else:
        print(f"WARN|HTTP {r.status_code} {text}")
except Exception as exc:
    print(f"WARN|{exc.__class__.__name__}: {exc}")
PY
}

check_gpu_policy() {
  local base="$GPU_POLICY_BASE_URL"
  if [[ -z "$base" && -n "$BASELINE_POLICY_URL" ]]; then
    base="${BASELINE_POLICY_URL%/chat/completions}"
  fi
  if [[ -z "$base" ]]; then
    echo "WARN|set GPU_POLICY_BASE_URL or BASELINE_POLICY_URL to validate GPU inference"
    return 0
  fi
  python3 - "$base" "$MODEL_NAME" <<'PY'
import sys
import requests
base, model = sys.argv[1].rstrip('/'), sys.argv[2]
try:
    models = requests.get(f'{base}/models', timeout=20)
    if not (200 <= models.status_code < 300):
        print(f"WARN|/models HTTP {models.status_code}: {models.text[:200]}")
        raise SystemExit(0)
    payload = {
        'model': model,
        'messages': [{'role': 'user', 'content': 'Reply exactly ARCH_VALIDATION_OK'}],
        'temperature': 0,
        'max_tokens': 16,
    }
    chat = requests.post(f'{base}/chat/completions', json=payload, timeout=120)
    body = chat.text[:300].replace('\n', ' ')
    if 200 <= chat.status_code < 300 and 'ARCH_VALIDATION_OK' in chat.text:
        print(f"PASS|GPU OpenAI-compatible endpoint responded with expected marker using model={model}")
    else:
        print(f"WARN|chat HTTP {chat.status_code}: {body}")
except Exception as exc:
    print(f"WARN|{exc.__class__.__name__}: {exc}")
PY
}

check_artifact() {
  local name="$1"
  local path="$2"
  if [[ -e "$path" ]]; then
    if [[ -f "$path" ]]; then
      status_line "$name" "PASS" "$path ($(wc -l < "$path" 2>/dev/null || echo '?') lines)"
    else
      status_line "$name" "PASS" "$path ($(find "$path" -type f | wc -l) files)"
    fi
  else
    status_line "$name" "WARN" "missing: $path"
  fi
}

write_header
{
  echo "Command and package checks"
  check_cmd "python3" python3
  check_cmd "psql/pg_isready" pg_isready
  check_cmd "docker" docker
  IFS='|' read -r status detail < <(check_python_imports)
  status_line "CPU Python imports" "$status" "$detail"
  echo

  echo "Service checks"
  IFS='|' read -r status detail < <(check_http "MCP health" "$MCP_URL/health")
  status_line "MCP health" "$status" "$detail"
  IFS='|' read -r status detail < <(check_http "MLflow health" "$MLFLOW_URL/health")
  status_line "MLflow health" "$status" "$detail"
  IFS='|' read -r status detail < <(check_postgres)
  status_line "PostgreSQL" "$status" "$detail"
  IFS='|' read -r status detail < <(check_postgres_synthetic_data)
  status_line "PostgreSQL synthetic data" "$status" "$detail"
  IFS='|' read -r status detail < <(check_mcp_profile)
  status_line "MCP profile tool" "$status" "$detail"
  IFS='|' read -r status detail < <(check_gpu_policy)
  status_line "GPU policy endpoint" "$status" "$detail"
  echo

  echo "Artifact checks"
  check_artifact "Synthetic task JSONL" "$REPO_DIR/data/synthetic/tasks.jsonl"
  check_artifact "Synthetic datasets" "$REPO_DIR/data/synthetic/datasets"
  check_artifact "Baseline trajectories" "$REPO_DIR/trajectories/baseline"
  check_artifact "Scored trajectories" "$REPO_DIR/trajectories/scored"
  check_artifact "GRPO dataset" "$REPO_DIR/data/grpo/grouped_rollouts.jsonl"
  check_artifact "Agent Lightning export" "$REPO_DIR/data/grpo/agent_lightning_transitions.jsonl"
  check_artifact "Comparison report" "$REPO_DIR/reports/baseline_vs_rl_tuned.md"
  check_artifact "MLflow store" "$REPO_DIR/mlruns"
  check_artifact "JSON fallback tracking" "$REPO_DIR/reports/mlflow_runs"
  check_artifact "Checkpoints" "$REPO_DIR/checkpoints"
  echo

  echo "Port snapshot"
  if command -v ss >/dev/null 2>&1; then
    ss -ltnp 2>/dev/null | grep -E ':(5432|5000|8090|8000)\b' || true
  else
    echo "ss not available"
  fi
  echo
  echo "Validation complete. WARN items are actionable gaps; PASS items are ready."
} | append_line

cat "$REPORT_PATH"
