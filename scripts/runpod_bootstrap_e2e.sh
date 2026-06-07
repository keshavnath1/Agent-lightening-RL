#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_DIR"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

log() {
  printf '\n[%s] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*"
}

need_env() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required environment variable: ${name}" >&2
    echo "Set DATABASE_URL, for example by creating .env from .env.example." >&2
    exit 2
  fi
}

wait_http() {
  local url="$1"
  local name="$2"
  local tries="${3:-90}"
  for _ in $(seq 1 "$tries"); do
    if python - "$url" <<'PY' >/dev/null 2>&1
import sys
import urllib.request
urllib.request.urlopen(sys.argv[1], timeout=2).read()
PY
    then
      log "${name} is healthy at ${url}"
      return 0
    fi
    sleep 2
  done
  echo "${name} did not become healthy at ${url}" >&2
  return 1
}

start_bg() {
  local name="$1"
  local logfile="$2"
  shift 2
  mkdir -p "$(dirname "$logfile")"
  log "Starting ${name}; log: ${logfile}"
  ("$@" >"$logfile" 2>&1 & echo $! >"${logfile%.log}.pid")
}

need_env DATABASE_URL

export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export WORKSPACE_DIR="${WORKSPACE_DIR:-$REPO_DIR}"
export VENV_DIR="${VENV_DIR:-$REPO_DIR/.venv}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$WORKSPACE_DIR/.cache/pip}"
export TMPDIR="${TMPDIR:-$WORKSPACE_DIR/.tmp}"
export MCP_DATABASE_URL="${MCP_DATABASE_URL:-$DATABASE_URL}"
export EXECUTION_DATABASE_URL="${EXECUTION_DATABASE_URL:-$DATABASE_URL}"
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-sqlite:///${REPO_DIR}/reports/mlflow_tracking.db}"
export CODE_INTERPRETER_MODE="${CODE_INTERPRETER_MODE:-host_subprocess}"
export LIGHTNING_SERVER_PORT="${LIGHTNING_SERVER_PORT:-19124}"
export LIGHTNING_SERVER_URL="${LIGHTNING_SERVER_URL:-http://127.0.0.1:${LIGHTNING_SERVER_PORT}}"
export LIGHTNING_MIN_ROLLOUTS="${LIGHTNING_MIN_ROLLOUTS:-1}"
export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-3B-Instruct}"
export TRACKB_CHECKPOINT_DIR="${TRACKB_CHECKPOINT_DIR:-$REPO_DIR/checkpoints/trackb_trl_grpo_runpod}"

if [[ -n "${HF_TOKEN:-}" && -z "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
  export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
fi

DEFAULT_OPENML_TASK_SPECS=$'31|openml_31_german_credit|mltask_openml_31_german_credit_baseline|auto|roc_auc|accuracy|10\n44|openml_44_spambase|mltask_openml_44_spambase_baseline|auto|roc_auc|accuracy|20\n1461|openml_1461_bank_marketing|mltask_openml_1461_bank_marketing_baseline|auto|roc_auc|accuracy|30\n1489|openml_1489_phoneme|mltask_openml_1489_phoneme_baseline|auto|roc_auc|accuracy|40'
: "${OPENML_TASK_SPECS:=$DEFAULT_OPENML_TASK_SPECS}"
: "${TASK_LIMIT:=$(printf '%s\n' "$OPENML_TASK_SPECS" | sed '/^[[:space:]]*$/d' | wc -l | tr -d ' ')}"
: "${TRACKA_INITIAL_ROLLOUTS:=4}"
: "${RUN_TRACKB:=1}"
: "${RUN_LIVE_POLICY:=0}"
: "${INSTALL_DEPS:=1}"
: "${INSTALL_GPU_DEPS:=auto}"
: "${START_DASHBOARD:=0}"
: "${DASHBOARD_PORT:=8888}"
: "${PYTORCH_VERSION:=2.8.0}"

mkdir -p reports/run_logs reports/service_logs reports/streamlit_logs data/grpo trajectories checkpoints artifacts "$PIP_CACHE_DIR" "$TMPDIR"

if [[ ! -d "$VENV_DIR" ]]; then
  log "Creating virtual environment at ${VENV_DIR}"
  mkdir -p "$(dirname "$VENV_DIR")"
  python3 -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

if [[ "$INSTALL_DEPS" == "1" ]]; then
  log "Installing CPU dependencies"
  python -m pip install --upgrade pip
  pip install -r requirements-cpu.txt

  if [[ "$INSTALL_GPU_DEPS" == "1" ]] || { [[ "$INSTALL_GPU_DEPS" == "auto" ]] && command -v nvidia-smi >/dev/null 2>&1; }; then
    log "Installing GPU dependencies"
    python - <<'PY' || pip install --upgrade "torch==${PYTORCH_VERSION}"
import torch
from torch.distributed.fsdp import FSDPModule
print(torch.__version__)
print(FSDPModule.__name__)
PY
    pip install -r requirements-gpu.txt
  fi
fi

log "Validating source wiring"
python scripts/validate/validate_agent_lightning_strict.py
python scripts/dev_validate_official_mcp_server.py
python scripts/dev_validate_trackb_wiring.py
python scripts/dev_validate_dashboard_refactor.py

log "Installing or refreshing PostgreSQL ML contract and OpenML tasks"
: > reports/run_logs/openml_ingest_latest.log
while IFS='|' read -r OPENML_ID DATASET_KEY TRACKA_TASK_ID TARGET_COLUMN PRIMARY_METRIC SECONDARY_METRIC PRIORITY; do
  [[ -z "${OPENML_ID//[[:space:]]/}" ]] && continue
  [[ "${OPENML_ID}" =~ ^[[:space:]]*# ]] && continue
  log "Ingesting OpenML ${OPENML_ID} as ${TRACKA_TASK_ID}"
  python scripts/ingest_openml_to_postgres.py \
    --openml-id "$OPENML_ID" \
    --dataset-key "$DATASET_KEY" \
    --task-id "$TRACKA_TASK_ID" \
    --target-column "${TARGET_COLUMN:-auto}" \
    --canonical-target-column target \
    --primary-metric "${PRIMARY_METRIC:-roc_auc}" \
    --secondary-metric "${SECONDARY_METRIC:-accuracy}" \
    --priority "${PRIORITY:-100}" \
    --if-exists replace \
    2>&1 | tee -a reports/run_logs/openml_ingest_latest.log
done <<< "$OPENML_TASK_SPECS"

log "Starting Lightning server"
start_bg lightning reports/service_logs/lightning_server_${LIGHTNING_SERVER_PORT}.log \
  env LIGHTNING_SERVER_PORT="$LIGHTNING_SERVER_PORT" \
      LIGHTNING_MIN_ROLLOUTS="$LIGHTNING_MIN_ROLLOUTS" \
      LIGHTNING_TRANSITIONS_DIR="$REPO_DIR/data/grpo" \
      LIGHTNING_CHECKPOINT_DIR="$TRACKB_CHECKPOINT_DIR" \
      VLLM_BASE_URL="http://127.0.0.1:18081" \
      python -m src.training.lightning_server_app
wait_http "${LIGHTNING_SERVER_URL}/health" "Lightning server"

log "Running initial Track A rollouts"
python -m src.agents.supervisor \
  --task-source postgres_mcp \
  --task-limit "$TASK_LIMIT" \
  --rollouts "$TRACKA_INITIAL_ROLLOUTS" \
  --policy-version baseline_initial \
  --output-dir trajectories/tracka_initial \
  --lightning-server-url "$LIGHTNING_SERVER_URL" \
  2>&1 | tee reports/run_logs/tracka_initial_latest.log

python -m src.rewards.scorer \
  --input-dir trajectories/tracka_initial \
  --output-dir trajectories/tracka_initial_scored

python -m src.training.prepare_grpo_dataset \
  --input-dir trajectories/tracka_initial_scored \
  --output data/grpo/grouped_rollouts.jsonl

python -m src.evaluation.compare_policies \
  --policy-dir initial_no_llm=trajectories/tracka_initial_scored \
  --output reports/tracka_initial_benchmark.md

if [[ "$RUN_TRACKB" == "1" ]]; then
  log "Running Track B TRL GRPO"
  TRAINER="${TRAINER:-trl_grpo}" \
  REWARD_MODE="${REWARD_MODE:-hybrid}" \
  NUM_GENERATIONS="${NUM_GENERATIONS:-4}" \
  MODEL_NAME="$MODEL_NAME" \
  GRPO_DATASET_PATH="$REPO_DIR/data/grpo/grouped_rollouts.jsonl" \
  POLICY_OUTPUT_DIR="$TRACKB_CHECKPOINT_DIR" \
  bash scripts/gpu/run_02_train_policy_qlora_grpo.sh \
  2>&1 | tee reports/run_logs/trackb_trl_grpo_latest.log
fi

if [[ "$RUN_LIVE_POLICY" == "1" ]]; then
  log "Starting baseline and tuned local OpenAI-compatible endpoints"
  start_bg baseline_policy reports/service_logs/baseline_policy_18080.log \
    env LOCAL_LLM_MODEL="$MODEL_NAME" LOCAL_LLM_MAX_NEW_TOKENS="${LOCAL_LLM_MAX_NEW_TOKENS:-256}" \
      uvicorn src.inference.local_openai_server:app --host 0.0.0.0 --port 18080
  start_bg tuned_policy reports/service_logs/tuned_policy_18081.log \
    env LOCAL_LLM_MODEL="$MODEL_NAME" LOCAL_LLM_ADAPTER_PATH="$TRACKB_CHECKPOINT_DIR" \
      LOCAL_LLM_MAX_NEW_TOKENS="${LOCAL_LLM_MAX_NEW_TOKENS:-256}" \
      uvicorn src.inference.local_openai_server:app --host 0.0.0.0 --port 18081
  wait_http "http://127.0.0.1:18080/health" "baseline policy"
  wait_http "http://127.0.0.1:18081/health" "tuned policy"

  BASELINE_POLICY_URL="http://127.0.0.1:18080/v1/chat/completions" \
  BASELINE_POLICY_MODEL="$MODEL_NAME" \
  python -m src.agents.supervisor \
    --task-source postgres_mcp \
    --task-limit "$TASK_LIMIT" \
    --policy-version baseline \
    --require-live-policy \
    --output-dir trajectories/tracka_baseline_live \
    --lightning-server-url "$LIGHTNING_SERVER_URL" \
    2>&1 | tee reports/run_logs/tracka_baseline_live_latest.log
  python -m src.rewards.scorer \
    --input-dir trajectories/tracka_baseline_live \
    --output-dir trajectories/tracka_baseline_live_scored

  TUNED_POLICY_URL="http://127.0.0.1:18081/v1/chat/completions" \
  TUNED_POLICY_MODEL="$MODEL_NAME" \
  python -m src.agents.supervisor \
    --task-source postgres_mcp \
    --task-limit "$TASK_LIMIT" \
    --policy-version rl_tuned \
    --require-live-policy \
    --output-dir trajectories/tracka_tuned_live \
    --lightning-server-url "$LIGHTNING_SERVER_URL" \
    2>&1 | tee reports/run_logs/tracka_tuned_live_latest.log
  python -m src.rewards.scorer \
    --input-dir trajectories/tracka_tuned_live \
    --output-dir trajectories/tracka_tuned_live_scored

  python -m src.evaluation.compare_policies \
    --policy-dir initial_no_llm=trajectories/tracka_initial_scored \
    --policy-dir baseline_llm=trajectories/tracka_baseline_live_scored \
    --policy-dir trackb_redeploy_llm=trajectories/tracka_tuned_live_scored \
    --output reports/tracka_initial_vs_baseline_vs_trackb_redeploy.md
else
  log "Skipping live policy rerun; set RUN_LIVE_POLICY=1 to compare baseline/tuned endpoints"
  python -m src.evaluation.compare_policies \
    --policy-dir initial_no_llm=trajectories/tracka_initial_scored \
    --output reports/tracka_initial_vs_baseline_vs_trackb_redeploy.md
fi

if [[ "$START_DASHBOARD" == "1" ]]; then
  log "Starting Streamlit dashboard on ${DASHBOARD_PORT}"
  start_bg streamlit "reports/streamlit_logs/dashboard_${DASHBOARD_PORT}.log" \
    streamlit run scripts/demo_dashboard.py \
      --server.port "$DASHBOARD_PORT" \
      --server.address 0.0.0.0 \
      --server.headless true \
      --server.enableCORS false \
      --server.enableXsrfProtection false
fi

log "E2E complete"
echo "Reports:"
echo "  reports/tracka_initial_benchmark.md"
echo "  reports/tracka_initial_vs_baseline_vs_trackb_redeploy.md"
echo "Logs:"
echo "  reports/run_logs/"
echo "  reports/service_logs/"
