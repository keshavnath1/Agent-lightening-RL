#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_DIR"

: "${RULER_JUDGE_MODEL:=Qwen/Qwen2.5-3B-Instruct}"
: "${RULER_JUDGE_SERVED_MODEL:=$RULER_JUDGE_MODEL}"
: "${RULER_JUDGE_HOST:=0.0.0.0}"
: "${RULER_JUDGE_PORT:=8001}"
: "${RULER_JUDGE_BASE_URL:=http://127.0.0.1:${RULER_JUDGE_PORT}/v1}"
: "${RULER_JUDGE_API_KEY:=EMPTY}"
: "${RULER_JUDGE_TENSOR_PARALLEL_SIZE:=1}"
: "${RULER_JUDGE_GPU_MEMORY_UTILIZATION:=0.35}"
: "${RULER_JUDGE_MAX_MODEL_LEN:=4096}"

usage() {
  cat <<USAGE
Usage: bash scripts/gpu/start_vllm_ruler_judge.sh [--health|--print-env|--serve]

Starts or checks a local OpenAI-compatible vLLM endpoint used only as the
RULER judge. The endpoint is intentionally separate from the policy endpoint so
judge traffic does not accidentally change the hosted policy server.

Environment variables:
  RULER_JUDGE_MODEL                    default: Qwen/Qwen2.5-3B-Instruct
  RULER_JUDGE_SERVED_MODEL             default: same as model
  RULER_JUDGE_HOST                     default: 0.0.0.0
  RULER_JUDGE_PORT                     default: 8001
  RULER_JUDGE_BASE_URL                 default: http://127.0.0.1:8001/v1
  RULER_JUDGE_API_KEY                  default: EMPTY
  RULER_JUDGE_TENSOR_PARALLEL_SIZE     default: 1
  RULER_JUDGE_GPU_MEMORY_UTILIZATION   default: 0.35
  RULER_JUDGE_MAX_MODEL_LEN            default: 4096
USAGE
}

health() {
  python - <<'PY'
import json, os, sys, urllib.request
base = os.environ.get("RULER_JUDGE_BASE_URL", "http://127.0.0.1:8001/v1").rstrip("/")
url = base + "/models"
try:
    req = urllib.request.Request(url, headers={"Authorization": "Bearer " + os.environ.get("RULER_JUDGE_API_KEY", "EMPTY")})
    with urllib.request.urlopen(req, timeout=5) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    models = [m.get("id", "unknown") for m in payload.get("data", [])]
    print(json.dumps({"ok": True, "base_url": base, "models": models}, indent=2))
except Exception as exc:
    print(json.dumps({"ok": False, "base_url": base, "error": str(exc)}, indent=2))
    sys.exit(1)
PY
}

print_env() {
  cat <<ENV
export RULER_MODE=vllm_judge
export RULER_JUDGE_BASE_URL="$RULER_JUDGE_BASE_URL"
export RULER_JUDGE_API_KEY="$RULER_JUDGE_API_KEY"
export RULER_JUDGE_MODEL="$RULER_JUDGE_SERVED_MODEL"
ENV
}

case "${1:---help}" in
  --help|-h)
    usage
    exit 0
    ;;
  --health)
    health
    exit 0
    ;;
  --print-env)
    print_env
    exit 0
    ;;
  --serve|"")
    ;;
  *)
    echo "Unknown option: $1" >&2
    usage >&2
    exit 2
    ;;
esac

if ! python - <<'PY' >/dev/null 2>&1
import vllm
PY
then
  echo "vLLM is not installed in this environment." >&2
  echo "Install the GPU environment first, then retry: bash scripts/setup_environment.sh --target gpu" >&2
  exit 78
fi

exec python -m vllm.entrypoints.openai.api_server \
  --model "$RULER_JUDGE_MODEL" \
  --served-model-name "$RULER_JUDGE_SERVED_MODEL" \
  --host "$RULER_JUDGE_HOST" \
  --port "$RULER_JUDGE_PORT" \
  --tensor-parallel-size "$RULER_JUDGE_TENSOR_PARALLEL_SIZE" \
  --gpu-memory-utilization "$RULER_JUDGE_GPU_MEMORY_UTILIZATION" \
  --max-model-len "$RULER_JUDGE_MAX_MODEL_LEN"
