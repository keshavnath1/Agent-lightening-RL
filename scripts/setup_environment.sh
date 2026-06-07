#!/usr/bin/env bash
set -euo pipefail

TARGET="cpu"
PYTHON_BIN="${PYTHON_BIN:-python3}"
INSTALL_TORCH="${INSTALL_TORCH:-auto}"
PYTORCH_VERSION="${PYTORCH_VERSION:-2.8.0}"
WITH_OPTIONAL="0"

usage() {
  cat <<'USAGE'
Usage: bash scripts/setup_environment.sh [--target cpu|gpu] [--python python3.11] [--install-torch yes|no|auto] [--with-optional]

Creates a local virtual environment for the self-improving ML agent.

Targets:
  cpu  Streamlit dashboard, Track A orchestration, reports, and local validation.
  gpu  Track B TRL GRPO training and OpenAI-compatible validation serving.

Examples:
  bash scripts/setup_environment.sh --target cpu
  bash scripts/setup_environment.sh --target gpu --install-torch auto
  PYTORCH_VERSION=2.8.0 bash scripts/setup_environment.sh --target gpu
  bash scripts/setup_environment.sh --target cpu --with-optional
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target) TARGET="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --install-torch) INSTALL_TORCH="$2"; shift 2 ;;
    --with-optional) WITH_OPTIONAL="1"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ "$TARGET" != "cpu" && "$TARGET" != "gpu" ]]; then
  echo "--target must be cpu or gpu" >&2
  exit 2
fi

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_DIR"

if [[ "$TARGET" == "gpu" ]]; then
  VENV_DIR="${VENV_DIR:-.venv-gpu}"
  REQ_FILE="requirements-gpu.txt"
else
  VENV_DIR="${VENV_DIR:-.venv}"
  REQ_FILE="requirements-cpu.txt"
fi

"$PYTHON_BIN" -m venv "$VENV_DIR"
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip setuptools wheel

if [[ "$TARGET" == "gpu" ]]; then
  if [[ "$INSTALL_TORCH" == "yes" || "$INSTALL_TORCH" == "auto" ]]; then
    if ! python - <<'PY' >/dev/null 2>&1
import torch
from torch.distributed.fsdp import FSDPModule
print(torch.__version__)
print(FSDPModule.__name__)
PY
    then
      echo "Installing PyTorch ${PYTORCH_VERSION}. Set INSTALL_TORCH=no if your base image already provides a compatible torch."
      python -m pip install --upgrade "torch==${PYTORCH_VERSION}"
    else
      echo "Compatible torch already importable; keeping existing torch install."
    fi
  fi
fi

python -m pip install -r "$REQ_FILE"
if [[ "$WITH_OPTIONAL" == "1" ]]; then
  python -m pip install -r requirements-optional.txt
fi
python scripts/verify_environment.py --target "$TARGET"

echo ""
echo "Environment ready: $VENV_DIR"
echo "Activate with: source $VENV_DIR/bin/activate"
