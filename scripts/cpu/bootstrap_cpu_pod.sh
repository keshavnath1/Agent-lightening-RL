#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/workspace/self-improving-ml-agent}"
cd "$REPO_DIR"

export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export WORKSPACE_DIR="$REPO_DIR"

python3 --version
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements-cpu.txt

# Optional extras used by the validated local Track A benchmark. If the image
# already has these packages, pip will no-op or upgrade only when needed.
python3 -m pip install pyarrow scikit-learn xgboost lightgbm

mkdir -p data/synthetic data/grpo artifacts trajectories/baseline trajectories/scored reports checkpoints logs

python3 -m py_compile $(find src -name '*.py' -print)

echo "CPU pod bootstrap complete. Repository: $REPO_DIR"
echo "Next smoke-test sequence:"
echo "  export TASK_COUNT=5 TASK_LIMIT=2 ROLLOUTS_PER_TASK=2"
echo "  scripts/cpu/run_01_generate_synthetic.sh"
echo "  scripts/cpu/run_03_run_baseline_workflow.sh"
echo "  scripts/cpu/run_04_score_trajectories.sh"
echo "  scripts/cpu/run_05_prepare_grpo_dataset.sh"
echo "  scripts/cpu/run_05b_export_agent_lightning_transitions.sh"
