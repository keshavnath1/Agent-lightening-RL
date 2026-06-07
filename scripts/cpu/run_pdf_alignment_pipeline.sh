#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
python -m src.training.generate_tool_scenarios --output data/synthetic/tool_scenarios.jsonl --num-scenarios 32 || true
bash scripts/cpu/run_01_generate_synthetic.sh
bash scripts/cpu/run_02_load_synthetic_to_postgres.sh
bash scripts/cpu/run_03_run_baseline_workflow.sh
bash scripts/cpu/run_04_score_trajectories.sh
bash scripts/cpu/run_05_prepare_grpo_dataset.sh
: "${RULER_JUDGE_BASE_URL:?Set RULER_JUDGE_BASE_URL before running the RULER scoring stage}"
python -m src.rewards.apply_ruler_scores --input data/grpo/grouped_rollouts.jsonl --output data/grpo/ruler_scored_groups.jsonl --mode vllm_judge --judge-model "${RULER_JUDGE_MODEL:-Qwen/Qwen2.5-3B-Instruct}" --judge-base-url "$RULER_JUDGE_BASE_URL" --judge-api-key "${RULER_JUDGE_API_KEY:-EMPTY}" --alpha 0.7 --beta 0.3
python -m src.training.build_heldout_policy_eval --input data/grpo/ruler_scored_groups.jsonl --output-dir data/grpo
python -m src.evaluation.tool_use_metrics --trajectories trajectories/scored --output reports/tool_use_metrics.md
python -m src.evaluation.flywheel_report --output reports/self_improving_flywheel.md
python scripts/dev_validate_pdf_alignment.py
