#!/usr/bin/env bash
set -euo pipefail
IN=${1:-trajectories/tool_learning_rollouts}
SCORED=${2:-trajectories/tool_learning_rollouts_scored}
GROUPED=${3:-data/grpo/tool_learning_grouped.jsonl}
python -m src.rewards.scorer --input-dir "$IN" --output-dir "$SCORED"
python -m src.training.prepare_grpo_dataset --input-dir "$SCORED" --output "$GROUPED"
python -m src.rewards.tool_ruler_evaluator --input "$GROUPED" --output data/grpo/tool_ruler_scored_groups.jsonl --report reports/tool_ruler_evaluation.md
python -m src.evaluation.tool_use_metrics --trajectories "$SCORED" --output reports/tool_use_metrics.md
