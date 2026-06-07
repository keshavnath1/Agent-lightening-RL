#!/usr/bin/env bash
set -euo pipefail
DATASET=${1:-data/grpo/tool_ruler_scored_groups.jsonl}
python -m src.training.train_policy_qlora_grpo --dataset "$DATASET" --reward-mode ruler_relative --output-dir checkpoints/tool_policy_art_ruler "${@}"
