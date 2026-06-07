#!/usr/bin/env bash
set -euo pipefail
TASKS=${1:-data/synthetic/tool_scenarios.jsonl}
OUT=${2:-trajectories/tool_learning_rollouts}
python -m src.agents.supervisor --tasks "$TASKS" --output-dir "$OUT" --policy-version tool_learning --rollouts "${ROLLOUTS:-2}"
