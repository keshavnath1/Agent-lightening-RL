#!/usr/bin/env bash
set -euo pipefail
python -m src.training.generate_tool_scenarios "${@}"
