# Feedback Implementation Checklist

The new feedback says the repository is an architecture-aligned MVP but not a full implementation. The highest-priority implementation item is Track A: real tabular model benchmarking and experiment tracking. The implementation pass should therefore focus on practical CPU-plane improvements that can be validated in the current environment and later run on RunPod.

## Priority implementation items

| Priority | Feedback requirement | Implementation action |
|---|---|---|
| 1 | Implement real GBM benchmarking | Add `src/tools/gbm_benchmark.py` to train/evaluate sklearn HistGradientBoosting plus optional XGBoost/LightGBM when dependencies exist; write metrics, models, and champion metadata. |
| 2 | Replace MLflow stub with real MLflow | Update the tracker to use real `mlflow` when installed, with a JSON fallback for minimal environments. |
| 3 | Profile-driven preprocessing | Make the data engineer derive preprocessing settings from `profile_summary.json` rather than a static config. |
| 4 | High-cardinality model selection | Make the gradient boosting specialist read high-cardinality indicators from profile metadata. |
| 5 | Guardrail validation | Add `src/governance/guardrails.py` and connect reviewer/reward logic to artifact and raw-data-leakage checks. |
| 6 | Stronger reward scoring | Add `R_data`, `R_model`, `R_tracking`, `R_sandbox`, `R_reproducibility`, and `R_violation` components. |
| 7 | Multiple rollouts per task | Allow baseline workflow to generate multiple deterministic variants per task for grouped relative ranking. |
| 8 | True Docker execution | Add Docker command path to `DockerizedCodeInterpreter`, retaining local fallback where Docker is unavailable. |

## RunPod instruction questions to answer

The user asked whether to connect by SSH to the pod and whether to use VS Code or JupyterLab. The answer should be: use SSH/VS Code Remote SSH for editing and running scripts; use JupyterLab mainly for exploration/debugging notebooks; use terminal scripts for reproducible CPU/GPU pipeline execution. CPU and GPU pods should share the same RunPod Network Volume mounted at `/workspace` so outputs from one pod are visible to the other.
