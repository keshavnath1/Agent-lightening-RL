# Streamlit Dashboard Runbook and GRPO E2E Update

**Author:** Manus AI  
**Last updated:** June 07, 2026

## Purpose

The Streamlit dashboard is the operator-facing handoff surface for the self-improving ML agent. It summarizes the CPU/GPU split, shows Track A readiness, explains Track B TRL GRPO trainer choices, and renders the latest policy-improvement reports. This update documents the reproducible **Track A → Track B** run and aligns the dashboard with committed sanitized E2E evidence files.

## Validated E2E state

| Area | Validated state | Evidence |
|---|---|---|
| Baseline policy benchmark | Track A runs against the baseline local OpenAI-compatible endpoint before any redeploy. | `reports/e2e_tracka_baseline_latest.md` |
| TRL GRPO training | Track B trains a PEFT LoRA adapter with TRL GRPO from grouped, scored rollouts. | `checkpoints/trackb_trl_grpo_runpod/adapter_metadata.json` |
| Adapter redeploy | The local OpenAI-compatible validation endpoint loads the trained adapter at process start through `LOCAL_LLM_ADAPTER_PATH`. | `reports/service_logs/` |
| Post-redeploy benchmark | Track A reruns after adapter redeploy for like-for-like comparison. | `reports/e2e_trackb_adapter_latest.md` |
| Before/after comparison | The comparison report records baseline vs tuned reward, endpoint use, and task-level deltas. | `reports/e2e_tracka_vs_trackb_summary_latest.json` |

## How to run the dashboard

From the repository root, create the environment with the setup script and then launch the dashboard. The same script supports CPU-only dashboard/report review and GPU training environments.

```bash
cd /workspace/self-improving-ml-agent
bash scripts/setup_environment.sh --target cpu
source .venv/bin/activate
streamlit run scripts/demo_dashboard.py   --server.port 8501   --server.address 0.0.0.0   --server.headless true   --server.enableCORS false   --server.enableXsrfProtection false
```

For a GPU pod that will train or serve the policy, use the GPU target. If CUDA-specific PyTorch wheels are needed, install PyTorch from the official wheel index for the pod's CUDA image first, then run the project requirements.

```bash
cd /workspace/self-improving-ml-agent
bash scripts/setup_environment.sh --target gpu
source .venv-gpu/bin/activate
```

## Dashboard pages after this update

| Page | What to verify |
|---|---|
| Overview | Confirms task, trajectory, grouped rollout, GPU, and service readiness. |
| Setup & Handoff | Shows CPU/GPU package checks and recommends the environment setup script plus Track A/Track B commands. |
| Track A | Shows the tabular workflow stages, champion artifacts, and grouped rollout readiness. |
| Track B | Shows trainer selection, reward mode configuration, dry-run command generation, and adapter evidence. |
| Results | Reads the newest E2E Track A/Track B markdown reports before older baseline reports. |
| Advanced Debug | Available in Developer mode for raw service diagnostics. |

## Reproducible command sequence

The next session can recreate the environment and resume from the same workflow with the following high-level commands.

```bash
# CPU/reporting environment
bash scripts/setup_environment.sh --target cpu
source .venv/bin/activate
python scripts/verify_environment.py --target cpu
streamlit run scripts/demo_dashboard.py --server.port 8501 --server.address 0.0.0.0

# GPU/training environment
bash scripts/setup_environment.sh --target gpu
source .venv-gpu/bin/activate
python scripts/verify_environment.py --target gpu
export TRAINER=trl_grpo
export REWARD_MODE=hybrid
export NUM_GENERATIONS=4
export MODEL_NAME=Qwen/Qwen2.5-3B-Instruct
export GRPO_DATASET_PATH=/workspace/self-improving-ml-agent/data/grpo/grouped_rollouts.jsonl
export POLICY_OUTPUT_DIR=/workspace/self-improving-ml-agent/checkpoints/trackb_trl_grpo_runpod
bash scripts/gpu/run_02_train_policy_qlora_grpo.sh
```

## Git handoff note

The clean archive generated for check-in intentionally excludes Python caches, virtual environments, model weights, checkpoints, raw dataset extracts, local service state, and secrets. It keeps source code, scripts, docs, requirements files, sanitized logs, trajectory JSONL, grouped-rollout JSONL, and benchmark reports needed to review the E2E run in Streamlit after checkout.


## Optional integrations

Experimental official Agent Lightning and ART/RULER integrations are isolated in `requirements-optional.txt`. Install them with `bash scripts/setup_environment.sh --target cpu --with-optional` only when those integrations are required.


## RULER vLLM judge and TRL handoff update

The dashboard now aligns with a three-track RULER handoff. **Track 1** scores grouped rollouts using either the deterministic heuristic scorer or an OpenAI-compatible local vLLM judge. **Track 2** verifies that scored RULER metadata is available to the `ruler_relative` reward mode while staying out of prompts. **Track 3** adds safe GPU operator scripts that stop at a dry-run gate unless training is explicitly enabled.

| Step | Command | Expected result |
|---|---|---|
| Check judge endpoint | `bash scripts/gpu/start_vllm_ruler_judge.sh --health` | Returns JSON with `ok: true` when a local judge is running, or an explicit connection error when it is not. |
| Score grouped rollouts | `RULER_MODE=vllm_judge bash scripts/cpu/run_06_ruler_score_groups.sh` | Writes `data/grpo/ruler_scored_groups.jsonl` and `reports/ruler_scoring_summary.md`. |
| Dry-run TRL handoff | `bash scripts/gpu/run_ruler_trl_handoff.sh --dry-run` | Scores data if needed, validates RULER field coverage, checks prompt leakage, and prints the exact training command. |
| Launch GPU training | `bash scripts/gpu/run_ruler_trl_handoff.sh --train` | Runs the same validated handoff and then launches `scripts/gpu/run_02_train_policy_qlora_grpo.sh`. |

The Streamlit Track B page should be used as the operator explanation surface, while the shell scripts remain the source of truth for actual GPU execution.
