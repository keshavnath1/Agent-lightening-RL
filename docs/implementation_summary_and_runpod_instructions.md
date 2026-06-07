# Implementation Summary and RunPod MVP Instructions

**Author:** Manus AI  
**Date:** May 30, 2026

## Executive summary

This implementation converts the feedback into a concrete, runnable MVP for a **self-improving tabular ML agent**. The repository now separates the system into a **CPU control and tool plane** and a **GPU policy and training plane**, which is the right operational split for a RunPod demonstration. The CPU plane owns synthetic task creation, tabular tooling, orchestration, telemetry, reward scoring, and evaluation reports. The GPU plane owns baseline policy inference, LoRA/GRPO-style policy training, and tuned-policy inference. This split also keeps the project honest about the distinction between **training task models** such as XGBoost or LightGBM and **training the LLM agent policy** that decides which tools to call and how to improve the tabular workflow.

The first implementation pass is intentionally lightweight. It includes runnable scripts, typed data contracts, trajectory logging, reward scoring, grouped-rollout dataset preparation, and a GPU training stub that can be replaced by the full `transformers`, `peft`, and `trl` implementation once the RunPod GPU image is available. The local validation run completed successfully in the sandbox after installing the Parquet dependency, proving that the MVP control loop works from synthetic task creation through comparison reporting.

## What was implemented

The repository at `/home/ubuntu/self-improving-ml-agent` now contains a runnable scaffold for the revised architecture. The codebase is organized so that each phase of the demo can be executed independently, persisted to shared storage, and resumed across separate CPU and GPU pods.

| Area | Implemented artifact | Purpose |
|---|---|---|
| Configuration | `config/settings.toml`, `.env.example`, Dockerfiles, requirements files | Defines ports, paths, model IDs, RunPod workspace assumptions, and separate CPU/GPU dependency sets. |
| Agent orchestration | `src/agents/*` | Provides baseline supervisor, data engineer, model specialist, reviewer, telemetry, and sandbox execution components. |
| Tool plane | `src/tools/*`, `src/mcp_server/server.py` | Establishes interfaces for SQL, profiling, experiment tracking, code execution, and tool registry/MCP exposure. |
| Synthetic workload | `src/synthetic_data/generate_tasks.py`, `load_to_postgres.py` | Creates tabular benchmark tasks and provides a PostgreSQL loader path for the full CPU pod. |
| Policy inference | `src/inference/*` | Defines the OpenAI-compatible policy client and batch inference entrypoint for baseline and tuned policies. |
| Reward and RL data | `src/rewards/scorer.py`, `src/training/prepare_grpo_dataset.py` | Scores trajectories and converts grouped rollouts into an RL-ready dataset. |
| Training | `src/training/train_policy_qlora_grpo.py` | Provides a safe MVP training placeholder that writes adapter metadata and documents where the full QLoRA/GRPO trainer will plug in. |
| Evaluation | `src/evaluation/compare_policies.py` | Produces the baseline-vs-tuned report from scored trajectories. |
| Execution scripts | `scripts/cpu/*.sh`, `scripts/gpu/*.sh` | Provides CPU and GPU commands for the first end-to-end MVP loop. |

The generated repository follows the feedback’s main design constraint: **LLMs receive metadata, summaries, and artifact references rather than raw tabular rows**. This is reflected in the synthetic task contracts, trajectory schema, and code execution wrapper.

## Validation completed

A lightweight validation run was executed against the repository. The run compiled the source modules and exercised the MVP loop with six synthetic tasks and four baseline trajectories. The GPU phase was validated through the training stub rather than an actual GRPO job, because the sandbox does not provide a RunPod GPU runtime.

| Validation step | Command family | Result |
|---|---|---|
| Python syntax check | `python3.11 -m compileall -q src` | Passed. |
| Synthetic task generation | `scripts/cpu/run_01_generate_synthetic.sh` | Wrote `data/synthetic/tasks.jsonl`. |
| Baseline workflow | `scripts/cpu/run_03_run_baseline_workflow.sh` | Wrote four baseline trajectories. |
| Reward scoring | `scripts/cpu/run_04_score_trajectories.sh` | Wrote scored trajectories under `trajectories/scored`. |
| GRPO dataset preparation | `scripts/cpu/run_05_prepare_grpo_dataset.sh` | Wrote `data/grpo/grouped_rollouts.jsonl`. |
| Policy training placeholder | `scripts/gpu/run_02_train_policy_qlora_grpo.sh` | Wrote adapter metadata under `checkpoints/qwen25-3b-agent-lora`. |
| Baseline-vs-tuned comparison | `scripts/cpu/run_06_compare_baseline_vs_tuned.sh` | Wrote `reports/baseline_vs_rl_tuned.md`. |

The first validation attempt exposed that the sandbox lacked `pyarrow`, which is required for Parquet writes. The repository already lists `pyarrow` in the CPU requirements, so the missing dependency was an environment issue rather than an application logic issue.

## Recommended RunPod deployment layout

RunPod network volumes provide storage that persists independently of compute resources and are mounted at `/workspace` for pods, which makes them a suitable handoff point between the CPU and GPU planes.[1] The recommended MVP setup is a shared RunPod network volume with this repository checked out under `/workspace/self-improving-ml-agent`.

| Resource | Recommended role | Notes |
|---|---|---|
| RunPod Network Volume | Shared `/workspace` | Stores repository, synthetic data, trajectories, checkpoints, reports, and model cache. |
| CPU Pod | Tool/control plane | Runs PostgreSQL, MCP server, MLflow, synthetic workload, workflow execution, scoring, and comparison. |
| GPU Pod | Policy plane | Runs vLLM baseline/tuned inference and policy fine-tuning. |
| Model cache | `/workspace/.cache/huggingface` | Prevents repeated model downloads across pod restarts. |
| Experiment tracking | MLflow plus optional DVC | MLflow can track experiments and LLM/agent workflows, while DVC provides Git-like version control for data, models, and experiments.[6] [7] |

The selected base policy is `Qwen/Qwen2.5-3B-Instruct`. Its model card documents the Qwen2.5 3B instruction-tuned model, the Transformers loading path, and a vLLM serving example.[3] vLLM is appropriate for the inference plane because it exposes an OpenAI-compatible serving interface, which matches the policy client abstraction already implemented in the repository.[2] The RL training plan uses LoRA/QLoRA-style adapter training to avoid full-model fine-tuning; LoRA freezes the base weights and trains smaller update matrices, reducing trainable parameters and memory requirements.[5] TRL’s GRPO trainer is the intended full training target because it supports GRPO for language model post-training using reward functions over sampled completions.[4]

## CPU pod quick start

Start by deploying a CPU pod with the shared network volume attached. After opening a terminal in the pod, place the repository under `/workspace` and create an environment from the CPU dependency file.

```bash
cd /workspace
# If using the attached archive, unzip it here. If using Git later, clone the repo here.
cd /workspace/self-improving-ml-agent
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-cpu.txt
cp .env.example .env
```

For the full CPU-plane services, initialize PostgreSQL, start MLflow, and start the MCP server in separate terminals. The service scripts are included, but the local sandbox validation used the file-based MVP path rather than standing up PostgreSQL.

```bash
bash scripts/cpu/start_postgres.sh
bash scripts/cpu/start_mlflow.sh
bash scripts/cpu/start_mcp_server.sh
```

Then run the CPU-side workload and trajectory preparation flow.

```bash
export WORKSPACE_DIR=/workspace/self-improving-ml-agent
export TASK_COUNT=100
export TASK_LIMIT=25

bash scripts/cpu/run_01_generate_synthetic.sh
bash scripts/cpu/run_02_load_synthetic_to_postgres.sh
bash scripts/cpu/run_03_run_baseline_workflow.sh
bash scripts/cpu/run_04_score_trajectories.sh
bash scripts/cpu/run_05_prepare_grpo_dataset.sh
```

For an initial smoke test, use `TASK_COUNT=6` and `TASK_LIMIT=4`. For a stronger demo, scale to at least 100 synthetic tasks and keep a held-out set that is not used to prepare training rollouts.

## GPU pod quick start

Deploy a GPU pod against the same network volume. Use a CUDA-enabled base image and install the GPU requirements. If Hugging Face access is required for gated models in later experiments, configure the appropriate token in the environment before model download.

```bash
cd /workspace/self-improving-ml-agent
python3.11 -m venv .venv-gpu
source .venv-gpu/bin/activate
pip install --upgrade pip
pip install -r requirements-gpu.txt
export HF_HOME=/workspace/.cache/huggingface
export WORKSPACE_DIR=/workspace/self-improving-ml-agent
```

Start the baseline inference server first. The current scripts are designed around the OpenAI-compatible vLLM endpoint and the Qwen model family.[2] [3]

```bash
bash scripts/gpu/start_baseline_inference.sh
```

In another GPU-pod terminal, run baseline batch inference if you want policy-generated rollouts rather than the deterministic local baseline. Then run policy training and tuned inference.

```bash
bash scripts/gpu/run_01_baseline_batch_inference.sh
bash scripts/gpu/run_02_train_policy_qlora_grpo.sh
bash scripts/gpu/start_tuned_inference.sh
bash scripts/gpu/run_03_tuned_batch_inference.sh
```

The MVP training script currently writes adapter metadata and validates the handoff contract. The next implementation step is to replace the placeholder with a concrete TRL `GRPOTrainer` job that loads `data/grpo/grouped_rollouts.jsonl`, applies a PEFT LoRA configuration, and writes adapter weights to `checkpoints/qwen25-3b-agent-lora`.[4] [5]

## Final CPU comparison

After the tuned-policy trajectories are available on the shared volume, return to the CPU pod and run the comparison report.

```bash
cd /workspace/self-improving-ml-agent
source .venv/bin/activate
export WORKSPACE_DIR=/workspace/self-improving-ml-agent
bash scripts/cpu/run_06_compare_baseline_vs_tuned.sh
```

The report is written to `reports/baseline_vs_rl_tuned.md`. For a credible demonstration, the report should compare held-out task success, mean reward, compile/runtime failures, tool-call counts, and artifact quality. The current MVP report already computes the first-order reward comparison and can be extended with task-model metrics once the XGBoost/LightGBM training tools are expanded.

## Immediate next implementation steps

The next pass should convert the training placeholder into a real GPU training job. That means implementing the PEFT LoRA configuration, wiring TRL’s `GRPOTrainer`, and adding a reward-function adapter that reads the existing trajectory reward fields. The inference phase should then generate multiple rollouts per task so that grouped relative rewards are meaningful rather than copied from a single baseline trajectory. This is also the right point to add DVC stages for synthetic data generation, trajectory scoring, GRPO dataset creation, training, and evaluation, because the user preference is to keep ML workflow artifacts versioned with DVC where practical.[7]

A second pass should harden the CPU plane. The PostgreSQL loader should be exercised against a real RunPod CPU pod, the MCP server should expose the tool registry over the selected protocol, and the code interpreter wrapper should be replaced with a locked-down container execution path. These steps are necessary before allowing arbitrary generated Python to run beyond controlled smoke tests.

A third pass should improve evaluation. The current evaluator compares reward summaries; it should also measure downstream tabular model quality, artifact completeness, reproducibility, and violation counts for guardrails such as raw-row leakage. This will make the final demo more persuasive because it will show that policy improvement changes both agent behavior and tabular workflow outcomes.

## References

[1]: https://docs.runpod.io/storage/network-volumes "RunPod Documentation: Network volumes"  
[2]: https://docs.vllm.ai/en/stable/serving/openai_compatible_server/ "vLLM Documentation: OpenAI-compatible server"  
[3]: https://huggingface.co/Qwen/Qwen2.5-3B-Instruct "Hugging Face: Qwen/Qwen2.5-3B-Instruct"  
[4]: https://huggingface.co/docs/trl/en/grpo_trainer "Hugging Face TRL Documentation: GRPO Trainer"  
[5]: https://huggingface.co/docs/peft/main/en/conceptual_guides/lora "Hugging Face PEFT Documentation: LoRA"  
[6]: https://mlflow.org/docs/latest/ "MLflow Documentation"  
[7]: https://dvc.org/doc "DVC Documentation"
