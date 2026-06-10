# RULER vLLM Judge to TRL GRPO Handoff Runbook

**Author:** Manus AI  
**Last updated:** June 10, 2026

This runbook documents the one-track-at-a-time RULER handoff path. The goal is to score grouped rollouts with a RULER-style judge, preserve those scores as reward metadata, and hand the scored dataset to the repository-native **TRL GRPO** training entrypoint without leaking score labels into model prompts. The default fresh RunPod path uses `REWARD_MODE=hybrid`; use this handoff when you explicitly want `REWARD_MODE=ruler_relative`.

| Track | Implementation area | Primary files | Operator gate |
|---|---|---|---|
| Track 1 | RULER scoring foundation with optional local vLLM judge | `src/config/ruler_config.py`, `src/rewards/ruler_vllm_judge.py`, `src/rewards/apply_ruler_scores.py` | `python -m src.rewards.apply_ruler_scores --help` and a scoring dry-run write JSONL plus Markdown summary. |
| Track 2 | TRL reward integration and metadata safety checks | `src/training/train_policy_qlora_grpo.py`, `src/rewards/policy_reward.py` | `--dry-run` reports non-null RULER field coverage and confirms scores do not leak into prompts. |
| Track 3 | GPU handoff scripts and Streamlit-facing documentation | `scripts/gpu/start_vllm_ruler_judge.sh`, `scripts/gpu/run_ruler_trl_handoff.sh`, `docs/streamlit_dashboard_runbook.md` | Handoff script runs in safe dry-run mode and prints the exact training command before any GPU job starts. |

## Step-by-step commands

The default path is intentionally safe. It checks the judge endpoint, scores grouped rollouts, validates TRL input construction, and then stops before launching a long GPU training job.

```bash
cd /workspace/self-improving-ml-agent
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

# Optional: launch a separate local judge endpoint in another shell.
RULER_JUDGE_MODEL=Qwen/Qwen2.5-3B-Instruct RULER_JUDGE_PORT=8001 bash scripts/gpu/start_vllm_ruler_judge.sh --serve

# Check whether the judge endpoint is reachable.
bash scripts/gpu/start_vllm_ruler_judge.sh --health || true

# Safe RULER -> TRL dry-run. This does not launch training.
RULER_MODE=vllm_judge RULER_JUDGE_BASE_URL=http://127.0.0.1:8001/v1 RULER_JUDGE_API_KEY=EMPTY bash scripts/gpu/run_ruler_trl_handoff.sh --dry-run
```

After reviewing the generated `data/grpo/ruler_scored_groups.jsonl`, `reports/ruler_vllm_scoring_summary.md`, and the TRL dry-run output, run the same handoff with training enabled.

```bash
RULER_MODE=vllm_judge RULER_JUDGE_BASE_URL=http://127.0.0.1:8001/v1 RULER_JUDGE_API_KEY=EMPTY TRAINER=trl_grpo REWARD_MODE=ruler_relative bash scripts/gpu/run_ruler_trl_handoff.sh --train
```

## Privacy and prompt-safety rule

RULER scores are reward metadata only. The training dry-run should show coverage for fields such as `ruler_relative_score`, `final_hybrid_reward`, `ruler_rank`, and `ruler_reason`, while the prompt-leakage check should remain clean. If the leakage check fails, stop before training and inspect the dataset builder.
