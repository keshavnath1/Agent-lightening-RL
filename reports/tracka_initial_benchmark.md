# Track A Initial Training Evidence

This report summarizes the initial Track A rollout set used to create scored
trajectory evidence and grouped rollout data for TRL GRPO.

This file is **not** the baseline-vs-tuned benchmark. The actual benchmark
comparison is:

`reports/tracka_initial_vs_baseline_vs_trackb_redeploy.md`

## Summary

| evidence_set | scored_trajectories | avg_reward | task_success_rate | valid_tool_rate |
|:--|--:|--:|--:|--:|
| tracka_training_evidence | 16 | 0.85 | 1.0 | 1.0 |

## Per-Task Reward Matrix

| task_id                                    | tracka_training_evidence |
|:-------------------------------------------|-------------------------:|
| mltask_openml_1461_bank_marketing_baseline |                   0.8776 |
| mltask_openml_1489_phoneme_baseline        |                   0.7710 |
| mltask_openml_31_german_credit_baseline    |                   0.8430 |
| mltask_openml_44_spambase_baseline         |                   0.9085 |

## Purpose

The initial Track A rollout set is used upstream:

1. Generate multiple trajectories per OpenML task.
2. Score each trajectory with the project reward function.
3. Group rollouts by task.
4. Write `data/grpo/grouped_rollouts.jsonl`.
5. Train Track B with TRL GRPO.

After Track B training and redeploy, the benchmark compares only:

1. Track A with baseline LLM.
2. Track A with tuned LLM.
