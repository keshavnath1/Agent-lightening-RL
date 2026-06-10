# Track A Baseline LLM vs Track A Tuned LLM

This report compares the two live-policy Track A evaluation passes:

1. **Track A with baseline LLM**: the base hosted vLLM policy endpoint.
2. **Track A with tuned LLM**: the same Track A benchmark after Track B TRL GRPO
   training and adapter redeploy.

The initial Track A rollouts are still important, but they are not shown as a
benchmark policy here. They are training/evidence data: they are scored, grouped,
and used to produce the Track B GRPO training dataset.

## End-To-End Flow

1. Run initial Track A rollouts to create scored trajectory evidence.
2. Build grouped rollout data from those scored trajectories.
3. Train Track B with TRL GRPO.
4. Run Track A with the baseline LLM endpoint.
5. Redeploy the Track B tuned adapter behind vLLM.
6. Run Track A with the tuned LLM endpoint.
7. Compare baseline LLM vs tuned LLM on the same OpenML tasks.

## Summary Metrics

Both rows below are live vLLM policy endpoint evaluations. The
`live_policy_endpoint_rate` should be `1.0` for both, because both benchmark
passes use the strict `PolicyClient.chat` endpoint wrapper.

| policy       | display_name              | scored_trajectories | avg_reward | task_success_rate | valid_tool_rate | live_policy_endpoint_rate |
|:-------------|:--------------------------|--------------------:|-----------:|------------------:|----------------:|--------------------------:|
| baseline_llm | Track A with baseline LLM |                   4 |       0.85 |               1.0 |             1.0 |                       1.0 |
| tuned_llm    | Track A with tuned LLM    |                   4 |       0.85 |               1.0 |             1.0 |                       1.0 |

## Per-Task Reward Matrix

| task_id                                    | baseline_llm | tuned_llm |
|:-------------------------------------------|-------------:|----------:|
| mltask_openml_1461_bank_marketing_baseline |       0.8776 |    0.8776 |
| mltask_openml_1489_phoneme_baseline        |       0.7710 |    0.7710 |
| mltask_openml_31_german_credit_baseline    |       0.8430 |    0.8430 |
| mltask_openml_44_spambase_baseline         |       0.9085 |    0.9085 |

## Interpretation

The tuned LLM completed the same Track A benchmark as the baseline LLM and did
not regress. The average reward is identical in this run:

| Comparison | Value |
|:--|--:|
| Baseline LLM average reward | 0.85 |
| Tuned LLM average reward | 0.85 |
| Delta | +0.00 |

So the correct claim is:

**Track B TRL GRPO training and redeploy completed successfully, and the tuned
LLM matched the baseline LLM on this four-task Track A benchmark.**

The current result should not be presented as measured policy improvement yet,
because the tuned and baseline rewards are identical on every task.

## What This Proves

1. The baseline LLM endpoint can run Track A through the strict policy wrapper.
2. Track B TRL GRPO produced a trained adapter.
3. The tuned adapter was redeployed behind vLLM.
4. The tuned LLM endpoint can run Track A through the same strict policy wrapper.
5. The tuned LLM did not regress versus the baseline LLM on the checked-in
   four-task benchmark.

## What The Initial Track A Rollouts Are Used For

The initial Track A rollouts are not part of this baseline-vs-tuned benchmark
table. They are used upstream:

| Artifact | Purpose |
|:--|:--|
| `trajectories/tracka_initial/` | Raw initial Track A rollout traces. |
| `trajectories/tracka_initial_scored/` | Scored initial trajectories. |
| `data/grpo/grouped_rollouts.jsonl` | Grouped rollout dataset used by TRL GRPO. |
| `reports/tracka_initial_benchmark.md` | Evidence-only report for the initial rollout data. |

## Demo Script

Use this language in the demo:

> We first generate scored Track A trajectories to create the GRPO training
> signal. Then Track B trains a TRL GRPO adapter from those grouped rollouts.
> After training, we compare two live Track A evaluations: baseline LLM versus
> tuned LLM. In this run the tuned LLM successfully redeployed and matched the
> baseline reward, so we can claim end-to-end RL infrastructure validation and
> no regression. We should not claim policy lift yet because the scores are the
> same.

## Next Benchmark To Show Lift

To demonstrate real improvement instead of no-regression:

1. Add held-out OpenML tasks that were not used to build the GRPO grouped
   rollout dataset.
2. Run multiple baseline and tuned LLM rollouts per task.
3. Report reward components separately.
4. Compare tool ordering, guardrail violations, MLflow completeness, and
   explainability artifacts.
5. Track min/mean/max or confidence intervals by policy.
