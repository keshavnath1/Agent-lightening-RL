# Multi-Policy Benchmark Evaluation

This report compares named policy trajectory directories, for example baseline, V2, and RL-tuned GPU endpoint runs. The `live_policy_endpoint_rate` column verifies whether each trajectory used the strict live OpenAI-compatible policy endpoint wrapper.

| policy         |   tasks |   avg_reward |   task_success_rate |   valid_tool_rate |   live_policy_endpoint_rate |
|:---------------|--------:|-------------:|--------------------:|------------------:|----------------------------:|
| initial_no_llm |      16 |         0.85 |                   1 |                 1 |                           0 |

## Per-Task Reward Matrix

| task_id                                    |   initial_no_llm |
|:-------------------------------------------|-----------------:|
| mltask_openml_1461_bank_marketing_baseline |           0.8776 |
| mltask_openml_1489_phoneme_baseline        |           0.771  |
| mltask_openml_31_german_credit_baseline    |           0.843  |
| mltask_openml_44_spambase_baseline         |           0.9085 |
