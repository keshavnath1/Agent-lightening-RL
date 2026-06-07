# Multi-Policy Benchmark Evaluation

This report compares named policy trajectory directories, for example baseline, V2, and RL-tuned GPU endpoint runs. The `live_policy_endpoint_rate` column verifies whether each trajectory used the strict live OpenAI-compatible policy endpoint wrapper.

| policy              |   tasks |   avg_reward |   task_success_rate |   valid_tool_rate |   live_policy_endpoint_rate |
|:--------------------|--------:|-------------:|--------------------:|------------------:|----------------------------:|
| initial_no_llm      |      16 |         0.85 |                   1 |                 1 |                           0 |
| baseline_llm        |       4 |         0.85 |                   1 |                 1 |                           1 |
| trackb_redeploy_llm |       4 |         0.85 |                   1 |                 1 |                           1 |

## Per-Task Reward Matrix

| task_id                                    |   baseline_llm |   initial_no_llm |   trackb_redeploy_llm |
|:-------------------------------------------|---------------:|-----------------:|----------------------:|
| mltask_openml_1461_bank_marketing_baseline |         0.8776 |           0.8776 |                0.8776 |
| mltask_openml_1489_phoneme_baseline        |         0.771  |           0.771  |                0.771  |
| mltask_openml_31_german_credit_baseline    |         0.843  |           0.843  |                0.843  |
| mltask_openml_44_spambase_baseline         |         0.9085 |           0.9085 |                0.9085 |
