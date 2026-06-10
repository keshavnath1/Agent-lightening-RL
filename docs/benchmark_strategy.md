# Benchmark Strategy

The current benchmark compares **Track A with the baseline LLM** against **Track A after the tuned Track B adapter is redeployed**. The environment, OpenML task set, PostgreSQL/MCP boundary, scoring rubric, and artifact contracts remain fixed. The policy endpoint is the main variable: first the baseline hosted model, then the tuned model loaded with the Track B TRL GRPO adapter.

The default fresh-RunPod command is `bash scripts/runpod_bootstrap_e2e.sh`. Set `RUN_LIVE_POLICY=1` when the run should start baseline and tuned local OpenAI-compatible endpoints and write the before/after comparison.

| Stage | Evidence |
|---|---|
| Track A grouped training evidence | `reports/tracka_initial_benchmark.md` |
| Track B TRL GRPO training log | `reports/run_logs/trackb_trl_grpo_latest.log` |
| Baseline LLM vs tuned LLM comparison | `reports/tracka_initial_vs_baseline_vs_trackb_redeploy.md` |
| Run summary for Streamlit | `reports/e2e_grpo_run_summary.json` |

| Metric | Direction | Meaning |
|---|---:|---|
| End-to-end task success rate | Higher | The full workflow completed with required artifacts. |
| Average reward score | Higher | Composite score across task completion, tool use, SQL, code, ML quality, logging, and efficiency. |
| Valid tool-call rate | Higher | The policy selected known tools with valid arguments. |
| SQL execution success rate | Higher | SQL calls executed without errors and followed the read/write contract. |
| Code execution success rate | Higher | Controlled code execution completed successfully. |
| Profiling completion rate | Higher | Profile summary artifact was produced. |
| MLflow/DVC completeness | Higher | Experiment metadata and artifacts were logged. |
| Retries per task | Lower | The policy required fewer failed attempts. |
| Cost/time proxy per success | Lower | Successful workflows required less execution time and fewer calls. |
