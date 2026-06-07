# Benchmark Strategy

The benchmark compares the baseline policy and the RL-tuned policy on held-out synthetic tabular ML tasks. The environment, tool layer, task split, scoring rubric, and artifact contracts remain fixed. The only variable that changes is the policy version used by the multi-agent runtime.

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
