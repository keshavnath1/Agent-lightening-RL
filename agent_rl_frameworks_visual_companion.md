# Visual Companion: RL Frameworks and Our Self-Improving ML Agent

Use this file while recording or presenting the podcast. It follows the same learning arc: RL basics, current project pipeline, SFT as contrast, current TRL GRPO Track B, Agent Lightning, MCP, scoring, benchmark evidence, and repo structure.

---

## 1. Current Project Pipeline

The current E2E path is:

```mermaid
flowchart TD
  A["Hosted base/instruct model<br/>vLLM or local OpenAI-compatible endpoint"]
  B["Track A rollouts<br/>agent executes ML workflow"]
  C["Reward/RULER-style scoring<br/>trajectory becomes learning signal"]
  D["Grouped rollout dataset<br/>data/grpo/grouped_rollouts.jsonl"]
  E["Track B TRL GRPO<br/>train LoRA adapter from grouped rewards"]
  F["Redeploy tuned adapter<br/>local policy endpoint"]
  G["Rerun Track A"]
  H["Compare baseline vs tuned<br/>reward, tool validity, success, artifacts"]

  A --> B --> C --> D --> E --> F --> G --> H
```

Speaker note:

The project uses RL vocabulary because the system is built around trajectories, rewards, grouped rollouts, and policy improvement. The default Track B implementation is now TRL GRPO. SFT remains useful as a teaching contrast, not as the current fresh RunPod default.

---

## 2. RL Basics Mapped to the Project

| RL concept | Simple meaning | In our project |
|---|---|---|
| Agent | Decision maker | Supervisor/specialist workflow policy |
| State | What the agent sees | task metadata, schema, target profile, prior actions |
| Action | What the agent does | MCP call, plan step, execution, logging, review |
| Environment | World that responds | Postgres, MCP tools, code runtime, ML libraries |
| Reward | Score for behavior | scorer result from artifacts, safety, tracking, model quality |
| Policy | Action-selection behavior | hosted model or tuned adapter |
| Trajectory | Full episode | all steps, tool calls, artifacts, final status, reward |

```mermaid
flowchart LR
  S["State<br/>task + metadata + previous steps"]
  P["Policy<br/>base model or tuned adapter"]
  A["Action<br/>next agent/tool step"]
  E["Environment<br/>MCP + Postgres + runtime"]
  R["Reward<br/>scorer/RULER result"]
  T["Trajectory log<br/>training evidence"]

  S --> P --> A --> E --> R --> T
  E --> S
```

---

## 3. No Training vs SFT vs GRPO

```mermaid
flowchart TD
  subgraph N["No project training yet"]
    N1["Host base/instruct model"]
    N2["Run Track A"]
    N3["Observe behavior"]
    N1 --> N2 --> N3
  end

  subgraph S["SFT contrast: imitation learning"]
    S1["Take selected good examples"]
    S2["Build prompt/completion pairs"]
    S3["Train with cross-entropy loss"]
    S4["Redeploy and compare"]
    S1 --> S2 --> S3 --> S4
  end

  subgraph G["Current Track B: TRL GRPO"]
    G1["Group rollouts by task"]
    G2["Generate multiple completions"]
    G3["Score with hybrid reward"]
    G4["Compute group-relative advantage"]
    G5["Train LoRA adapter"]
    G6["Redeploy and compare"]
    G1 --> G2 --> G3 --> G4 --> G5 --> G6
  end

  subgraph F["Future scale-up"]
    F1["Richer RULER/ART judging"]
    F2["More tasks and rollouts"]
    F3["veRL / distributed GRPO"]
    F1 --> F2 --> F3
  end
```

Key distinction:

| Method | Learns from | Training signal | Project status |
|---|---|---|---|
| Hosted baseline | general pretraining/instruction tuning | none from our project | implemented |
| SFT | prompt/completion examples | cross-entropy imitation loss | concept/previous alternative |
| TRL GRPO | grouped rollouts and reward function | relative reward/advantage | current Track B default |
| ART/RULER | judged trajectory groups | ranking/process quality | evaluation and future richer reward path |

---

## 3A. LLM RL Technique Map

Use this section when you want to explain the wider framework landscape before narrowing back to our implementation.

| Technique | Core idea | What it optimizes | How it relates to our project |
|---|---|---|---|
| SFT | Train on prompt/completion examples | Imitation loss | Conceptual baseline, not current default |
| QLoRA | 4-bit base model plus LoRA adapter | Memory-efficient fine-tuning | Useful adapter technique, not the active trainer mode |
| RLHF | Human feedback becomes reward/preference signal | Human-preferred behavior | Conceptual ancestor of reward-based LLM tuning |
| RLAIF | AI judge supplies feedback | Scalable judged preference | Similar spirit to RULER-style evaluation |
| Reward model | Model predicts reward for outputs | Learned scoring | Our scorer is rule/component based, not a learned reward model |
| PPO | Policy-gradient RL with critic/value model | Expected reward under KL control | Heavier full-RL option for LLMs |
| GRPO | Compare outputs within a group | Relative advantage without separate critic | Current Track B trainer through TRL |
| DPO | Learn from chosen/rejected pairs | Preference alignment without rollout RL | Could train from RULER chosen vs rejected pairs |
| Rejection sampling | Generate many, keep best | Quality filtering | Can seed better groups or selected examples |
| KL penalty | Keep tuned model near reference model | Stability and anti-drift | Important in PPO/GRPO-style training |
| DDP | Replicate model across GPUs | Distributed throughput | Scaling infrastructure, not the learning algorithm |
| FSDP | Shard model weights across GPUs | Larger model training | Scaling infrastructure for bigger models |
| DeepSpeed | Optimized distributed training runtime | Memory and speed | Scaling infrastructure |
| veRL | RL training framework/runtime | Large-scale RL pipelines | Project has handoff mode for official veRL command |
| Agent Lightning | Agent execution/training bridge | Trajectory and transition plumbing | Connects agent rollouts to policy optimization |

```mermaid
flowchart TD
  A["LLM behavior improvement"]
  B["Imitation path<br/>SFT / QLoRA"]
  C["Preference path<br/>DPO / chosen-rejected"]
  D["RL path<br/>PPO / GRPO"]
  E["Judging path<br/>reward model / RULER / RLAIF"]
  F["Scaling path<br/>DDP / FSDP / DeepSpeed / veRL"]
  G["Agent framework path<br/>Agent Lightning"]

  A --> B
  A --> C
  A --> D
  A --> E
  A --> F
  A --> G
```

Project position:

```text
Implemented default:
  grouped rollouts -> TRL GRPO LoRA adapter -> redeploy comparison

Supported/future scale-up:
  richer RULER/ART rewards -> larger GRPO batches -> veRL/distributed handoff
```

---

## 4. Track A Agent Flow

```mermaid
flowchart TD
  T["Postgres registry task<br/>mltask_openml_31_german_credit_baseline"]
  M1["MCP metadata tools<br/>schema, summary, target profile"]
  S["SupervisorAgent"]
  D["DataEngineerAgent<br/>safe profile and preprocessing plan"]
  G["GBMSpecialistAgent<br/>GBM benchmark plan"]
  X["SandboxExecutionAgent<br/>runs code against execution dataset source"]
  A["Artifacts<br/>metrics, champion model, profile, MLflow metadata"]
  C["CriticAgent<br/>review completeness and violations"]
  R["Reward scorer<br/>component breakdown"]
  L["Trajectory JSONL<br/>steps + tool calls + reward"]

  T --> M1 --> S --> D --> G --> X --> A --> C --> R --> L
```

Speaker note:

The LLM policy is not supposed to see raw rows. It reasons from safe metadata and asks the execution boundary to materialize rows only inside the runtime.

---

## 5. MCP and Postgres Boundary

```mermaid
flowchart LR
  subgraph LLM["LLM / Agent reasoning layer"]
    A["Can see safe metadata"]
    B["Cannot see raw rows"]
  end

  subgraph MCP["Project MCP tools"]
    M1["postgres_get_task_metadata"]
    M2["postgres_get_dataset_schema"]
    M3["postgres_get_dataset_summary"]
    M4["postgres_get_column_profile"]
    M5["postgres_get_target_profile"]
    M6["postgres_get_execution_dataset_source"]
  end

  subgraph DB["PostgreSQL"]
    R1["ml_registry.real_benchmark_tasks"]
    R2["ml_registry.real_dataset_summaries"]
    R3["ml_execution.dataset_sources"]
    R4["ml_data.openml_31_german_credit"]
  end

  subgraph EXEC["Execution boundary"]
    E1["Loads rows for model training"]
    E2["Writes benchmark artifacts"]
  end

  A --> M1 --> R1
  A --> M2 --> R2
  A --> M3 --> R2
  A --> M4 --> R2
  A --> M5 --> R2
  B -. "blocked by design" .- R4
  M6 --> R3 --> EXEC --> R4
```

Safe rule:

```text
MCP/agent layer sees metadata.
Execution layer sees rows.
The LLM should not receive raw dataset rows.
```

---

## 6. OpenML Ingestion

The fresh RunPod bootstrap can ingest a list of OpenML tasks. The current GRPO validation used four:

| OpenML ID | Dataset key | Task id | Primary metric | Sort order |
|---:|---|---|---|---:|
| 31 | `openml_31_german_credit` | `mltask_openml_31_german_credit_baseline` | `roc_auc` | 10 |
| 44 | `openml_44_spambase` | `mltask_openml_44_spambase_baseline` | `roc_auc` | 20 |
| 1461 | `openml_1461_bank_marketing` | `mltask_openml_1461_bank_marketing_baseline` | `roc_auc` | 30 |
| 1489 | `openml_1489_phoneme` | `mltask_openml_1489_phoneme_baseline` | `roc_auc` | 40 |

The ingestion command shape is:

```bash
python scripts/ingest_openml_to_postgres.py \
  --openml-id "$OPENML_ID" \
  --dataset-key "$DATASET_KEY" \
  --task-id "$TRACKA_TASK_ID" \
  --target-column "$TARGET_COLUMN" \
  --primary-metric "$PRIMARY_METRIC" \
  --secondary-metric accuracy \
  --if-exists replace
```

The ingestion script fetches live data through `sklearn.datasets.fetch_openml`, then writes:

| Destination | Purpose |
|---|---|
| `ml_data.openml_*` | actual dataset rows for execution |
| `ml_registry.real_benchmark_tasks` | task contract for agent/MCP |
| `ml_registry.real_dataset_summaries` | safe aggregate metadata |
| `ml_execution.dataset_sources` | execution-only dataset pointer |

```mermaid
flowchart TD
  O["OpenML tasks<br/>31, 44, 1461, 1489"]
  I["scripts/ingest_openml_to_postgres.py"]
  D["ml_data.openml_*<br/>raw rows for execution"]
  T["ml_registry.real_benchmark_tasks<br/>task metadata"]
  S["ml_registry.real_dataset_summaries<br/>safe profiles"]
  E["ml_execution.dataset_sources<br/>source pointer"]

  O --> I
  I --> D
  I --> T
  I --> S
  I --> E
```

---

## 7. Scoring and Reward Breakdown

The scorer uses weighted components:

| Component | Weight | What it checks |
|---|---:|---|
| `R_data` | 0.18 | profile/schema/preprocessing/data-source artifacts exist |
| `R_model` | 0.25 | benchmark model and useful metrics exist |
| `R_tracking` | 0.17 | MLflow/tracking metadata and metrics exist |
| `R_sandbox` | 0.15 | intended execution boundary was used |
| `R_reproducibility` | 0.15 | plan, metrics, champion, tracking are replayable |
| `R_violation` | 0.09 | raw-row leakage and critical review issues avoided |
| `R_postgres_tool` | 0.01 | PostgreSQL/MCP tool pattern was correct |

Formula shape:

```text
base_reward =
  0.18 * R_data
  + 0.25 * R_model
  + 0.17 * R_tracking
  + 0.15 * R_sandbox
  + 0.15 * R_reproducibility
  + 0.09 * R_violation
  + 0.01 * R_postgres_tool

reward = base_reward * task_completion * max(0.5, tool_validity)
```

Example reward breakdown:

| Component | traj_A | traj_B | traj_C | traj_D |
|---|---:|---:|---:|---:|
| `R_data` | 1.00 | 1.00 | 0.80 | 0.20 |
| `R_model` | 0.90 | 0.80 | 0.40 | 0.10 |
| `R_tracking` | 1.00 | 1.00 | 0.00 | 0.00 |
| `R_sandbox` | 0.75 | 0.75 | 0.75 | 0.00 |
| `R_reproducibility` | 1.00 | 0.80 | 0.30 | 0.00 |
| `R_postgres_tool` | 1.00 | 1.00 | 0.70 | 0.00 |
| `R_violation` | 1.00 | 1.00 | 1.00 | 0.00 |

---

## 8. RULER-Style Ranking View

For one task, multiple trajectories can be judged:

```text
Task: openml_31_german_credit

Rank 1: traj_A
Reason: Correct classifier, safe MCP use, strong metrics, MLflow logging, SHAP artifacts.

Rank 2: traj_B
Reason: Good workflow but weaker explainability.

Rank 3: traj_C
Reason: Missing MLflow and incomplete artifact manifest.

Rank 4: traj_D
Reason: Raw-row boundary violation and failed execution.
```

```mermaid
flowchart TD
  Task["openml_31_german_credit"]
  A["traj_A<br/>reward 0.92<br/>rank 1"]
  B["traj_B<br/>reward 0.75<br/>rank 2"]
  C["traj_C<br/>reward 0.45<br/>rank 3"]
  D["traj_D<br/>reward 0.05<br/>rank 4"]
  R["RULER/scorer ranking<br/>A > B > C > D"]

  Task --> A
  Task --> B
  Task --> C
  Task --> D
  A --> R
  B --> R
  C --> R
  D --> R
```

---

## 9. GRPO Explanation With Group Advantage

GRPO trains from grouped rollouts and relative rewards.

| Trajectory | RULER score | Group mean | Advantage |
|---|---:|---:|---:|
| `traj_A` | 1.00 | 0.52 | +0.48 |
| `traj_B` | 0.72 | 0.52 | +0.20 |
| `traj_C` | 0.34 | 0.52 | -0.18 |
| `traj_D` | 0.00 | 0.52 | -0.52 |

```mermaid
flowchart TD
  P["One task prompt/state"]
  R1["Rollout A<br/>advantage positive"]
  R2["Rollout B<br/>advantage positive"]
  R3["Rollout C<br/>advantage negative"]
  R4["Rollout D<br/>advantage negative"]
  U["GRPO update<br/>increase A/B-like behavior<br/>decrease C/D-like behavior"]

  P --> R1 --> U
  P --> R2 --> U
  P --> R3 --> U
  P --> R4 --> U
```

For our current default:

```text
Track B uses TRL GRPO today.
The demo should claim training/redeploy/no-regression on a small gate.
It should not claim broad model superiority from four tasks.
```

---

## 10. Track B TRL GRPO Training

Current default bootstrap:

```bash
TRAINER="${TRAINER:-trl_grpo}" \
REWARD_MODE="${REWARD_MODE:-hybrid}" \
NUM_GENERATIONS="${NUM_GENERATIONS:-4}" \
MODEL_NAME="$MODEL_NAME" \
GRPO_DATASET_PATH="$REPO_DIR/data/grpo/grouped_rollouts.jsonl" \
POLICY_OUTPUT_DIR="$REPO_DIR/checkpoints/trackb_trl_grpo_runpod" \
bash scripts/gpu/run_02_train_policy_qlora_grpo.sh
```

The trainer loop is:

```mermaid
flowchart TD
  G["grouped_rollouts.jsonl"]
  P["Prompt dataset<br/>task + trajectory context"]
  C["Generate N completions<br/>NUM_GENERATIONS=4"]
  R["Hybrid reward function<br/>format + tool + trajectory signals"]
  A["Group-relative advantage"]
  Q["TRL GRPO update<br/>LoRA adapter"]
  O["checkpoints/trackb_trl_grpo_runpod"]

  G --> P --> C --> R --> A --> Q --> O
```

Example GRPO record:

```json
{
  "prompt": "SYSTEM... STATE_JSON: {task_id, policy_version, step_index, previous_agent_actions}",
  "reference_completion": "{\"agent_name\":\"GBMSpecialistAgent\",\"action\":\"create_reproducible_gbm_benchmark_plan\",...}",
  "reward": 0.92,
  "task_id": "mltask_openml_31_german_credit_baseline"
}
```

What GRPO learns:

```text
Given a compact workflow state, increase probability of completions
that score above their rollout group and decrease weaker alternatives.
```

What the redeploy test still must prove:

```text
The adapter trained successfully.
Now rerun Track A and check whether real agent behavior improved.
```

---

## 11. Train, Validation, Test Split Story

The repo has a task-level split helper:

```text
src/training/split_policy_dataset.py
```

It splits by `task_id`, not by individual row, to avoid leakage:

```mermaid
flowchart TD
  A["grouped rollout records"]
  B["hash task_id"]
  C["train task groups"]
  D["validation task groups"]
  E["test task groups"]
  F["leakage report"]

  A --> B
  B --> C
  B --> D
  B --> E
  C --> F
  D --> F
  E --> F
```

Recommended evaluation framing:

| Layer | Metric | Meaning |
|---|---|---|
| GRPO training | reward mean/std, advantage signs | whether grouped rewards are producing a usable learning signal |
| GRPO stability | KL/stability signal, grad norm, runtime | whether the adapter update stayed controlled |
| Agentic test | average reward, success rate, tool validity | whether redeployed adapter improves Track A |
| Safety test | raw-row violations, invalid tool calls | whether behavior stayed inside guardrails |

Honest current-state note:

```text
The split helper exists.
The default fresh RunPod bootstrap records GRPO logs and adapter metadata.
The strongest proof is baseline-vs-tuned Track A comparison after redeploy.
```

---

## 12. How To Say "How Good Was Training?"

Use a two-layer answer:

### Layer 1: GRPO training quality

Good signs:

```text
reward mean is non-trivial
reward variance exists across generated completions
advantages create positive and negative learning pressure
grad_norm is stable
no training failure in adapter_metadata.json
adapter files are written
```

Example wording:

```text
The adapter update is healthy if GRPO sees usable reward spread, stable gradients, and writes adapter metadata without fallback.
```

### Layer 2: Agentic behavior quality

Good signs:

```text
average Track A reward improves after redeploy
task_success_rate improves
valid_tool_rate improves
raw-row violations stay zero
artifact completeness improves
MLflow/tracking completeness improves
```

Example wording:

```text
The tuned adapter is useful only if it improves the agent workflow after redeploy. GRPO reward statistics are training signals. Track A rerun is the behavioral test.
```

---

## 13. Agent Lightning Purpose

Agent Lightning is useful because multi-agent workflows have delayed, multi-step credit assignment.

Without a framework, we write custom glue for:

```text
trajectory capture
transition formatting
reward association
policy version tracking
checkpoint bookkeeping
training handoff
rollout comparison
```

With a framework boundary:

```mermaid
flowchart LR
  A["Agent runtime<br/>LangGraph/workflow"]
  B["Trajectory and transition capture"]
  C["Training server / optimizer"]
  D["Checkpointed policy"]
  E["Redeploy"]

  A --> B --> C --> D --> E --> A
```

Speaker note:

Agent Lightning is not the reward itself. It is framework plumbing for connecting agent execution to optimization.

---

## 14. LangGraph-Style View

```mermaid
stateDiagram-v2
  [*] --> Supervisor
  Supervisor --> DataEngineer: inspect safe metadata
  DataEngineer --> GBMSpecialist: propose benchmark plan
  GBMSpecialist --> SandboxExecutor: execute plan
  SandboxExecutor --> ArtifactLogger: write metrics/models/reports
  ArtifactLogger --> Critic: review workflow
  Critic --> RewardScorer: score trajectory
  RewardScorer --> LightningServer: emit transition/reward
  RewardScorer --> [*]
```

Speaker note:

This is the agentic equivalent of an RL episode. Each node contributes to the final trajectory.

---

## 15. Monorepo Structure

```mermaid
flowchart TD
  Repo["Agent-lightening-RL"]
  Agents["src/agents<br/>supervisor and specialist agents"]
  Tools["src/tools<br/>MCP, Postgres, execution tools"]
  Rewards["src/rewards<br/>scorer and reward components"]
  Training["src/training<br/>grouping, GRPO, Lightning, ART hooks"]
  Eval["src/evaluation<br/>baseline vs tuned comparison"]
  Scripts["scripts<br/>RunPod bootstrap and operations"]
  Prompts[".github/prompts<br/>fresh RunPod instructions"]
  Data["data/grpo<br/>grouped rollouts"]
  Traj["trajectories<br/>Track A logs"]
  Ckpt["checkpoints<br/>LoRA adapters"]
  Reports["reports<br/>logs, metrics, benchmarks"]

  Repo --> Agents
  Repo --> Tools
  Repo --> Rewards
  Repo --> Training
  Repo --> Eval
  Repo --> Scripts
  Repo --> Prompts
  Repo --> Data
  Repo --> Traj
  Repo --> Ckpt
  Repo --> Reports
```

Folder purpose table:

| Area | Purpose |
|---|---|
| `src/agents` | agent orchestration and specialist roles |
| `src/tools` | safe tool interface, Postgres/MCP, execution helpers |
| `src/rewards` | component scoring and reward metadata |
| `src/training` | rollout grouping, TRL GRPO, Lightning/ART modes |
| `src/evaluation` | compare policy trajectory directories |
| `scripts/runpod_bootstrap_e2e.sh` | fresh pod E2E path |
| `.github/prompts/fresh-runpod-e2e.prompt.md` | instructions for future Codex/runpod runs |
| `trajectories` | raw/scored Track A rollouts |
| `data/grpo` | grouped rollout dataset |
| `checkpoints` | trained adapters |
| `reports` | human-readable evidence |

---

## 16. E2E Fresh RunPod Story

```mermaid
sequenceDiagram
  participant User
  participant RunPod
  participant GitHub
  participant Postgres
  participant Agent
  participant Trainer
  participant Dashboard

  User->>RunPod: export DATABASE_URL and optional HF_TOKEN
  RunPod->>GitHub: clone repo and checkout E2E_grpo_4openml_streamlit_evidence
  RunPod->>Postgres: ingest four OpenML tasks and safe metadata
  RunPod->>Agent: run Track A rollouts
  Agent->>Postgres: call MCP metadata tools
  Agent->>RunPod: write artifacts and trajectories
  RunPod->>Trainer: prepare grouped_rollouts.jsonl
  Trainer->>RunPod: train TRL GRPO adapter
  RunPod->>Agent: redeploy tuned endpoint
  Agent->>RunPod: rerun Track A
  RunPod->>Dashboard: show logs, reports, comparison metrics
```

Command:

```bash
export DATABASE_URL='postgres://USER:PASSWORD@HOST:PORT/DB?sslmode=require'
export HF_TOKEN='hf_...'  # optional but useful

cd /workspace
git clone https://github.com/keshavnath1/Agent-lightening-RL.git
cd Agent-lightening-RL
git checkout E2E_grpo_4openml_streamlit_evidence

RUN_LIVE_POLICY=1 START_DASHBOARD=1 DASHBOARD_PORT=8503 \
bash scripts/runpod_bootstrap_e2e.sh
```

---

## 17. Benchmark Evidence From Current Run

The latest checked-in evidence on branch `E2E_grpo_4openml_streamlit_evidence`:

| Phase | Tasks / rollouts | Avg reward | Success | Valid tool rate | Live endpoint rate |
|---|---:|---:|---:|---:|---:|
| Initial Track A, no LLM | 16 rollouts | 0.85 | 1.00 | 1.00 | 0.00 |
| Baseline live LLM | 4 tasks | 0.85 | 1.00 | 1.00 | 1.00 |
| Track B redeploy LLM | 4 tasks | 0.85 | 1.00 | 1.00 | 1.00 |

```mermaid
flowchart LR
  I["Initial Track A<br/>16 rollouts<br/>avg reward 0.85"]
  B["Baseline live endpoint<br/>4 tasks<br/>avg reward 0.85"]
  T["Track B redeploy endpoint<br/>4 tasks<br/>avg reward 0.85"]
  C["Conclusion<br/>no regression<br/>no measured lift on small gate"]

  I --> B --> T --> C
```

Per-task rewards were flat after redeploy:

```mermaid
flowchart TD
  A["bank marketing<br/>0.8776"]
  B["phoneme<br/>0.7710"]
  C["german credit<br/>0.8430"]
  D["spambase<br/>0.9085"]
  E["Track B redeploy<br/>same task rewards"]

  A --> E
  B --> E
  C --> E
  D --> E
```

Speaker note:

The demo should say: "Track B GRPO trained and redeployed successfully; on this four-task benchmark it preserved behavior but did not yet produce a measured lift." That is the clean scientific claim.

---

## 18. Final Teaching Summary

```text
RL basics:
  agent chooses actions to maximize reward.

Our project:
  agent chooses workflow actions to maximize safe, reproducible task success.

Track A:
  generate trajectories and evidence.

Scoring/RULER:
  turn trajectories into ranked learning signal.

Track B today:
  TRL GRPO adapter learns from grouped rollout rewards.

Scale-up path:
  richer RULER/ART rewards, more tasks, more rollouts, veRL/distributed GRPO.

Redeploy comparison:
  the tuned adapter must improve real agent behavior, not just training loss.
```

The most honest one-line explanation:

```text
We are building a measured loop from agent workflow evidence to reward-trained policy adapters, then testing them by redeploying and rerunning Track A.
```
