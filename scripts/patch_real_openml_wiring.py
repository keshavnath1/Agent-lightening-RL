from __future__ import annotations

from pathlib import Path

ROOT = Path('/workspace/self-improving-ml-agent')


def replace(path: str, old: str, new: str) -> None:
    p = ROOT / path
    text = p.read_text(encoding='utf-8')
    if old not in text:
        print(f'[skip] pattern not found in {path}: {old[:80]!r}')
        return
    p.write_text(text.replace(old, new), encoding='utf-8')
    print(f'[patch] {path}')

# CPU Track A default task source: keep legacy env var name as an override alias, but default to real OpenML.
replace(
    'scripts/cpu/run_03_run_baseline_workflow.sh',
    ': "${SYNTHETIC_TASKS_PATH:=$WORKSPACE_DIR/data/synthetic/tasks.jsonl}"\n',
    ': "${REAL_TASKS_PATH:=$WORKSPACE_DIR/data/real_openml/tasks.jsonl}"\n: "${SYNTHETIC_TASKS_PATH:=$REAL_TASKS_PATH}"  # backwards-compatible alias\n',
)

# GPU tuned inference default task source: preserve the legacy variable but point it at the real task list.
run_tuned = ROOT / 'scripts/gpu/run_03_tuned_batch_inference.sh'
if run_tuned.exists():
    replace(
        'scripts/gpu/run_03_tuned_batch_inference.sh',
        ': "${SYNTHETIC_TASKS_PATH:=$WORKSPACE_DIR/data/synthetic/tasks.jsonl}"\n: "${TUNED_POLICY_JOBS:=$WORKSPACE_DIR/data/synthetic/tuned_policy_jobs.jsonl}"\n',
        ': "${REAL_TASKS_PATH:=$WORKSPACE_DIR/data/real_openml/tasks.jsonl}"\n: "${SYNTHETIC_TASKS_PATH:=$REAL_TASKS_PATH}"  # backwards-compatible alias\n: "${TUNED_POLICY_JOBS:=$WORKSPACE_DIR/data/real_openml/tuned_policy_jobs.jsonl}"\n',
    )

# Track B default Agent Lightning task source for dry-run or official modes.
replace(
    'scripts/gpu/run_02_train_policy_qlora_grpo.sh',
    ': "${AGENT_LIGHTNING_TASKS:=$WORKSPACE_DIR/data/synthetic/tasks.jsonl}"\n',
    ': "${AGENT_LIGHTNING_TASKS:=$WORKSPACE_DIR/data/real_openml/tasks.jsonl}"\n',
)

# Training CLI default for direct invocations.
replace(
    'src/training/train_policy_qlora_grpo.py',
    "    parser.add_argument('--agent-lightning-tasks', default='data/synthetic/tasks.jsonl')\n",
    "    parser.add_argument('--agent-lightning-tasks', default='data/real_openml/tasks.jsonl')\n",
)

# Dashboard data loader: count real OpenML task files first and include openml_* artifact directories.
replace(
    'src/ui/view_models/dashboard_state.py',
    '    for task_path in [\n        ROOT / "data" / "synthetic" / "tasks.jsonl",\n        ROOT / "data" / "synthetic" / "tasks_dataset.jsonl",\n    ]:\n',
    '    for task_path in [\n        ROOT / "data" / "real_openml" / "tasks.jsonl",\n        ROOT / "data" / "synthetic" / "tasks.jsonl",\n        ROOT / "data" / "synthetic" / "tasks_dataset.jsonl",\n    ]:\n',
)
replace(
    'src/ui/view_models/dashboard_state.py',
    '        ds.artifacts = sorted(artifacts_dir.glob("task_*/champion_model.json"))\n',
    '        ds.artifacts = sorted(list(artifacts_dir.glob("task_*/champion_model.json")) + list(artifacts_dir.glob("openml_*/champion_model.json")))\n',
)
replace(
    'src/ui/view_models/dashboard_state.py',
    '    for task_dir in sorted(artifacts_dir.glob("task_*")):\n',
    '    for task_dir in sorted(list(artifacts_dir.glob("task_*")) + list(artifacts_dir.glob("openml_*"))):\n',
)

# Results page: prefer new real-data report names before old dated reports.
replace(
    'src/ui/pages/results.py',
    '    report_files = [\n        ROOT / "reports" / "e2e_tracka_vs_trackb_adapter_first4_20260604_053507.md",\n',
    '    report_files = [\n        ROOT / "reports" / "e2e_real_openml_tracka_vs_trackb.md",\n        ROOT / "reports" / "baseline_vs_rl_tuned.md",\n        ROOT / "reports" / "baseline_v2_tuned_benchmark.md",\n        ROOT / "reports" / "e2e_tracka_vs_trackb_adapter_first4_20260604_053507.md",\n',
)

print('Real OpenML wiring patch complete.')
