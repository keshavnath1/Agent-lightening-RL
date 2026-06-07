# Strict MCP/Agent Boundary Refactor

## Non-negotiable rules

1. Agent inputs are task-oriented (`task_id`), not table-oriented.
2. Agents never open direct PostgreSQL connections.
3. Official Project MCP is the only agent-facing metadata/task access layer.
4. `raw_rows_exposed_to_llm=false` is required for planning/metadata tools.
5. Full-row access is Docker/materializer-only and must never enter agent state, RULER prompts, or GRPO prompts.
6. Production Track A fails closed when the registry/MCP layer is unavailable, empty, or invalid.
7. Local JSONL/synthetic fallbacks are not part of the production Track A path.
8. RULER/GRPO training must fail if required reward/ranking fields are missing; no reward-mode fallback is allowed.
9. MCP server handlers are thin wrappers; PostgreSQL business logic lives in shared tooling.
10. Any exception in the agent graph is logged to `reports/agent_failures.jsonl` and re-raised.

## PostgreSQL contract

Real data lives in PostgreSQL, but each role sees a different part of the database.

```text
ml_registry.real_benchmark_tasks       task queue and target/metric metadata
ml_registry.real_dataset_summaries     safe schema, summary, column profiles, target profile
ml_execution.dataset_sources           execution-only source contract
ml_data.<dataset table>                real rows, readable only by execution role
```

The MCP runtime should use `MCP_DATABASE_URL` and have read access to `ml_registry` plus `ml_execution.dataset_sources`.
The Docker/materializer runtime should use `EXECUTION_DATABASE_URL` and have row access to `ml_data`.

## Flow

```text
OpenML ingestion
  -> writes rows to ml_data.<dataset>
  -> writes safe metadata to ml_registry
  -> writes source contract to ml_execution
  -> MCP lists tasks and safe metadata
  -> agent plans from metadata only
  -> Docker receives execution dataset source contract
  -> Docker loads rows with EXECUTION_DATABASE_URL
  -> benchmark writes metrics/model artifacts
```

## Canonical agent-facing MCP tools

- `postgres_get_next_task`
- `postgres_list_tasks`
- `postgres_get_task_metadata`
- `postgres_get_dataset_schema`
- `postgres_get_dataset_summary`
- `postgres_get_column_profile`
- `postgres_get_target_profile`
- `postgres_get_rollout_status`
- `postgres_get_reward_history`

## Execution-only tool

- `postgres_get_execution_dataset_source`

This returns a contract like:

```json
{
  "contract_version": "mcp_execution_dataset_source_v1",
  "task_id": "mltask_openml_31_german_credit_baseline",
  "dataset_key": "openml_31_german_credit",
  "storage_backend": "postgres_table",
  "source_schema": "ml_data",
  "source_table": "openml_31_german_credit",
  "raw_rows_exposed_to_llm": false,
  "execution_only": true
}
```

It must not return rows or credentials, and it is not for planning prompts.

## Disabled tools

- `get_dataset_rows`
- `get_dataset_sample`
- `get_dataset_as_frame_spec`
- `postgres_get_artifact_manifest`
- `get_table_overview(table_name)`
- `profile_column(table_name, ...)`
- raw SQL query tools, except developer-only `postgres_debug_readonly_sql` gated by `ALLOW_RAW_SQL_TOOL=1`

## Validation

Run:

```bash
python scripts/validate/validate_strict_mcp_agent_boundary.py
python -m compileall -q src services packages apps scripts tests
pytest -q tests/training/test_tracka_postgres_loader.py tests/tools/test_sql_safety.py
python scripts/dev_validate_trackb_wiring.py
```
