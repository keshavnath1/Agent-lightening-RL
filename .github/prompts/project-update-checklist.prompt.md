# Project Update Checklist Prompt

You are modifying the `self-improving-ml-agent` monorepo. Make the smallest safe change, preserve existing behavior, and validate the area you touched.

## Before Changing Code

Identify the target area first: dashboard, official MCP server, PostgreSQL tools, rewards, RULER scoring, TRL GRPO training, Agent Lightning bridge, or monorepo structure. Inspect relevant files before editing and do not do broad rewrites.

## Validation After Changes

Run the most relevant validation commands:

```bash
python scripts/validate/validate_monorepo_phase1.py
python scripts/dev_validate_official_mcp_server.py
python scripts/dev_validate_trackb_wiring.py
python scripts/dev_validate_dashboard_refactor.py
```

If a change affects shell scripts, syntax-check them:

```bash
find scripts -name "*.sh" -print0 | xargs -0 -I{} bash -n {}
```

## Report Format

When finished, report:

```text
Changed files
Reason for change
Validation run
Validation result
Remaining risks
Next recommended step
```

Strictly avoid secrets, raw SQL exposure, raw data leakage, unallowlisted shell execution, unnecessary GPU dependency reinstalls, and expensive training runs unless explicitly requested.
