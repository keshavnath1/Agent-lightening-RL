#!/usr/bin/env python3
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FAILURES: list[str] = []

AGENT_PATHS = [ROOT / 'src' / 'agents', ROOT / 'apps' / 'rollout_worker']
FORBIDDEN_AGENT_IMPORTS = {'sqlalchemy', 'psycopg2', 'asyncpg'}
FORBIDDEN_AGENT_TOKENS = ['SQLAlchemyConnector', 'pd.read_sql', 'SELECT *', 'DATABASE_URL']
FORBIDDEN_MCP_TOOL_NAMES = ['get_dataset_sample', 'get_dataset_rows', 'get_dataset_as_frame_spec', 'list_ready_tasks', 'get_task_context', 'list_datasets', 'get_dataset_metadata', 'postgres_get_artifact_manifest']
SERVER_FILES = [
    ROOT / 'services' / 'project_mcp_server' / 'src' / 'project_mcp_server' / 'server.py',
    ROOT / 'src' / 'mcp_official' / 'server.py',
    ROOT / 'src' / 'mcp_server' / 'server.py',
]


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def fail(message: str) -> None:
    FAILURES.append(message)


def iter_py(paths: list[Path]):
    for base in paths:
        if base.exists():
            yield from base.rglob('*.py')


def check_agent_layer() -> None:
    for path in iter_py(AGENT_PATHS):
        text = path.read_text(encoding='utf-8', errors='ignore')
        try:
            tree = ast.parse(text)
        except SyntaxError as exc:
            fail(f'{rel(path)} has syntax error: {exc}')
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split('.')[0]
                    if root in FORBIDDEN_AGENT_IMPORTS:
                        fail(f'{rel(path)} imports forbidden DB library {alias.name}')
            elif isinstance(node, ast.ImportFrom) and node.module:
                root = node.module.split('.')[0]
                if root in FORBIDDEN_AGENT_IMPORTS:
                    fail(f'{rel(path)} imports forbidden DB library {node.module}')
        for token in FORBIDDEN_AGENT_TOKENS:
            if token in text:
                fail(f'{rel(path)} contains forbidden agent-layer token {token!r}')
        if re.search(r'\btable_name\b', text) and 'debug' not in path.name:
            fail(f'{rel(path)} contains table_name in agent/rollout layer')


def check_mcp_servers() -> None:
    for path in SERVER_FILES:
        if not path.exists():
            fail(f'missing MCP server file: {rel(path)}')
            continue
        text = path.read_text(encoding='utf-8', errors='ignore')
        for name in FORBIDDEN_MCP_TOOL_NAMES:
            if f'@log_mcp_tool_call("{name}")' in text or f"@log_mcp_tool_call('{name}')" in text:
                fail(f'{rel(path)} exposes deprecated/unsafe MCP tool {name}')
        if re.search(r'def\s+\w+\([^)]*table_name\s*:', text):
            fail(f'{rel(path)} exposes table_name in MCP function signature')
        if 'get_dataset_rows' in text and '@log_mcp_tool_call("get_dataset_rows")' in text:
            fail(f'{rel(path)} exposes get_dataset_rows')


def check_training_no_silent_fallback() -> None:
    files = [
        ROOT / 'src' / 'training' / 'train_policy_qlora_grpo.py',
        ROOT / 'apps' / 'ruler_scorer' / 'src' / 'ruler_scorer' / 'vllm_judge.py',
        ROOT / 'apps' / 'trainer' / 'src' / 'trainer_app' / 'art_ruler_training.py',
    ]
    bad_patterns = [
        'using all groups',
        'Fallback deterministic reward ranking',
        '--allow-heuristic-ruler-fallback',
        'heuristic_fallback',
        'return _fallback_rank',
    ]
    for path in files:
        if not path.exists():
            fail(f'missing expected strict file: {rel(path)}')
            continue
        text = path.read_text(encoding='utf-8', errors='ignore')
        for pattern in bad_patterns:
            if pattern in text:
                fail(f'{rel(path)} contains silent fallback pattern {pattern!r}')



def check_execution_and_tracking_no_fallback() -> None:
    files = [
        ROOT / 'src' / 'tools' / 'dockerized_code_interpreter.py',
        ROOT / 'src' / 'tools' / 'mlflow_dvc_tracker.py',
        ROOT / 'src' / 'agents' / 'experiment_tracking.py',
        ROOT / 'src' / 'agents' / 'gradient_boosting_specialist.py',
    ]
    bad_patterns = [
        'local_subprocess_fallback',
        'json_fallback',
        'sklearn_hist_gradient_boosting_fallback',
        'log_experiment_to_mlflow_or_json_fallback',
    ]
    for path in files:
        if not path.exists():
            fail(f'missing expected strict execution/tracking file: {rel(path)}')
            continue
        text = path.read_text(encoding='utf-8', errors='ignore')
        for pattern in bad_patterns:
            if pattern in text:
                fail(f'{rel(path)} contains execution/tracking fallback pattern {pattern!r}')


def check_postgres_execution_contract() -> None:
    tooling = ROOT / 'src' / 'tools' / 'postgres_tooling.py'
    loader = ROOT / 'src' / 'tools' / 'sandbox_postgres_data_loader.py'
    docker = ROOT / 'src' / 'tools' / 'dockerized_code_interpreter.py'
    data_engineer = ROOT / 'src' / 'agents' / 'data_engineer.py'
    sandbox = ROOT / 'src' / 'agents' / 'sandbox_execution.py'
    required = {
        tooling: ['ml_registry.real_benchmark_tasks', 'ml_registry.real_dataset_summaries', 'ml_execution.dataset_sources', 'postgres_get_execution_dataset_source'],
        loader: ['EXECUTION_DATABASE_URL', 'load_dataset_frame_from_source'],
        docker: ['EXECUTION_DATABASE_URL'],
        data_engineer: ['mcp_safe_metadata_only', 'docker_execution_dataset_source_contract'],
        sandbox: ['postgres_get_execution_dataset_source', 'execution_dataset_source_json'],
    }
    for path, tokens in required.items():
        if not path.exists():
            fail(f'missing expected strict contract file: {rel(path)}')
            continue
        text = path.read_text(encoding='utf-8', errors='ignore')
        for token in tokens:
            if token not in text:
                fail(f'{rel(path)} missing PostgreSQL execution contract token {token!r}')


def check_graph_fail_closed() -> None:
    path = ROOT / 'src' / 'agents' / 'graph.py'
    text = path.read_text(encoding='utf-8', errors='ignore')
    required = ['_log_and_raise', 'reports', 'agent_failures.jsonl', 'strict_fail_closed']
    for token in required:
        if token not in text:
            fail(f'{rel(path)} missing fail-closed token {token!r}')
    if "return {'error'" in text or 'become a no-op' in text:
        fail(f'{rel(path)} still converts node failures into no-op/error state instead of raising')


def main() -> int:
    check_agent_layer()
    check_mcp_servers()
    check_training_no_silent_fallback()
    check_graph_fail_closed()
    check_execution_and_tracking_no_fallback()
    check_postgres_execution_contract()
    if FAILURES:
        print('STRICT MCP/AGENT BOUNDARY VALIDATION FAILED')
        for item in FAILURES:
            print(f'- {item}')
        return 1
    print('STRICT MCP/AGENT BOUNDARY VALIDATION PASSED')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
