from __future__ import annotations

import json
from pathlib import Path

from src.agents.base import BaseAgent, AgentContext
from src.telemetry.schema import AgentStep, ToolCallRecord
from src.tools.dockerized_code_interpreter import DockerizedCodeInterpreter
from src.tools.postgres_tooling import postgres_get_execution_dataset_source


class SandboxExecutionAgent(BaseAgent):
    name = 'SandboxExecutionAgent'

    def step(self, context: AgentContext) -> AgentStep:
        task_id = context.task['task_id']
        artifact_dir = (Path('artifacts') / task_id).resolve()
        artifact_dir.mkdir(parents=True, exist_ok=True)
        script_path = artifact_dir / 'run_gbm_benchmark.py'

        dataset_source_payload = postgres_get_execution_dataset_source(task_id)
        dataset_source = dataset_source_payload['result']
        dataset_source_path = artifact_dir / 'execution_dataset_source.json'
        dataset_source_path.write_text(json.dumps(dataset_source, indent=2), encoding='utf-8')
        context.artifacts['execution_dataset_source_json'] = str(dataset_source_path)

        schema_metadata = Path(context.artifacts['schema_metadata_json']).resolve().relative_to(artifact_dir)
        profile_summary = Path(context.artifacts['profile_summary_json']).resolve().relative_to(artifact_dir)
        preprocessing_config = Path(context.artifacts['preprocessing_config_json']).resolve().relative_to(artifact_dir)
        execution_dataset_source = dataset_source_path.resolve().relative_to(artifact_dir)

        script_path.write_text(
            "from src.tools.gbm_benchmark import run_gbm_benchmark_from_postgres\n"
            "import json\n"
            "from pathlib import Path\n"
            "work_dir = Path.cwd()\n"
            "payload = run_gbm_benchmark_from_postgres(\n"
            f"    dataset_source=str(work_dir / {str(execution_dataset_source)!r}),\n"
            f"    schema_metadata=str(work_dir / {str(schema_metadata)!r}),\n"
            f"    profile_summary=str(work_dir / {str(profile_summary)!r}),\n"
            f"    preprocessing_config=str(work_dir / {str(preprocessing_config)!r}),\n"
            "    output_dir=str(work_dir),\n"
            f"    metric={context.task.get('metric') or context.task.get('primary_metric', 'roc_auc')!r},\n"
            ")\n"
            "print(json.dumps({'champion': payload['champion']}, indent=2))\n",
            encoding='utf-8',
        )
        result = DockerizedCodeInterpreter(artifact_dir).run_python_file(str(script_path.resolve()), timeout_seconds=300)
        status = 'success' if result.returncode == 0 else 'failed'
        metrics_path = artifact_dir / 'benchmark_metrics.json'
        champion_path = artifact_dir / 'champion_model.json'
        if metrics_path.exists():
            context.artifacts['benchmark_metrics_json'] = str(metrics_path.resolve())
        if champion_path.exists():
            context.artifacts['champion_model_json'] = str(champion_path.resolve())
            try:
                champion = json.loads(champion_path.read_text(encoding='utf-8'))
                if champion.get('model_path'):
                    context.artifacts['champion_model_artifact'] = champion['model_path']
            except Exception as exc:
                raise RuntimeError(
                    f'Invalid champion artifact JSON at {champion_path}: {exc.__class__.__name__}: {exc}'
                ) from exc
        context.tool_outputs['sandbox_execution'] = {
            'status': status,
            'execution_mode': result.execution_mode,
            'stdout': result.stdout[-2000:],
            'stderr': result.stderr[-2000:],
            'dataset_source_contract': {
                'contract_version': dataset_source.get('contract_version'),
                'dataset_key': dataset_source.get('dataset_key'),
                'storage_backend': dataset_source.get('storage_backend'),
                'execution_only': dataset_source.get('execution_only'),
                'raw_rows_exposed_to_llm': False,
            },
        }
        return AgentStep(
            agent_name=self.name,
            action='execute_real_gbm_benchmark_from_postgres_contract',
            reasoning_summary='Executed the generated GBM benchmark through Docker using an execution-only PostgreSQL dataset source contract.',
            tool_calls=[
                ToolCallRecord(
                    tool_name='postgres_get_execution_dataset_source',
                    arguments={
                        'task_id': task_id,
                        'execution_only': True,
                        'raw_rows_exposed_to_llm': False,
                    },
                    status='success',
                    output_ref=str(dataset_source_path),
                ),
                ToolCallRecord(
                    tool_name='DockerizedCodeInterpreter',
                    arguments={'script_path': str(script_path), 'execution_mode': result.execution_mode},
                    status=status,
                    output_ref=str(metrics_path if metrics_path.exists() else script_path),
                    error=result.stderr if result.stderr else None,
                ),
            ],
        )
