from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from dataclasses import dataclass


@dataclass
class CodeExecutionResult:
    status: str
    stdout: str
    stderr: str
    returncode: int
    execution_mode: str = 'docker'


class DockerizedCodeInterpreter:
    """Controlled benchmark execution wrapper.

    Docker remains the default strict isolation mode. Some RunPod containers do
    not allow nested container mounts; those runs can opt in explicitly with
    CODE_INTERPRETER_MODE=host_subprocess. The host mode is recorded in
    telemetry and still keeps raw rows outside the LLM/MCP planning context.
    """

    def __init__(self, work_dir: str | Path):
        self.work_dir = Path(work_dir).resolve()
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def _repo_root(self) -> Path:
        return Path(__file__).resolve().parents[2]

    def _execution_database_url(self) -> str:
        execution_database_url = os.getenv('EXECUTION_DATABASE_URL')
        if not execution_database_url:
            raise RuntimeError(
                'EXECUTION_DATABASE_URL is required for execution in strict PostgreSQL data-contract mode.'
            )
        return execution_database_url

    def _run_docker(self, script: Path, timeout_seconds: int) -> CodeExecutionResult:
        docker = shutil.which('docker')
        if not docker:
            raise RuntimeError('Docker is required for code interpreter execution in strict mode; no local fallback is allowed.')
        image = os.getenv('CODE_INTERPRETER_IMAGE', 'self-improving-ml-agent-cpu:latest')
        memory = os.getenv('CODE_INTERPRETER_MEMORY', '4g')
        cpus = os.getenv('CODE_INTERPRETER_CPUS', '2')
        network = os.getenv('CODE_INTERPRETER_NETWORK', 'bridge')
        execution_database_url = self._execution_database_url()
        repo_root = self._repo_root()
        try:
            script_rel = script.resolve().relative_to(self.work_dir)
        except ValueError:
            raise ValueError(f'Script {script} must be inside interpreter work_dir {self.work_dir} for Docker execution.')
        container_script = Path('/workspace') / script_rel
        cmd = [
            docker, 'run', '--rm', '--network', network, '--memory', memory, '--cpus', cpus,
            '--workdir', '/workspace',
            '-v', f'{self.work_dir}:/workspace',
            '-v', f'{repo_root}:/repo:ro',
            '-e', 'PYTHONPATH=/repo',
            '-e', f'EXECUTION_DATABASE_URL={execution_database_url}',
            image, 'python', str(container_script),
        ]
        proc = subprocess.run(
            cmd,
            cwd=self.work_dir,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        return CodeExecutionResult(
            status='success' if proc.returncode == 0 else 'failed',
            stdout=proc.stdout,
            stderr=proc.stderr,
            returncode=proc.returncode,
            execution_mode='docker',
        )

    def _run_host_subprocess(self, script: Path, timeout_seconds: int) -> CodeExecutionResult:
        try:
            script.resolve().relative_to(self.work_dir)
        except ValueError:
            raise ValueError(f'Script {script} must be inside interpreter work_dir {self.work_dir} for host execution.')
        repo_root = self._repo_root()
        env = os.environ.copy()
        env['EXECUTION_DATABASE_URL'] = self._execution_database_url()
        existing_pythonpath = env.get('PYTHONPATH')
        env['PYTHONPATH'] = str(repo_root) if not existing_pythonpath else f'{repo_root}:{existing_pythonpath}'
        proc = subprocess.run(
            [sys.executable, str(script)],
            cwd=self.work_dir,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        return CodeExecutionResult(
            status='success' if proc.returncode == 0 else 'failed',
            stdout=proc.stdout,
            stderr=proc.stderr,
            returncode=proc.returncode,
            execution_mode='host_subprocess',
        )

    def run_python_file(self, script_path: str, timeout_seconds: int = 120) -> CodeExecutionResult:
        script = Path(script_path).resolve()
        if not script.exists():
            raise FileNotFoundError(script)
        mode = os.getenv('CODE_INTERPRETER_MODE', 'docker').strip().lower()
        if mode == 'host_subprocess':
            return self._run_host_subprocess(script, timeout_seconds)
        if mode != 'docker':
            raise RuntimeError(
                f'Unsupported CODE_INTERPRETER_MODE={mode!r}. Use docker or host_subprocess.'
            )
        return self._run_docker(script, timeout_seconds)
