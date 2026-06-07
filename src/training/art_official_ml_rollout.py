from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import art  # type: ignore
    from art.langgraph import init_chat_model, wrap_rollout  # noqa: F401
except Exception:  # keep CPU/import-only paths working
    art = None  # type: ignore
    init_chat_model = None  # type: ignore
    wrap_rollout = None  # type: ignore

try:
    from pydantic import BaseModel, Field
except Exception:  # pragma: no cover
    BaseModel = object  # type: ignore
    def Field(default_factory=None, default=None):  # type: ignore
        return default_factory() if default_factory else default


if BaseModel is object:
    class MLWorkflowScenario:  # lightweight fallback when pydantic is unavailable
        def __init__(self, task_id: str, task_payload: dict[str, Any] | None = None, expected_artifacts: list[str] | None = None, difficulty: str | None = None, step: int = 0):
            self.task_id = task_id
            self.task_payload = task_payload or {}
            self.expected_artifacts = expected_artifacts or []
            self.difficulty = difficulty
            self.step = step

        def model_dump(self) -> dict[str, Any]:
            return {
                'task_id': self.task_id,
                'task_payload': self.task_payload,
                'expected_artifacts': self.expected_artifacts,
                'difficulty': self.difficulty,
                'step': self.step,
            }

        def copy(self, update: dict[str, Any] | None = None):
            data = self.model_dump()
            data.update(update or {})
            return MLWorkflowScenario(**data)
else:
    class MLWorkflowScenario(BaseModel):
        task_id: str
        task_payload: dict[str, Any] = Field(default_factory=dict)
        expected_artifacts: list[str] = Field(default_factory=list)
        difficulty: str | None = None
        step: int = 0


if art is not None:
    class MLWorkflowTrajectory(art.Trajectory):  # type: ignore[misc]
        task_id: str | None = None
        policy_version: str | None = None
        final_status: str | None = None
        workflow_reward_metadata: dict[str, Any] = {}
        final_artifacts: dict[str, Any] = {}
else:
    class MLWorkflowTrajectory:  # fallback for import/validation without ART installed
        def __init__(self, reward: float = 0.0, messages_and_choices: list[dict[str, Any]] | None = None, metadata: dict[str, Any] | None = None, **kwargs: Any):
            self.reward = reward
            self.messages_and_choices = messages_and_choices or []
            self.metadata = metadata or {}
            self.metrics: dict[str, Any] = {}
            for k, v in kwargs.items():
                setattr(self, k, v)


def _require_art() -> None:
    if art is None or init_chat_model is None or wrap_rollout is None:
        raise RuntimeError('Official ART backend selected but openpipe-art[backend,langgraph]>=0.4.9 is not installed.')


def _scenario_to_task(scenario: MLWorkflowScenario) -> dict[str, Any]:
    payload = dict(scenario.task_payload or {})
    payload.setdefault('task_id', scenario.task_id)
    payload.setdefault('difficulty', scenario.difficulty)
    return payload


async def rollout(model: Any, scenario: MLWorkflowScenario) -> MLWorkflowTrajectory:
    """Compatibility-converted official ART rollout for the existing ML workflow.

    The first implementation records official ART trajectories while running the
    repository's current SupervisorAgent. It does not expose chain-of-thought or
    raw dataset rows; it stores concise action/tool summaries and reward metrics.
    """
    _require_art()
    traj = MLWorkflowTrajectory(
        reward=0.0,
        messages_and_choices=[],
        metadata={'task_id': scenario.task_id, 'step': scenario.step, 'workflow': 'self_improving_ml_agent', 'conversion': 'official ART trajectory/training path with compatibility-converted workflow messages'},
    )
    try:
        _ = init_chat_model(model.get_inference_name(), temperature=1.0)
    except Exception as exc:
        traj.metadata['init_chat_model_warning'] = f'{exc.__class__.__name__}: {exc}'

    from src.agents.supervisor import SupervisorAgent
    from src.rewards.scorer import score_trajectory

    supervisor = SupervisorAgent(policy_version=f'art_official_step_{scenario.step}')
    custom = supervisor.run_task(_scenario_to_task(scenario))
    custom_dict = custom.to_dict() if hasattr(custom, 'to_dict') else dict(custom)
    for idx, step in enumerate(custom_dict.get('steps', []) or []):
        traj.messages_and_choices.append({
            'role': 'assistant',
            'agent_name': step.get('agent_name'),
            'content': step.get('action') or '',
            'choice': {'reasoning_summary': step.get('reasoning_summary')},
            'metadata': {'step_index': idx},
        })
        for call in step.get('tool_calls', []) or []:
            traj.messages_and_choices.append({
                'role': 'tool',
                'tool_name': call.get('tool_name'),
                'status': call.get('status'),
                'content': call.get('output_ref') or call.get('error') or '',
            })
    try:
        reward, reward_meta = score_trajectory(custom)
    except Exception:
        reward = float(custom_dict.get('reward') or 0.0)
        reward_meta = custom_dict.get('reward_metadata') or {}
    traj.reward = float(reward or 0.0)
    traj.metrics = dict(reward_meta or {})
    traj.metrics['workflow_reward'] = traj.reward
    traj.task_id = scenario.task_id
    traj.policy_version = custom_dict.get('policy_version')
    traj.final_status = custom_dict.get('final_status')
    traj.workflow_reward_metadata = dict(reward_meta or {})
    traj.final_artifacts = custom_dict.get('artifacts') or {}
    traj.metadata['final_status'] = traj.final_status
    traj.metadata['artifacts'] = traj.final_artifacts
    return traj
