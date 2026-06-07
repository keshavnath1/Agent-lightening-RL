"""Local ART-like compatibility objects for custom trajectory telemetry.

This module deliberately does not import the official ``art`` package. It is a
stable offline adapter that converts the repository's custom trajectory schema
into structures that look like ART trajectories/groups for demos, dataset
building, and relative reward scoring.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict, is_dataclass
from typing import Any

try:  # optional typing aid only
    from src.telemetry.schema import Trajectory
except Exception:  # pragma: no cover
    Trajectory = Any  # type: ignore


@dataclass
class ArtLikeHistory:
    name: str
    messages_and_choices: list[dict[str, Any]]
    tools: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ArtLikeTrajectory:
    trajectory_id: str
    task_id: str
    policy_version: str
    messages_and_choices: list[dict[str, Any]]
    additional_histories: list[ArtLikeHistory]
    reward: float | None
    metrics: dict[str, Any]
    metadata: dict[str, Any]


@dataclass
class ArtLikeTrajectoryGroup:
    task_id: str
    trajectories: list[ArtLikeTrajectory]
    metadata: dict[str, Any] = field(default_factory=dict)


def _to_dict(obj: dict[str, Any] | Trajectory) -> dict[str, Any]:
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, 'to_dict'):
        return obj.to_dict()  # type: ignore[no-any-return]
    if is_dataclass(obj):
        return asdict(obj)
    raise TypeError(f'Unsupported trajectory object: {type(obj)!r}')


def _tool_message(call: dict[str, Any]) -> dict[str, Any]:
    return {
        'role': 'tool',
        'tool_name': call.get('tool_name'),
        'status': call.get('status'),
        'content': call.get('output_ref') or call.get('error') or '',
        'metadata': {
            'has_error': bool(call.get('error')),
            'arguments': call.get('arguments') or {},
            'started_at': call.get('started_at'),
            'completed_at': call.get('completed_at'),
        },
    }


def _step_to_messages(step: dict[str, Any], index: int) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [{
        'role': 'assistant',
        'agent_name': step.get('agent_name'),
        'content': step.get('action') or '',
        'choice': {
            'action': step.get('action'),
            'reasoning_summary': step.get('reasoning_summary'),
        },
        'metadata': {'step_index': index, 'timestamp': step.get('timestamp')},
    }]
    for call in step.get('tool_calls', []) or []:
        messages.append(_tool_message(call))
    return messages


def _lightweight_histories_from_steps(data: dict[str, Any]) -> list[ArtLikeHistory]:
    by_agent: dict[str, list[dict[str, Any]]] = {}
    tools_by_agent: dict[str, list[dict[str, Any]]] = {}
    alias = {
        'policy_router': 'policy_router_history',
        'data_engineer': 'data_engineer_history',
        'gbm_specialist': 'gbm_specialist_history',
        'sandbox_executor': 'sandbox_execution_history',
        'experiment_tracker': 'experiment_tracking_history',
        'reviewer': 'reviewer_critic_history',
        'critic': 'reviewer_critic_history',
    }
    for idx, step in enumerate(data.get('steps', []) or []):
        agent = str(step.get('agent_name') or 'unknown_agent')
        name = alias.get(agent, f'{agent}_history')
        by_agent.setdefault(name, []).append({
            'role': 'assistant',
            'agent_name': agent,
            'content': step.get('action') or '',
            'choice': {'reasoning_summary': step.get('reasoning_summary')},
            'metadata': {'step_index': idx, 'timestamp': step.get('timestamp')},
        })
        for call in step.get('tool_calls', []) or []:
            tools_by_agent.setdefault(name, []).append({
                'tool_name': call.get('tool_name'),
                'status': call.get('status'),
                'output_ref': call.get('output_ref'),
                'has_error': bool(call.get('error')),
            })
    return [ArtLikeHistory(name=k, messages_and_choices=v, tools=tools_by_agent.get(k, [])) for k, v in sorted(by_agent.items())]


def custom_to_art_like(trajectory: dict[str, Any] | Trajectory) -> ArtLikeTrajectory:
    data = _to_dict(trajectory)
    messages: list[dict[str, Any]] = []
    for idx, step in enumerate(data.get('steps', []) or []):
        messages.extend(_step_to_messages(step, idx))

    histories: list[ArtLikeHistory] = []
    for h in data.get('additional_histories', []) or []:
        histories.append(ArtLikeHistory(
            name=str(h.get('name') or 'additional_history'),
            messages_and_choices=list(h.get('messages_and_choices') or []),
            tools=list(h.get('tools') or []),
            metadata=dict(h.get('metadata') or {}),
        ))
    if not histories:
        histories = _lightweight_histories_from_steps(data)

    reward_metadata = dict(data.get('reward_metadata') or {})
    metadata = {
        'task_id': data.get('task_id'),
        'policy_version': data.get('policy_version'),
        'final_status': data.get('final_status'),
        'created_at': data.get('created_at'),
    }
    metadata.update(dict(data.get('metadata') or {}))
    return ArtLikeTrajectory(
        trajectory_id=str(data.get('trajectory_id') or ''),
        task_id=str(data.get('task_id') or ''),
        policy_version=str(data.get('policy_version') or 'unknown'),
        messages_and_choices=messages,
        additional_histories=histories,
        reward=data.get('reward'),
        metrics=reward_metadata,
        metadata=metadata,
    )


def group_custom_trajectories_by_task(trajectories: list[dict[str, Any] | Trajectory]) -> list[ArtLikeTrajectoryGroup]:
    grouped: dict[str, list[ArtLikeTrajectory]] = {}
    for traj in trajectories:
        art_like = custom_to_art_like(traj)
        grouped.setdefault(art_like.task_id, []).append(art_like)
    return [
        ArtLikeTrajectoryGroup(task_id=task_id, trajectories=rows, metadata={'group_size': len(rows)})
        for task_id, rows in sorted(grouped.items())
    ]


def art_like_to_training_example(traj: ArtLikeTrajectory) -> dict[str, Any]:
    return {
        'trajectory_id': traj.trajectory_id,
        'task_id': traj.task_id,
        'policy_version': traj.policy_version,
        'messages_and_choices': traj.messages_and_choices,
        'additional_histories': [asdict(h) for h in traj.additional_histories],
        'reward': traj.reward,
        'reward_metadata': traj.metrics,
        'metadata': traj.metadata,
    }
