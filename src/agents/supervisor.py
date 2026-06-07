from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.agents.graph import build_agent_graph, AgentGraphState
from src.inference.lightning_sidecar import LightningClientSidecar, RolloutReport
from src.inference.policy_client import PolicyClient
from src.telemetry.schema import AgentStep, ToolCallRecord, Trajectory
from src.telemetry.logger import TrajectoryLogger
from src.tools.task_registry import (
    TrackATaskLoadError,
    load_tracka_tasks_from_postgres,
    TRACKA_DEFAULT_TASK_LIMIT,
)


def _dict_to_step(d: dict[str, Any]) -> AgentStep:
    """Reconstruct an AgentStep dataclass from its serialised dict form."""
    tool_calls = [
        ToolCallRecord(
            tool_name=tc['tool_name'],
            arguments=tc['arguments'],
            status=tc['status'],
            output_ref=tc.get('output_ref'),
            error=tc.get('error'),
            started_at=tc.get('started_at', ''),
            completed_at=tc.get('completed_at', ''),
        )
        for tc in d.get('tool_calls', [])
    ]
    return AgentStep(
        agent_name=d['agent_name'],
        action=d['action'],
        reasoning_summary=d['reasoning_summary'],
        tool_calls=tool_calls,
        timestamp=d.get('timestamp', ''),
    )



def _attach_compact_additional_histories(trajectory: Trajectory, final_state: dict[str, Any]) -> None:
    """Attach ART-style compact auxiliary histories for sub-agents and workflow compaction.

    The histories intentionally store action summaries, tool names/statuses, and compact
    metadata only. They do not expose hidden chain-of-thought or raw dataset rows.
    """
    by_agent: dict[str, list[AgentStep]] = {}
    for step in trajectory.steps:
        by_agent.setdefault(step.agent_name, []).append(step)

    for agent_name, steps in sorted(by_agent.items()):
        messages_and_choices: list[dict[str, Any]] = []
        tools: list[dict[str, Any]] = []
        for idx, step in enumerate(steps):
            messages_and_choices.append({
                'role': 'assistant',
                'agent_name': agent_name,
                'content': step.action,
                'choice': {'reasoning_summary': step.reasoning_summary, 'action': step.action},
                'metadata': {'step_index': idx, 'timestamp': step.timestamp, 'history_type': 'sub_agent_compact'},
            })
            for call in step.tool_calls:
                tools.append({
                    'tool_name': call.tool_name,
                    'status': call.status,
                    'output_ref': call.output_ref,
                    'has_error': bool(call.error),
                    'started_at': call.started_at,
                    'completed_at': call.completed_at,
                })
        trajectory.add_history(
            name=f'sub_agent:{agent_name}',
            messages_and_choices=messages_and_choices,
            tools=tools,
            metadata={
                'history_type': 'sub_agent_compact',
                'agent_name': agent_name,
                'step_count': len(steps),
                'tool_call_count': len(tools),
            },
        )

    ordered_steps = [
        {
            'step_index': idx,
            'agent_name': step.agent_name,
            'action': step.action,
            'tool_call_count': len(step.tool_calls),
            'failed_tool_call_count': sum(1 for call in step.tool_calls if call.status not in ('success', 'ok', 'completed') or call.error),
        }
        for idx, step in enumerate(trajectory.steps)
    ]
    trajectory.add_history(
        name='workflow_compaction',
        messages_and_choices=[{
            'role': 'assistant',
            'content': 'Compact trajectory history for replay, reward judging, and training without raw data exposure.',
            'choice': {
                'final_status': trajectory.final_status,
                'ordered_steps': ordered_steps,
                'artifact_keys': sorted((final_state.get('artifacts') or {}).keys()),
            },
            'metadata': {'history_type': 'compaction', 'step_count': len(trajectory.steps)},
        }],
        tools=[
            {'tool_name': call.tool_name, 'status': call.status, 'output_ref': call.output_ref, 'has_error': bool(call.error)}
            for step in trajectory.steps
            for call in step.tool_calls
        ],
        metadata={'history_type': 'compaction', 'raw_rows_exposed_to_llm': False, 'workflow_error': final_state.get('error')},
    )

    trajectory.add_history(
        name='multi_agent_workflow',
        messages_and_choices=[{
            'role': 'system',
            'content': 'policy_router → data_engineer → gbm_specialist → sandbox_execution → experiment_tracking → reviewer_critic',
            'choice': {'agent_path': [step.agent_name for step in trajectory.steps], 'task_id': trajectory.task_id, 'policy_version': trajectory.policy_version},
            'metadata': {'history_type': 'multi_agent_workflow', 'supports_complex_workflow_replay': True},
        }],
        tools=[],
        metadata={'history_type': 'multi_agent_workflow', 'agent_count': len(by_agent), 'step_count': len(trajectory.steps)},
    )


class SupervisorAgent:
    """
    Orchestrator for the Agent Lightning CPU execution plane.

    Runs the multi-agent LangGraph pipeline for each task, optionally wraps
    the PolicyClient in a LightningClientSidecar to intercept all LLM calls,
    and reports completed rollouts to the GPU Lightning Server for GRPO/veRL
    training.

    LIGHTNING_SERVER_URL must be configured. Reporting failures are fatal so
    rollouts are never silently dropped.
    """

    def __init__(
        self,
        require_live_policy: bool = False,
        policy_decision_dir: str | Path = 'artifacts/policy_decisions',
        lightning_server_url: str | None = None,
    ) -> None:
        self.require_live_policy  = require_live_policy
        self.policy_decision_dir  = Path(policy_decision_dir)
        self.lightning_server_url = lightning_server_url or os.getenv('LIGHTNING_SERVER_URL', '')

    # ── Sidecar factory ────────────────────────────────────────────────────

    def _make_sidecar(self, policy_version: str | None = None) -> LightningClientSidecar:
        """
        Build a LightningClientSidecar for one rollout.

        When require_live_policy is True a real PolicyClient is created and
        monkey-patched by the sidecar so all LLM calls (including those fired
        inside PolicyClient.decision()) are intercepted non-intrusively.
        """
        if not self.lightning_server_url:
            raise RuntimeError(
                'SupervisorAgent requires LIGHTNING_SERVER_URL in strict mode.'
            )

        client: PolicyClient | None = None
        if self.require_live_policy and policy_version:
            client = PolicyClient(policy_version=policy_version)
        return LightningClientSidecar(
            policy_client=client,
            server_url=self.lightning_server_url,
        )

    # ── Lightning Server reporting (Optimization Framework entry point) ────

    def _report_to_lightning(
        self,
        trajectory: Trajectory,
        sidecar: LightningClientSidecar,
    ) -> None:
        """
        Score the trajectory inline and POST a RolloutReport to the Lightning
        Server's reporting API.

        Converts the trajectory into Agent Lightning transition tuples
        (state_t, action_t, reward_t, state_t+1) and includes all intercepted
        LLM spans and error types for the server's credit-assignment and
        error-monitoring algorithms.

        Strict mode raises explicit errors whenever reporting fails.
        """
        if not sidecar.enabled:
            raise RuntimeError(
                'SupervisorAgent reporting failed: sidecar is disabled. Ensure LIGHTNING_SERVER_URL is configured.'
            )

        from src.rewards.scorer import score_trajectory
        from src.training.agent_lightning_export import trajectory_to_transitions
        import agentlightning as agl  # type: ignore

        try:
            traj_dict                  = trajectory.to_dict()
            reward, reward_meta        = score_trajectory(traj_dict)
            trajectory.reward          = reward
            trajectory.reward_metadata = reward_meta
            traj_dict                  = trajectory.to_dict()   # refresh with reward

            transitions = trajectory_to_transitions(traj_dict)
            error_types = [
                tc.get('tool_name', '')
                for step in traj_dict.get('steps', [])
                for tc in step.get('tool_calls', [])
                if tc.get('status') == 'failed'
            ]

            try:
                agl.emit_reward(reward)
                reward_meta['agentlightning_emit_reward_status'] = 'emitted'
            except RuntimeError as emit_exc:
                if 'No active tracer found' not in str(emit_exc):
                    raise
                reward_meta['agentlightning_emit_reward_status'] = 'skipped_no_active_tracer'
                reward_meta['agentlightning_emit_reward_error'] = str(emit_exc)

            report = RolloutReport(
                task_id=traj_dict['task_id'],
                trajectory_id=traj_dict['trajectory_id'],
                policy_version=traj_dict['policy_version'],
                llm_spans=[asdict(s) for s in sidecar.get_spans()],
                transitions=transitions,
                final_reward=reward,
                reward_metadata=reward_meta,
                error_types=error_types,
                trajectory_dict=traj_dict,
            )
            sidecar.report_rollout(report)
        except Exception as exc:
            raise RuntimeError(
                'SupervisorAgent reporting failed in strict mode: '
                f'{exc.__class__.__name__}: {exc}'
            ) from exc
        finally:
            sidecar.clear_spans()

    # ── Main execution ─────────────────────────────────────────────────────

    def run_task(self, task: dict[str, Any], policy_version: str) -> Trajectory:
        """
        Execute one task through the LangGraph agent pipeline.

        Graph topology (see src/agents/graph.py):
            policy_router → data_engineer → gbm_specialist
            → sandbox_execution → experiment_tracking → reviewer_critic

        After the graph completes, the trajectory is reported to the Lightning
        Server (Stage 2 → Stage 3 of the Agent Lightning loop).
        """
        task_payload = dict(task)
        sidecar      = self._make_sidecar(policy_version)
        graph        = build_agent_graph(
            sidecar=sidecar,
            policy_decision_dir=self.policy_decision_dir,
        )

        initial: AgentGraphState = {
            'task':                task_payload,
            'policy_version':      policy_version,
            'artifacts':           {},
            'tool_outputs':        {},
            'steps':               [],
            'require_live_policy': self.require_live_policy,
            'error':               None,
        }

        final      = graph.invoke(initial)
        trajectory = Trajectory(task_id=task_payload['task_id'], policy_version=policy_version)

        for step_dict in final.get('steps', []):
            trajectory.steps.append(_dict_to_step(step_dict))

        trajectory.final_status = 'error' if final.get('error') else 'completed'
        _attach_compact_additional_histories(trajectory, final)
        self._report_to_lightning(trajectory, sidecar)
        return trajectory


# ── Module-level runner ────────────────────────────────────────────────────────

def run_tasks(
    output_dir: str,
    policy_version: str,
    task_limit: int | None = None,
    rollouts: int = 1,
    require_live_policy: bool = False,
    lightning_server_url: str | None = None,
) -> None:
    """Execute Track A over tasks loaded from PostgreSQL via MCP.

    This is the strict, fail-closed Track A execution path. PostgreSQL/MCP is
    the only supported task source.
    """
    if not lightning_server_url:
        raise TrackATaskLoadError(
            'Strict Agent Lightning mode requires --lightning-server-url. '
            'Set LIGHTNING_SERVER_URL or pass --lightning-server-url.'
        )

    source_mode = str(os.getenv('TRACKA_TASK_SOURCE', 'postgres_registry')).strip().lower()
    if source_mode not in {'postgres_registry', 'postgres_mcp'}:
        raise TrackATaskLoadError(
            f'Unsupported TRACKA_TASK_SOURCE={source_mode!r}. '
            "Allowed values are 'postgres_registry' or 'postgres_mcp'."
        )

    # ── Task loading: PostgreSQL/MCP only ────────────────────────────────────
    try:
        task_records = load_tracka_tasks_from_postgres(limit=task_limit)
    except TrackATaskLoadError:
        raise
    except Exception as exc:
        raise TrackATaskLoadError(
            'Unexpected failure while loading Track A tasks from PostgreSQL via MCP. '
            f'Cause: {exc.__class__.__name__}.'
        ) from exc

    supervisor = SupervisorAgent(
        require_live_policy=require_live_policy,
        lightning_server_url=lightning_server_url,
    )
    logger = TrajectoryLogger(output_dir)
    count  = 0

    for task in task_records:
        for rollout_index in range(max(1, rollouts)):
            task_variant                   = dict(task)
            task_variant['rollout_index']  = rollout_index
            task_variant['rollout_seed']   = 42 + rollout_index
            task_variant['policy_profile'] = policy_version
            effective = policy_version if rollouts == 1 else f'{policy_version}_r{rollout_index:02d}'
            traj = supervisor.run_task(task_variant, policy_version=effective)
            logger.write(traj)
            count += 1

    print(
        f'Wrote {count} trajectories for {len(task_records)} PostgreSQL/MCP tasks '
        f'(strict mode) to {output_dir}'
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Track A supervisor — PostgreSQL/MCP-first strict mode.',
    )
    parser.add_argument(
        '--task-source',
        default='postgres_mcp',
        choices=['postgres_mcp'],
        help='Task source.  Only postgres_mcp is supported in production.'
    )
    parser.add_argument(
        '--task-limit',
        dest='task_limit',
        type=int,
        default=None,
        help=f'Maximum tasks to load from PostgreSQL (default: TRACKA_DEFAULT_TASK_LIMIT={TRACKA_DEFAULT_TASK_LIMIT}).'
    )
    parser.add_argument('--output-dir',          default='trajectories/track_a_postgres')
    parser.add_argument('--policy-version',      default='baseline')
    parser.add_argument('--rollouts',            type=int, default=1)
    parser.add_argument('--require-live-policy', action='store_true')
    parser.add_argument('--lightning-server-url', default=None,
                        help='GPU Lightning Server URL (overrides LIGHTNING_SERVER_URL env var)')
    args = parser.parse_args()
    run_tasks(
        output_dir=args.output_dir,
        policy_version=args.policy_version,
        task_limit=args.task_limit,
        rollouts=args.rollouts,
        require_live_policy=args.require_live_policy,
        lightning_server_url=args.lightning_server_url,
    )
