"""
LangGraph agent execution graph – Agent Lightning "agent side".

Replaces the hand-rolled sequential loop in SupervisorAgent with a
declarative, observable StateGraph that mirrors the Microsoft Agent
Lightning architecture diagram.

Graph topology
--------------

  START
    └─► policy_router        (LLM call via sidecar – intercepted non-intrusively)
    └─► data_engineer        (DataEngineerAgent   – profiling + schema)
    └─► gbm_specialist       (GradientBoostingSpecialistAgent – model selection)
    └─► sandbox_execution    (SandboxExecutionAgent – dockerised benchmark)
    └─► experiment_tracking  (ExperimentTrackingAgent – MLflow / JSON log)
    └─► reviewer_critic      (ReviewerCriticAgent – governance guardrails)
    └─► END

Each node is a plain function; error propagation uses a "first-error-wins"
pattern: a failing node stores its error string in state['error'] and all
downstream nodes become no-ops, preserving partial artefacts.

Optimization Framework connection
----------------------------------
After graph.invoke() returns, SupervisorAgent.run_task() calls
_report_to_lightning(), which converts the accumulated steps into Agent
Lightning-style (state_t, action_t, reward_t, state_t+1) transition tuples
and POSTs them to the GPU Lightning Server for GRPO/veRL training.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from dataclasses import asdict
from pathlib import Path
from typing import Any, TYPE_CHECKING

from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

from src.agents.base import AgentContext
from src.agents.data_engineer import DataEngineerAgent
from src.agents.gradient_boosting_specialist import GradientBoostingSpecialistAgent
from src.agents.sandbox_execution import SandboxExecutionAgent
from src.agents.experiment_tracking import ExperimentTrackingAgent
from src.agents.reviewer import ReviewerCriticAgent
from src.telemetry.schema import AgentStep, ToolCallRecord

if TYPE_CHECKING:
    from src.inference.lightning_sidecar import LightningClientSidecar


# ── Graph State ────────────────────────────────────────────────────────────────

class AgentGraphState(TypedDict):
    """
    Shared state threaded through every node of the agent graph.

    artifacts and tool_outputs are passed by value through LangGraph so
    each node receives a fresh copy of what the previous node produced.
    steps accumulates serialised AgentStep dicts; the supervisor
    reconstructs AgentStep objects from them when building the Trajectory.
    """
    task:                dict[str, Any]
    policy_version:      str
    artifacts:           dict[str, str]        # artifact-key → file path
    tool_outputs:        dict[str, Any]        # tool-key → result payload
    steps:               list[dict[str, Any]]  # serialised AgentStep dicts
    require_live_policy: bool
    error:               str | None            # first-error-wins


# ── Internal helpers ───────────────────────────────────────────────────────────

def _ctx(state: AgentGraphState) -> AgentContext:
    """Build an AgentContext (mutable within a node) from the current state."""
    return AgentContext(
        task=state['task'],
        policy_version=state['policy_version'],
        artifacts=dict(state['artifacts']),
        tool_outputs=dict(state['tool_outputs']),
    )


# ── Policy Router node (LLM call, intercepted by sidecar) ─────────────────────

def _build_policy_router(
    policy_decision_dir: Path,
    sidecar: LightningClientSidecar | None,
):
    """
    Factory that closes over the sidecar so the returned node is a plain
    function with no extra arguments — as required by LangGraph.

    When require_live_policy=False, the node returns {} (no-op) and
    LangGraph leaves every other state key unchanged.

    When a sidecar is provided, its monkey-patched client intercepts the
    LLM call inside client.decision() without modifying PolicyClient code.
    """
    def policy_router(state: AgentGraphState) -> dict[str, Any]:
        if state.get('error') or not state.get('require_live_policy'):
            return {}

        task           = state['task']
        policy_version = state['policy_version']
        endpoint_profile = str(task.get('policy_profile') or policy_version)

        try:
            from src.inference.policy_client import PolicyClient
            client   = sidecar if sidecar is not None else PolicyClient(policy_version=endpoint_profile)
            decision = client.decision(task, policy_version=endpoint_profile)
        except Exception as exc:
            _log_and_raise('PolicyRouter', state, exc)

        policy_decision_dir.mkdir(parents=True, exist_ok=True)
        safe_id  = ''.join(c if c.isalnum() or c in ('-', '_') else '_' for c in str(task.get('task_id', 'task')))
        safe_pol = ''.join(c if c.isalnum() or c in ('-', '_') else '_' for c in endpoint_profile)
        out_path = policy_decision_dir / f'{safe_id}_{safe_pol}.json'
        out_path.write_text(json.dumps(decision, indent=2), encoding='utf-8')

        step = AgentStep(
            agent_name='PolicyRouter',
            action='queried_live_policy_endpoint',
            reasoning_summary=(
                'Resolved the named policy profile to its OpenAI-compatible GPU endpoint '
                'and captured a compact policy decision for this benchmark rollout.'
            ),
            tool_calls=[
                ToolCallRecord(
                    tool_name='PolicyClient.chat',
                    arguments={
                        'policy_version':        policy_version,
                        'model_name':            decision.get('model_name'),
                        'endpoint_url_redacted': decision.get('endpoint_url_redacted'),
                    },
                    status='success',
                    output_ref=str(out_path),
                )
            ],
        )
        return {
            'tool_outputs': {**state['tool_outputs'], 'policy_decision': decision},
            'artifacts':    {**state['artifacts'], 'policy_decision_json': str(out_path)},
            'steps':        [*state['steps'], asdict(step)],
            'error':        None,
        }

    return policy_router




# ── Strict failure logging ─────────────────────────────────────────────────────

_AGENT_FAILURE_LOG = Path('reports') / 'agent_failures.jsonl'


def _safe_error_record(component: str, state: AgentGraphState, exc: Exception) -> dict[str, Any]:
    task = state.get('task', {}) if isinstance(state, dict) else {}
    return {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'component': component,
        'task_id': task.get('task_id'),
        'rollout_index': task.get('rollout_index'),
        'policy_version': state.get('policy_version') if isinstance(state, dict) else None,
        'error_type': exc.__class__.__name__,
        'error_message': str(exc),
        'raw_rows_exposed_to_llm': False,
        'direct_postgres_used_by_agent': False,
        'strict_fail_closed': True,
    }


def _log_and_raise(component: str, state: AgentGraphState, exc: Exception) -> None:
    _AGENT_FAILURE_LOG.parent.mkdir(parents=True, exist_ok=True)
    with _AGENT_FAILURE_LOG.open('a', encoding='utf-8') as f:
        f.write(json.dumps(_safe_error_record(component, state, exc), default=str) + '\n')
    raise RuntimeError(f'{component} failed in strict mode: {exc.__class__.__name__}: {exc}') from exc

# ── Generic agent node factory ─────────────────────────────────────────────────

def _make_agent_node(agent: Any):
    """
    Return a LangGraph node function for any BaseAgent subclass.

    Error handling: strict fail-closed. Any exception is logged to reports/agent_failures.jsonl and re-raised; downstream nodes do not run and no fake trajectory is emitted.
    """
    def node(state: AgentGraphState) -> dict[str, Any]:
        if state.get('error'):
            raise RuntimeError(f"{agent.name} refused to run because upstream error exists: {state.get('error')}")
        ctx = _ctx(state)
        try:
            step = agent.step(ctx)
        except Exception as exc:
            _log_and_raise(agent.name, state, exc)
        return {
            'artifacts':   ctx.artifacts,
            'tool_outputs': ctx.tool_outputs,
            'steps':        [*state['steps'], asdict(step)],
            'error':        None,
        }

    node.__name__ = agent.name.lower()
    return node


# ── Graph builder ──────────────────────────────────────────────────────────────

def build_agent_graph(
    *,
    sidecar: LightningClientSidecar | None = None,
    policy_decision_dir: str | Path = 'artifacts/policy_decisions',
):
    """
    Compile and return the agent execution StateGraph.

    Parameters
    ----------
    sidecar:
        LightningClientSidecar instance wrapping a PolicyClient.  When
        provided, ALL LLM calls (including those fired indirectly inside
        PolicyClient.decision()) are intercepted via the sidecar's
        monkey-patch — no changes to agent code required.
    policy_decision_dir:
        Directory where the PolicyRouter writes its decision JSON artefact.

    Returns
    -------
    A compiled LangGraph CompiledStateGraph ready for .invoke() calls.
    """
    policy_decision_dir = Path(policy_decision_dir)

    graph = StateGraph(AgentGraphState)

    # ── Nodes ──────────────────────────────────────────────────────────────
    graph.add_node('policy_router',       _build_policy_router(policy_decision_dir, sidecar))
    graph.add_node('data_engineer',       _make_agent_node(DataEngineerAgent()))
    graph.add_node('gbm_specialist',      _make_agent_node(GradientBoostingSpecialistAgent()))
    graph.add_node('sandbox_execution',   _make_agent_node(SandboxExecutionAgent()))
    graph.add_node('experiment_tracking', _make_agent_node(ExperimentTrackingAgent()))
    graph.add_node('reviewer_critic',     _make_agent_node(ReviewerCriticAgent()))

    # ── Edges — linear pipeline ────────────────────────────────────────────
    graph.add_edge(START,                 'policy_router')
    graph.add_edge('policy_router',       'data_engineer')
    graph.add_edge('data_engineer',       'gbm_specialist')
    graph.add_edge('gbm_specialist',      'sandbox_execution')
    graph.add_edge('sandbox_execution',   'experiment_tracking')
    graph.add_edge('experiment_tracking', 'reviewer_critic')
    graph.add_edge('reviewer_critic',     END)

    return graph.compile()
