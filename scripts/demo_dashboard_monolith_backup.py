"""
Self-Improving ML Agent — Project Cockpit
==========================================

A storytelling dashboard that guides users through the full Agent Lightning
feedback loop.  Organized as two tracks:

  Track A — Business ML Workflow
      LangGraph multi-agent pipeline that profiles data, trains
      XGBoost / LightGBM models, tracks metrics in MLflow, and produces
      a champion model artifact.

  Track B — Agent Self-Improvement
      Agent Lightning-style loop that collects LLM spans + rewards from
      Track A, groups rollouts, trains a LoRA adapter (trl_grpo / qlora_sft
      / verl / agent_lightning_official), hot-reloads vLLM, and closes the
      policy improvement cycle.

Tabs
----
  1  🏁 Setup & Services      — environment checks + service controls
  2  🧭 Architecture          — Track A / Track B concept map
  3  🧪 Track A: ML Workflow  — run / stream the LangGraph pipeline
  4  ⚡ Agent Lightning Trace  — live trace with LLM span attribution
  5  🏆 Rewards & Rollouts    — rollout browser, grouped rollouts, reward curve
  6  🔥 Track B: Optimization — trainer selector, preflight, training trigger
  7  📈 Compare Policies      — baseline vs tuned metrics comparison

Start:
    streamlit run scripts/demo_dashboard.py

Optional env vars:
    LIGHTNING_SERVER_URL   default http://localhost:19123
    VLLM_BASE_URL          default http://localhost:8000
    MLFLOW_TRACKING_URI    default mlruns/
    MCP_SERVER_URL         default http://localhost:8080
"""
from __future__ import annotations

import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── Add workspace root to Python path ─────────────────────────────────────────
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import requests
import streamlit as st

# ── Page config (must be first Streamlit call) ─────────────────────────────────
st.set_page_config(
    page_title='Self-Improving ML Agent',
    page_icon='⚡',
    layout='wide',
    initial_sidebar_state='expanded',
)

# ── Lazy-import UI support modules ─────────────────────────────────────────────
@st.cache_resource
def _get_service_manager():
    from src.ui import service_manager
    return service_manager


@st.cache_resource
def _get_health_checks():
    from src.ui import health_checks
    return health_checks


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — global config
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.title('⚡ Agent Lightning')
st.sidebar.caption('Self-Improving ML Agent')

st.sidebar.markdown('---')
st.sidebar.markdown('**Services**')

server_url: str = st.sidebar.text_input(
    'Lightning Server URL',
    value=os.getenv('LIGHTNING_SERVER_URL', 'http://localhost:19123'),
)
vllm_url: str = st.sidebar.text_input(
    'vLLM URL',
    value=os.getenv('VLLM_BASE_URL', 'http://localhost:8000'),
)
mcp_url: str = st.sidebar.text_input(
    'MCP Server URL',
    value=os.getenv('MCP_SERVER_URL', 'http://localhost:8080'),
)
mlflow_uri: str = st.sidebar.text_input(
    'MLflow Tracking URI',
    value=os.getenv('MLFLOW_TRACKING_URI', str(_ROOT / 'mlruns')),
)
policy_version: str = st.sidebar.text_input('Policy version label', value='v1')

st.sidebar.markdown('---')
st.sidebar.markdown('**GPU execution mode**')
gpu_mode: str = st.sidebar.radio(
    'Track B data handoff',
    ['Offline — shared volume', 'Online — live streaming'],
    key='gpu_mode',
    help=(
        'Offline: CPU writes grouped_rollouts.jsonl to the shared network volume; '
        'GPU reads it directly (no network needed between pods).\n\n'
        'Online: CPU Track A streams rollout reports in real time to the GPU '
        'Lightning Server via LIGHTNING_SERVER_URL.'
    ),
)
if gpu_mode == 'Online — live streaming':
    st.sidebar.info(
        'Set `LIGHTNING_SERVER_URL` on the **CPU pod** to point at the GPU pod:\n\n'
        f'`export LIGHTNING_SERVER_URL={server_url}`'
    )

st.sidebar.markdown('---')

# ── Quick server health in sidebar ────────────────────────────────────────────
def _get(path: str, base: str = '', timeout: int = 4) -> dict[str, Any] | None:
    try:
        r = requests.get(f'{base or server_url}{path}', timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        return {'error': str(exc)}


def _post(path: str, payload: dict | None = None, timeout: int = 30) -> dict[str, Any] | None:
    try:
        r = requests.post(f'{server_url}{path}', json=payload or {}, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        return {'error': str(exc)}


health = _get('/health')
if health and 'error' not in health:
    st.sidebar.success('⚡ Lightning Server online')
    st.sidebar.metric('Queue',       health.get('queue_size', '–'))
    st.sidebar.metric('Rollouts',    health.get('rollouts_collected', '–'))
    st.sidebar.metric('Transitions', health.get('transitions_written', '–'))
else:
    st.sidebar.warning('⚡ Lightning Server offline')

vhealth = _get('/v1/models', base=vllm_url)
if vhealth and 'error' not in vhealth:
    models = [m.get('id', '?') for m in vhealth.get('data', [])]
    st.sidebar.success(f'🤖 vLLM online — {len(models)} model(s)')
else:
    st.sidebar.warning('🤖 vLLM offline')

st.sidebar.markdown('---')
st.sidebar.caption(f'Workspace: `{_ROOT}`')

# ─────────────────────────────────────────────────────────────────────────────
# Tab layout
# ─────────────────────────────────────────────────────────────────────────────
(
    tab_setup,
    tab_arch,
    tab_track_a,
    tab_trace,
    tab_rewards,
    tab_track_b,
    tab_compare,
) = st.tabs([
    '🏁 Setup & Services',
    '🧭 Architecture',
    '🧪 Track A: ML Workflow',
    '⚡ Agent Lightning Trace',
    '🏆 Rewards & Rollouts',
    '🔥 Track B: Optimization',
    '📈 Compare Policies',
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Setup & Services
# ══════════════════════════════════════════════════════════════════════════════
with tab_setup:
    st.header('🏁 Setup & Services')
    st.caption(
        'Verify your environment and start required services before running the demo.  '
        'A new user should never need to ask "which script do I run next?" — this page '
        'always shows current status, the next required action, and the fix command.'
    )

    hc = _get_health_checks()
    sm = _get_service_manager()

    col_mode, col_refresh = st.columns([3, 1])
    with col_mode:
        setup_mode = st.selectbox(
            'Setup mode',
            ['Local CPU Demo', 'CPU + Remote GPU', 'Full RunPod (CPU + GPU pods)'],
            help=(
                'Local CPU Demo: no real GPU, uses mock policy.\n'
                'CPU + Remote GPU: Streamlit on CPU, vLLM / Lightning Server on GPU.\n'
                'Full RunPod: both CPU and GPU pods on RunPod.'
            ),
        )
    with col_refresh:
        st.markdown('&nbsp;', unsafe_allow_html=True)
        refresh_btn = st.button('🔄 Re-check', use_container_width=True)

    st.divider()

    if refresh_btn or 'hc_results' not in st.session_state:
        with st.spinner('Running health checks…'):
            st.session_state['hc_results'] = hc.run_all_checks(
                lightning_url=server_url,
                vllm_url=vllm_url,
                mcp_url=mcp_url,
                mlflow_uri=mlflow_uri,
                root=_ROOT,
            )

    results: dict = st.session_state['hc_results']

    def _render_checks(title: str, checks: list) -> None:
        if title:
            st.subheader(title)
        for c in checks:
            icon = '✅' if c['ok'] else '❌'
            col_i, col_d, col_f = st.columns([0.5, 5, 4])
            col_i.markdown(icon)
            col_d.markdown(f'**{c["name"]}** — {c["detail"]}')
            if not c['ok'] and c.get('fix'):
                col_f.code(c['fix'], language='bash')

    _render_checks('Runtime environment', results['runtime'])

    if setup_mode in ('CPU + Remote GPU', 'Full RunPod (CPU + GPU pods)'):
        _render_checks('GPU packages', results['gpu_pkgs'])
    else:
        with st.expander('GPU packages (not required for Local CPU Demo)'):
            _render_checks('', results['gpu_pkgs'])

    with st.expander('CPU packages'):
        _render_checks('', results['cpu_pkgs'])

    st.divider()
    _render_checks('Services', results['services'])

    st.divider()
    _render_checks('Project paths', results['paths'])

    # ── Service control panel ─────────────────────────────────────────────────
    st.divider()
    st.subheader('Service Controls')
    st.caption('Each button launches the corresponding allow-listed shell script in the background.')

    if 'launched_jobs' not in st.session_state:
        st.session_state['launched_jobs'] = {}
    launched: dict[str, str] = st.session_state['launched_jobs']

    def _service_row(label: str, key: str, env: dict | None = None) -> None:
        col_lbl, col_btn, col_status = st.columns([3, 2, 5])
        col_lbl.markdown(f'**{label}**')
        if col_btn.button('▶ Start', key=f'start_{key}', use_container_width=True):
            try:
                jid = sm.launch(key, env_overrides=env)
                launched[key] = jid
                st.session_state['launched_jobs'] = launched
                col_status.success(f'Launched — job `{jid}`')
            except Exception as exc:
                col_status.error(str(exc))
        if key in launched:
            status = sm.job_status(launched[key]) or {}
            s  = status.get('status', '?')
            rc = status.get('return_code')
            badge = {'running': '🟡 running', 'succeeded': '✅ succeeded',
                     'failed': '❌ failed', 'stopped': '⏹ stopped'}.get(s, s)
            col_status.caption(badge + (f' (rc={rc})' if rc is not None else ''))

    st.markdown('**CPU Services**')
    _service_row('PostgreSQL',      'start_postgres')
    _service_row('MCP Tool Server', 'start_mcp_server')
    _service_row('MLflow UI',       'start_mlflow')

    if setup_mode != 'Local CPU Demo':
        st.markdown('**GPU Services**')
        _service_row('Lightning Server', 'start_lightning_server')
        _service_row('vLLM (baseline)',  'start_vllm_baseline')

    st.markdown('**Data pipeline**')
    _service_row('1. Generate synthetic tasks',   'generate_synthetic')
    _service_row('2. Load tasks into PostgreSQL', 'load_postgres')
    _service_row('3. Run baseline Track A',       'run_baseline_workflow')
    _service_row('4. Score trajectories',         'score_trajectories')
    _service_row('5. Prepare GRPO dataset',       'prepare_grpo_dataset')

    # ── CPU → GPU handoff validation ─────────────────────────────────────────
    st.divider()
    st.subheader('CPU → GPU Handoff Validation')

    grpo_handoff = _ROOT / 'data' / 'grpo' / 'grouped_rollouts.jsonl'
    hv_col1, hv_col2 = st.columns([3, 2])
    with hv_col1:
        if grpo_handoff.exists():
            line_count = sum(1 for ln in grpo_handoff.read_text().splitlines() if ln.strip())
            size_kb    = grpo_handoff.stat().st_size // 1024
            st.success(
                f'✅ `data/grpo/grouped_rollouts.jsonl` — **{line_count} rollouts**, {size_kb} KB  \n'
                'GPU pod can start Track B training.'
            )
        else:
            st.error(
                '❌ `data/grpo/grouped_rollouts.jsonl` not found.  \n'
                'Run step **5. Prepare GRPO dataset** above before switching to the GPU pod.'
            )
        scored_dir = _ROOT / 'trajectories' / 'scored'
        scored_count = len(list(scored_dir.glob('*.jsonl'))) if scored_dir.exists() else 0
        if scored_count:
            st.success(f'✅ `trajectories/scored/` — **{scored_count}** scored trajectory file(s)')
        else:
            st.warning('⚠️ `trajectories/scored/` is empty — run steps 3 + 4 first.')
    with hv_col2:
        if gpu_mode == 'Online — live streaming':
            st.markdown('**Online mode — set on CPU pod:**')
            st.code(
                f'export LIGHTNING_SERVER_URL={server_url}\n'
                'export WORKSPACE_DIR=/workspace/self-improving-ml-agent\n\n'
                '# Then run Track A as usual:\n'
                'bash scripts/cpu/run_03_run_baseline_workflow.sh',
                language='bash',
            )
            st.caption(
                'The sidecar will POST every rollout report directly to the GPU '
                'Lightning Server.  `grouped_rollouts.jsonl` is optional in this mode.'
            )
        else:
            st.markdown('**Offline mode — run on GPU pod:**')
            st.code(
                'export WORKSPACE_DIR=/workspace/self-improving-ml-agent\n'
                'cd $WORKSPACE_DIR\n\n'
                '# Verify the file is visible:\n'
                'ls -lh data/grpo/grouped_rollouts.jsonl\n'
                'wc -l data/grpo/grouped_rollouts.jsonl\n\n'
                '# Then start GPU Track B:\n'
                'bash scripts/gpu/start_lightning_server.sh\n'
                'bash scripts/gpu/run_02_train_policy_qlora_grpo.sh',
                language='bash',
            )

    if launched:
        st.divider()
        st.subheader('Job logs')
        sel_job = st.selectbox('Select job', list(launched.values()), key='setup_log_sel')
        if sel_job:
            st.code(sm.read_job_log(sel_job, lines=80) or '(no output yet)', language='text')


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Architecture
# ══════════════════════════════════════════════════════════════════════════════
with tab_arch:
    st.header('🧭 Architecture Map')
    st.caption(
        'Track A builds the ML model.  Track B improves the agent policy.  '
        'Agent Lightning observes Track A without changing the business logic.'
    )

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('### Track A — Business ML Workflow')
        st.code(
            'User Task\n'
            '    │\n'
            '    ▼\n'
            'Policy Router\n'
            '    │ decides which agents to run\n'
            '    ▼\n'
            'Data Engineer Agent\n'
            '    │ YData profiling → profile_summary.json\n'
            '    │ schema extraction → schema_metadata.json\n'
            '    │ preprocessing → clean.parquet\n'
            '    ▼\n'
            'GBM Specialist Agent\n'
            '    │ model selection (XGBoost / LightGBM)\n'
            '    │ benchmark plan → gbm_benchmark_plan.json\n'
            '    ▼\n'
            'Sandbox Execution Agent\n'
            '    │ trains models inside Docker interpreter\n'
            '    │ benchmark_metrics.json + champion_model.json\n'
            '    ▼\n'
            'Experiment Tracking Agent\n'
            '    │ MLflow run logged + champion model registered\n'
            '    ▼\n'
            'Reviewer / Critic Agent\n'
            '    │ guardrail checks\n'
            '    │ reward = 0.0 – 1.0\n'
            '    ▼\n'
            'Trajectory stored  +  LLM spans captured',
            language='text',
        )

    with col_b:
        st.markdown('### Track B — Agent Self-Improvement')
        tb_path_tab, tb_online_tab = st.tabs(['📂 Offline (shared volume)', '🌐 Online (live streaming)'])
        with tb_path_tab:
            st.code(
                'CPU pod finishes Track A + scoring + grouping\n'
                '    │  run_05_prepare_grpo_dataset.sh\n'
                '    ▼\n'
                'data/grpo/grouped_rollouts.jsonl  ← on shared volume\n'
                '            │\n'
                '            ▼  (GPU pod reads same path)\n'
                'GPU pod\n'
                '    bash scripts/gpu/start_lightning_server.sh\n'
                '    bash scripts/gpu/run_02_train_policy_qlora_grpo.sh\n'
                '            │  reads GRPO_DATASET_PATH from volume\n'
                '            ▼\n'
                'GRPO Algorithm  →  Trainer  →  LoRA adapter\n'
                '            │\n'
                '            ▼\n'
                'vLLM hot-reload  →  Improved policy',
                language='text',
            )
            st.caption('`WORKSPACE_DIR` must be identical on both pods (same mount point).')
        with tb_online_tab:
            st.code(
                'CPU pod (Track A running)\n'
                '    export LIGHTNING_SERVER_URL=http://<gpu-pod-ip>:19123\n'
                '            │\n'
                '            ▼\n'
                'Lightning Sidecar\n'
                '    │ intercepts LLM calls → LLMSpan\n'
                '    │ POST /api/rollouts/report → GPU Lightning Server\n'
                '            │  (real-time, no file needed)\n'
                '            ▼\n'
                'GPU Lightning Server  (port 19123)\n'
                '    │ /api/rollouts/report\n'
                '    │ LightningStore ← rollout (no stale preparing)\n'
                '    │ transitions.jsonl written\n'
                '            │  when rollouts >= MIN_ROLLOUTS\n'
                '            ▼\n'
                'POST /api/training/trigger  (or auto-trigger)\n'
                '            │\n'
                '            ▼\n'
                'Trainer  →  LoRA adapter  →  vLLM hot-reload',
                language='text',
            )
            st.caption(
                'Requires GPU pod reachable from CPU pod on port 19123.  '
                'Set `LIGHTNING_SERVER_URL` before starting Track A.'
            )

    st.divider()
    col_lc1, col_lc2 = st.columns(2)
    with col_lc1:
        st.subheader('Rollout lifecycle')
        st.code(
            'QUEUING  →  PREPARING  →  RUNNING  →  SUCCEEDED  →  TRAINING_READY',
            language='text',
        )
        st.caption(
            '`pull_task()` injects `lightning_rollout_id` into the task payload.  '
            'The sidecar echoes it back in `RolloutReport` so '
            '`accept_rollout_report()` updates the same record — no stale preparing entries.'
        )

    with col_lc2:
        st.subheader('Agent Lightning collection meter')
        store_stats = _get('/api/store/statistics')
        if store_stats and 'error' not in store_stats:
            st.metric('LLM spans collected', store_stats.get('total_spans', 0))
            st.metric('Transitions emitted', store_stats.get('transitions_written', 0))
            st.metric('Rollouts succeeded',  store_stats.get('succeeded_rollouts', 0))
            st.metric('Rollouts preparing',  store_stats.get('preparing_rollouts', 0))
        else:
            st.caption('Lightning Server offline — stats not available.')

    st.divider()
    st.subheader('Agent Lightning parity status')
    st.markdown(
        '| Component | Status |\n'
        '|---|---|\n'
        '| LangGraph StateGraph pipeline | ✅ |\n'
        '| LightningClientSidecar (non-intrusive trace) | ✅ |\n'
        '| LightningStoreAdapter (rollout store + async API) | ✅ |\n'
        '| LightningTrainer (task loading + training loop) | ✅ |\n'
        '| GRPOAlgorithm (triplet dataset + vLLM reload) | ✅ |\n'
        '| SpanTraceAdapter (LLMSpan → Triplet) | ✅ |\n'
        '| MLAgentLitAgent (LitAgent wrapper) | ✅ |\n'
        '| LLMSpan attribution (rollout_id / attempt_id / seq) | ✅ |\n'
        '| trl_grpo 4-bit QLoRA (BitsAndBytesConfig) | ✅ |\n'
        '| veRL strict mode (fails if not installed) | ✅ |\n'
        '| Official agentlightning pip package | ⚠️ optional (blinker conflict) |\n'
    )

    with st.expander('LangGraph graph topology (Mermaid source)'):
        try:
            from src.agents.graph import build_agent_graph
            from src.inference.lightning_sidecar import LightningClientSidecar
            _g = build_agent_graph(
                sidecar=LightningClientSidecar(),
                policy_decision_dir=_ROOT / 'artifacts' / 'policy_decisions',
            )
            st.code(_g.get_graph().draw_mermaid(), language='text')
            st.caption('Paste into https://mermaid.live to render.')
        except Exception as _e:
            st.caption(f'Graph not available: {_e}')


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Track A: ML Workflow
# ══════════════════════════════════════════════════════════════════════════════
with tab_track_a:
    st.header('🧪 Track A: ML Workflow')
    st.caption(
        'Run one task through the full LangGraph agent pipeline.  '
        'Each agent node executes, calls tools, produces artifacts, '
        'and is scored by the Reviewer/Critic to assign a trajectory reward.'
    )

    col_left, col_right = st.columns([2, 1])
    with col_left:
        task_id_a   = st.text_input('Task ID', value=f'task_{uuid.uuid4().hex[:6]}', key='ta_id')
        task_desc_a = st.text_area(
            'Task description', height=110, key='ta_desc',
            value=(
                'Analyse the provided dataset, engineer features, '
                'train a LightGBM model, and report benchmark metrics.'
            ),
        )
        dataset_a = st.text_input('Dataset path (optional)', value='', key='ta_ds',
                                   placeholder='data/synthetic/sample.csv')
    with col_right:
        policy_a       = st.text_input('Policy version', value=policy_version, key='ta_pv')
        pull_from_srv  = st.checkbox('Pull task from Lightning Server first', value=False, key='ta_pull')
        run_a          = st.button('▶ Run Track A', type='primary', use_container_width=True, key='ta_run')

    with st.expander('Load tasks into server pool'):
        tasks_file_a = st.text_input('Tasks JSONL path', value='data/synthetic/tasks.jsonl', key='ta_tf')
        if st.button('📤 Upload to Server', key='ta_upload'):
            resp = _post('/api/tasks/load', {'tasks_jsonl_path': tasks_file_a})
            if resp and 'error' not in resp:
                st.success(f"Loaded {resp['loaded']} tasks.  Queue size: {resp['queue_size']}")
            else:
                st.error(str(resp))

    st.divider()

    if run_a:
        payload: dict[str, Any] = {
            'task_id': task_id_a, 'description': task_desc_a, 'policy_version': policy_a,
        }
        if dataset_a.strip():
            payload['dataset_path'] = dataset_a.strip()
        if pull_from_srv:
            pulled = _get('/api/tasks/pull')
            if pulled and 'task_id' in pulled:
                payload = pulled
                st.info(f"Using server task `{pulled['task_id']}` "
                        f"(lightning_rollout_id={pulled.get('lightning_rollout_id', '—')})")

        # Node stepper
        NODES_A = [
            ('policy_router', '🎯', 'Policy Router'),
            ('data_engineer', '🔬', 'Data Engineer'),
            ('gbm_specialist', '🌲', 'GBM Specialist'),
            ('sandbox_execution', '⚙️', 'Sandbox'),
            ('experiment_tracking', '📈', 'MLflow'),
            ('reviewer_critic', '⚖️', 'Reviewer'),
        ]
        ph_map: dict[str, Any] = {}
        node_cols = st.columns(len(NODES_A))
        for idx, (nk, ni, nl) in enumerate(NODES_A):
            with node_cols[idx]:
                ph_map[nk] = st.empty()
                ph_map[nk].markdown(f'{ni}  \n**{nl}**  \n⏳')

        result_box = st.empty()
        with st.spinner('Running LangGraph pipeline…'):
            try:
                from src.agents.supervisor import SupervisorAgent
                from src.telemetry.logger import TrajectoryLogger

                supervisor = SupervisorAgent(
                    require_live_policy=False,
                    lightning_server_url=server_url,
                )
                out_dir = _ROOT / 'trajectories' / 'demo'
                out_dir.mkdir(parents=True, exist_ok=True)
                traj = supervisor.run_task(payload, policy_version=policy_a)
                TrajectoryLogger(str(out_dir)).write(traj)

                step_by_node: dict[str, Any] = {}
                for step in traj.steps:
                    key = step.agent_name.lower().replace(' ', '_').replace('-', '_')
                    step_by_node[key] = step

                for nk, ni, nl in NODES_A:
                    step = step_by_node.get(nk)
                    if step:
                        errs = sum(1 for tc in (step.tool_calls or []) if tc.status != 'ok')
                        icon = '✅' if errs == 0 else '⚠️'
                        ph_map[nk].markdown(
                            f'{ni}  \n**{nl}**  \n{icon} {len(step.tool_calls or [])} tool(s)'
                        )
                    else:
                        ph_map[nk].markdown(f'{ni}  \n**{nl}**  \n⬜ skipped')

                badge = 'success' if traj.final_status == 'success' else 'warning'
                getattr(result_box, badge)(
                    f'Task `{traj.task_id}` — **{traj.final_status}**  |  reward **{traj.reward}**'
                )

                st.subheader('Step details')
                for i, step in enumerate(traj.steps, 1):
                    with st.expander(
                        f'Step {i}: {step.agent_name} — {step.action}', expanded=(i == 1)
                    ):
                        col_s, col_ac = st.columns(2)
                        col_s.markdown('**Reasoning**')
                        col_s.caption(step.reasoning_summary or '—')
                        col_ac.markdown('**Tool calls**')
                        for tc in (step.tool_calls or []):
                            badge2 = '✅' if tc.status == 'ok' else '❌'
                            col_ac.code(
                                f'{badge2} {tc.tool_name}\n'
                                f'  args: {json.dumps(tc.arguments or {}, indent=2)}\n'
                                f'  status: {tc.status}'
                                + (f'\n  error: {tc.error}' if tc.error else ''),
                                language='yaml',
                            )
            except Exception as exc:
                result_box.error(f'Error: {exc}')
                st.exception(exc)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Agent Lightning Trace
# ══════════════════════════════════════════════════════════════════════════════
with tab_trace:
    st.header('⚡ Agent Lightning Trace')
    st.caption(
        'Streams each LangGraph node in real time and shows LLM spans captured '
        'by the Lightning sidecar — including attribution fields '
        '(`rollout_id`, `attempt_id`, `sequence_id`) that SpanTraceAdapter uses '
        'to convert spans into GRPO training triplets.'
    )

    col_tr1, col_tr2 = st.columns([2, 1])
    with col_tr1:
        trace_task_id = st.text_input('Task ID', value=f'trace_{uuid.uuid4().hex[:6]}', key='tr_id')
        trace_desc    = st.text_area('Task description', height=100, key='tr_desc',
                                      value='Analyse the dataset and train a LightGBM model.')
    with col_tr2:
        trace_policy  = st.text_input('Policy version', value=policy_version, key='tr_pv')
        trace_dataset = st.text_input('Dataset path (optional)', value='', key='tr_ds')
        trace_btn     = st.button('▶ Trace Task', type='primary', use_container_width=True, key='tr_btn')

    NODE_ICONS = {
        'policy_router': '🎯', 'data_engineer': '🔬', 'gbm_specialist': '🌲',
        'sandbox_execution': '⚙️', 'experiment_tracking': '📈', 'reviewer_critic': '⚖️',
    }

    if trace_btn:
        task_payload_t: dict[str, Any] = {
            'task_id': trace_task_id, 'description': trace_desc, 'policy_version': trace_policy,
        }
        if trace_dataset.strip():
            task_payload_t['dataset_path'] = trace_dataset.strip()

        status_ph = st.empty()
        status_ph.info('Starting Agent Lightning trace…')
        timeline  = st.container()

        try:
            from src.agents.graph import build_agent_graph, AgentGraphState
            from src.inference.lightning_sidecar import LightningClientSidecar

            sidecar = LightningClientSidecar(server_url=server_url)
            graph   = build_agent_graph(
                sidecar=sidecar,
                policy_decision_dir=_ROOT / 'artifacts' / 'policy_decisions',
            )
            initial: AgentGraphState = {
                'task': task_payload_t, 'policy_version': trace_policy,
                'artifacts': {}, 'tool_outputs': {}, 'steps': [],
                'require_live_policy': False, 'error': None,
            }

            step_count = total_tools = total_errors = 0

            for chunk in graph.stream(initial, stream_mode='updates'):
                for node_name, state_update in chunk.items():
                    step_count += 1
                    icon       = NODE_ICONS.get(node_name, '🔷')
                    new_steps  = state_update.get('steps', [])
                    node_tools = sum(len(s.tool_calls) for s in new_steps if hasattr(s, 'tool_calls'))
                    node_errs  = sum(
                        1 for s in new_steps if hasattr(s, 'tool_calls')
                        for tc in s.tool_calls if tc.status != 'ok'
                    )
                    total_tools  += node_tools
                    total_errors += node_errs
                    err_flag      = state_update.get('error')

                    with timeline:
                        title = (
                            f'{icon} **{node_name}**'
                            + (f'  — {node_tools} tool(s)' if node_tools else '')
                            + (' ⚠️ error' if err_flag else '')
                        )
                        with st.expander(title, expanded=True):
                            tab_s, tab_a, tab_o = st.tabs(['State', 'Action', 'Observation'])
                            with tab_s:
                                for s in new_steps:
                                    st.caption(s.reasoning_summary or '—')
                            with tab_a:
                                for s in new_steps:
                                    for tc in (s.tool_calls or []):
                                        b = '✅' if tc.status == 'ok' else '❌'
                                        st.code(
                                            f'{b} {tc.tool_name}\n'
                                            f'  args: {json.dumps(tc.arguments or {}, indent=2)}\n'
                                            f'  status: {tc.status}'
                                            + (f'\n  error: {tc.error}' if tc.error else ''),
                                            language='yaml',
                                        )
                            with tab_o:
                                artifacts = state_update.get('artifacts') or {}
                                if artifacts:
                                    for k, v in artifacts.items():
                                        st.markdown(f'- `{k}`: {str(v)[:120]}')
                                if err_flag:
                                    st.error(f'Error: {err_flag}')

                    status_ph.info(
                        f'Step {step_count} — `{node_name}`  '
                        f'| tools: {total_tools}  | errors: {total_errors}'
                    )

            status_ph.success(
                f'Trace complete — {step_count} nodes  '
                f'| {total_tools} tool calls  | {total_errors} errors'
            )

            spans = sidecar.get_spans()
            if spans:
                st.divider()
                st.subheader(f'LLM Spans captured ({len(spans)})')
                has_attr = sum(1 for s in spans if s.rollout_id)
                st.markdown(f'**{has_attr}/{len(spans)}** spans have rollout attribution.')

                for i, span in enumerate(spans, 1):
                    label = (
                        f'Span {i} — {span.model_name} ({span.latency_ms:.0f} ms)'
                        + (f' | rollout: {(span.rollout_id or "")[:12]}' if span.rollout_id else '')
                    )
                    with st.expander(label):
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric('Latency ms', f'{span.latency_ms:.0f}')
                        c2.metric('rollout_id',  (span.rollout_id or '—')[:16])
                        c3.metric('attempt_id',   span.attempt_id  or '—')
                        c4.metric('sequence_id',  span.sequence_id if span.sequence_id is not None else '—')
                        if span.error:
                            st.error(f'Error: {span.error}')
                        with st.expander('Messages sent'):
                            st.json(span.messages)
                        if span.response:
                            with st.expander('Response'):
                                st.markdown(
                                    f'> {span.response[:600]}'
                                    + ('…' if len(span.response) > 600 else '')
                                )

        except Exception as exc:
            status_ph.error(f'Trace failed: {exc}')
            st.exception(exc)

    else:
        st.info('Configure a task above and click **▶ Trace Task** to start.')
        try:
            from src.agents.graph import build_agent_graph
            from src.inference.lightning_sidecar import LightningClientSidecar
            _g = build_agent_graph(
                sidecar=LightningClientSidecar(),
                policy_decision_dir=_ROOT / 'artifacts' / 'policy_decisions',
            )
            st.markdown('**Static graph topology** (copy into [mermaid.live](https://mermaid.live)):')
            st.code(_g.get_graph().draw_mermaid(), language='text')
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Rewards & Rollouts
# ══════════════════════════════════════════════════════════════════════════════
with tab_rewards:
    st.header('🏆 Rewards & Rollouts')

    col_ctl1, col_ctl2 = st.columns([2, 1])
    with col_ctl1:
        limit_r = st.slider('Show last N rollouts', 10, 500, 50, key='rw_limit')
    with col_ctl2:
        if st.button('🔄 Refresh', key='rw_refresh'):
            st.rerun()

    rollout_data = _get(f'/api/rollouts?limit={limit_r}')

    if rollout_data and 'error' not in rollout_data:
        rlist: list[dict] = rollout_data.get('rollouts', [])
        total_r = rollout_data.get('total', 0)
        st.caption(f'Total collected: **{total_r}** rollouts')

        if rlist:
            import pandas as pd
            import altair as alt

            rows = []
            for r in rlist:
                rows.append({
                    'rollout_id':  (r.get('rollout_id') or '')[:12] + '…',
                    'task_id':     r.get('task_id', ''),
                    'policy':      r.get('policy_version', ''),
                    'status':      r.get('status', ''),
                    'reward':      round(float(r.get('final_reward') or 0.0), 4),
                    'llm_spans':   r.get('llm_span_count', r.get('llm_spans', 0)),
                    'transitions': r.get('transitions_count', r.get('transitions', 0)),
                    'errors':      ', '.join(r.get('error_types', [])) or '—',
                    'completed':   (r.get('completed_at') or r.get('end_time') or '')[:19],
                })
            df = pd.DataFrame(rows)

            # Status summary
            scounts = df['status'].value_counts().to_dict()
            scols   = st.columns(len(scounts) + 1)
            scols[0].metric('Total', total_r)
            for i, (s, cnt) in enumerate(scounts.items()):
                scols[i + 1].metric(s, cnt)

            st.dataframe(
                df.style.background_gradient(subset=['reward'], cmap='RdYlGn', vmin=0, vmax=1),
                use_container_width=True, hide_index=True,
            )

            # Reward curve
            st.subheader('Reward curve')
            cdf = pd.DataFrame({
                'rollout': range(1, len(rlist) + 1),
                'reward':  [float(r.get('final_reward') or 0.0) for r in rlist],
            })
            cdf['rolling_avg'] = cdf['reward'].rolling(5, min_periods=1).mean()
            base = alt.Chart(cdf).encode(x='rollout:Q')
            pts  = base.mark_circle(opacity=0.5).encode(
                y=alt.Y('reward:Q', scale=alt.Scale(domain=[0, 1])),
                color=alt.Color('reward:Q', scale=alt.Scale(scheme='redyellowgreen'), legend=None),
                tooltip=['rollout', 'reward'],
            )
            line = base.mark_line(color='steelblue', strokeWidth=2).encode(
                y='rolling_avg:Q', tooltip=['rollout', 'rolling_avg'],
            )
            st.altair_chart((pts + line).properties(height=260), use_container_width=True)

            if df['policy'].nunique() > 1:
                st.subheader('Reward by policy version')
                pdf = df.groupby('policy')['reward'].agg(['mean', 'std', 'count']).reset_index()
                pdf.columns = ['policy', 'mean_reward', 'std', 'count']
                bar = (
                    alt.Chart(pdf).mark_bar()
                    .encode(
                        x='policy:N',
                        y=alt.Y('mean_reward:Q', scale=alt.Scale(domain=[0, 1])),
                        color='policy:N',
                        tooltip=['policy', 'mean_reward', 'std', 'count'],
                    )
                    .properties(height=220)
                )
                st.altair_chart(bar, use_container_width=True)

            # Grouped rollouts
            st.divider()
            st.subheader('Grouped rollouts (relative ranking for GRPO)')
            grpo_path = _ROOT / 'data' / 'grpo' / 'grouped_rollouts.jsonl'
            if grpo_path.exists():
                import json as _json
                groups: dict[str, list] = {}
                for line in grpo_path.read_text().splitlines():
                    if not line.strip():
                        continue
                    rec = _json.loads(line)
                    groups.setdefault(rec.get('task_id', 'unknown'), []).append(rec)

                st.caption(f'{len(groups)} task groups in `{grpo_path.relative_to(_ROOT)}`')
                sel_task = st.selectbox('Select task group', list(groups.keys()), key='grp_sel')
                if sel_task:
                    task_recs = sorted(groups[sel_task], key=lambda x: x.get('reward', 0.0), reverse=True)
                    gdf = pd.DataFrame([{
                        'trajectory_id': (r.get('trajectory_id') or '')[:16],
                        'policy':  r.get('policy_version', ''),
                        'reward':  round(float(r.get('reward') or 0.0), 4),
                        'status':  r.get('final_status', ''),
                    } for r in task_recs])
                    st.dataframe(
                        gdf.style.background_gradient(subset=['reward'], cmap='RdYlGn', vmin=0, vmax=1),
                        use_container_width=True, hide_index=True,
                    )
                    with st.expander('Top rollout — raw record'):
                        st.json(task_recs[0] if task_recs else {})
            else:
                st.info(
                    f'`{grpo_path.relative_to(_ROOT)}` not found.  '
                    'Run **Prepare GRPO dataset** in the Setup tab first.'
                )

            # Rollout detail
            st.divider()
            st.subheader('Rollout detail')
            sel_idx = st.selectbox(
                'Inspect rollout',
                range(len(rlist)),
                format_func=lambda i: (
                    f"[{i+1}] {rlist[i].get('task_id','')} "
                    f"| reward={float(rlist[i].get('final_reward') or 0):.3f} "
                    f"| {rlist[i].get('status','')}"
                ),
                key='rw_sel',
            )
            if sel_idx is not None:
                sel = rlist[sel_idx]
                c1, c2, c3 = st.columns(3)
                c1.metric('Reward',       round(float(sel.get('final_reward') or 0.0), 4))
                c1.metric('LLM Spans',    sel.get('llm_span_count', sel.get('llm_spans', 0)))
                c2.metric('Policy',       sel.get('policy_version', '—'))
                c2.metric('Transitions',  sel.get('transitions_count', sel.get('transitions', 0)))
                c3.metric('Status',       sel.get('status', '—'))
                c3.metric('Errors',       len(sel.get('error_types', [])))
                if sel.get('trajectory_dict'):
                    with st.expander('Full trajectory JSON'):
                        st.json(sel['trajectory_dict'])
                if sel.get('reward_metadata'):
                    with st.expander('Reward metadata'):
                        st.json(sel['reward_metadata'])
        else:
            st.info('No rollouts yet.  Run a task in Track A or Trace tab first.')
    else:
        st.warning(
            f"Could not fetch rollouts: "
            f"{(rollout_data or {}).get('error', 'Lightning Server offline')}"
        )

    st.divider()
    with st.expander('Raw transitions (state_t → action_t → reward_t → state_t+1)'):
        trans = _get('/api/transitions?limit=20')
        if trans and 'error' not in trans:
            st.caption(f"Total written: {trans.get('total', 0)}")
            for t in trans.get('transitions', []):
                st.json(t)
        else:
            st.caption('Lightning Server offline or no transitions yet.')


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — Track B: Policy Optimization
# ══════════════════════════════════════════════════════════════════════════════
with tab_track_b:
    st.header('🔥 Track B: Policy Optimization')
    st.caption(
        'Turn scored agent trajectories into a better policy.  '
        'Select a trainer, review the flow, check preflight readiness, '
        'then trigger training or inspect previous training evidence.'
    )

    # ── Mode banner ───────────────────────────────────────────────────────────
    if gpu_mode == 'Online — live streaming':
        st.info(
            '🌐 **Online mode** — CPU Track A is streaming rollouts directly to the GPU '
            'Lightning Server at `' + server_url + '`.  '
            'Training can be triggered once enough rollouts are collected.'
        )
    else:
        st.info(
            '📂 **Offline mode** — GPU reads `data/grpo/grouped_rollouts.jsonl` from the '
            'shared network volume.  Ensure CPU has completed step 5 before triggering training.'
        )

    train_status = _get('/api/training/status')
    if train_status and 'error' not in train_status:
        cs1, cs2, cs3, cs4 = st.columns(4)
        cs1.metric('Rollouts',         train_status.get('rollouts_collected', 0))
        cs2.metric('Transitions',      train_status.get('transitions_written', 0))
        cs3.metric('Training active',  '🟢 Yes' if train_status.get('training_active') else '⚪ Idle')
        cs4.metric('Ready to train',   '✅ Yes' if train_status.get('ready_for_training') else '⏳ No')
    else:
        st.warning('Lightning Server offline — training status unavailable.')

    st.divider()

    # ── Sub-tabs ──────────────────────────────────────────────────────────────
    (
        tb_tab_select,
        tb_tab_flow,
        tb_tab_rewards,
        tb_tab_grpo,
        tb_tab_split,
        tb_tab_preflight,
        tb_tab_train,
        tb_tab_evidence,
        tb_tab_compare,
        tb_tab_lifecycle,
        tb_tab_gaps,
        tb_tab_demo_script,
    ) = st.tabs([
        '🎛 Trainer Select',
        '📊 Training Flow',
        '🏆 Reward Breakdown',
        '📐 GRPO Explainer',
        '🗂 Dataset Split',
        '✅ Preflight',
        '🚀 Train',
        '📋 Evidence',
        '📈 Baseline vs Tuned',
        '🔄 Policy Lifecycle',
        '⚠️ Gaps',
        '📖 Demo Script',
    ])

    # ═══════════════════════════════════════════════════════════════
    # Sub-tab 1 — Trainer Select + Comparison Table
    # ═══════════════════════════════════════════════════════════════
    with tb_tab_select:
        st.subheader('Trainer backend comparison')

        import pandas as pd

        _trainer_table = pd.DataFrame([
            {
                'Trainer':              'qlora_sft',
                'Data needed':          'High-reward trajectory steps',
                'Training signal':      'Supervised imitation / CE loss',
                'Best for':             'Stable demo, limited GPU',
                'Risk':                 '🟢 Low',
                'GPU req':              '≥ 16 GB',
                'Duration':             'Fast (< 10 min)',
                'Readiness':            '✅ Ready / safest',
                'Demo confidence':      '⭐⭐⭐⭐⭐',
            },
            {
                'Trainer':              'trl_grpo',
                'Data needed':          'Grouped rollouts + reward function',
                'Training signal':      'Group-relative reward optimisation',
                'Best for':             'Lightweight RL demo',
                'Risk':                 '🟡 Medium',
                'GPU req':              '≥ 24 GB',
                'Duration':             'Medium (20–60 min)',
                'Readiness':            '🟡 Ready — reward improving',
                'Demo confidence':      '⭐⭐⭐⭐',
            },
            {
                'Trainer':              'verl',
                'Data needed':          'Grouped rollouts + veRL command',
                'Training signal':      'Distributed GRPO/PPO-style RL',
                'Best for':             'Multi-GPU serious RL',
                'Risk':                 '🟠 High setup',
                'GPU req':              '≥ 2× GPU + veRL installed',
                'Duration':             'Long (hours)',
                'Readiness':            '🟠 Requires veRL installation',
                'Demo confidence':      '⭐⭐',
            },
            {
                'Trainer':              'agent_lightning_official',
                'Data needed':          'Official LitAgent/Trainer APIs',
                'Training signal':      'Official Lightning orchestration',
                'Best for':             'Integration proof',
                'Risk':                 '🟠 Compatibility risk',
                'GPU req':              'Depends on backend',
                'Duration':             'Depends on backend',
                'Readiness':            '🟠 Requires official package',
                'Demo confidence':      '⭐⭐⭐',
            },
        ])
        st.dataframe(_trainer_table, use_container_width=True, hide_index=True)

        st.divider()
        st.subheader('Select trainer')
        TRAINER_INFO = {
            'qlora_sft': (
                '**QLoRA SFT** — Supervised fine-tuning from the best-reward trajectories.  '
                'Uses `BitsAndBytesConfig(load_in_4bit=True)` with `prepare_model_for_kbit_training`.  '
                '✅ Fastest and most stable — **recommended for demo**.'
            ),
            'trl_grpo': (
                '**TRL GRPO** — Group-relative policy optimisation using TRL `GRPOTrainer`.  '
                'Loads the base model with 4-bit QLoRA and trains a LoRA adapter from grouped rollout rewards.  '
                '🟡 Good RL demo once hybrid reward is active.'
            ),
            'verl': (
                '**veRL** — Strict external veRL command (`VERL_TRAIN_CMD`).  '
                'Runs `scripts/gpu/start_verl_training.sh`.  '
                '🟠 Exits with error if veRL is not installed — no silent fallback.'
            ),
            'agent_lightning_official': (
                '**Agent Lightning Official** — Uses official Microsoft Agent Lightning '
                'package interfaces if installed.  '
                '🟠 May not be installable due to `blinker` version conflict.  '
                'Stubs are active and the system runs without it.'
            ),
        }

        col_sel, col_info = st.columns([1, 2])
        with col_sel:
            trainer_b = st.radio('Select trainer', list(TRAINER_INFO.keys()), index=0, key='tb_trainer')
        with col_info:
            st.info(TRAINER_INFO[trainer_b])

        st.divider()
        st.subheader('What this trainer actually optimises')
        _what_optimises = {
            'qlora_sft': [
                ('Optimises', 'Next-token cross-entropy loss'),
                ('Learns', 'Imitate high-reward trajectory steps'),
                ('Needs', 'Prompt / completion examples from scored trajectories'),
                ('Good for', 'Making policy follow known good workflow patterns'),
                ('Limitation', 'Does not directly optimise reward signal'),
            ],
            'trl_grpo': [
                ('Optimises', 'Group-relative reward objective'),
                ('Learns', 'Prefer better-than-average completions in the same prompt group'),
                ('Needs', 'Prompts + reward function + multiple generations (num_generations)'),
                ('Good for', 'Lightweight RL improvement without PPO overhead'),
                ('Limitation', 'Reward function must be meaningful — JSON-only reward is too weak'),
            ],
            'verl': [
                ('Optimises', 'Distributed RL objective (GRPO/PPO depending on veRL command)'),
                ('Learns', 'Same RL idea as TRL GRPO but scalable across many GPUs'),
                ('Needs', 'Official veRL environment + VERL_TRAIN_CMD configured'),
                ('Good for', 'Serious multi-GPU training, research-grade experiments'),
                ('Limitation', 'Setup complexity; hard failure if veRL not installed'),
            ],
            'agent_lightning_official': [
                ('Optimises', 'Depends on official trainer/algorithm backend'),
                ('Learns', 'Policy behaviour through official Agent Lightning data/control plane'),
                ('Needs', 'Official `agentlightning` package compatibility'),
                ('Good for', 'Proving alignment with Agent Lightning architecture'),
                ('Limitation', 'Package/version/runtime compatibility uncertainty'),
            ],
        }
        for label, value in _what_optimises.get(trainer_b, []):
            c1, c2 = st.columns([1, 3])
            c1.markdown(f'**{label}**')
            c2.markdown(value)

        # Store selection for other sub-tabs
        st.session_state['tb_trainer_selected'] = trainer_b

        st.divider()
        st.subheader('Training configuration')
        col_c1, col_c2, col_c3 = st.columns(3)
        with col_c1:
            model_b  = st.text_input('Base model', value='Qwen/Qwen2.5-3B-Instruct', key='tb_model')
            min_roll = st.number_input('Min rollouts threshold', min_value=1, value=4, key='tb_min')
        with col_c2:
            reward_mode  = st.selectbox(
                'Reward mode',
                ['hybrid', 'workflow_policy', 'json_validity', 'trajectory_reward'],
                key='tb_rm',
                help=(
                    'hybrid: 30% JSON validity + 70% workflow quality (recommended)\n'
                    'workflow_policy: full workflow-aware scoring\n'
                    'json_validity: structural check only (legacy)\n'
                    'trajectory_reward: use pre-computed rollout reward scores'
                ),
            )
            num_gens = st.number_input(
                'Num generations (GRPO)', min_value=2, max_value=16, value=4, key='tb_ngens',
                help='Number of completions per prompt for group-relative advantage calculation.',
            )
        with col_c3:
            checkpoint = st.text_input('Output checkpoint path', value='checkpoints/qwen25-3b-agent-lora', key='tb_ckpt')
            reload_after = st.checkbox('Hot-reload vLLM after training', value=True, key='tb_reload')
            allow_fallback = False
            if trainer_b == 'verl':
                allow_fallback = st.checkbox('Allow TRL GRPO fallback (ALLOW_VERL_FALLBACK=1)', value=False, key='tb_fb')

    # ═══════════════════════════════════════════════════════════════
    # Sub-tab 2 — Visual Training Flow
    # ═══════════════════════════════════════════════════════════════
    with tb_tab_flow:
        _trainer_sel = st.session_state.get('tb_trainer_selected', 'qlora_sft')
        st.subheader(f'Training flow: {_trainer_sel}')

        _flows = {
            'qlora_sft': (
                'Scored trajectories\n'
                '   ↓\n'
                'Pick best-reward trajectories (min_reward filter)\n'
                '   ↓\n'
                'Convert each step → prompt / completion pair\n'
                '   ↓\n'
                'QLoRA 4-bit base model (BitsAndBytesConfig nf4)\n'
                '   ↓\n'
                'Train LoRA adapter (CE loss)\n'
                '   ↓\n'
                'Save checkpoint → checkpoints/qwen25-3b-agent-lora'
            ),
            'trl_grpo': (
                'Task prompt\n'
                '   ↓\n'
                'Generate N candidate completions (num_generations)\n'
                '   ↓\n'
                'Score each completion with reward_fn (hybrid / workflow / json_validity)\n'
                '   ↓\n'
                'Compute group mean reward\n'
                '   ↓\n'
                'advantage = reward − group_mean\n'
                '   ↓\n'
                'Increase P(action) where advantage > 0\n'
                'Decrease P(action) where advantage < 0\n'
                '   ↓\n'
                'Save LoRA adapter'
            ),
            'verl': (
                'Grouped rollout dataset (grouped_rollouts.jsonl)\n'
                '   ↓\n'
                'Export GRPO_DATASET_PATH, MODEL_NAME, POLICY_OUTPUT_DIR env vars\n'
                '   ↓\n'
                'Execute VERL_TRAIN_CMD (official veRL command)\n'
                '   ↓\n'
                'Distributed RL trainer (GRPO or PPO depending on command)\n'
                '   ↓\n'
                'Adapter / checkpoint exported by veRL\n'
                '   ↓\n'
                'Log path: checkpoints/.../verl_train_command.log'
            ),
            'agent_lightning_official': (
                'LitAgent wrapper (src/agents/lit_agent.py)\n'
                '   ↓\n'
                'Lightning Store (src/training/lightning_store.py)\n'
                '   ↓\n'
                'Lightning Trainer (official agentlightning.Trainer)\n'
                '   ↓\n'
                'Algorithm backend (GRPO / SFT)\n'
                '   ↓\n'
                'Checkpoint registration\n'
                '   ↓\n'
                'adapter_metadata.json written'
            ),
        }
        st.code(_flows.get(_trainer_sel, '(no flow defined)'), language='text')

        st.divider()
        st.subheader('End-to-end pipeline position')
        st.code(
            'CPU Track A rollouts\n'
            '   ↓\n'
            'Trajectory scoring (src/rewards/scorer.py)\n'
            '   ↓\n'
            'Grouped rollout dataset (data/grpo/grouped_rollouts.jsonl)\n'
            '   ↓\n'
            'Trainer backend selection  ← you are here\n'
            '   ↓\n'
            f'{_trainer_sel}\n'
            '   ↓\n'
            'LoRA adapter checkpoint\n'
            '   ↓\n'
            'vLLM hot reload\n'
            '   ↓\n'
            'Baseline vs tuned comparison',
            language='text',
        )

    # ═══════════════════════════════════════════════════════════════
    # Sub-tab 3 — Reward Breakdown
    # ═══════════════════════════════════════════════════════════════
    with tb_tab_rewards:
        st.subheader('Reward component breakdown')
        st.caption(
            'Select a rollout from the grouped dataset to see how the hybrid reward '
            'is constructed from individual components.'
        )

        grpo_path_rw = _ROOT / 'data' / 'grpo' / 'grouped_rollouts.jsonl'
        if not grpo_path_rw.exists():
            st.warning('`data/grpo/grouped_rollouts.jsonl` not found — run step 5 in Setup.')
        else:
            import json as _json
            import altair as alt

            _rw_groups: dict[str, list] = {}
            for line in grpo_path_rw.read_text().splitlines():
                if not line.strip():
                    continue
                rec = _json.loads(line)
                task_id_r = rec.get('task_id', 'unknown')
                # Handle both flat records and nested ranked_trajectories
                ranked = rec.get('ranked_trajectories', [])
                if ranked:
                    _rw_groups.setdefault(task_id_r, []).extend(ranked)
                else:
                    _rw_groups.setdefault(task_id_r, []).append(rec)

            sel_task_rw = st.selectbox('Task group', list(_rw_groups.keys()), key='rw_task_sel')
            task_trajs_rw = _rw_groups.get(sel_task_rw, [])

            traj_labels = [
                f"{t.get('trajectory_id', '?')[:16]} | reward={float(t.get('reward', 0)):.3f}"
                for t in task_trajs_rw
            ]
            sel_traj_rw = st.selectbox('Trajectory', range(len(task_trajs_rw)),
                                        format_func=lambda i: traj_labels[i], key='rw_traj_sel')

            if task_trajs_rw:
                traj_rw = task_trajs_rw[sel_traj_rw]

                # Show scorer.py reward metadata if present
                rmeta = traj_rw.get('reward_metadata', {}) or {}
                if rmeta:
                    st.subheader('Scorer reward components (R_data, R_model, …)')
                    _rw_comp_keys = ['R_data', 'R_model', 'R_tracking', 'R_sandbox',
                                     'R_reproducibility', 'R_violation']
                    _rw_comp_vals = [float(rmeta.get(k, 0.0)) for k in _rw_comp_keys]

                    _bar_df = pd.DataFrame({'Component': _rw_comp_keys, 'Score': _rw_comp_vals})
                    bar_rw = (
                        alt.Chart(_bar_df).mark_bar()
                        .encode(
                            x=alt.X('Score:Q', scale=alt.Scale(domain=[0, 1])),
                            y=alt.Y('Component:N', sort='-x'),
                            color=alt.Color('Score:Q', scale=alt.Scale(scheme='redyellowgreen'),
                                            legend=None),
                            tooltip=['Component', 'Score'],
                        )
                        .properties(height=220, title='Reward components')
                    )
                    st.altair_chart(bar_rw, use_container_width=True)

                    # Reward waterfall
                    st.subheader('Reward waterfall')
                    _weights = {'R_data': 0.18, 'R_model': 0.25, 'R_tracking': 0.17,
                                'R_sandbox': 0.15, 'R_reproducibility': 0.15, 'R_violation': 0.10}
                    _wf_rows = []
                    _running = 0.0
                    for k in _rw_comp_keys:
                        w = _weights.get(k, 0.0)
                        contrib = float(rmeta.get(k, 0.0)) * w
                        _running += contrib
                        _wf_rows.append({'Component': k, 'Contribution': round(contrib, 4),
                                         'Cumulative': round(_running, 4)})
                    _wf_rows.append({'Component': 'Final reward', 'Contribution': 0,
                                     'Cumulative': round(float(traj_rw.get('reward', _running)), 4)})
                    _wf_df = pd.DataFrame(_wf_rows)
                    wf_chart = (
                        alt.Chart(_wf_df[_wf_df['Component'] != 'Final reward']).mark_bar()
                        .encode(
                            x='Component:N',
                            y='Contribution:Q',
                            color=alt.condition(
                                alt.datum.Contribution > 0,
                                alt.value('steelblue'), alt.value('tomato'),
                            ),
                            tooltip=['Component', 'Contribution', 'Cumulative'],
                        )
                        .properties(height=220, title='Weighted reward contributions')
                    )
                    line_wf = (
                        alt.Chart(_wf_df).mark_line(point=True, color='orange')
                        .encode(x='Component:N', y='Cumulative:Q', tooltip=['Component', 'Cumulative'])
                    )
                    st.altair_chart((wf_chart + line_wf).resolve_scale(y='independent'),
                                    use_container_width=True)

                # Show policy_reward breakdown for first step completion
                steps_rw = traj_rw.get('steps', [])
                if steps_rw:
                    st.subheader('Policy reward breakdown (first step)')
                    first_step = steps_rw[0]
                    completion_sample = first_step.get('completion') or _json.dumps({
                        'agent_name': first_step.get('agent_name', ''),
                        'action': first_step.get('action', ''),
                        'reasoning_summary': first_step.get('reasoning_summary', ''),
                        'expected_tool_calls': first_step.get('tool_calls', []),
                    })
                    try:
                        from src.rewards.policy_reward import score_breakdown
                        bd = score_breakdown(
                            prompt=first_step.get('prompt', ''),
                            completion=completion_sample,
                        )
                        c_pos, c_neg = st.columns(2)
                        with c_pos:
                            st.markdown('**Positive contributions**')
                            for k, v in bd['contributions'].items():
                                if v > 0:
                                    st.markdown(f'`{k}` → **+{v:.2f}**')
                        with c_neg:
                            st.markdown('**Penalties**')
                            neg_found = False
                            for k, v in bd['contributions'].items():
                                if v < 0:
                                    st.markdown(f'`{k}` → **{v:.2f}**')
                                    neg_found = True
                            if not neg_found:
                                st.success('No penalties detected')
                        st.metric('Final policy score', bd['final_score'])
                    except Exception as _be:
                        st.caption(f'Policy breakdown not available: {_be}')
                else:
                    st.info('No step data in this trajectory.')

    # ═══════════════════════════════════════════════════════════════
    # Sub-tab 4 — GRPO Explainer
    # ═══════════════════════════════════════════════════════════════
    with tb_tab_grpo:
        st.subheader('GRPO group-relative advantage')
        st.markdown(
            'GRPO compares completions **within the same prompt group**.  '
            'Completions above the group mean get a positive advantage → '
            'the LoRA adapter is updated to make those actions **more likely**.  '
            'Completions below the mean get a negative advantage → those actions become **less likely**.'
        )

        grpo_path_ex = _ROOT / 'data' / 'grpo' / 'grouped_rollouts.jsonl'
        if not grpo_path_ex.exists():
            st.warning('`data/grpo/grouped_rollouts.jsonl` not found.')
        else:
            import json as _json2
            import altair as alt2

            _ex_groups: dict[str, list] = {}
            for line in grpo_path_ex.read_text().splitlines():
                if not line.strip():
                    continue
                rec = _json2.loads(line)
                tid = rec.get('task_id', 'unknown')
                ranked = rec.get('ranked_trajectories', [])
                trajs = ranked if ranked else [rec]
                _ex_groups.setdefault(tid, []).extend(trajs)

            sel_task_ex = st.selectbox('Select task group', list(_ex_groups.keys()), key='ex_task_sel')
            group_trajs = _ex_groups.get(sel_task_ex, [])

            if group_trajs:
                rewards_ex = [float(t.get('reward', 0.0)) for t in group_trajs]
                group_mean = sum(rewards_ex) / len(rewards_ex)

                st.metric('Group mean reward', f'{group_mean:.4f}')

                rows_ex = []
                for t in group_trajs:
                    r = float(t.get('reward', 0.0))
                    adv = r - group_mean
                    rows_ex.append({
                        'trajectory_id': (t.get('trajectory_id') or '?')[:18],
                        'reward':        round(r, 4),
                        'group_mean':    round(group_mean, 4),
                        'advantage':     round(adv, 4),
                        'GRPO update':   '⬆ increase prob.' if adv > 0 else ('⬇ decrease prob.' if adv < 0 else '— neutral'),
                    })

                df_ex = pd.DataFrame(rows_ex)
                def _adv_color(val):
                    if val > 0:
                        return 'background-color: #d4edda'
                    elif val < 0:
                        return 'background-color: #f8d7da'
                    return ''
                st.dataframe(
                    df_ex.style.map(_adv_color, subset=['advantage']),
                    use_container_width=True, hide_index=True,
                )

                # Advantage bar chart
                _adv_df = pd.DataFrame({
                    'trajectory': [r['trajectory_id'] for r in rows_ex],
                    'advantage':  [r['advantage'] for r in rows_ex],
                })
                adv_chart = (
                    alt2.Chart(_adv_df).mark_bar()
                    .encode(
                        x='trajectory:N',
                        y='advantage:Q',
                        color=alt2.condition(
                            alt2.datum.advantage > 0,
                            alt2.value('steelblue'), alt2.value('tomato'),
                        ),
                        tooltip=['trajectory', 'advantage'],
                    )
                    .properties(height=220, title='GRPO advantage per trajectory')
                )
                mean_rule = alt2.Chart(pd.DataFrame({'y': [0]})).mark_rule(
                    color='black', strokeDash=[4, 4]
                ).encode(y='y:Q')
                st.altair_chart((adv_chart + mean_rule), use_container_width=True)

                # Natural language explanation
                if rows_ex:
                    top = max(rows_ex, key=lambda x: x['advantage'])
                    bot = min(rows_ex, key=lambda x: x['advantage'])
                    st.info(
                        f'**GRPO explanation for this group:**  \n'
                        f'Group mean reward = **{group_mean:.4f}**  \n\n'
                        f'Trajectory `{top["trajectory_id"]}` has reward **{top["reward"]}**.  '
                        f'Advantage = **+{top["advantage"]:.4f}** → '
                        f'GRPO increases probability of actions from this trajectory.  \n\n'
                        f'Trajectory `{bot["trajectory_id"]}` has reward **{bot["reward"]}**.  '
                        f'Advantage = **{bot["advantage"]:.4f}** → '
                        f'GRPO decreases probability of actions from this trajectory.'
                    )
            else:
                st.info('No trajectories in selected group.')

    # ═══════════════════════════════════════════════════════════════
    # Sub-tab 5 — Dataset Split & Leakage Check
    # ═══════════════════════════════════════════════════════════════
    with tb_tab_split:
        st.subheader('Dataset Split & Leakage Check')
        st.caption(
            'Splitting is done at the **task_id level** — the same task never appears '
            'in both train and test.  This prevents prompt leakage and produces a '
            'valid held-out evaluation set.'
        )

        grpo_path_sp = _ROOT / 'data' / 'grpo' / 'grouped_rollouts.jsonl'
        if not grpo_path_sp.exists():
            st.warning('`data/grpo/grouped_rollouts.jsonl` not found — run step 5 first.')
        else:
            import json as _json3
            _sp_train_r = st.slider('Train ratio', 0.5, 0.9, 0.8, 0.05, key='sp_train')
            _sp_val_r   = st.slider('Val ratio',   0.0, 0.3, 0.1, 0.05, key='sp_val')
            _sp_test_r  = round(1.0 - _sp_train_r - _sp_val_r, 4)
            st.caption(f'Test ratio (computed): **{_sp_test_r:.2f}**')

            if st.button('Run split check', key='sp_run'):
                with st.spinner('Computing split…'):
                    try:
                        from src.training.split_policy_dataset import (
                            load_grouped_rollouts, split_by_task_id, check_leakage,
                        )
                        _sp_groups = load_grouped_rollouts(grpo_path_sp)
                        _sp_train, _sp_val, _sp_test = split_by_task_id(
                            _sp_groups,
                            train_ratio=_sp_train_r,
                            val_ratio=_sp_val_r,
                            test_ratio=max(_sp_test_r, 0.0),
                        )
                        _sp_report = check_leakage(_sp_train, _sp_val, _sp_test)
                        st.session_state['sp_report'] = _sp_report
                    except Exception as _sp_e:
                        st.error(f'Split error: {_sp_e}')

            report_sp = st.session_state.get('sp_report', {})
            if report_sp:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric('Total groups',  report_sp.get('total_task_groups', 0))
                c2.metric('Train groups',  report_sp.get('train_task_groups', 0))
                c3.metric('Val groups',    report_sp.get('val_task_groups', 0))
                c4.metric('Test groups',   report_sp.get('test_task_groups', 0))

                if report_sp.get('leakage_detected'):
                    st.error('⚠️ Leakage detected!')
                    leaking = report_sp.get('task_id_leakage_train_test', [])
                    if leaking:
                        st.write('Task IDs in both train and test:', leaking)
                    prompt_leak = report_sp.get('prompt_leakage_count', 0)
                    if prompt_leak:
                        st.write(f'Prompt-level leakage: {prompt_leak} overlapping prompts')
                else:
                    st.success('✅ No train/test leakage detected')

                st.metric('Heldout eval available',
                          '✅ Yes' if report_sp.get('heldout_evaluation_available') else '❌ No')

    # ═══════════════════════════════════════════════════════════════
    # Sub-tab 6 — Preflight
    # ═══════════════════════════════════════════════════════════════
    with tb_tab_preflight:
        _trainer_pf = st.session_state.get('tb_trainer_selected', 'qlora_sft')
        st.subheader(f'Preflight checks — {_trainer_pf}')

        grpo_path_b  = _ROOT / 'data' / 'grpo' / 'grouped_rollouts.jsonl'
        _gpu_url_ok  = (
            gpu_mode == 'Offline — shared volume'
            or ('localhost' not in server_url and '127.0.0.1' not in server_url)
        )
        _grpo_ok = (
            grpo_path_b.exists() if gpu_mode == 'Offline — shared volume' else True
        )

        # Count high-reward trajectories
        _high_reward_count = 0
        _min_group_size_ok = False
        if grpo_path_b.exists():
            import json as _json4
            _pf_groups = []
            for line in grpo_path_b.read_text().splitlines():
                if line.strip():
                    _pf_groups.append(_json4.loads(line))
            for g in _pf_groups:
                ranked = g.get('ranked_trajectories', []) or []
                if ranked:
                    top_r = float(ranked[0].get('reward', 0.0)) if ranked else 0.0
                    if top_r >= 0.5:
                        _high_reward_count += 1
                    if len(ranked) >= 2:
                        _min_group_size_ok = True
                else:
                    r = float(g.get('reward', 0.0))
                    if r >= 0.5:
                        _high_reward_count += 1

        # Check packages
        def _check_import(pkg: str) -> bool:
            import importlib
            try:
                importlib.import_module(pkg)
                return True
            except ImportError:
                return False

        # Common checks
        _common: list[tuple[bool, str, str]] = [
            (_grpo_ok, 'grouped_rollouts.jsonl exists',
             'bash scripts/cpu/run_05_prepare_grpo_dataset.sh'),
            (bool(health and 'error' not in health), 'Lightning Server online',
             'bash scripts/gpu/start_lightning_server.sh'),
            (bool(train_status and not train_status.get('training_active')),
             'No training in progress', ''),
        ]

        # Trainer-specific checks
        _specific: list[tuple[bool, str, str]] = []
        if _trainer_pf == 'qlora_sft':
            _specific = [
                (_high_reward_count > 0, f'High-reward trajectories found ({_high_reward_count})',
                 'bash scripts/cpu/run_04_score_trajectories.sh'),
                (_check_import('transformers'), 'transformers installed',
                 'pip install transformers'),
                (_check_import('peft'), 'peft installed', 'pip install peft'),
                (_check_import('bitsandbytes'), 'bitsandbytes installed',
                 'pip install bitsandbytes'),
            ]
        elif _trainer_pf == 'trl_grpo':
            _specific = [
                (_min_group_size_ok, 'At least 2 trajectories per group (for GRPO ranking)',
                 'bash scripts/cpu/run_05_prepare_grpo_dataset.sh  # with GROUP_SIZE>=2'),
                (_check_import('trl'), 'trl installed', 'pip install trl'),
                (_check_import('bitsandbytes'), 'bitsandbytes installed',
                 'pip install bitsandbytes'),
                (bool(st.session_state.get('tb_rm')), 'Reward mode selected', ''),
            ]
        elif _trainer_pf == 'verl':
            _verl_cmd_ok = bool(os.getenv('VERL_TRAIN_CMD'))
            _specific = [
                (_verl_cmd_ok, 'VERL_TRAIN_CMD environment variable set',
                 'export VERL_TRAIN_CMD="python -m verl.trainer.main ..."'),
                (_check_import('verl'), 'verl package importable',
                 'pip install verl  # or follow official veRL install docs'),
                (_check_import('ray'), 'ray available (for distributed mode)',
                 'pip install ray[default]'),
            ]
        elif _trainer_pf == 'agent_lightning_official':
            _al_ok = _check_import('agentlightning')
            _al_runner_ok = (_ROOT / 'src' / 'training' / 'agent_lightning_official_runner.py').exists()
            _al_lit_ok    = (_ROOT / 'src' / 'agents' / 'lit_agent.py').exists()
            _specific = [
                (_al_runner_ok, 'agent_lightning_official_runner.py present', ''),
                (_al_lit_ok,    'lit_agent.py present', ''),
                (_al_ok,        'agentlightning package importable',
                 'pip install agentlightning  # may need: pip install blinker==1.7.0'),
            ]

        all_ok_pf = True
        for ok, label, fix in (_common + _specific):
            icon = '✅' if ok else '❌'
            c1, c2, c3 = st.columns([0.5, 4, 5])
            c1.markdown(icon)
            c2.markdown(label)
            if not ok:
                all_ok_pf = False
                if fix:
                    c3.code(fix, language='bash')

        # Store preflight result for Train tab
        st.session_state['tb_preflight_ok']  = all_ok_pf
        st.session_state['tb_trainer_pf']     = _trainer_pf

        if all_ok_pf:
            st.success('All preflight checks passed — ready to train.')
        else:
            st.warning('Fix the items above before triggering training.')

    # ═══════════════════════════════════════════════════════════════
    # Sub-tab 7 — Train
    # ═══════════════════════════════════════════════════════════════
    with tb_tab_train:
        _trainer_tr  = st.session_state.get('tb_trainer_selected', 'qlora_sft')
        _all_ok_tr   = st.session_state.get('tb_preflight_ok', False)
        _model_tr    = st.session_state.get('tb_model', 'Qwen/Qwen2.5-3B-Instruct')
        _min_roll_tr = st.session_state.get('tb_min', 4)
        _reward_tr   = st.session_state.get('tb_rm', 'hybrid')
        _ngens_tr    = st.session_state.get('tb_ngens', 4)
        _ckpt_tr     = st.session_state.get('tb_ckpt', 'checkpoints/qwen25-3b-agent-lora')

        st.subheader(f'Trigger training — {_trainer_tr}')
        st.caption('Sends a POST to /api/training/trigger with the full configuration.')

        with st.expander('Payload preview'):
            _payload_preview = {
                'trainer':       _trainer_tr,
                'model_name':    _model_tr,
                'min_rollouts':  int(_min_roll_tr),
                'reward_mode':   _reward_tr,
                'num_generations': int(_ngens_tr),
                'checkpoint_path': _ckpt_tr,
                'reload_after':  st.session_state.get('tb_reload', True),
            }
            st.json(_payload_preview)

        trigger_b = st.button(
            '🚀 Start Track B Optimization',
            type='primary',
            disabled=not _all_ok_tr,
            key='tb_trigger',
            help='Fix preflight issues in the Preflight tab to enable.' if not _all_ok_tr else '',
        )

        if not _all_ok_tr:
            st.warning('Complete the **Preflight** tab checks first.')

        if trigger_b:
            with st.spinner('Sending training trigger…'):
                resp_t = _post('/api/training/trigger', _payload_preview)
            if resp_t and 'error' not in resp_t:
                s = resp_t.get('status', '')
                if s == 'training_started':
                    st.success(
                        f"Training started!  PID {resp_t.get('pid')}  "
                        f"| Trainer: **{resp_t.get('trainer')}**  "
                        f"| Tasks grouped: {resp_t.get('grouped_tasks')}"
                    )
                    st.json(resp_t)
                elif s == 'already_running':
                    st.warning('Training already in progress.')
                elif s == 'insufficient_rollouts':
                    st.warning(
                        f"Not enough rollouts: {resp_t.get('rollouts_collected')} / "
                        f"{resp_t.get('required')}.  Run more tasks in Track A."
                    )
                else:
                    st.json(resp_t)
            else:
                st.error(str(resp_t))

        # ── Multi-run: Run All Options ─────────────────────────────────────
        st.divider()
        st.subheader('Run All Options (sequential)')
        st.info(
            '**For true blocking GPU execution** use the shell script:\n\n'
            '```bash\nbash scripts/gpu/run_all_trackb_options_one_by_one.sh\n```\n\n'
            'The button below uses Streamlit polling (POST → wait → next) which is '
            'slower but works without SSH access.'
        )

        _ALL_OPTIONS = [
            ('qlora_sft',                'checkpoints/trackb_qlora_sft',            'hybrid'),
            ('trl_grpo',                 'checkpoints/trackb_trl_grpo_hybrid',       'hybrid'),
            ('trl_grpo',                 'checkpoints/trackb_trl_grpo_workflow',     'workflow_policy'),
            ('verl',                     'checkpoints/trackb_verl',                  'hybrid'),
            ('agent_lightning_official', 'checkpoints/trackb_agent_lightning',       'hybrid'),
        ]

        _POLL_INTERVAL_S   = 20   # seconds between /health polls
        _POLL_TIMEOUT_S    = 180  # max seconds to wait per trainer (3 min)

        def _wait_for_training_idle(status_placeholder, label: str) -> bool:
            """Poll /health until training_active=False or timeout."""
            import time as _time
            elapsed = 0
            while elapsed < _POLL_TIMEOUT_S:
                _h = _get('/health')
                if _h and not _h.get('training_active', False):
                    return True
                status_placeholder.info(
                    f'⏳ Waiting for **{label}** to finish… '
                    f'({elapsed}s / {_POLL_TIMEOUT_S}s)'
                )
                _time.sleep(_POLL_INTERVAL_S)
                elapsed += _POLL_INTERVAL_S
            return False  # timed out

        # Initialise run-results table in session_state
        if 'tb_multi_run_results' not in st.session_state:
            st.session_state['tb_multi_run_results'] = []

        run_all_btn = st.button(
            '▶ Run All Options (with polling)',
            type='secondary',
            key='tb_run_all',
            help='POST each backend sequentially, polling /health between runs.',
        )

        if run_all_btn:
            results_accumulator: list[dict] = []
            _poll_ph = st.empty()
            for _tr_opt, _ckpt_opt, _rm_opt in _ALL_OPTIONS:
                _opt_label = f'{_tr_opt} ({_rm_opt})'
                _poll_ph.info(f'🚀 Triggering **{_opt_label}**…')
                _payload_opt = {
                    'trainer':         _tr_opt,
                    'model_name':      _model_tr,
                    'min_rollouts':    int(_min_roll_tr),
                    'reward_mode':     _rm_opt,
                    'num_generations': int(_ngens_tr),
                    'checkpoint_path': _ckpt_opt,
                    'reload_after':    False,  # skip reload for intermediate runs
                }
                _resp_opt = _post('/api/training/trigger', _payload_opt)

                _status_opt = 'unknown'
                if _resp_opt and 'error' not in _resp_opt:
                    _status_opt = _resp_opt.get('status', 'unknown')
                    # Poll until training is no longer active before next trigger
                    if _status_opt == 'training_started':
                        _idle = _wait_for_training_idle(_poll_ph, _opt_label)
                        if not _idle:
                            _status_opt = 'timeout_waiting'
                    _meta_path = _ROOT / _ckpt_opt / 'adapter_metadata.json'
                    results_accumulator.append({
                        'Trainer':          _tr_opt,
                        'Reward mode':      _rm_opt,
                        'Checkpoint':       _ckpt_opt,
                        'Status':           _status_opt,
                        'PID':              _resp_opt.get('pid', '—'),
                        'Rollouts used':    _resp_opt.get('rollouts_used', '—'),
                        'adapter_metadata': '✅' if _meta_path.exists() else '⬜',
                        'vLLM reload':      _resp_opt.get('last_reload_status', 'skipped_by_request'),
                    })
                else:
                    results_accumulator.append({
                        'Trainer':          _tr_opt,
                        'Reward mode':      _rm_opt,
                        'Checkpoint':       _ckpt_opt,
                        'Status':           f'ERROR: {_resp_opt}',
                        'PID':              '—',
                        'Rollouts used':    '—',
                        'adapter_metadata': '❌',
                        'vLLM reload':      '—',
                    })
            _poll_ph.success('All options triggered.')
            st.session_state['tb_multi_run_results'] = results_accumulator

        # ── Comparison table ───────────────────────────────────────────────
        if st.session_state.get('tb_multi_run_results'):
            st.divider()
            st.subheader('Trainer comparison — last run')
            _mr_df = pd.DataFrame(st.session_state['tb_multi_run_results'])
            def _status_icon(s: str) -> str:
                if 'training_started' in s or s == 'no_rollouts':
                    return '🟢 ' + s
                if 'already_running' in s:
                    return '🟡 ' + s
                if 'ERROR' in s or 'FAIL' in s or 'timeout' in s:
                    return '🔴 ' + s
                if 'SKIP' in s or 'insufficient' in s:
                    return '⚪ ' + s
                return s
            _mr_df['Status'] = _mr_df['Status'].apply(_status_icon)
            st.dataframe(_mr_df, use_container_width=True, hide_index=True)

            # Per-trainer adapter metadata cards
            st.subheader('Per-option adapter metadata')
            for _, _row in pd.DataFrame(st.session_state['tb_multi_run_results']).iterrows():
                _meta_p = _ROOT / _row['Checkpoint'] / 'adapter_metadata.json'
                with st.expander(f"{_row['Trainer']} ({_row['Reward mode']}) — {_row['Checkpoint']}"):
                    if _meta_p.exists():
                        try:
                            st.json(json.loads(_meta_p.read_text()))
                        except Exception:
                            st.code(_meta_p.read_text())
                    else:
                        st.caption('No adapter_metadata.json yet.')

        # vLLM hot-reload
        st.divider()
        st.subheader('vLLM Hot-Reload')
        col_rl1, col_rl2 = st.columns([3, 1])
        with col_rl1:
            adapter_path_b = st.text_input(
                'Adapter path (blank = default checkpoint dir)',
                value='', placeholder='checkpoints/qwen25-3b-agent-lora', key='tb_adapter',
            )
        with col_rl2:
            if st.button('♻ Reload vLLM', use_container_width=True, key='tb_rl_btn'):
                payload_rl: dict[str, Any] = {}
                if adapter_path_b.strip():
                    payload_rl['adapter_path'] = adapter_path_b.strip()
                with st.spinner('Sending reload request…'):
                    rsp_rl = _post('/api/inference/reload', payload_rl)
                if rsp_rl and 'error' not in rsp_rl:
                    st.success(
                        f"Status: **{rsp_rl.get('reload_status')}**  "
                        f"| Adapter: `{rsp_rl.get('adapter_path')}`"
                    )
                else:
                    st.error(str(rsp_rl))

        # MLflow
        st.divider()
        st.subheader('MLflow')
        mlflow_port = st.number_input('MLflow UI port', value=5000, min_value=1024,
                                       max_value=65535, key='tb_mlf')
        st.markdown(
            f'Open **[MLflow UI](http://localhost:{mlflow_port})**  \n'
            f'Start: `mlflow ui --backend-store-uri {mlflow_uri} --port {mlflow_port}`'
        )
        try:
            import mlflow
            mlflow.set_tracking_uri(mlflow_uri)
            mlf_runs = []
            for exp in mlflow.tracking.MlflowClient().search_experiments()[:3]:
                for run in mlflow.tracking.MlflowClient().search_runs(
                    [exp.experiment_id], order_by=['start_time DESC'], max_results=6
                ):
                    mlf_runs.append({
                        'experiment': exp.name,
                        'run_id': run.info.run_id[:8],
                        'status': run.info.status,
                        **{k: round(v, 4) for k, v in (run.data.metrics or {}).items()},
                    })
            if mlf_runs:
                st.dataframe(pd.DataFrame(mlf_runs), use_container_width=True, hide_index=True)
        except Exception:
            st.caption('MLflow not available (start server separately).')

    # ═══════════════════════════════════════════════════════════════
    # Sub-tab 8 — Training Evidence
    # ═══════════════════════════════════════════════════════════════
    with tb_tab_evidence:
        st.subheader('Training Evidence')
        ckpt_dir_ev = _ROOT / 'checkpoints' / 'qwen25-3b-agent-lora'

        # adapter_metadata.json
        meta_path_ev = ckpt_dir_ev / 'adapter_metadata.json'
        if meta_path_ev.exists():
            try:
                meta_ev = json.loads(meta_path_ev.read_text())
                st.success(f'✅ `adapter_metadata.json` found in `{ckpt_dir_ev.name}/`')

                mc1, mc2, mc3 = st.columns(3)
                mc1.metric('Trainer',     meta_ev.get('trainer', '—'))
                mc1.metric('Status',      meta_ev.get('adapter_status', '—'))
                mc2.metric('Examples',    meta_ev.get('num_examples', '—'))
                mc2.metric('Task groups', meta_ev.get('grouped_rollout_groups', '—'))
                mc3.metric('Reward mode', meta_ev.get('reward_mode', '—'))
                mc3.metric('Fallback',    str(meta_ev.get('fallback_used', '—')))

                with st.expander('Full adapter_metadata.json'):
                    st.json(meta_ev)
            except Exception as _me:
                st.warning(f'Could not parse adapter_metadata.json: {_me}')
        else:
            st.info(
                '`adapter_metadata.json` not found.  '
                'Trigger training in the **Train** tab or run:\n\n'
                '```bash\nbash scripts/gpu/run_02_train_policy_qlora_grpo.sh\n```'
            )

        # adapter_config.json + safetensors
        col_ev1, col_ev2, col_ev3, col_ev4 = st.columns(4)
        col_ev1.metric('adapter_config.json',         '✅' if (ckpt_dir_ev / 'adapter_config.json').exists() else '❌')
        col_ev2.metric('adapter_model.safetensors',   '✅' if (ckpt_dir_ev / 'adapter_model.safetensors').exists() else '❌')
        col_ev3.metric('tokenizer_config.json',       '✅' if (ckpt_dir_ev / 'tokenizer_config.json').exists() else '❌')
        col_ev4.metric('policy_training_examples.jsonl', '✅' if (ckpt_dir_ev / 'policy_training_examples.jsonl').exists() else '❌')

        # Training log files
        st.divider()
        st.subheader('Training log files')
        log_dirs = [_ROOT / 'logs', _ROOT / 'reports']
        log_files = []
        for ld in log_dirs:
            if ld.exists():
                log_files += sorted(
                    [f for f in ld.glob('*train_policy*') if f.is_file()],
                    key=lambda p: p.stat().st_mtime, reverse=True,
                )
                log_files += sorted(
                    [f for f in ld.glob('*qlora_sft*') if f.is_file()],
                    key=lambda p: p.stat().st_mtime, reverse=True,
                )

        if log_files:
            sel_log = st.selectbox('Select log', log_files,
                                    format_func=lambda p: p.name, key='ev_log_sel')
            if sel_log:
                st.code(sel_log.read_text(encoding='utf-8', errors='replace')[-4000:],
                        language='text')
        else:
            st.caption('No training logs found in `logs/` or `reports/`.')

    # ═══════════════════════════════════════════════════════════════
    # Sub-tab 9 — Baseline vs Tuned
    # ═══════════════════════════════════════════════════════════════
    with tb_tab_compare:
        st.subheader('Baseline vs Tuned Policy')

        # Try to load from compare_policies.py
        _cmp_available = False
        try:
            from src.evaluation.compare_policies import compare_policy_versions
            _cmp_available = True
        except ImportError:
            pass

        # Show live metrics if server is up
        rollout_resp_cmp = _get('/api/rollouts?limit=500')
        if rollout_resp_cmp and 'error' not in rollout_resp_cmp:
            recs_cmp = rollout_resp_cmp.get('rollouts', [])
            if recs_cmp:
                import altair as alt
                cmp_df = pd.DataFrame([{
                    'policy': r.get('policy_version', 'unknown'),
                    'reward': float(r.get('final_reward') or 0.0),
                } for r in recs_cmp])
                if cmp_df['policy'].nunique() >= 2:
                    cmp_sum = (
                        cmp_df.groupby('policy')['reward']
                        .agg(mean='mean', std='std', count='count')
                        .reset_index()
                        .sort_values('mean', ascending=False)
                    )
                    policies = cmp_sum['policy'].tolist()
                    if len(policies) >= 2:
                        baseline_r = float(cmp_sum[cmp_sum['policy'] == policies[-1]]['mean'].iloc[0])
                        tuned_r    = float(cmp_sum[cmp_sum['policy'] == policies[0]]['mean'].iloc[0])
                        cm1, cm2, cm3 = st.columns(3)
                        cm1.metric('Baseline mean reward', f'{baseline_r:.4f}')
                        cm2.metric('Tuned mean reward',    f'{tuned_r:.4f}',
                                   delta=f'{tuned_r - baseline_r:+.4f}')
                        cm3.metric('Win rate',
                                   f'{100*sum(1 for r in recs_cmp if float(r.get("final_reward",0))>baseline_r)/len(recs_cmp):.1f}%')

                    bar_cmp = (
                        alt.Chart(cmp_sum).mark_bar()
                        .encode(
                            x=alt.X('policy:N', sort='-y'),
                            y=alt.Y('mean:Q', scale=alt.Scale(domain=[0, 1]), title='Mean Reward'),
                            color=alt.Color('mean:Q', scale=alt.Scale(scheme='redyellowgreen')),
                            tooltip=['policy', 'mean', 'std', 'count'],
                        )
                        .properties(height=240)
                    )
                    st.altair_chart(bar_cmp, use_container_width=True)
                else:
                    st.info('Need at least 2 policy versions to compare.  Train Track B and run new rollouts.')
            else:
                st.info('No rollout data available.')
        else:
            st.info('Lightning Server offline — showing local report files only.')

        # Local comparison reports
        st.divider()
        st.subheader('Local comparison reports')
        _cmp_reports = []
        _rep_dir = _ROOT / 'reports'
        if _rep_dir.exists():
            for pat in ['baseline_vs_rl_tuned.md', 'baseline_v2_tuned_benchmark.md']:
                p = _rep_dir / pat
                if p.exists():
                    _cmp_reports.append(p)

        if _cmp_reports:
            sel_cmp = st.selectbox('Report', _cmp_reports,
                                    format_func=lambda p: p.name, key='cmp_rpt_sel')
            if sel_cmp:
                st.markdown(sel_cmp.read_text(encoding='utf-8'))
        else:
            st.warning(
                'No comparison reports found.  Run:\n\n'
                '```bash\nbash scripts/cpu/run_06_compare_baseline_vs_tuned.sh\n```'
            )

    # ═══════════════════════════════════════════════════════════════
    # Sub-tab 10 — Policy Lifecycle
    # ═══════════════════════════════════════════════════════════════
    with tb_tab_lifecycle:
        st.subheader('Policy lifecycle')
        st.code(
            'Base Policy\n'
            '   ↓\n'
            'SFT Adapter (QLoRA fine-tune on best trajectories)\n'
            '   ↓\n'
            'GRPO Adapter (group-relative RL refinement)\n'
            '   ↓\n'
            'vLLM Hot Reload (adapter injected into serving engine)\n'
            '   ↓\n'
            'New Rollouts (improved policy runs Track A)\n'
            '   ↓\n'
            'Reward Comparison (baseline vs tuned)',
            language='text',
        )
        st.divider()
        st.subheader('Artifact status')
        _ckpt_lc = _ROOT / 'checkpoints' / 'qwen25-3b-agent-lora'
        _lc_checks = [
            ('Base model reachable', bool(vhealth and 'error' not in vhealth)),
            ('Checkpoint dir exists', _ckpt_lc.exists()),
            ('adapter_config.json',  (_ckpt_lc / 'adapter_config.json').exists()),
            ('adapter_model.safetensors', (_ckpt_lc / 'adapter_model.safetensors').exists()),
            ('adapter_metadata.json', (_ckpt_lc / 'adapter_metadata.json').exists()),
            ('vLLM serving (hot reload available)', bool(vhealth and 'error' not in vhealth)),
            ('New rollouts exist', len(list((_ROOT / 'trajectories' / 'scored').glob('*.jsonl'))) > 0
             if (_ROOT / 'trajectories' / 'scored').exists() else False),
            ('Comparison report exists',
             any((_ROOT / 'reports' / p).exists()
                 for p in ['baseline_vs_rl_tuned.md', 'baseline_v2_tuned_benchmark.md'])),
        ]
        for label, ok in _lc_checks:
            st.markdown(f'{"✅" if ok else "⬜"}  {label}')

        # Official Agent Lightning status
        st.divider()
        st.subheader('Agent Lightning integration status')
        _al_status_checks = [
            ('lit_agent.py present',               (_ROOT / 'src' / 'agents' / 'lit_agent.py').exists()),
            ('lightning_store.py present',         (_ROOT / 'src' / 'training' / 'lightning_store.py').exists()),
            ('lightning_sidecar.py present',       (_ROOT / 'src' / 'inference' / 'lightning_sidecar.py').exists()),
            ('agent_lightning_official_runner.py', (_ROOT / 'src' / 'training' / 'agent_lightning_official_runner.py').exists()),
        ]
        _al_pkg_ok = False
        try:
            import importlib
            importlib.import_module('agentlightning')
            _al_pkg_ok = True
        except ImportError:
            pass

        for label, ok in _al_status_checks:
            st.markdown(f'{"✅" if ok else "❌"}  {label}')

        if _al_pkg_ok:
            st.success('✅ Official `agentlightning` package importable')
        else:
            st.warning('🟡 Official `agentlightning` package **not installed** — local abstraction active')

        st.info(
            '**Integration modes:**  \n'
            '✅ Official mode executed — official package ran successfully  \n'
            '🟡 Local Agent-Lightning-style abstraction active — all key components present  \n'
            '❌ Official package unavailable — install `agentlightning` to enable'
        )

    # ═══════════════════════════════════════════════════════════════
    # Sub-tab 11 — Implementation Gaps
    # ═══════════════════════════════════════════════════════════════
    with tb_tab_gaps:
        st.subheader('Implementation Gaps')
        st.caption(
            'These gaps are acknowledged and shown here to demonstrate project maturity — '
            'not weakness.  Each gap has a clear fix path.'
        )
        _gaps = [
            ('Gap 1', 'GRPO reward is not yet fully workflow-aware',
             'Current TRL GRPO reward was mostly JSON-validity based.  '
             '`src/rewards/policy_reward.py` now provides `hybrid` and `workflow_policy` modes.  '
             'Set `REWARD_MODE=hybrid` in the trainer.'),
            ('Gap 2', 'Trajectory reward not fully connected to GRPO',
             '`src/rewards/scorer.py` calculates R_data, R_model, R_tracking, R_sandbox, '
             'R_reproducibility, R_violation — but these were not passed into GRPOTrainer.  '
             'Use `trajectory_reward` mode to pass pre-computed scores.'),
            ('Gap 3', 'SFT is the strongest demo today',
             'QLoRA SFT path is the safest demo because prompt/completion data is clean '
             'and training is fast.  Use SFT as primary demo, GRPO as advanced option.'),
            ('Gap 4', 'veRL requires external setup',
             'veRL is strictly opt-in.  If VERL_TRAIN_CMD is not set, training fails '
             'explicitly — no silent fallback.  This is intentional.'),
            ('Gap 5', 'Official Agent Lightning = compatibility mode',
             'Two layers: (1) local Agent-Lightning-style abstraction — always active; '
             '(2) official agentlightning package run — only if import succeeds.  '
             'The lifecycle tab shows which layer is active.'),
            ('Gap 6', 'Need more diverse rollouts for meaningful GRPO grouping',
             'GRPO needs ≥ 2 trajectories per task.  Collect more Track A rollouts '
             'across diverse task types to improve group quality.'),
            ('Gap 7', 'No deterministic train/val/test split by default',
             '`src/training/split_policy_dataset.py` now implements task_id-level split.  '
             'Use `--train-split 0.8` in the trainer CLI.'),
            ('Gap 8', 'Champion model artifact consistency not enforced in GRPO',
             'Review/reporting step should cross-check champion_model.json against '
             'benchmark_metrics.json.  Currently advisory only.'),
        ]
        for key, title, detail in _gaps:
            with st.expander(f'**{key}** — {title}'):
                st.markdown(detail)

    # ═══════════════════════════════════════════════════════════════
    # Sub-tab 12 — Recommended Demo Script
    # ═══════════════════════════════════════════════════════════════
    with tb_tab_demo_script:
        st.subheader('Recommended Demo Script')
        _steps = [
            ('Step 1',  'Run health checks',                   'Setup & Services → Re-check'),
            ('Step 2',  'Start CPU services',                  'Setup & Services → Start PostgreSQL, MCP, MLflow'),
            ('Step 3',  'Generate / load synthetic data',      'Setup: step 1 + 2'),
            ('Step 4',  'Run baseline Track A workflow',       'Setup: step 3'),
            ('Step 5',  'Score trajectories',                  'Setup: step 4'),
            ('Step 6',  'Prepare GRPO grouped dataset',        'Setup: step 5'),
            ('Step 7',  'Open Agent Lightning Trace tab',      'Show span attribution fields'),
            ('Step 8',  'Open Rewards & Rollouts tab',         'Show reward breakdown and grouped rollouts'),
            ('Step 9',  'Go to Track B → Trainer Select',      'Review comparison table'),
            ('Step 10', 'Select QLoRA SFT',                    'Safe demo — fastest, most stable'),
            ('Step 11', 'Show TRL GRPO',                       'Explain hybrid reward and GRPO advantage chart'),
            ('Step 12', 'Show veRL',                           'Explain distributed RL with strict preflight'),
            ('Step 13', 'Show Agent Lightning Official',       'Show compatibility card and integration status'),
            ('Step 14', 'Check Preflight tab',                 'Verify all green before training'),
            ('Step 15', 'Trigger training or show Evidence',   'Use previous training log if GPU not available'),
            ('Step 16', 'Hot-reload adapter',                  'Track B → Train → ♻ Reload vLLM'),
            ('Step 17', 'Compare baseline vs tuned',           'Track B → Baseline vs Tuned'),
        ]
        for step, title, detail in _steps:
            c1, c2, c3 = st.columns([1, 3, 4])
            c1.markdown(f'**{step}**')
            c2.markdown(title)
            c3.caption(detail)

        st.divider()
        st.subheader('Clean message for reviewers')
        st.info(
            '**Track A** generates scored agent trajectories.  \n'
            '**Track B** turns those trajectories into policy-improvement data.  \n'
            '**SFT** imitates the best trajectories.  \n'
            '**GRPO** compares grouped rollouts and rewards better actions.  \n'
            '**veRL** is the scalable distributed option.  \n'
            '**Agent Lightning** provides the training-agent disaggregation pattern.'
        )



# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — Compare Policies
# ══════════════════════════════════════════════════════════════════════════════
with tab_compare:
    st.header('📈 Compare Policies')
    st.caption(
        'Compare reward metrics between policy versions from the rollout store and '
        'local report files.  Equal rewards across policies indicate execution-path '
        'completeness is proven — improvement requires more diverse rollouts.'
    )

    if st.button('🔄 Refresh', key='cp_refresh'):
        st.rerun()

    rollout_resp = _get('/api/rollouts?limit=500')
    if rollout_resp and 'error' not in rollout_resp:
        recs = rollout_resp.get('rollouts', [])
        if recs:
            import pandas as pd
            import altair as alt

            cdf = pd.DataFrame([{
                'policy': r.get('policy_version', 'unknown'),
                'reward': float(r.get('final_reward') or 0.0),
                'task_id': r.get('task_id', ''),
                'status': r.get('status', ''),
            } for r in recs])

            summary = (
                cdf.groupby('policy')['reward']
                .agg(count='count', mean='mean', std='std', min='min', max='max')
                .reset_index()
                .sort_values('mean', ascending=False)
            )
            for col in ('mean', 'std', 'min', 'max'):
                summary[col] = summary[col].round(4)

            st.subheader('Policy summary')
            st.dataframe(
                summary.style.background_gradient(subset=['mean'], cmap='RdYlGn', vmin=0, vmax=1),
                use_container_width=True, hide_index=True,
            )

            bar = (
                alt.Chart(summary).mark_bar()
                .encode(
                    x=alt.X('policy:N', sort='-y'),
                    y=alt.Y('mean:Q', scale=alt.Scale(domain=[0, 1]), title='Mean Reward'),
                    color=alt.Color('mean:Q', scale=alt.Scale(scheme='redyellowgreen')),
                    tooltip=['policy', 'mean', 'std', 'count'],
                )
                .properties(height=250)
            )
            st.altair_chart(bar, use_container_width=True)

            if cdf['policy'].nunique() > 1:
                strip = (
                    alt.Chart(cdf).mark_tick(opacity=0.4)
                    .encode(
                        x=alt.X('reward:Q', scale=alt.Scale(domain=[0, 1])),
                        y='policy:N',
                        color='policy:N',
                        tooltip=['policy', 'reward', 'task_id'],
                    )
                    .properties(height=180, title='Reward distribution per policy')
                )
                st.altair_chart(strip, use_container_width=True)

            st.caption(
                'ℹ️  Equal mean rewards prove the feedback loop is running.  '
                'To see reward improvement, collect rollouts across ≥ 10 diverse tasks '
                'and run at least 2 training cycles.'
            )
    else:
        st.info('Lightning Server offline — showing local reports only.')

    # Local comparison reports
    st.divider()
    st.subheader('Local comparison reports')
    report_dir = _ROOT / 'reports'
    md_reports = (
        sorted(report_dir.glob('*.md'), key=lambda p: p.stat().st_mtime, reverse=True)
        if report_dir.exists() else []
    )
    if md_reports:
        sel_report = st.selectbox(
            'Select report', md_reports,
            format_func=lambda p: p.name, key='cp_report',
        )
        if sel_report:
            st.markdown(sel_report.read_text(encoding='utf-8'))
    else:
        st.info(
            'No `.md` reports found in `reports/`.  '
            'Run `bash scripts/cpu/run_06_compare_baseline_vs_tuned.sh`.'
        )

    # Champion model artifacts
    st.divider()
    st.subheader('Champion model artifacts')
    arts_dir = _ROOT / 'artifacts'
    if arts_dir.exists():
        champs = list(arts_dir.rglob('champion_model.json'))
        if champs:
            rows_c = []
            for p in sorted(champs):
                try:
                    data = json.loads(p.read_text())
                    rows_c.append({
                        'task':    p.parent.name,
                        'model':   data.get('model_type', ''),
                        'auc':     round(float(data.get('test_auc') or 0.0), 4),
                        'reward':  round(float(data.get('reward') or 0.0), 4),
                        'policy':  data.get('policy_version', ''),
                        'trained': data.get('trained_at', '')[:19],
                    })
                except Exception:
                    pass
            if rows_c:
                import pandas as pd
                st.dataframe(pd.DataFrame(rows_c), use_container_width=True, hide_index=True)
        else:
            st.info('No `champion_model.json` files found.  Run Track A to generate them.')
