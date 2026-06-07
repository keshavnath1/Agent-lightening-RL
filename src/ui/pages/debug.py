"""debug.py — Advanced Debug page: raw URLs, packages, logs, JSON payloads, MLflow."""
from __future__ import annotations
import json
import subprocess
import sys
import streamlit as st
from pathlib import Path
from src.ui.view_models.dashboard_state import (
    DataStatus, GPUStatus, ServiceHealth, ROOT,
)


def render(
    ds: DataStatus,
    gs: GPUStatus,
    sh: ServiceHealth,
    server_url: str,
    vllm_url: str,
    mlflow_uri: str,
) -> None:
    st.header("🔬 Advanced Debug")
    st.caption("Full diagnostic view — raw service data, logs, JSON payloads, MLflow runs.")

    # ── Service raw info ──────────────────────────────────────────────────────
    with st.expander("Service URLs & Health"):
        st.markdown(f"**Lightning Server:** `{server_url}`")
        st.markdown(f"**vLLM:**            `{vllm_url}`")
        st.markdown(f"**MLflow:**          `{mlflow_uri}`")
        st.markdown("")
        col1, col2, col3 = st.columns(3)
        col1.metric("Lightning", "🟢 ok" if sh.lightning_ok else "🔴 offline")
        col2.metric("vLLM",      "🟢 ok" if sh.vllm_ok else "🔴 offline")
        col3.metric("MLflow",    "🟢 ok" if sh.mlflow_ok else "🔴 offline")
        if sh.lightning_ok:
            col1.metric("Rollouts collected", sh.rollouts_collected)
            col1.metric("Training active",    str(sh.training_active))
            col1.metric("Queue size",         sh.queue_size)

        if st.button("Refresh service health"):
            st.rerun()

    # ── Package versions ──────────────────────────────────────────────────────
    with st.expander("Package Versions"):
        PKGS = [
            "torch", "peft", "transformers", "trl", "bitsandbytes",
            "streamlit", "fastapi", "vllm", "verl", "agentlightning",
            "mlflow", "lightgbm", "pandas", "numpy", "altair",
        ]
        import importlib.metadata as _imeta
        rows = []
        for pkg in PKGS:
            try:
                __import__(pkg)
                try:
                    ver = _imeta.version(pkg)
                except Exception:
                    ver = "installed"
                rows.append({"package": pkg, "version": ver, "status": "✅"})
            except ImportError:
                rows.append({"package": pkg, "version": "—", "status": "⬜"})
        import pandas as pd
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── GPU info ──────────────────────────────────────────────────────────────
    with st.expander("GPU Details"):
        if gs.available:
            try:
                import torch
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    st.markdown(
                        f"**GPU {i}:** {props.name} — "
                        f"{props.total_memory/1e9:.1f} GB total, "
                        f"{props.multi_processor_count} SMs"
                    )
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,temperature.gpu",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5,
                )
                if result.returncode == 0:
                    st.text(result.stdout.strip())
            except Exception as e:
                st.warning(f"Error fetching GPU details: {e}")
        else:
            st.info("No CUDA GPU detected.")

    # ── Streamlit log ─────────────────────────────────────────────────────────
    with st.expander("Streamlit Log (last 60 lines)"):
        streamlit_logs = [Path("/tmp/streamlit.log")]
        streamlit_log_dir = ROOT / "reports" / "streamlit_logs"
        if streamlit_log_dir.exists():
            streamlit_logs.extend(
                sorted(streamlit_log_dir.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
            )
        log_path = next((p for p in streamlit_logs if p.exists()), None)
        if log_path:
            st.caption(str(log_path))
            lines = log_path.read_text().splitlines()
            st.text("\n".join(lines[-60:]))
        else:
            st.info("No Streamlit log file found.")

    # ── Training logs ─────────────────────────────────────────────────────────
    with st.expander("Training Logs (last 80 lines)"):
        log_dirs = [ROOT / "logs", ROOT / "reports" / "run_logs", ROOT / "reports" / "service_logs"]
        log_files = sorted(
            [p for log_dir in log_dirs if log_dir.exists() for p in log_dir.glob("**/*.log")],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if log_files:
            chosen = st.selectbox("Select log file", [str(p.relative_to(ROOT)) for p in log_files[:10]])
            lp = ROOT / chosen
            lines = lp.read_text().splitlines()
            st.text("\n".join(lines[-80:]))
        else:
            st.info("No .log files found in logs/")

    # ── Raw rollout JSON ──────────────────────────────────────────────────────
    with st.expander("Raw Grouped Rollouts (first 3)"):
        grp_path = ROOT / "data" / "grpo" / "grouped_rollouts.jsonl"
        if grp_path.exists():
            rows = [json.loads(l) for l in grp_path.read_text().splitlines() if l.strip()]
            for i, row in enumerate(rows[:3]):
                st.json(row)
        else:
            st.info("grouped_rollouts.jsonl not found")

    # ── Adapter metadata ──────────────────────────────────────────────────────
    with st.expander("Adapter Metadata (all checkpoints)"):
        for meta_path in sorted((ROOT / "checkpoints").glob("*/adapter_metadata.json")):
            st.markdown(f"**{meta_path.parent.name}**")
            try:
                st.json(json.loads(meta_path.read_text()))
            except Exception:
                st.warning(f"Could not parse {meta_path}")

    # ── MLflow runs ───────────────────────────────────────────────────────────
    with st.expander("MLflow Runs"):
        if not sh.mlflow_ok:
            st.warning(f"MLflow not accessible at `{mlflow_uri}`")
        else:
            try:
                import mlflow
                mlflow.set_tracking_uri(mlflow_uri)
                client = mlflow.tracking.MlflowClient()
                exps = client.search_experiments()
                for exp in exps:
                    runs = client.search_runs(exp.experiment_id, max_results=10,
                                              order_by=["start_time DESC"])
                    if runs:
                        import pandas as pd
                        rows = []
                        for run in runs:
                            rows.append({
                                "run_id":  run.info.run_id[:8],
                                "status":  run.info.status,
                                "start":   run.info.start_time,
                                **{k: v for k, v in run.data.metrics.items()},
                            })
                        st.markdown(f"**{exp.name}**")
                        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            except Exception as e:
                st.error(f"MLflow error: {e}")

    # ── Fix / diagnostic commands ─────────────────────────────────────────────
    with st.expander("Fix Commands"):
        st.code(
            "# Restart Streamlit\n"
            "pkill -f 'streamlit run' && \\\n"
            "  python -m streamlit run scripts/demo_dashboard.py \\\n"
            "  --server.port 8503 --server.address 0.0.0.0 \\\n"
            "  --server.headless true --server.enableCORS false \\\n"
            "  --server.enableXsrfProtection false > reports/streamlit_logs/dashboard_8503.log 2>&1 &\n\n"
            "# Check health\n"
            "curl http://localhost:8503/_stcore/health\n"
            "curl http://localhost:19124/health\n"
            "curl http://localhost:18081/v1/models\n\n"
            "# Re-run validation (27 checks)\n"
            "python scripts/dev_validate_trackb_wiring.py\n\n"
            "# Re-run dashboard refactor validation\n"
            "python scripts/dev_validate_dashboard_refactor.py",
            language="bash",
        )

    # ── Implementation gaps ───────────────────────────────────────────────────
    with st.expander("Implementation Gaps / Known Issues"):
        st.markdown(
            """
| Gap | Severity | Notes |
|-----|----------|-------|
| `trajectory_reward` batch column | Medium | Needs real GPU TRL validation |
| veRL not installed | Low | `pip install verl` on GPU pod |
| agentlightning not installed | Low | `pip install agentlightning` |
| MLflow tracking URI env var | Low | Set `MLFLOW_TRACKING_URI` |
| vLLM hot-reload after SFT | Medium | Requires matching LoRA architecture |
"""
        )


# Alignment note inserted for presentation mode
def _render_pdf_alignment_note() -> None:
    import streamlit as st
    st.markdown('### Advanced ART/RULER Debug')
    st.markdown('''PDF alignment: ART-like trajectories, grouped rollouts, GRPO, MCP tool loop partial, fail-closed vLLM-backed RULER judge, checkpoint forking metadata, behavioral policy drift, and GSPO roadmap only. Official ART/RULER is available only when openpipe-art and official RULER are installed.''')
