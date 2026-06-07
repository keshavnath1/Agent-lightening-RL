"""Streamlit dashboard router for the self-improving ML agent.

The router owns the page configuration, sidebar state, service URL inputs,
shared data loading, and page dispatch. Thin entrypoints import and call
``render_dashboard()`` without executing Streamlit code at import time.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import requests
import streamlit as st


PAGES = [
    "Overview",
    "Architecture",
    "Setup & Handoff",
    "Track A",
    "Tool Learning",
    "Track B",
    "Results",
    "Monorepo Health",
    "Advanced Debug",
]


def _get(path: str, base: str = "", timeout: int = 4) -> dict[str, Any]:
    _base = base or st.session_state.get("server_url", "http://localhost:19123")
    try:
        r = requests.get(f"{_base}{path}", timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        return {"error": str(exc)}


def _post(path: str, payload: dict | None = None, timeout: int = 30) -> dict[str, Any]:
    _base = st.session_state.get("server_url", "http://localhost:19123")
    try:
        r = requests.post(f"{_base}{path}", json=payload or {}, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        return {"error": str(exc)}


def _dot(ok: bool) -> str:
    return "OK" if ok else "--"


def _nav(page: str) -> None:
    st.session_state["page"] = page
    st.session_state["_sidebar_nav"] = page
    st.rerun()


def _set_page_config_once() -> None:
    if not st.session_state.get("_dashboard_page_configured"):
        st.set_page_config(
            page_title="Self-Improving ML Agent",
            page_icon="ML",
            layout="wide",
            initial_sidebar_state="expanded",
        )
        st.session_state["_dashboard_page_configured"] = True


def render_dashboard() -> None:
    """Render the dashboard shell and dispatch to the selected page."""
    _set_page_config_once()

    st.sidebar.title("Agent Lightning")
    st.sidebar.caption("Self-Improving ML Agent")
    st.sidebar.divider()

    if "page" not in st.session_state:
        st.session_state["page"] = "Overview"
    if st.session_state.get("page") not in PAGES:
        st.session_state["page"] = "Overview"

    selected_page = st.sidebar.radio(
        "Navigate",
        PAGES,
        index=PAGES.index(st.session_state.get("page", "Overview")),
        key="_sidebar_nav",
    )
    st.session_state["page"] = selected_page

    st.sidebar.divider()
    mode = st.sidebar.radio(
        "Mode",
        ["Presentation", "Developer"],
        index=0 if st.session_state.get("mode", "Presentation") == "Presentation" else 1,
        key="_mode_radio",
        help="Presentation: clean demo view. Developer: full debug information.",
    )
    st.session_state["mode"] = mode

    st.sidebar.divider()
    with st.sidebar.expander("Service URLs", expanded=(mode == "Developer")):
        st.text_input(
            "Lightning Server URL",
            value=os.getenv("LIGHTNING_SERVER_URL", "http://localhost:19123"),
            key="server_url",
        )
        st.text_input(
            "vLLM URL",
            value=os.getenv("VLLM_BASE_URL", "http://localhost:8000"),
            key="vllm_url",
        )
        st.text_input(
            "MLflow Tracking URI",
            value=os.getenv("MLFLOW_TRACKING_URI", str(ROOT / "mlruns")),
            key="mlflow_uri",
        )

    from src.ui.view_models.dashboard_state import (
        load_data_status,
        load_gpu_status,
        load_service_health,
    )

    ds = load_data_status()
    gs = load_gpu_status()
    sh = load_service_health(
        st.session_state.get("server_url", "http://localhost:19123"),
        st.session_state.get("vllm_url", "http://localhost:8000"),
        st.session_state.get("mlflow_uri", str(ROOT / "mlruns")),
    )

    st.sidebar.caption(
        f"{_dot(sh.lightning_ok)} Lightning  "
        f"{_dot(sh.vllm_ok)} vLLM  "
        f"{_dot(sh.mlflow_ok)} MLflow"
    )

    page = st.session_state["page"]

    if page == "Overview":
        from src.ui.pages import overview
        overview.render(ds, gs, sh, nav_callback=_nav)
    elif page == "Architecture":
        from src.ui.pages import architecture
        architecture.render()
    elif page == "Setup & Handoff":
        from src.ui.pages import setup
        setup.render(ds, gs, sh)
    elif page == "Track A":
        from src.ui.pages import track_a
        track_a.render(ds)
    elif page == "Tool Learning":
        from src.ui.pages import tool_learning
        tool_learning.render()
    elif page == "Track B":
        from src.ui.pages import track_b
        track_b.render(ds, sh, post_fn=_post, get_fn=_get)
    elif page == "Results":
        from src.ui.pages import results
        results.render(sh, server_url=st.session_state.get("server_url", "http://localhost:19123"))
    elif page == "Monorepo Health":
        from src.ui.pages import monorepo_health
        monorepo_health.render()
    elif page == "Advanced Debug":
        if mode == "Presentation":
            st.warning("Advanced Debug is only visible in Developer mode. Toggle the mode in the sidebar.")
        else:
            from src.ui.pages import debug
            debug.render(
                ds,
                gs,
                sh,
                server_url=st.session_state.get("server_url", "http://localhost:19123"),
                vllm_url=st.session_state.get("vllm_url", "http://localhost:8000"),
                mlflow_uri=st.session_state.get("mlflow_uri", str(ROOT / "mlruns")),
            )
