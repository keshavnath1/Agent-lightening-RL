"""Monorepo health page for package and validation visibility."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.ui.components.validation_panel import render_validation_panel

ROOT = Path(__file__).resolve().parents[3]

PROJECT_AREAS = [
    "apps/dashboard",
    "apps/trainer",
    "apps/rollout_worker",
    "apps/ruler_scorer",
    "services/project_mcp_server",
    "packages/contracts",
    "packages/ml_tools",
    "packages/rewards",
    "packages/lightning_bridge",
    "packages/mcp_client_bridge",
]


def render() -> None:
    """Render package-level health and validation guidance."""
    st.header("Monorepo Health")
    st.caption("Package and service inventory with focused validation commands.")

    rows = []
    for rel in PROJECT_AREAS:
        path = ROOT / rel
        rows.append({
            "Area": rel,
            "Present": "yes" if path.exists() else "no",
            "pyproject.toml": "yes" if (path / "pyproject.toml").exists() else "no",
        })
    st.table(rows)

    render_validation_panel()
