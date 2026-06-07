"""Validation status panel used by the dashboard health page."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[3]

VALIDATION_COMMANDS = [
    ("Monorepo validation", "python scripts/validate/validate_monorepo_phase1.py"),
    ("Official MCP validation", "python scripts/dev_validate_official_mcp_server.py"),
    ("Track B wiring validation", "python scripts/dev_validate_trackb_wiring.py"),
    ("Dashboard validation", "python scripts/dev_validate_dashboard_refactor.py"),
    ("RULER handoff dry-run", "bash scripts/gpu/run_ruler_trl_handoff.sh --dry-run"),
]


def _latest_report() -> Path | None:
    report_dir = ROOT / "reports"
    if not report_dir.exists():
        return None
    candidates = sorted(report_dir.glob("*validation*"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def render_validation_panel() -> None:
    """Render validation commands and latest-report guidance."""
    st.subheader("Validation panel")
    st.caption("Run these commands after dashboard or monorepo wiring changes.")
    st.table({"Check": [c[0] for c in VALIDATION_COMMANDS], "Command": [c[1] for c in VALIDATION_COMMANDS]})

    latest = _latest_report()
    if latest is None:
        st.info("No validation report was found. Run the commands above and save the output under reports/.")
    else:
        st.success(f"Latest validation-like report: {latest.relative_to(ROOT)}")
