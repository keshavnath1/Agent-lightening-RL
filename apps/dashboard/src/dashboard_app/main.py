"""Entrypoint for the packaged Streamlit dashboard application."""
from __future__ import annotations

from dashboard_app.router import render_dashboard


def run() -> None:
    """Run the Streamlit dashboard."""
    render_dashboard()


if __name__ == "__main__":
    run()
