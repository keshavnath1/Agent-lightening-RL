"""Thin Streamlit wrapper for the packaged dashboard app.

Usage:
    streamlit run scripts/demo_dashboard.py
    streamlit run scripts/demo_dashboard.py --server.port 8501 --server.address 0.0.0.0
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DASHBOARD_SRC = ROOT / "apps" / "dashboard" / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(DASHBOARD_SRC) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_SRC))

from dashboard_app.main import run

run()
