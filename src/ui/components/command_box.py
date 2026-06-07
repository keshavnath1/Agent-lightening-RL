"""command_box.py — clean command display with copy hint."""
from __future__ import annotations
import streamlit as st


def render_command(cmd: str, label: str = "Command", language: str = "bash") -> None:
    """Show a labelled code block."""
    st.caption(label)
    st.code(cmd, language=language)


def render_command_card(cmd: str, label: str, note: str = "") -> None:
    """Render a command in a bordered card with optional note."""
    note_html = f'<br/><span style="color:#888; font-size:0.75em;">{note}</span>' if note else ""
    st.markdown(
        f"<div style='border:1px solid #333; border-radius:6px; padding:10px 14px; "
        f"background:#0d0d0d; margin:4px 0;'>"
        f"<span style='color:#aaa; font-size:0.8em;'>{label}</span>"
        f"{note_html}"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.code(cmd, language="bash")
