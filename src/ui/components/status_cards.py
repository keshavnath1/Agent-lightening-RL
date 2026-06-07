"""status_cards.py — compact status badge components."""
from __future__ import annotations
import streamlit as st
from typing import Any


def status_badge(ok: bool | None, label: str, detail: str = "", severity: str = "required") -> None:
    """
    Render a single status row.

    severity: "required" | "optional" | "advanced"
    """
    if ok is True:
        icon = "✅"
    elif ok is False:
        if severity == "required":
            icon = "❌"
        elif severity == "optional":
            icon = "⚠️"
        else:
            icon = "⬜"
    else:
        icon = "⏳"

    severity_tag = {"required": "", "optional": " *(optional)*", "advanced": " *(advanced)*"}.get(severity, "")
    st.markdown(f"{icon} &nbsp; **{label}**{severity_tag}" + (f" — {detail}" if detail else ""),
                unsafe_allow_html=True)


def render_status_grid(checks: list[dict]) -> None:
    """
    checks: list of dicts with keys: label, ok, detail, severity
    """
    for c in checks:
        status_badge(
            ok=c.get("ok"),
            label=c["label"],
            detail=c.get("detail", ""),
            severity=c.get("severity", "required"),
        )


def render_stage_card(
    stage: str,
    description: str,
    status: str,          # "complete" | "ready" | "pending" | "error"
    metric: str = "",
    cta_label: str = "",
    cta_page: str = "",
) -> bool:
    """Render a coloured stage card. Returns True if CTA was clicked."""
    color = {"complete": "#1e7e34", "ready": "#0d6efd", "pending": "#6c757d", "error": "#dc3545"}.get(status, "#6c757d")
    icon  = {"complete": "✅", "ready": "🟢", "pending": "⬜", "error": "❌"}.get(status, "⬜")

    st.markdown(
        f"""<div style="border-left:4px solid {color}; padding:10px 14px; margin:6px 0; border-radius:4px; background:#1a1a2e;">
        <span style="font-size:1.1em; font-weight:600;">{icon} {stage}</span><br/>
        <span style="color:#bbb; font-size:0.9em;">{description}</span>
        {"<br/><span style='color:#aef; font-size:0.85em;'>"+metric+"</span>" if metric else ""}
        </div>""",
        unsafe_allow_html=True,
    )
    if cta_label:
        return st.button(cta_label, key=f"cta_{stage.replace(' ','_')}")
    return False
