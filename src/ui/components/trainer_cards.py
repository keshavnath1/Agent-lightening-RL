"""trainer_cards.py — trainer option card components for Track B."""
from __future__ import annotations
import streamlit as st
from typing import Any


TRAINER_CONFIGS: dict[str, dict[str, Any]] = {
    "trl_grpo": {
        "label":          "TRL GRPO",
        "badge":          "✅ Default",
        "badge_color":    "#1e7e34",
        "use_when":       "Default branch path: group-relative RL reward optimisation",
        "signal":         "Advantage-weighted update over grouped rollouts",
        "gpu_req":        "≥ 24 GB",
        "reward_modes":   ["hybrid", "workflow_policy", "trajectory_reward", "ruler_relative"],
        "default_rm":     "hybrid",
        "duration":       "Medium (20–60 min)",
        "availability":   "always",
    },
    "verl": {
        "label":          "veRL",
        "badge":          "🟠 Advanced",
        "badge_color":    "#664d03",
        "use_when":       "Distributed multi-GPU RL training",
        "signal":         "Official veRL distributed GRPO/PPO",
        "gpu_req":        "≥ 2× GPU + veRL installed",
        "reward_modes":   ["hybrid"],
        "default_rm":     "hybrid",
        "duration":       "Long (hours)",
        "availability":   "verl",   # checked at runtime
    },
    "official_art_ruler": {
        "label":          "Official ART + RULER",
        "badge":          "🟠 Official",
        "badge_color":    "#664d03",
        "use_when":       "Official openpipe-art LangGraph trajectory groups and RULER scoring",
        "signal":         "art.TrajectoryGroup + official RULER + art.LocalBackend",
        "gpu_req":        "ART installed; depends on backend",
        "reward_modes":   ["hybrid"],
        "default_rm":     "hybrid",
        "duration":       "Depends on backend",
        "availability":   "art",
    },
    "agent_lightning_official": {
        "label":          "Agent Lightning Official",
        "badge":          "🟠 Advanced",
        "badge_color":    "#664d03",
        "use_when":       "Official agentlightning package integration proof",
        "signal":         "Official Lightning Trainer + Algorithm APIs",
        "gpu_req":        "Depends on backend",
        "reward_modes":   ["hybrid"],
        "default_rm":     "hybrid",
        "duration":       "Depends on backend",
        "availability":   "agentlightning",
    },
}


def _check_availability(key: str) -> tuple[bool, str]:
    avail = TRAINER_CONFIGS[key]["availability"]
    if avail == "always":
        return True, ""
    try:
        __import__(avail)
        return True, ""
    except ImportError:
        return False, ('Official ART/RULER not installed — `uv pip install -U "openpipe-art[backend,langgraph]>=0.4.9"`' if avail == 'art' else f'`{avail}` not installed — `pip install {avail}`')


def render_trainer_selector() -> str:
    """Render trainer option cards. Returns the selected trainer key."""
    selected = st.session_state.get("tb_trainer_selected", "trl_grpo")

    cols = st.columns(len(TRAINER_CONFIGS))
    for col, (key, cfg) in zip(cols, TRAINER_CONFIGS.items()):
        with col:
            avail, reason = _check_availability(key)
            border = "2px solid #0d6efd" if selected == key else "1px solid #444"
            bg     = "#0d1b4b" if selected == key else "#1a1a1a"
            st.markdown(
                f"""<div style="border:{border}; background:{bg}; border-radius:8px; padding:12px; min-height:140px;">
                <div style="font-weight:700; font-size:1em;">{cfg['label']}</div>
                <div style="font-size:0.8em; color:{cfg['badge_color']}; margin:4px 0;">{cfg['badge']}</div>
                <div style="font-size:0.78em; color:#aaa;">{cfg['use_when']}</div>
                <div style="font-size:0.75em; color:#888; margin-top:4px;">GPU: {cfg['gpu_req']}</div>
                {"<div style='font-size:0.72em; color:#f88; margin-top:4px;'>"+reason+"</div>" if reason else ""}
                </div>""",
                unsafe_allow_html=True,
            )
            st.markdown("")
            label = f"{'✓ ' if selected==key else ''}Select {cfg['label'].split()[0]}"
            if st.button(label, key=f"sel_{key}", disabled=not avail, use_container_width=True):
                st.session_state["tb_trainer_selected"] = key
                selected = key

    return selected


def render_trainer_detail(key: str) -> None:
    """Render the detail card for a selected trainer."""
    cfg = TRAINER_CONFIGS.get(key, {})
    if not cfg:
        return
    with st.expander(f"About {cfg['label']}", expanded=False):
        col1, col2 = st.columns(2)
        col1.markdown(f"**Training signal:** {cfg['signal']}")
        col1.markdown(f"**GPU requirement:** {cfg['gpu_req']}")
        col2.markdown(f"**Typical duration:** {cfg['duration']}")
        col2.markdown(f"**Reward modes:** {', '.join(cfg['reward_modes'])}")
