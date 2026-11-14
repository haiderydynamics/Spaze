"""Persistent Sidebar Navigation Component for SPAZE.

Renders brand identity, stage selection switcher, workflow completion indicators,
and architectural branding illustration matching the visual reference image.
"""
from pathlib import Path
import streamlit as st
import torch

from cvae.config import CVAEConfig
from ui.state import STAGE_NAMES, StateManager


def render_sidebar() -> str:
    """Render the persistent sidebar navigation and return selected stage ID."""
    with st.sidebar:
        # 1. Branding Header matching reference
        st.markdown(
            """
            <div style="display:flex; align-items:center; gap:0.65rem; padding: 0.35rem 0.2rem 1.0rem 0.2rem; border-bottom: 1px solid #E9E7F2; margin-bottom: 1.15rem;">
                <div style="width:36px; height:36px; border-radius:10px; background: linear-gradient(135deg, #7047F5 0%, #388BFF 100%); display:flex; align-items:center; justify-content:center; color:#FFFFFF; font-size:1.15rem; font-weight:900; box-shadow: 0 4px 10px rgba(112, 71, 245, 0.25);">
                    📐
                </div>
                <div>
                    <div style="font-size:1.25rem; font-weight:900; color:#19213A; letter-spacing:-0.4px; line-height:1.1;">SPAZE</div>
                    <div style="font-size:0.68rem; color:#66718B; font-weight:600; letter-spacing:0.2px;">Architectural GenAI Workspace</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 2. Workspace Navigation Stages
        current_stage = StateManager.get_stage()
        stage_keys = list(STAGE_NAMES.keys())

        # Exact icons matching reference
        stage_icons = {
            "00_workspace": "🏠 Workspace Overview",
            "01_floorplan": "📐 01 Floor Plan Studio",
            "02_generation": "✨ 02 AI Layout Generator",
            "03_visualization": "🧊 03 3D Visualization",
            "04_evaluation": "📊 04 Evaluation Lab",
            "05_export": "📥 05 Project Export",
            "06_docs": "📖 06 Research & Docs",
        }

        # Render stage switcher using customized radio buttons (circle hidden via CSS)
        selected_stage = st.radio(
            "Navigate Workflow",
            options=stage_keys,
            format_func=lambda s: stage_icons.get(s, STAGE_NAMES[s]),
            index=stage_keys.index(current_stage) if current_stage in stage_keys else 0,
            label_visibility="collapsed",
        )

        if selected_stage != current_stage:
            StateManager.set_stage(selected_stage)
            st.rerun()

        # 3. Architectural Villa Illustration in Sidebar
        st.markdown(
            """
            <div style="margin-top: 2.2rem; padding: 0.8rem 0.4rem; text-align: center; border-radius: 16px; background: linear-gradient(180deg, rgba(238, 232, 255, 0.4) 0%, rgba(244, 240, 255, 0.95) 100%); border: 1px solid #E9E7F2;">
                <div style="font-size: 2.8rem; line-height: 1.1; margin-bottom: 0.35rem;">🏡</div>
                <div style="font-size: 0.76rem; font-weight: 700; color: #19213A;">Autonomous CAD Studio</div>
                <div style="font-size: 0.68rem; color: #66718B; margin-top: 0.15rem;">SPAZE GenAI Platform v1.0</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        return selected_stage
