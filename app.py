"""SPAZE — Unified Architectural GenAI Workspace.

Single entry point coordinating the complete architectural layout workflow:
00 Workspace -> 01 Floor Plan -> 02 AI Generation -> 03 3D Visualization -> 04 Evaluation -> 05 Export -> 06 Docs.
"""
from __future__ import annotations

import streamlit as st

from config.settings import AppConfig
from ui.components import render_top_header
from ui.navigation import render_sidebar
from ui.pages import (
    render_docs_page,
    render_evaluation_page,
    render_export_page,
    render_floorplan_page,
    render_generation_page,
    render_overview_page,
    render_visualization_page,
)
from ui.state import StateManager
from ui.theme import apply_spaze_theme

# -----------------------------------------------------------------------------
# Streamlit Application Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title=f"{AppConfig.APP_NAME} — Architectural Layout Workspace",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main() -> None:
    """Master controller dispatching to active workspace pages."""
    # 1. Inject architectural theme styles
    apply_spaze_theme()

    # 2. Initialize persistent session state
    StateManager.initialize()

    # 3. Render persistent sidebar navigation
    active_stage = render_sidebar()

    # 4. Render top workspace header with horizontal progress indicator
    render_top_header(active_stage)

    # 5. Modular Page Dispatcher (renders only the selected workspace)
    if active_stage == "00_workspace":
        render_overview_page()
    elif active_stage == "01_floorplan":
        render_floorplan_page()
    elif active_stage == "02_generation":
        render_generation_page()
    elif active_stage == "03_visualization":
        render_visualization_page()
    elif active_stage == "04_evaluation":
        render_evaluation_page()
    elif active_stage == "05_export":
        render_export_page()
    elif active_stage == "06_docs":
        render_docs_page()
    else:
        render_floorplan_page()


if __name__ == "__main__":
    main()
