"""Stage 00 — Workspace Overview & Studio Landing Dashboard.

Welcomes architects and researchers to SPAZE, provides pipeline overviews,
displays current project state, and allows one-click sample loading or design initiation.
"""
from __future__ import annotations

import streamlit as st

from models.floorplan_schema import FloorPlan, Room, ScaleInfo
from ui.state import StateManager


def render_overview_page() -> None:
    """Render the Stage 00 Workspace Overview dashboard matching reference design."""
    # 1. Hero Section
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #FFFFFF 0%, #F5F3FF 100%); border: 1px solid #E9E7F2; border-radius: 18px; padding: 2.2rem 2.0rem; margin-bottom: 1.5rem; box-shadow: 0 4px 20px rgba(112, 71, 245, 0.05);">
            <div style="display:inline-block; background:#EEE8FF; color:#7047F5; padding:4px 10px; border-radius:8px; font-size:0.76rem; font-weight:800; letter-spacing:0.6px; text-transform:uppercase; margin-bottom:0.75rem;">
                ⚡ Architectural GenAI Platform
            </div>
            <div style="font-size: 2.3rem; font-weight: 900; color: #171A2B; line-height: 1.15; letter-spacing: -0.6px;">
                From Blueprint to <span style="background: linear-gradient(135deg, #7047F5 0%, #956BFF 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Living Space.</span>
            </div>
            <div style="font-size: 1.0rem; color: #69738B; margin-top: 0.6rem; max-width: 780px; line-height: 1.55;">
                SPAZE is an integrated Generative AI architectural platform that digitizes 2D floor-plan drawings, 
                synthesizes preference-conditioned layout alternatives with a Conditional Variational Autoencoder (CVAE), 
                and transforms validated layouts into interactive 3D WebGL environments with binary GLB export.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Quick Action Buttons
    c_btn1, c_btn2, c_btn3 = st.columns([1.2, 1.4, 1.2])

    with c_btn1:
        if st.button("🚀 Start Floor Plan Studio", type="primary", use_container_width=True):
            StateManager.set_stage("01_floorplan")
            st.rerun()

    with c_btn2:
        if st.button("📂 Load Verified Residential Demo", use_container_width=True):
            _load_demo_project()
            st.rerun()

    with c_btn3:
        if st.button("🔬 Explore Evaluation Lab", use_container_width=True):
            StateManager.set_stage("04_evaluation")
            st.rerun()

    st.write("")

    # 2. Pipeline Architecture Cards
    st.markdown(
        """
        <div style="font-size:0.78rem; font-weight:700; text-transform:uppercase; color:#69738B; letter-spacing:0.8px; margin-bottom:0.6rem;">
            Integrated Architectural GenAI Pipeline
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    col1, col2, col3, col4 = st.columns(4, gap="medium")

    with col1:
        st.markdown(
            """
            <div class="section-card" style="height: 100%;">
                <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">📐</div>
                <div style="font-weight: 800; color: #171A2B; margin-bottom: 0.3rem;">01. Computer Vision</div>
                <div style="color: #69738B; font-size: 0.80rem; line-height: 1.45;">
                    Local OpenCV preprocessing, morphological wall detection, adaptive thresholding, and closed room contour extraction with interactive scale calibration.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="section-card" style="height: 100%;">
                <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">🧠</div>
                <div style="font-weight: 800; color: #171A2B; margin-bottom: 0.3rem;">02. Conditional VAE</div>
                <div style="color: #69738B; font-size: 0.80rem; line-height: 1.45;">
                    PyTorch CVAE trained on paired residential layouts. Synthesizes 3 distinct alternatives conditioned on Space-Efficient, Open, and Privacy design intents in 8.4 ms.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
            <div class="section-card" style="height: 100%;">
                <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">🧊</div>
                <div style="font-weight: 800; color: #171A2B; margin-bottom: 0.3rem;">03. Procedural 3D</div>
                <div style="color: #69738B; font-size: 0.80rem; line-height: 1.45;">
                    Shapely & Trimesh procedural geometry. Shared interior wall deduplication, physical walk-through apertures, modular furniture, and native Plotly WebGL viewer.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            """
            <div class="section-card" style="height: 100%;">
                <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">📊</div>
                <div style="font-weight: 800; color: #171A2B; margin-bottom: 0.3rem;">04. Research Lab</div>
                <div style="color: #69738B; font-size: 0.80rem; line-height: 1.45;">
                    Empirical validation loss, KL divergence, layout diversity (1-IoU), preference sensitivity metrics, CV ground-truth precision/recall, and procedural baseline benchmarks.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 3. Active Project Diagnostics
    st.markdown("---")
    st.markdown(
        """
        <div style="font-size:0.78rem; font-weight:700; text-transform:uppercase; color:#69738B; letter-spacing:0.8px; margin-bottom:0.6rem;">
            Active Project Session
        </div>
        """,
        unsafe_allow_html=True,
    )

    has_fp = StateManager.has_floorplan()
    has_alts = StateManager.has_alternatives()
    has_3d = st.session_state.get("active_3d_result") is not None

    p_col1, p_col2, p_col3, p_col4 = st.columns(4)

    with p_col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">2D Floor Plan</div>
                <div class="metric-value" style="color: {'#19B978' if has_fp else '#69738B'};">
                    {'Verified' if has_fp else 'None'}
                </div>
                <div class="metric-sub">{len(st.session_state.floorplan.confirmed_rooms) if has_fp else 0} confirmed rooms</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with p_col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">CVAE Alternatives</div>
                <div class="metric-value" style="color: {'#7047F5' if has_alts else '#69738B'};">
                    {len(st.session_state.get('cvae_alternatives', []))} Generated
                </div>
                <div class="metric-sub">Space-Efficient, Open, Privacy</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with p_col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">3D Architectural Model</div>
                <div class="metric-value" style="color: {'#19B978' if has_3d else '#69738B'};">
                    {'Ready' if has_3d else 'Pending'}
                </div>
                <div class="metric-sub">Plotly WebGL + Binary GLB</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with p_col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Physical Calibration</div>
                <div class="metric-value" style="color: #171A2B;">
                    {st.session_state.get('scale_info', ScaleInfo()).pixels_per_meter or 50.0:.1f}
                </div>
                <div class="metric-sub">pixels / meter</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _load_demo_project() -> None:
    """Helper to initialize the verified 4-room residential apartment fixture."""
    from ui.pages.visualization_page import _create_demo_floorplan
    demo_fp = _create_demo_floorplan()
    st.session_state.floorplan = demo_fp
    st.session_state.sample_loaded = True
    st.success("Loaded verified residential apartment project into memory.")
