"""Reusable UI Components, Header Bar, and Metric Cards for SPAZE.

Implements reference-locked top bar with stage title and workflow ribbon,
as well as metric cards with icon boxes matching the visual reference image.
"""
from typing import Optional
import streamlit as st

from ui.state import StateManager


STAGE_HEADER_INFO = {
    "00_workspace": ("00", "Workspace Overview", "Mission control dashboard, pipeline health, and active architectural design sessions.", "📊"),
    "01_floorplan": ("Stage 01", "Floor Plan Processing & Extraction", "Upload a 2D floor plan or load a test fixture to extract walls, detect rooms, calibrate scale, and verify boundaries before AI generation.", "📐"),
    "02_generation": ("Stage 02", "AI-Powered Layout Generation", "Generate alternative architectural layouts using preference-conditioned generative AI.", "🧠"),
    "03_visualization": ("Stage 03", "Intelligent 3D Visualization", "Explore your AI-generated architectural layouts in an interactive three-dimensional environment.", "🧊"),
    "04_evaluation": ("Stage 04", "Model Evaluation & Performance", "Empirical quantitative evaluation across Computer Vision extraction, Conditional VAE convergence, and baseline benchmarks.", "🔬"),
    "05_export": ("Stage 05", "Project Export Hub", "Download production-ready architectural deliverables in industry-standard formats including binary glTF 2.0, JSON schemas, and blueprints.", "📦"),
    "06_docs": ("Stage 06", "Academic Research & Documentation", "Explore complete peer-reviewed academic specifications, system architectures, probabilistic formulations, and experimental methodologies.", "📚"),
}


def render_top_header(current_stage: str) -> None:
    """Render the top workspace bar matching the reference image with stage title on left and workflow ribbon on right."""
    stages = [
        ("01_floorplan", "01 Floor Plan"),
        ("02_generation", "02 AI Layout"),
        ("03_visualization", "03 3D Studio"),
        ("04_evaluation", "04 Evaluation"),
        ("05_export", "05 Export"),
    ]

    pipeline_html = '<div class="workflow-bar">'
    for idx, (s_key, s_label) in enumerate(stages):
        is_active = (s_key == current_stage)
        status = StateManager.get_stage_status(s_key)
        is_completed = (status in ("Completed", "Generated", "Ready", "Available"))

        cls = "step-item"
        if is_active:
            cls += " active"
        elif is_completed:
            cls += " completed"

        pipeline_html += f'<span class="{cls}">{s_label}</span>'
        if idx < len(stages) - 1:
            pipeline_html += '<span class="step-arrow">→</span>'
    pipeline_html += '</div>'

    info = STAGE_HEADER_INFO.get(current_stage, ("Stage", "Architectural Workspace", "SPAZE GenAI Workspace", "📐"))
    tag_prefix, title, subtitle, icon = info

    st.markdown(
        f"""
        <div class="spaze-top-bar">
            <div class="stage-title-wrap">
                <div class="stage-icon-box">{icon}</div>
                <div>
                    <div class="stage-title-text">{tag_prefix}: {title}</div>
                    <div class="stage-subtitle-text">{subtitle}</div>
                </div>
            </div>
            {pipeline_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(
    title: str,
    value: str,
    subtitle: str = "",
    val_color: str = "#19213A",
) -> None:
    """Render a styled compact architectural metric card."""
    st.markdown(
        f"""
        <div class="ref-metric-card">
            <div>
                <div class="ref-metric-label">{title}</div>
                <div class="ref-metric-value" style="color: {val_color};">{value}</div>
            </div>
            <div style="font-size:0.70rem; color:#8E98AF; margin-top:0.25rem;">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_reference_metric(
    icon: str,
    icon_type: str,  # "purple" | "blue" | "green" | "amber" | "cyan"
    value: str,
    label: str,
    badge_html: str = "",
) -> None:
    """Render a single metric card strictly matching the reference image layout."""
    st.markdown(
        f"""
        <div class="ref-metric-card">
            <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                <div class="ref-metric-icon-box ref-metric-icon-{icon_type}">{icon}</div>
                {badge_html}
            </div>
            <div>
                <div class="ref-metric-value">{value}</div>
                <div class="ref-metric-label">{label}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_page_header(stage_tag: str, title: str, subtitle: str) -> None:
    """Fallback page header helper for backward compatibility."""
    pass
