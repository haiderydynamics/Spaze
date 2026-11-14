"""Stage 05 — Project Export Hub & Deliverables Center.

Provides unified, production-grade export downloads for active architectural designs:
Binary 3D Model (.glb), Structured Layout Schema (.json), Comprehensive 3D Manifest (.json),
2D Blueprint Preview (.png), and CVAE Training Telemetry (.csv).
"""
from __future__ import annotations

import io
import json
from pathlib import Path
import streamlit as st

import geometry3d
from cvae.config import CVAEConfig
from geometry3d.exporter import ModelExporter
from geometry3d.model_generator import generate_3d_model
from models.floorplan_schema import FloorPlan
from ui.components import render_page_header, safe_html
from ui.state import StateManager
from utils.export import export_floorplan_to_json, generate_export_summary


def render_export_page() -> None:
    """Render the Stage 05 Project Export Hub workspace matching reference design."""
    render_page_header(
        stage_tag="Stage 05",
        title="Project Export Hub",
        subtitle="Download production-ready architectural deliverables in industry-standard formats including binary glTF 2.0 (.glb), structured schemas (.json), and high-resolution blueprints.",
    )

    active_fp = StateManager.get_selected_alternative()

    if active_fp is None:
        safe_html(
            """
            <div class="section-card" style="text-align: center; padding: 2.5rem 1.5rem;">
                <div style="font-size: 2.2rem; margin-bottom: 0.6rem;">📦</div>
                <div style="font-size: 1.1rem; font-weight: 700; color: #171A2B;">No Active Architectural Design in Memory</div>
                <div style="font-size: 0.85rem; color: #69738B; margin-top: 0.4rem; max-width: 520px; margin-left: auto; margin-right: auto; margin-bottom: 1.2rem;">
                    Please upload or load a floor plan in Stage 01 to generate downloadable CAD, 3D mesh, and telemetry deliverables.
                </div>
            </div>
            """
        )
        if st.button("⬅️ Go to Stage 01: Floor Plan Studio", type="primary"):
            StateManager.set_stage("01_floorplan")
            st.rerun()
        return

    # 1. Active Design Context Banner
    rooms = active_fp.confirmed_rooms + active_fp.candidate_rooms
    source_tag = f"Alternative {st.session_state.get('selected_alt_idx', 0) + 1} ({active_fp.preference.replace('_', ' ').title()})" if active_fp.is_generated else "Source Floor Plan"

    safe_html(
        f"""
        <div class="section-card" style="padding: 1.1rem 1.4rem; margin-bottom: 1.4rem;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <span style="font-size: 0.76rem; color: #69738B; font-weight: 700; text-transform: uppercase; letter-spacing: 0.6px;">Active Export Target</span>
                    <div style="font-size: 1.25rem; font-weight: 800; color: #171A2B; margin-top: 0.2rem;">
                        {source_tag}
                    </div>
                </div>
                <span class="badge badge-purple">{len(rooms)} Rooms • {active_fp.image_width}×{active_fp.image_height} px</span>
            </div>
            <div style="font-size: 0.82rem; color: #69738B; margin-top: 0.4rem;">
                Metric Scale: <b>{active_fp.scale_information.pixels_per_meter or 50.0:.1f} px/m</b> &nbsp;|&nbsp;
                Preference Mode: <b>{(active_fp.preference or 'source').replace('_', ' ').title()}</b>
            </div>
        </div>
        """
    )

    # 2. Multi-Format Deliverables Grid (6 Cards)
    safe_html(
        """
        <div style="font-size:0.78rem; font-weight:700; text-transform:uppercase; color:#69738B; letter-spacing:0.8px; margin-bottom:0.75rem;">
            Available Project Deliverables
        </div>
        """
    )

    row1_c1, row1_c2, row1_c3 = st.columns(3, gap="medium")

    # Card 1: 3D GLB Model
    with row1_c1:
        safe_html(
            """
            <div class="section-card" style="min-height: 250px; display: flex; flex-direction: column; justify-content: space-between; margin-bottom: 0.5rem;">
                <div>
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
                        <span style="font-size: 1.6rem;">🧊</span>
                        <span class="badge badge-purple">GLB 2.0</span>
                    </div>
                    <div style="font-weight: 800; font-size: 1.05rem; color: #171A2B;">Generated 3D Model</div>
                    <div style="color: #69738B; font-size: 0.80rem; margin-top: 0.35rem; line-height: 1.45;">
                        Watertight manifold 3D mesh with genuine door/window apertures, floor finishes, and procedural furniture.
                    </div>
                </div>
                <div style="margin-top: 0.8rem;">
                    <span class="badge badge-green">Ready for Download</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        active_3d = st.session_state.get("active_3d_result")
        if active_3d is not None:
            glb_bytes = active_3d.get_glb_bytes()
        else:
            with st.spinner("Synthesizing 3D GLB model..."):
                res = generate_3d_model(active_fp)
                glb_bytes = res.get_glb_bytes()

        st.download_button(
            label="📥 Download 3D Model (.glb)",
            data=glb_bytes,
            file_name=f"spaze_{active_fp.preference or 'source'}_model.glb",
            mime="model/gltf-binary",
            use_container_width=True,
            type="primary",
            help="Compatible with Blender, Unity, Unreal Engine, Three.js, and WebGL viewers.",
        )

    # Card 2: 2D Blueprint Layout PNG
    with row1_c2:
        safe_html(
            """
            <div class="section-card" style="min-height: 250px; display: flex; flex-direction: column; justify-content: space-between; margin-bottom: 0.5rem;">
                <div>
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
                        <span style="font-size: 1.6rem;">📐</span>
                        <span class="badge badge-purple">PNG Image</span>
                    </div>
                    <div style="font-weight: 800; font-size: 1.05rem; color: #171A2B;">Architectural Blueprint</div>
                    <div style="color: #69738B; font-size: 0.80rem; margin-top: 0.35rem; line-height: 1.45;">
                        High-resolution 2D architectural blueprint rendering with color-coded rooms, labels, and dimensions.
                    </div>
                </div>
                <div style="margin-top: 0.8rem;">
                    <span class="badge badge-green">Ready for Download</span>
                </div>
            </div>
            """
        )

        png_bytes = ModelExporter.export_floorplan_preview(active_fp)
        st.download_button(
            label="📥 Download Blueprint (.png)",
            data=png_bytes,
            file_name=f"spaze_{active_fp.preference or 'source'}_blueprint.png",
            mime="image/png",
            use_container_width=True,
            help="High-resolution presentation blueprint image.",
        )

    # Card 3: Structured Schema JSON
    with row1_c3:
        safe_html(
            """
            <div class="section-card" style="min-height: 250px; display: flex; flex-direction: column; justify-content: space-between; margin-bottom: 0.5rem;">
                <div>
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
                        <span style="font-size: 1.6rem;">📄</span>
                        <span class="badge badge-purple">JSON Schema</span>
                    </div>
                    <div style="font-weight: 800; font-size: 1.05rem; color: #171A2B;">Structured Floor Plan</div>
                    <div style="color: #69738B; font-size: 0.80rem; margin-top: 0.35rem; line-height: 1.45;">
                        Standardized SPAZE JSON schema encoding room bounding coordinates, physical metrics, and topological adjacencies.
                    </div>
                </div>
                <div style="margin-top: 0.8rem;">
                    <span class="badge badge-green">Ready for Download</span>
                </div>
            </div>
            """
        )

        json_str = export_floorplan_to_json(active_fp)
        st.download_button(
            label="📥 Download Schema (.json)",
            data=json_str,
            file_name=f"spaze_{active_fp.preference or 'source'}_schema.json",
            mime="application/json",
            use_container_width=True,
            help="Structured architectural data contract.",
        )

    st.write("")
    row2_c1, row2_c2, row2_c3 = st.columns(3, gap="medium")

    # Card 4: Architectural Metadata Manifest
    with row2_c1:
        safe_html(
            """
            <div class="section-card" style="min-height: 250px; display: flex; flex-direction: column; justify-content: space-between; margin-bottom: 0.5rem;">
                <div>
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
                        <span style="font-size: 1.6rem;">📋</span>
                        <span class="badge badge-blue">JSON Manifest</span>
                    </div>
                    <div style="font-weight: 800; font-size: 1.05rem; color: #171A2B;">Architectural Metadata</div>
                    <div style="color: #69738B; font-size: 0.80rem; margin-top: 0.35rem; line-height: 1.45;">
                        3D specification manifest recording triangle budgets, vertex counts, room area schedules, and wall lengths.
                    </div>
                </div>
                <div style="margin-top: 0.8rem;">
                    <span class="badge badge-green">Ready for Download</span>
                </div>
            </div>
            """
        )

        cfg_geo = geometry3d.Geometry3DConfig()
        if active_3d is not None:
            manifest_data = active_3d.manifest
        else:
            manifest_data = ModelExporter.export_manifest(res.scene, active_fp, cfg_geo)

        manifest_json_str = json.dumps(manifest_data, indent=2)
        st.download_button(
            label="📥 Download Manifest (.json)",
            data=manifest_json_str,
            file_name=f"spaze_{active_fp.preference or 'source'}_manifest.json",
            mime="application/json",
            use_container_width=True,
        )

    # Card 5: Evaluation Report & Telemetry CSV
    with row2_c2:
        history_path = Path("training_logs/training_history.csv")
        has_csv = history_path.is_file()

        safe_html(
            f"""
            <div class="section-card" style="min-height: 250px; display: flex; flex-direction: column; justify-content: space-between; margin-bottom: 0.5rem;">
                <div>
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
                        <span style="font-size: 1.6rem;">📊</span>
                        <span class="badge badge-blue">CSV Log</span>
                    </div>
                    <div style="font-weight: 800; font-size: 1.05rem; color: #171A2B;">Training Telemetry</div>
                    <div style="color: #69738B; font-size: 0.80rem; margin-top: 0.35rem; line-height: 1.45;">
                        Epoch-by-epoch loss progression, reconstruction Smooth L1 metrics, and KL divergence history across 45 epochs.
                    </div>
                </div>
                <div style="margin-top: 0.8rem;">
                    <span class="badge {'badge-green' if has_csv else 'badge-amber'}">
                        {'Ready for Download' if has_csv else 'Not Available'}
                    </span>
                </div>
            </div>
            """
        )

        if has_csv:
            with open(history_path, "r", encoding="utf-8") as f:
                csv_data = f.read()
            st.download_button(
                label="📥 Download Telemetry (.csv)",
                data=csv_data,
                file_name="spaze_cvae_training_history.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.button("Telemetry Unavailable", disabled=True, use_container_width=True)

    # Card 6: Academic Research Documentation
    with row2_c3:
        safe_html(
            """
            <div class="section-card" style="min-height: 250px; display: flex; flex-direction: column; justify-content: space-between; margin-bottom: 0.5rem;">
                <div>
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
                        <span style="font-size: 1.6rem;">📚</span>
                        <span class="badge badge-purple">Markdown Docs</span>
                    </div>
                    <div style="font-weight: 800; font-size: 1.05rem; color: #171A2B;">Academic Research Suite</div>
                    <div style="color: #69738B; font-size: 0.80rem; margin-top: 0.35rem; line-height: 1.45;">
                        Complete 17-chapter technical report, mathematical derivations, problem statement, and presentation script.
                    </div>
                </div>
                <div style="margin-top: 0.8rem;">
                    <span class="badge badge-green">Ready for Download</span>
                </div>
            </div>
            """
        )

        abstract_path = Path("docs/project_abstract.md")
        if abstract_path.is_file():
            with open(abstract_path, "r", encoding="utf-8") as f:
                doc_text = f.read()
            st.download_button(
                label="📥 Download Abstract (.md)",
                data=doc_text,
                file_name="spaze_project_abstract.md",
                mime="text/markdown",
                use_container_width=True,
            )
        else:
            st.button("Docs Available in Stage 06", use_container_width=True, disabled=True)
