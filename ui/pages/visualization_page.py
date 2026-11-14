"""Stage 03 — Intelligent 3D Visualization Workspace.

Interactive architectural 3D studio transforming 2D floor plans and preference-conditioned
CVAE layout alternatives into watertight, physical 3D building models.
Features real-time WebGL orbit/pan/zoom, granular semantic layer toggles, camera presets,
room-level hover diagnostics, and multi-format GLB/JSON/PNG export.
"""
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

import geometry3d
from cvae.visualization import render_single_floorplan_axes
from geometry3d.config import Geometry3DConfig
from geometry3d.exporter import ModelExporter
from geometry3d.model_generator import Model3DResult, generate_3d_model
from geometry3d.viewer_adapter import ViewerAdapter
from models.floorplan_schema import FloorPlan, Room, ScaleInfo
from ui.components import render_metric_card, render_page_header, safe_html
from ui.state import StateManager


def _create_demo_floorplan() -> FloorPlan:
    """Generate a realistic 4-room residential apartment layout for instant demonstration."""
    scale = ScaleInfo()
    scale.calibrate(pixel_distance=200.0, real_meters=4.0)  # 50 px/m

    rooms = [
        Room(
            room_id="r_liv",
            room_type="living_room",
            label="Living Room",
            x=100.0,
            y=380.0,
            width=420.0,
            height=320.0,
            physical_width=8.4,
            physical_height=6.4,
            physical_area=53.76,
        ),
        Room(
            room_id="r_kit",
            room_type="kitchen",
            label="Kitchen & Dining",
            x=540.0,
            y=380.0,
            width=320.0,
            height=320.0,
            physical_width=6.4,
            physical_height=6.4,
            physical_area=40.96,
        ),
        Room(
            room_id="r_bed",
            room_type="bedroom",
            label="Master Suite",
            x=100.0,
            y=80.0,
            width=420.0,
            height=280.0,
            physical_width=8.4,
            physical_height=5.6,
            physical_area=47.04,
        ),
        Room(
            room_id="r_bath",
            room_type="bathroom",
            label="Master Bath",
            x=540.0,
            y=80.0,
            width=320.0,
            height=280.0,
            physical_width=6.4,
            physical_height=5.6,
            physical_area=35.84,
        ),
    ]

    return FloorPlan(
        image_width=960,
        image_height=780,
        scale_information=scale,
        confirmed_rooms=rooms,
    )


def render_visualization_page() -> None:
    """Render the Stage 03 3D Visualization workspace matching the reference design."""
    render_page_header(
        stage_tag="Stage 03",
        title="Intelligent 3D Visualization",
        subtitle="Explore your AI-generated architectural layouts in an interactive three-dimensional environment.",
    )

    # 1. Active Floor Plan Verification
    has_fp = StateManager.has_floorplan()
    has_alts = StateManager.has_alternatives()

    if not has_fp and not has_alts:
        safe_html(
            """
            <div class="section-card" style="text-align: center; padding: 2.5rem 1.5rem;">
                <div style="font-size: 2.2rem; margin-bottom: 0.6rem;">🧊</div>
                <div style="font-size: 1.1rem; font-weight: 700; color: #171A2B;">No Active Floor Plan in Memory</div>
                <div style="font-size: 0.85rem; color: #69738B; margin-top: 0.4rem; max-width: 520px; margin-left: auto; margin-right: auto; margin-bottom: 1.2rem;">
                    Please upload an image in Stage 01 or load the pre-built residential demo layout below to inspect 3D geometry immediately.
                </div>
            </div>
            """
        )
        d_col1, d_col2 = st.columns([1, 2])
        with d_col1:
            if st.button("📂 Load Residential Demo Layout", type="primary", use_container_width=True):
                demo_fp = _create_demo_floorplan()
                st.session_state.floorplan = demo_fp
                st.session_state.sample_loaded = True
                st.rerun()
        with d_col2:
            if st.button("⬅️ Return to Stage 01: Floor Plan", use_container_width=True):
                StateManager.set_stage("01_floorplan")
                st.rerun()
        return

    # 2. Layout Selection Bar (Source vs CVAE Alternatives)
    top_c1, top_c2 = st.columns([2.5, 1.5], gap="large")

    with top_c1:
        alt_options: List[str] = []
        if has_fp:
            alt_options.append("Source Floor Plan")
        if has_alts:
            alts = st.session_state.get("cvae_alternatives", [])
            for idx, alt in enumerate(alts):
                pref_name = (alt.preference or f"Alternative {idx + 1}").replace("_", " ").title()
                alt_options.append(f"Alternative {idx + 1}: {pref_name}")

        curr_alt_idx = st.session_state.get("selected_alt_idx", 0)
        default_index = 0
        if has_alts and has_fp:
            default_index = min(curr_alt_idx + 1, len(alt_options) - 1)
        elif has_alts:
            default_index = min(curr_alt_idx, len(alt_options) - 1)

        selected_opt = st.selectbox(
            "Select Layout Variant to Inhabit in 3D:",
            options=alt_options,
            index=default_index,
            help="Select the verified input floor plan or any of the 3 CVAE-generated alternatives.",
        )

        if selected_opt == "Source Floor Plan":
            target_floorplan: FloorPlan = st.session_state.floorplan
            current_layout_key = "source"
        else:
            alt_num = int(selected_opt.split(":")[0].replace("Alternative", "").strip()) - 1
            st.session_state.selected_alt_idx = alt_num
            target_floorplan = st.session_state.cvae_alternatives[alt_num]
            current_layout_key = f"alt_{alt_num}_{target_floorplan.preference}"

    with top_c2:
        rooms = target_floorplan.confirmed_rooms + target_floorplan.candidate_rooms
        scale_label = f"{target_floorplan.scale_information.pixels_per_meter:.1f} px/m" if target_floorplan.scale_information.is_calibrated else "Default (50 px/m)"
        safe_html(
            f"""
            <div style="background: #FFFFFF; border: 1px solid #E9E7F2; border-radius: 12px; padding: 0.75rem 1.0rem; font-size: 0.82rem; box-shadow: 0 1px 3px rgba(23,26,43,0.02); margin-top: 1.45rem;">
                <span style="color:#69738B; font-weight:600;">Active Program:</span> <b style="color:#171A2B;">{len(rooms)} Rooms</b> &nbsp;|&nbsp; 
                <span style="color:#69738B; font-weight:600;">Scale:</span> <b style="color:#171A2B;">{scale_label}</b><br>
                <span style="color:#69738B; font-weight:600;">Preference:</span> <b style="color:#7047F5;">{(target_floorplan.preference or 'Source').replace('_', ' ').title()}</b>
            </div>
            """
        )

    # 3. Main Workspace: LEFT (2D Reference), CENTER (3D Canvas), RIGHT (Controls)
    st.markdown("---")
    col_2d, col_3d, col_config = st.columns([1.1, 2.4, 1.1], gap="medium")

    with col_2d:
        safe_html(
            """
            <div style="font-size:0.78rem; font-weight:700; text-transform:uppercase; color:#69738B; letter-spacing:0.8px; margin-bottom:0.4rem;">
                2D Layout Reference
            </div>
            """
        )
        fig_2d, ax_2d = plt.subplots(figsize=(4.2, 3.8), facecolor="#FFFFFF")
        title_tag = f"Alt {st.session_state.get('selected_alt_idx', 0) + 1}" if target_floorplan.is_generated else "Source"
        render_single_floorplan_axes(ax_2d, target_floorplan, title=f"2D Blueprint ({title_tag})")
        st.pyplot(fig_2d, use_container_width=True)
        plt.close(fig_2d)

        with st.expander("📋 Room Inventory Schedule", expanded=False):
            for r in rooms:
                dim_str = f"{r.physical_width:.1f}×{r.physical_height:.1f} m ({r.physical_area:.1f} m²)" if r.physical_width else f"{int(r.width)}×{int(r.height)} px"
                st.markdown(f"• **{r.label or r.room_type}**: {dim_str}")

    with col_config:
        safe_html(
            """
            <div style="font-size:0.78rem; font-weight:700; text-transform:uppercase; color:#69738B; letter-spacing:0.8px; margin-bottom:0.4rem;">
                3D Model Parameters
            </div>
            """
        )
        wall_height = st.slider(
            "Wall Height (m)",
            min_value=2.2,
            max_value=4.2,
            value=float(st.session_state.get("wall_height_3d", 2.80)),
            step=0.1,
            help="Ceiling and wall height in real-world meters.",
        )
        st.session_state.wall_height_3d = wall_height

        ext_thick = st.slider(
            "Exterior Wall (m)",
            min_value=0.15,
            max_value=0.40,
            value=float(st.session_state.get("wall_thickness_3d", 0.20)),
            step=0.01,
            help="Perimeter envelope wall thickness.",
        )
        st.session_state.wall_thickness_3d = ext_thick

        int_thick = st.slider(
            "Interior Wall (m)",
            min_value=0.08,
            max_value=0.20,
            value=0.12,
            step=0.01,
            help="Interior room partition thickness.",
        )

        safe_html(
            """
            <div style="font-size:0.78rem; font-weight:700; text-transform:uppercase; color:#69738B; letter-spacing:0.8px; margin-top:0.8rem; margin-bottom:0.4rem;">
                Semantic Layers
            </div>
            """
        )
        show_floors = st.checkbox("Floors & Foundation", value=True)
        show_ext_walls = st.checkbox("Exterior Envelope Walls", value=True)
        show_int_walls = st.checkbox("Interior Partition Walls", value=True)
        show_doors = st.checkbox("Door Openings & Frames", value=True)
        show_windows = st.checkbox("Window Openings & Glass", value=True)
        show_furniture = st.checkbox("Procedural Furniture", value=st.session_state.get("show_furniture_3d", True))
        st.session_state.show_furniture_3d = show_furniture

        cam_mode = st.selectbox(
            "Camera Perspective",
            options=["Isometric View", "Perspective View", "Top-Down 3D View"],
            index=0,
        )

        viewer_theme = st.selectbox(
            "Viewer Canvas Theme",
            options=["Light", "Dark"],
            index=0,
        )

        st.write("")
        trigger_regen = st.button("🔨 Recompute 3D Geometry", type="primary", use_container_width=True)

    # 4. Geometry Generation & Session Cache
    cache = st.session_state.get("scene_cache", {})
    cache_key = f"{current_layout_key}_{wall_height}_{ext_thick}_{int_thick}_{show_furniture}_{show_doors}_{show_windows}"

    if trigger_regen or cache_key not in cache:
        with st.spinner("Synthesizing 3D walls, floor slabs, physical apertures, and procedural furniture..."):
            cfg = Geometry3DConfig(
                default_wall_height=wall_height,
                exterior_wall_thickness=ext_thick,
                interior_wall_thickness=int_thick,
                generate_furniture=show_furniture,
                generate_doors=show_doors,
                generate_windows=show_windows,
            )
            result = generate_3d_model(target_floorplan, cfg, include_furniture=show_furniture)
            cache[cache_key] = result
            st.session_state.scene_cache = cache
            st.session_state.active_3d_result = result
    else:
        result = cache[cache_key]
        st.session_state.active_3d_result = result

    with col_3d:
        safe_html(
            """
            <div style="font-size:0.78rem; font-weight:700; text-transform:uppercase; color:#69738B; letter-spacing:0.8px; margin-bottom:0.4rem;">
                Interactive 3D Architectural Model
            </div>
            """
        )
        if result is not None:
            fig = ViewerAdapter.scene_to_plotly(
                scene=result.scene,
                camera_mode=cam_mode,
                theme=viewer_theme.lower(),
                show_floors=show_floors,
                show_exterior_walls=show_ext_walls,
                show_interior_walls=show_int_walls,
                show_doors=show_doors,
                show_windows=show_windows,
                show_furniture=show_furniture,
            )
            fig.update_layout(height=540)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("🖱️ **Controls**: Click & Drag to Orbit | Scroll to Zoom | Right-Click to Pan | Click Legend Items to Toggle Visibility")
        else:
            st.error("Could not construct 3D scene from active layout.")

    # 5. Diagnostic Metrics & Geometric Feasibility
    if result is not None:
        st.markdown("---")
        safe_html(
            """
            <div style="font-size:0.78rem; font-weight:700; text-transform:uppercase; color:#69738B; letter-spacing:0.8px; margin-bottom:0.6rem;">
                Architectural Geometry Diagnostics
            </div>
            """
        )
        m = result.metrics

        m1, m2, m3, m4, m5, m6 = st.columns(6)
        with m1:
            render_metric_card("Triangles", f"{m.total_triangles:,}", "Polygon count", val_color="#7047F5")
        with m2:
            render_metric_card("Vertices", f"{m.total_vertices:,}", "Vertex nodes", val_color="#171A2B")
        with m3:
            render_metric_card("Wall Length", f"{m.total_wall_length_m:.1f} m", "Total perimeter", val_color="#171A2B")
        with m4:
            area_str = f"{m.usable_floor_area_sqm:.1f} m²" if m.usable_floor_area_sqm > 0 else "N/A"
            render_metric_card("Usable Area", area_str, "Floor surfaces", val_color="#19B978")
        with m5:
            render_metric_card("Apertures", f"{m.door_count}D / {m.window_count}W", "Doors & windows", val_color="#388BFF")
        with m6:
            render_metric_card("Synthesis", f"{m.generation_time_ms:.1f} ms", "Mesh latency", val_color="#7047F5")

        # 6. Room Schedule Table
        st.markdown("---")
        st.markdown("#### 📋 3D Space Program Schedule")
        schedule_data = []
        for r in rooms:
            schedule_data.append({
                "Room ID": r.room_id,
                "Type": r.room_type.replace("_", " ").title(),
                "Label": r.label or r.room_type,
                "Width (m)": f"{r.physical_width:.2f}" if r.physical_width else "-",
                "Depth (m)": f"{r.physical_height:.2f}" if r.physical_height else "-",
                "Area (m²)": f"{r.physical_area:.2f}" if r.physical_area else "-",
                "Ceiling Height": f"{wall_height:.2f} m",
                "Status": "Synthesized 3D",
            })
        st.dataframe(schedule_data, use_container_width=True, hide_index=True)

        # 7. Multi-Format Architectural Deliverables
        st.markdown("---")
        st.markdown("#### 📦 Multi-Format Architectural Deliverables")
        exp1, exp2, exp3 = st.columns(3)

        with exp1:
            glb_data = result.get_glb_bytes()
            st.download_button(
                label="📥 Download 3D Model (.glb)",
                data=glb_data,
                file_name=f"spaze_{current_layout_key}_model.glb",
                mime="model/gltf-binary",
                use_container_width=True,
                type="primary",
                help="Binary GLTF 2.0 file ready for Blender, Unreal Engine, Unity, Three.js, and WebGL viewers.",
            )

        with exp2:
            manifest_json = json.dumps(result.manifest, indent=2)
            st.download_button(
                label="📥 Download Manifest (.json)",
                data=manifest_json,
                file_name=f"spaze_{current_layout_key}_manifest.json",
                mime="application/json",
                use_container_width=True,
                help="Structured architectural specification manifest with dimensions, areas, and triangle budgets.",
            )

        with exp3:
            png_bytes = ModelExporter.export_floorplan_preview(target_floorplan)
            st.download_button(
                label="📥 Download Blueprint (.png)",
                data=png_bytes,
                file_name=f"spaze_{current_layout_key}_blueprint.png",
                mime="image/png",
                use_container_width=True,
                help="High-resolution 2D architectural blueprint layout with dimension annotations.",
            )

    # 8. Navigation Footer
    st.markdown("---")
    nav1, nav2, nav3 = st.columns([1.5, 2, 1.5])
    with nav1:
        if st.button("⬅️ Stage 02: AI Generation", use_container_width=True):
            StateManager.set_stage("02_generation")
            st.rerun()
    with nav3:
        if st.button("👉 Stage 04: Evaluation Lab", type="primary", use_container_width=True):
            StateManager.set_stage("04_evaluation")
            st.rerun()
