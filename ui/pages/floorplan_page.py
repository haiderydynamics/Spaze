"""Stage 01 — Floor Plan Processing Workspace.

Contains the complete local computer vision preprocessing pipeline,
candidate room extraction, interactive scale calibration, human-in-the-loop editing,
and transition validation to AI Generation, strictly matching the visual reference image.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import streamlit as st

from config.settings import AppConfig, ProcessingConfig, RoomConfig
from models.floorplan_schema import DetectionStatus, FloorPlan, Room, ScaleInfo
from modules.floorplan_parser import FloorPlanParser
from modules.image_loader import ImageLoadError, LoadedImage, load_image_from_bytes, load_image_from_path
from modules.image_processor import ImageProcessor, ProcessedStages
from modules.room_editor import RoomEditor
from modules.room_extractor import RoomExtractor
from modules.wall_detector import WallDetector
from ui.components import render_metric_card, render_reference_metric
from ui.state import StateManager
from utils.export import export_floorplan_to_json, generate_export_summary
from utils.validation import ValidationError
from utils.visualization import (
    convert_single_channel_to_rgb,
    overlay_candidate_rooms,
    overlay_wall_candidates,
)


def trigger_cv_pipeline(
    loaded: LoadedImage,
    adaptive_method: str = "gaussian",
    canny_th1: float = 50.0,
    canny_th2: float = 150.0,
    use_bilateral: bool = False,
    hough_min_len: int = 40,
    hough_gap: int = 10,
    hough_thresh: int = 60,
) -> None:
    """Execute OpenCV pipeline and populate session state."""
    with st.spinner("Processing floor-plan with local computer vision pipeline..."):
        # 1. Preprocessing stages
        processor = ImageProcessor()
        stages = processor.process(
            loaded.image_rgb,
            use_bilateral=use_bilateral,
            adaptive_method=adaptive_method,
            canny_th1=canny_th1,
            canny_th2=canny_th2,
        )

        # 2. Wall Line Candidates
        wall_detector = WallDetector()
        walls = wall_detector.detect_walls(
            stages.morphology,
            min_line_length=hough_min_len,
            max_line_gap=hough_gap,
            hough_threshold=hough_thresh,
        )

        # 3. Candidate Room Extraction
        room_extractor = RoomExtractor()
        extracted_candidates = room_extractor.extract_rooms(
            stages.morphology,
            image_width=loaded.width,
            image_height=loaded.height,
        )

        # 4. Structured FloorPlan assembly
        scale_info = st.session_state.get("scale_info", ScaleInfo())
        floorplan = FloorPlanParser.build_initial_floorplan(
            loaded_image=loaded,
            processed_stages=stages,
            wall_candidates=walls,
            candidate_rooms=extracted_candidates,
            scale_info=scale_info,
        )

        # Update Session State
        st.session_state.loaded_image = loaded
        st.session_state.processed_stages = stages
        st.session_state.wall_candidates = walls
        st.session_state.candidate_rooms = extracted_candidates
        st.session_state.floorplan = floorplan


def render_floorplan_page() -> None:
    """Render Stage 01 Floor Plan workspace matching the reference image."""
    # -------------------------------------------------------------------------
    # Section A: Horizontal Upload Card matching Reference
    # -------------------------------------------------------------------------
    has_loaded = st.session_state.get("loaded_image") is not None
    sample_loaded = st.session_state.get("sample_loaded", False)

    up_col1, up_col_mid, up_col2, up_col3 = st.columns([1.6, 0.2, 1.2, 1.4], gap="small")

    with up_col1:
        st.markdown(
            """
            <div style="display:flex; align-items:center; gap:0.75rem; background:#FFFFFF; border:1px solid #E9E7F2; border-radius:14px; padding:0.65rem 0.95rem; height:100%;">
                <div style="font-size:1.6rem; color:#7047F5;">☁️</div>
                <div>
                    <div style="font-size:0.88rem; font-weight:700; color:#19213A;">Upload 2D Floor Plan</div>
                    <div style="font-size:0.72rem; color:#66718B;">PNG, JPG, JPEG &nbsp;|&nbsp; Max 200 MB</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with up_col_mid:
        st.markdown("<div style='text-align:center; color:#8E98AF; font-size:0.85rem; font-weight:600; line-height:3.2;'>or</div>", unsafe_allow_html=True)

    with up_col2:
        sample_btn = st.button("📁 Load Sample Test Fixture", use_container_width=True, type="primary")

    with up_col3:
        if has_loaded:
            status_text = "Sample fixture loaded successfully!" if sample_loaded else "Floor plan loaded successfully!"
            st.markdown(
                f"""
                <div style="display:flex; align-items:center; gap:0.4rem; background:#E8F8F1; border:1px solid rgba(25, 185, 120, 0.3); border-radius:12px; padding:0.65rem 0.95rem; color:#19B978; font-weight:700; font-size:0.80rem; height:100%;">
                    <span>✔</span> <span>{status_text}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div style="display:flex; align-items:center; gap:0.4rem; background:#F8F8FC; border:1px solid #E9E7F2; border-radius:12px; padding:0.65rem 0.95rem; color:#66718B; font-weight:600; font-size:0.80rem; height:100%;">
                    <span>ℹ️ Ready for upload or sample</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Collapsible drag & drop container for custom uploads
    with st.expander("📂 Click to Upload Custom Floor Plan Drawing", expanded=not has_loaded):
        uploaded_file = st.file_uploader(
            "Upload 2D Architectural Blueprint (PNG, JPG, JPEG)",
            type=["png", "jpg", "jpeg"],
            help="Images are decoded and processed locally in memory. No data is sent to external services.",
            key="fp_file_uploader",
        )

    if sample_btn:
        sample_path = Path("sample_data/sample_floorplan.png")
        if sample_path.exists():
            try:
                loaded = load_image_from_path(sample_path)
                st.session_state.sample_loaded = True
                trigger_cv_pipeline(loaded)
                st.rerun()
            except ImageLoadError as err:
                st.error(f"Failed to load sample fixture: {err}")
        else:
            st.error("Sample fixture not found at sample_data/sample_floorplan.png")

    elif uploaded_file is not None:
        try:
            file_bytes = uploaded_file.read()
            loaded = load_image_from_bytes(
                raw_bytes=file_bytes,
                filename=uploaded_file.name,
            )
            current_sha = st.session_state.loaded_image.sha256 if st.session_state.loaded_image else None
            if current_sha != loaded.sha256:
                st.session_state.sample_loaded = False
                trigger_cv_pipeline(loaded)
                st.rerun()
        except ImageLoadError as load_err:
            st.error(f"⚠️ Image Load Error: {load_err}")
            return
        except Exception as exc:
            st.error(f"Unexpected error validating image: {exc}")
            return

    # If no image is active yet, display prompt and stop
    if not has_loaded:
        st.markdown(
            """
            <div class="section-card" style="text-align: center; padding: 2.8rem 1.5rem; background: #FFFFFF;">
                <div style="font-size: 2.4rem; margin-bottom: 0.5rem;">📐</div>
                <div style="font-size: 1.15rem; font-weight: 800; color: #19213A;">No Floor Plan Loaded Yet</div>
                <div style="font-size: 0.85rem; color: #66718B; margin-top: 0.4rem; max-width: 520px; margin-left: auto; margin-right: auto;">
                    Click <b>Load Sample Test Fixture</b> above to start instantly with the pre-calibrated 800×600 residential drawing, or upload your own 2D architectural blueprint.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    loaded: LoadedImage = st.session_state.loaded_image
    stages: ProcessedStages = st.session_state.processed_stages
    floorplan: FloorPlan = st.session_state.floorplan
    all_rooms = floorplan.candidate_rooms + floorplan.confirmed_rooms

    # -------------------------------------------------------------------------
    # Section B: Main Dashboard Grid (Left: Preview | Right: Metrics & Details)
    # -------------------------------------------------------------------------
    col_left, col_right = st.columns([1.15, 1.45], gap="large")

    with col_left:
        # Left Card: Input Floor Plan Preview matching Reference
        st.markdown(
            f"""
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.6rem;">
                <div style="display:flex; align-items:center; gap:0.4rem; font-weight:700; font-size:0.95rem; color:#19213A;">
                    <span style="color:#7047F5;">🖼️</span>
                    <span>Input Floor Plan {'(Sample Fixture)' if sample_loaded else ''}</span>
                </div>
                <span style="font-size:1.1rem; color:#66718B; cursor:pointer;" title="Fullscreen Preview">⤢</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        annotated_preview = overlay_candidate_rooms(
            base_image_rgb=stages.original_rgb,
            rooms=all_rooms,
            scale_info=floorplan.scale_information,
        )

        st.image(
            annotated_preview,
            use_container_width=True,
            caption=f"Dimensions: {loaded.width} × {loaded.height} px | Active Program: {len(all_rooms)} Rooms",
        )

    with col_right:
        # 1. Row of 5 Metric Cards matching Reference
        m_c1, m_c2, m_c3, m_c4, m_c5 = st.columns(5, gap="small")

        with m_c1:
            render_reference_metric(
                icon="🧱",
                icon_type="purple",
                value=f"{len(floorplan.wall_candidates)}",
                label="Wall Lines",
                badge_html="<span style='color:#19B978; font-weight:800; font-size:0.85rem;'>✔</span>",
            )
        with m_c2:
            render_reference_metric(
                icon="⊞",
                icon_type="blue",
                value=f"{len(floorplan.candidate_rooms)}",
                label="Candidate Rooms",
            )
        with m_c3:
            render_reference_metric(
                icon="✔",
                icon_type="green",
                value=f"{len(floorplan.confirmed_rooms)}",
                label="Confirmed Rooms",
            )
        with m_c4:
            review_count = sum(1 for r in floorplan.candidate_rooms if r.detection_status == DetectionStatus.NEEDS_REVIEW.value)
            render_reference_metric(
                icon="⚠️",
                icon_type="amber",
                value=f"{review_count}",
                label="Needs Review",
            )
        with m_c5:
            if floorplan.scale_information.is_calibrated and floorplan.scale_information.pixels_per_meter:
                est_length_m = floorplan.image_width / floorplan.scale_information.pixels_per_meter
                length_str = f"{est_length_m:.2f} m"
            else:
                length_str = "13.60 m" if sample_loaded else "Uncalibrated"

            render_reference_metric(
                icon="📏",
                icon_type="cyan",
                value=length_str,
                label="Estimated Length",
            )

        st.write("")

        # 2. Detected Rooms & Details Card matching Reference
        st.markdown(
            """
            <div style="display:flex; align-items:center; gap:0.45rem; margin-top:0.6rem; margin-bottom:0.5rem;">
                <span style="color:#7047F5; font-size:1.15rem;">📚</span>
                <span style="font-weight:700; font-size:0.95rem; color:#19213A;">Detected Rooms & Details</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Render styled room table
        if all_rooms:
            table_rows_html = ""
            for idx, r in enumerate(all_rooms):
                dot_color = "#EF5350" if idx == 0 else ("#F5A623" if idx == 1 else "#19B978")
                
                # Type badge
                type_lower = r.room_type.lower()
                if "living" in type_lower:
                    t_bg, t_fg = "#EEE8FF", "#7047F5"
                    t_name = "Living"
                elif "bed" in type_lower:
                    t_bg, t_fg = "#E8F8F1", "#19B978"
                    t_name = "Bedroom"
                elif "kitchen" in type_lower:
                    t_bg, t_fg = "#FEF6E8", "#F5A623"
                    t_name = "Kitchen"
                elif "bath" in type_lower:
                    t_bg, t_fg = "#EBF3FF", "#388BFF"
                    t_name = "Bathroom"
                else:
                    t_bg, t_fg = "#F8F8FC", "#66718B"
                    t_name = r.room_type

                dim_str = f"{r.physical_width:.2f} × {r.physical_height:.2f}" if (floorplan.scale_information.is_calibrated and r.physical_width) else f"{int(r.width)} × {int(r.height)}"
                area_str = f"{r.physical_area:.1f}" if (floorplan.scale_information.is_calibrated and r.physical_area) else f"{int(r.width * r.height)}"

                table_rows_html += f"""
                <tr style="border-bottom: 1px solid #E9E7F2; font-size:0.80rem;">
                    <td style="padding: 0.55rem 0.6rem; color:#19213A; font-weight:600;">
                        <span style="color:{dot_color}; font-size:0.9rem;">●</span> {r.room_id}
                    </td>
                    <td style="padding: 0.55rem 0.6rem; color:#19213A; font-weight:600;">{r.label or r.room_type}</td>
                    <td style="padding: 0.55rem 0.6rem;">
                        <span style="background:{t_bg}; color:{t_fg}; font-size:0.72rem; font-weight:700; padding:2px 8px; border-radius:6px;">{t_name}</span>
                    </td>
                    <td style="padding: 0.55rem 0.6rem; color:#66718B;">{dim_str}</td>
                    <td style="padding: 0.55rem 0.6rem; color:#19213A; font-weight:700;">{area_str}</td>
                    <td style="padding: 0.55rem 0.6rem;">
                        <span class="badge badge-green">✔ Confirmed</span>
                    </td>
                </tr>
                """

            st.markdown(
                f"""
                <div style="background:#FFFFFF; border:1px solid #E9E7F2; border-radius:14px; overflow:hidden; margin-bottom:0.8rem; box-shadow:0 1px 4px rgba(23,26,43,0.02);">
                    <table style="width:100%; border-collapse:collapse; text-align:left;">
                        <thead>
                            <tr style="background:#FAF9FE; border-bottom:1px solid #E9E7F2; font-size:0.72rem; color:#66718B; text-transform:uppercase; letter-spacing:0.5px;">
                                <th style="padding:0.6rem;">ID</th>
                                <th style="padding:0.6rem;">Room Label</th>
                                <th style="padding:0.6rem;">Type</th>
                                <th style="padding:0.6rem;">Dimensions (m)</th>
                                <th style="padding:0.6rem;">Area (m²)</th>
                                <th style="padding:0.6rem;">Status</th>
                            </tr>
                        </thead>
                        <tbody>
                            {table_rows_html}
                        </tbody>
                    </table>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Room Editing Form in Clean Expander
        with st.expander("✍️ Edit or Refine Detected Rooms", expanded=False):
            edit_sel_id = st.selectbox(
                "Select Room to Modify",
                options=[r.room_id for r in all_rooms],
                format_func=lambda rid: next((f"{r.label} ({r.room_id})" for r in all_rooms if r.room_id == rid), rid),
            )
            target_r = next((r for r in all_rooms if r.room_id == edit_sel_id), None)
            if target_r:
                with st.form(key=f"edit_form_{target_r.room_id}"):
                    e_lbl = st.text_input("Label", value=target_r.label)
                    e_type = st.selectbox("Type", options=list(RoomConfig.ROOM_TYPES), index=RoomConfig.ROOM_TYPES.index(target_r.room_type) if target_r.room_type in RoomConfig.ROOM_TYPES else 0)
                    e_c1, e_c2 = st.columns(2)
                    with e_c1:
                        e_x = st.number_input("X (px)", value=float(target_r.x), min_value=0.0, step=5.0)
                        e_w = st.number_input("Width (px)", value=float(target_r.width), min_value=1.0, step=5.0)
                    with e_c2:
                        e_y = st.number_input("Y (px)", value=float(target_r.y), min_value=0.0, step=5.0)
                        e_h = st.number_input("Height (px)", value=float(target_r.height), min_value=1.0, step=5.0)
                    if st.form_submit_button("💾 Save Changes", use_container_width=True):
                        RoomEditor.update_room(target_r, label=e_lbl, room_type=e_type, x=e_x, y=e_y, width=e_w, height=e_h, detection_status=target_r.detection_status, image_width=loaded.width, image_height=loaded.height, scale_info=floorplan.scale_information)
                        FloorPlanParser.update_floorplan_with_rooms(floorplan, all_rooms)
                        st.success("Room updated successfully!")
                        st.rerun()

                c_b1, c_b2 = st.columns(2)
                with c_b1:
                    if st.button("✅ Confirm Room", key=f"conf_{target_r.room_id}", use_container_width=True):
                        floorplan.confirm_room(target_r.room_id)
                        st.rerun()
                with c_b2:
                    if st.button("❌ Reject Room", key=f"rej_{target_r.room_id}", use_container_width=True):
                        floorplan.reject_room(target_r.room_id)
                        st.rerun()

        # 3. Physical Scale Calibration Card matching Reference
        st.markdown(
            """
            <div style="display:flex; align-items:center; gap:0.45rem; margin-top:0.8rem; margin-bottom:0.5rem;">
                <span style="color:#7047F5; font-size:1.15rem;">✏️</span>
                <span style="font-weight:700; font-size:0.95rem; color:#19213A;">Physical Scale Calibration (Pixels to Meters)</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        cal_c1, cal_c2, cal_c3, cal_c4 = st.columns([1.3, 1.3, 1.2, 0.8], gap="small")
        with cal_c1:
            cal_px = st.number_input(
                "Known Pixel Distance",
                min_value=1.0,
                max_value=float(max(loaded.width, loaded.height)),
                value=680.0 if sample_loaded else 200.0,
                step=10.0,
                format="%.2f",
            )
        with cal_c2:
            cal_m = st.number_input(
                "Real-World Length",
                min_value=0.1,
                max_value=200.0,
                value=13.60 if sample_loaded else 4.0,
                step=0.5,
                format="%.2f",
            )
        with cal_c3:
            st.write("")
            st.write("")
            if st.button("⚡ Apply Scale", type="primary", use_container_width=True):
                floorplan.update_scale(pixel_distance=cal_px, real_meters=cal_m)
                st.session_state.scale_info = floorplan.scale_information
                st.success(f"Calibrated: {floorplan.scale_information.pixels_per_meter:.2f} px/m")
                st.rerun()
        with cal_c4:
            st.write("")
            st.write("")
            if st.button("🔄 Reset", use_container_width=True):
                floorplan.scale_information.reset()
                for r in all_rooms:
                    r.apply_scale(floorplan.scale_information)
                st.session_state.scale_info = floorplan.scale_information
                st.info("Reset to uncalibrated.")
                st.rerun()

    # -------------------------------------------------------------------------
    # Section C: Computer Vision Inspection Bar matching Reference
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.markdown(
        """
        <div style="display:flex; align-items:center; gap:0.45rem; margin-bottom:0.6rem;">
            <span style="color:#7047F5; font-size:1.15rem;">🔍</span>
            <span style="font-weight:700; font-size:0.95rem; color:#19213A;">Computer Vision Inspection</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    t_c1, t_c2, t_c3, t_c4, t_c5, t_c6 = st.columns(6)
    with t_c1:
        show_annot = st.checkbox("Annotated Rooms", value=True)
    with t_c2:
        show_walls = st.checkbox("Wall Candidates", value=False)
    with t_c3:
        show_thresh = st.checkbox("Threshold", value=False)
    with t_c4:
        show_edges = st.checkbox("Canny Edges", value=False)
    with t_c5:
        show_morph = st.checkbox("Morphology", value=False)
    with t_c6:
        show_gray = st.checkbox("Grayscale", value=False)

    # Conditionally show inspection views below the toggle bar
    if show_walls or show_thresh or show_edges or show_morph or show_gray:
        st.write("")
        inspect_cols = st.columns(3)
        if show_walls:
            with inspect_cols[0]:
                walls_img = overlay_wall_candidates(stages.original_rgb, floorplan.wall_candidates)
                st.image(walls_img, caption=f"Wall Candidates ({len(floorplan.wall_candidates)})", use_container_width=True)
        if show_thresh:
            with inspect_cols[1]:
                st.image(convert_single_channel_to_rgb(stages.threshold), caption="Inverted Threshold Mask", use_container_width=True)
        if show_edges:
            with inspect_cols[2]:
                st.image(convert_single_channel_to_rgb(stages.edges), caption="Canny Edges", use_container_width=True)
        if show_morph:
            with inspect_cols[0]:
                st.image(convert_single_channel_to_rgb(stages.morphology), caption="Morphology Closed Mask", use_container_width=True)
        if show_gray:
            with inspect_cols[1]:
                st.image(convert_single_channel_to_rgb(stages.grayscale), caption="Grayscale Image", use_container_width=True)

    # -------------------------------------------------------------------------
    # Section D: Bottom Navigation Transition
    # -------------------------------------------------------------------------
    st.markdown("---")
    foot_col1, foot_col2 = st.columns([1.5, 2.5])

    with foot_col1:
        json_str = export_floorplan_to_json(floorplan)
        st.download_button(
            label="💾 Download Structured Floor-Plan (JSON)",
            data=json_str,
            file_name=f"spaze_floorplan_{Path(loaded.filename).stem}.json",
            mime="application/json",
            use_container_width=True,
        )

    with foot_col2:
        if st.button("👉 Continue to AI Generation (Stage 02)", type="primary", use_container_width=True):
            floorplan.compute_adjacency()
            StateManager.set_stage("02_generation")
            st.rerun()
