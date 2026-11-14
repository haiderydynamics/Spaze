"""Stage 02 — AI Generation Workspace.

Invokes the trained Conditional VAE to synthesize three geometrically validated
layout alternatives conditioned on user preferences, displays side-by-side comparisons,
and allows layout selection for downstream 3D visualization.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

from cvae.config import CVAEConfig
from cvae.infer import LayoutGenerationEngine
from cvae.visualization import render_single_floorplan_axes
from models.floorplan_schema import FloorPlan
from ui.components import render_metric_card, render_page_header
from ui.state import StateManager
from utils.export import export_floorplan_to_json


def render_generation_page() -> None:
    """Render the Stage 02 Generative AI workspace matching the reference design."""
    render_page_header(
        stage_tag="Stage 02",
        title="AI-Powered Layout Generation",
        subtitle="Generate alternative architectural layouts using preference-conditioned generative AI.",
    )

    # 1. Verification of Input Floor Plan
    if not StateManager.has_floorplan():
        st.markdown(
            """
            <div class="section-card" style="text-align: center; padding: 2.5rem 1.5rem;">
                <div style="font-size: 2.2rem; margin-bottom: 0.6rem;">⚠️</div>
                <div style="font-size: 1.1rem; font-weight: 700; color: #171A2B;">No Active Floor Plan Detected</div>
                <div style="font-size: 0.85rem; color: #69738B; margin-top: 0.4rem; max-width: 520px; margin-left: auto; margin-right: auto; margin-bottom: 1.2rem;">
                    Please return to Stage 01 (Floor Plan Studio) to upload or load an architectural floor plan before running AI generation.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("⬅️ Return to Stage 01: Floor Plan Studio", type="primary"):
            StateManager.set_stage("01_floorplan")
            st.rerun()
        return

    src_fp: FloorPlan = st.session_state.floorplan
    active_rooms = src_fp.confirmed_rooms + src_fp.candidate_rooms

    cfg = CVAEConfig()
    ckpt_file = cfg.CHECKPOINT_DIR / cfg.BEST_MODEL_FILENAME

    if not ckpt_file.is_file():
        st.error(
            f"⚠️ **Model Checkpoint Missing**: No trained model found at `{ckpt_file}`.  \n"
            "Please verify that `checkpoints/spaze_cvae_best.pth` exists."
        )
        return

    # -------------------------------------------------------------------------
    # Main Dashboard Grid: LEFT (Input Plan & Program) | RIGHT (Preferences & Generate)
    # -------------------------------------------------------------------------
    dash_col1, dash_col2 = st.columns([1.2, 1.8], gap="large")

    with dash_col1:
        st.markdown(
            """
            <div style="font-size:0.78rem; font-weight:700; text-transform:uppercase; color:#69738B; letter-spacing:0.8px; margin-bottom:0.5rem;">
                Input Architectural Program
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Source Blueprint Card
        fig_src, ax_src = plt.subplots(figsize=(4.2, 3.6), facecolor="#FFFFFF")
        render_single_floorplan_axes(ax_src, src_fp, title=f"Source ({len(active_rooms)} Rooms)")
        st.pyplot(fig_src, use_container_width=True)
        plt.close(fig_src)

        st.markdown(
            f"""
            <div class="section-card" style="padding: 1.0rem; margin-top: 0.6rem; font-size: 0.82rem;">
                <div style="display:flex; justify-content:space-between; margin-bottom:0.4rem;">
                    <span style="color:#69738B;">Program Count:</span>
                    <span style="color:#7047F5; font-weight:700;">{len(active_rooms)} Rooms</span>
                </div>
                <div style="display:flex; justify-content:space-between; margin-bottom:0.4rem;">
                    <span style="color:#69738B;">Dimensions:</span>
                    <span style="color:#171A2B; font-weight:600;">{src_fp.image_width} × {src_fp.image_height} px</span>
                </div>
                <div style="display:flex; justify-content:space-between;">
                    <span style="color:#69738B;">Room Types:</span>
                    <span style="color:#171A2B; font-weight:600;">{len({r.room_type for r in active_rooms})} categories</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("✏️ Edit Input Floor Plan", use_container_width=True):
            StateManager.set_stage("01_floorplan")
            st.rerun()

    with dash_col2:
        st.markdown(
            """
            <div style="font-size:0.78rem; font-weight:700; text-transform:uppercase; color:#69738B; letter-spacing:0.8px; margin-bottom:0.5rem;">
                Preference Configuration
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Preference Cards
        pref_options = ["space_efficient", "open_layout", "privacy_focused"]
        curr_pref = st.session_state.get("selected_preference", "space_efficient")

        selected_pref = st.radio(
            "Select Design Preference",
            options=pref_options,
            format_func=lambda p: {
                "space_efficient": "📐 Space-Efficient — Minimal circulation, compact room clusters",
                "open_layout": "🛋️ Open Layout — Connected common spaces, shared boundary borders",
                "privacy_focused": "🚪 Privacy-Focused — Isolated bedroom clusters, corridor buffering",
            }.get(p, p),
            index=pref_options.index(curr_pref) if curr_pref in pref_options else 0,
            label_visibility="collapsed",
        )
        st.session_state.selected_preference = selected_pref

        st.write("")

        # Sampling Controls Card
        with st.expander("⚙️ Advanced Latent Sampling Controls", expanded=False):
            temp = st.slider(
                "Latent Sampling Temperature (τ)",
                min_value=0.5,
                max_value=2.0,
                value=float(st.session_state.get("latent_temperature", 1.0)),
                step=0.1,
                help="Higher temperature yields greater structural divergence from the source layout.",
            )
            seed_val = st.number_input(
                "Random Seed",
                value=int(st.session_state.get("random_seed", 42)),
                step=1,
                help="Ensures 100% reproducible latent sampling.",
            )
            st.session_state.latent_temperature = temp
            st.session_state.random_seed = seed_val

        st.write("")
        gen_clicked = st.button("🚀 Generate 3 Alternative Layouts", type="primary", use_container_width=True)

    # -------------------------------------------------------------------------
    # Execution of CVAE Inference
    # -------------------------------------------------------------------------
    if gen_clicked:
        with st.spinner("Executing CVAE latent sampling & geometric constraint verification..."):
            try:
                engine = LayoutGenerationEngine(checkpoint_path=ckpt_file, config=cfg)
                alts, metrics = engine.generate_alternatives(
                    source_floorplan=src_fp,
                    preference=selected_pref,
                    num_alternatives=3,
                    temperature=st.session_state.latent_temperature,
                    random_seed=int(st.session_state.random_seed),
                )
                st.session_state.cvae_alternatives = alts
                st.session_state.cvae_metrics = metrics
                st.session_state.selected_alt_idx = 0
                st.success(
                    f"Generated {len(alts)} distinct alternatives in {metrics['inference_time_ms']:.1f}ms! "
                    f"(Diversity: {metrics['layout_diversity_score']:.2f}, Validity: {metrics['mean_validity_score']:.2f})"
                )
            except Exception as exc:
                st.error(f"Inference error: {exc}")

    # -------------------------------------------------------------------------
    # Results Comparison: 3 Distinct Alternative Layout Cards
    # -------------------------------------------------------------------------
    if StateManager.has_alternatives():
        alts: List[FloorPlan] = st.session_state.cvae_alternatives
        metrics = st.session_state.cvae_metrics
        curr_selected = st.session_state.get("selected_alt_idx", 0)

        st.markdown("---")
        st.markdown(
            """
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:1.0rem;">
                <div>
                    <span style="font-size:1.20rem; font-weight:800; color:#171A2B;">💡 Generated Layout Alternatives</span>
                    <div style="font-size:0.82rem; color:#69738B;">Side-by-side comparison of 3 CVAE-synthesized layout options under active spatial constraints.</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Metric summary banner
        mb1, mb2, mb3, mb4 = st.columns(4)
        with mb1:
            render_metric_card("Diversity Score", f"{metrics.get('layout_diversity_score', 0.0):.2f}", "Spatial 1 - IoU", val_color="#7047F5")
        with mb2:
            val_pct = metrics.get('geometric_validity_rate', 0.0) * 100
            render_metric_card("Validity Rate", f"{val_pct:.0f}%", "Passed constraints", val_color="#19B978")
        with mb3:
            render_metric_card("Inference Latency", f"{metrics.get('inference_time_ms', 0.0):.1f} ms", "CVAE sample time", val_color="#388BFF")
        with mb4:
            render_metric_card("Conditioning", selected_pref.replace("_", " ").title(), "Design intent", val_color="#7047F5")

        st.write("")

        # 3 Columns for 3 Alternatives
        cols = st.columns(3, gap="large")

        for idx, alt in enumerate(alts):
            is_active = (idx == curr_selected)
            r_cnt = len(alt.confirmed_rooms + alt.candidate_rooms)
            tot_m2 = sum(r.physical_area for r in alt.confirmed_rooms if r.physical_area)
            area_str = f"{tot_m2:.1f} m² usable" if tot_m2 > 0 else f"{alt.image_width}×{alt.image_height} px"

            border_style = "border: 2px solid #7047F5; box-shadow: 0 4px 18px rgba(112, 71, 245, 0.15);" if is_active else "border: 1px solid #E9E7F2;"

            with cols[idx]:
                st.markdown(
                    f"""
                    <div class="section-card" style="{border_style} padding: 1.0rem; margin-bottom: 0.6rem;">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.5rem;">
                            <span style="font-weight:700; font-size:0.95rem; color:#171A2B;">Alternative {idx + 1}</span>
                            <span class="{'badge badge-purple' if is_active else 'badge badge-green'}">
                                {'Active 3D Choice' if is_active else '✅ 100% Valid'}
                            </span>
                        </div>
                        <div style="font-size:0.75rem; color:#69738B;">
                            Rooms: <b>{r_cnt}</b> &nbsp;|&nbsp; Area: <b>{area_str}</b>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Matplotlib blueprint axes
                fig_alt, ax_alt = plt.subplots(figsize=(4.2, 3.6), facecolor="#FFFFFF")
                render_single_floorplan_axes(ax_alt, alt, title=f"Alt {idx + 1} ({r_cnt} Rooms)")
                st.pyplot(fig_alt, use_container_width=True)
                plt.close(fig_alt)

                if st.button(f"{'✅ Selected' if is_active else '👉 Select Alt ' + str(idx + 1)}", key=f"sel_alt_{idx}", use_container_width=True, type="primary" if is_active else "secondary"):
                    st.session_state.selected_alt_idx = idx
                    st.rerun()

        # ---------------------------------------------------------------------
        # Section F: Preferred Layout Selection & Export
        # ---------------------------------------------------------------------
        st.markdown("---")
        st.markdown("### 📥 Selected Layout Details & Downstream Transition")

        chosen_alt = alts[st.session_state.get("selected_alt_idx", 0)]
        sel_col1, sel_col2 = st.columns([2, 1], gap="large")

        with sel_col1:
            alt_table = [
                {
                    "Room ID": r.room_id,
                    "Type": r.room_type,
                    "Label": r.label,
                    "Bounds": f"({int(r.x)}, {int(r.y)})",
                    "Dimensions": f"{r.physical_width:.1f}×{r.physical_height:.1f} m" if r.physical_width else f"{int(r.width)}×{int(r.height)} px",
                    "Adjacency": ", ".join(r.adjacent_room_ids) if r.adjacent_room_ids else "None",
                }
                for r in chosen_alt.confirmed_rooms
            ]
            st.dataframe(alt_table, use_container_width=True, hide_index=True)

        with sel_col2:
            alt_json = export_floorplan_to_json(chosen_alt)
            st.download_button(
                label=f"💾 Download Alt {curr_selected + 1} (JSON)",
                data=alt_json,
                file_name=f"spaze_layout_alt_{curr_selected + 1}_{chosen_alt.preference}.json",
                mime="application/json",
                use_container_width=True,
            )

            st.write("")
            if st.button("👉 Proceed to 3D Visualization (Stage 03)", type="primary", use_container_width=True):
                StateManager.set_stage("03_visualization")
                st.rerun()
    else:
        st.markdown(
            """
            <div class="section-card" style="text-align: center; padding: 2.0rem; background: #FFFFFF; margin-top: 1.0rem;">
                <div style="font-size: 1.8rem; margin-bottom: 0.5rem;">💡</div>
                <div style="font-size: 1.0rem; font-weight: 700; color: #171A2B;">Ready for Layout Generation</div>
                <div style="font-size: 0.82rem; color: #69738B; margin-top: 0.3rem;">
                    Click <b>Generate 3 Alternative Layouts</b> above to synthesize 3 preference-controlled floor plans.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
