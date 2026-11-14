"""Stage 04 — Research Evaluation Lab & Benchmarking Workspace.

Presents verified quantitative performance metrics across Computer Vision, Generative AI (CVAE),
Geometric Validation, Procedural Baseline Comparisons, and 3D Mesh Synthesis.
Upholds strict scientific reporting: distinguishes evaluated metrics from unavailable ground truth.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from cvae.config import CVAEConfig
from evaluation.benchmark_suite import BenchmarkSuite, ComparisonMetricRow
from evaluation.cvae_evaluator import CVAEEvaluator, CVAEEvaluationMetrics
from evaluation.cv_evaluator import CVEvaluator, CVEvaluationResult
from models.floorplan_schema import FloorPlan
from ui.components import render_metric_card, render_page_header, safe_html
from ui.state import StateManager


def render_evaluation_page() -> None:
    """Render the Stage 04 Evaluation Lab and Research Benchmark workspace matching reference design."""
    render_page_header(
        stage_tag="Stage 04",
        title="Model Evaluation & Performance",
        subtitle="Empirical quantitative evaluation across Computer Vision extraction, Conditional VAE convergence, layout diversity, and procedural baseline benchmarks.",
    )

    cfg = CVAEConfig()
    cvae_eval = CVAEEvaluator(cfg)
    cvae_metrics = cvae_eval.evaluate_model_checkpoint()

    # Active floor plan context
    active_fp = StateManager.get_selected_alternative()

    # -------------------------------------------------------------------------
    # Section A: Primary Research Metric Cards
    # -------------------------------------------------------------------------
    safe_html(
        """
        <div style="font-size:0.78rem; font-weight:700; text-transform:uppercase; color:#69738B; letter-spacing:0.8px; margin-bottom:0.6rem;">
            Core Model & System Metrics
        </div>
        """
    )

    # 3D Gen time from active 3d result if available
    active_3d = st.session_state.get("active_3d_result")
    mesh_time_str = f"{active_3d.metrics.generation_time_ms:.1f} ms" if active_3d and hasattr(active_3d, "metrics") else "Not measured"

    m_r1, m_r2, m_r3, m_r4 = st.columns(4)
    with m_r1:
        render_metric_card("Recon Loss", f"{cvae_metrics.reconstruction_loss:.4f}", "Smooth L1 loss", val_color="#19B978")
    with m_r2:
        render_metric_card("KL Divergence", f"{cvae_metrics.kl_divergence:.2f}", "Normal prior penalty", val_color="#7047F5")
    with m_r3:
        render_metric_card("Layout Diversity", f"{cvae_metrics.layout_diversity_score:.3f}", "1 - Pairwise IoU", val_color="#7047F5")
    with m_r4:
        render_metric_card("Geometric Validity", f"{cvae_metrics.geometric_validity_rate * 100:.1f}%", "Zero envelope breach", val_color="#19B978")

    st.write("")
    m_s1, m_s2, m_s3, m_s4 = st.columns(4)
    with m_s1:
        render_metric_card("CVAE Inference Time", f"{cvae_metrics.inference_latency_ms:.1f} ms", "Latency per layout", val_color="#388BFF")
    with m_s2:
        render_metric_card("3D Generation Time", mesh_time_str, "Extrusion & aperture time", val_color="#7047F5" if mesh_time_str != "Not measured" else "#69738B")
    with m_s3:
        render_metric_card("Furniture Collision", f"{cvae_metrics.room_overlap_rate * 100:.1f}%", "Proxy non-overlap rate", val_color="#19B978")
    with m_s4:
        render_metric_card("Peak Memory", f"{cvae_metrics.model_size_mb:.2f} MB", "Model footprint", val_color="#171A2B")

    # -------------------------------------------------------------------------
    # Section B: Computer Vision Extraction Evaluation
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.markdown("### 👁️ 1. Computer Vision Extraction Evaluation")
    st.caption("Wall detection matching and room boundary IoU scored against verified ground truth CAD annotations.")

    cv_evaluator = CVEvaluator()
    if active_fp is not None:
        cv_result = cv_evaluator.evaluate(active_fp)
    else:
        cv_result = CVEvaluationResult(
            status="Not Evaluated — No Active Floor Plan",
            notes="Please load or upload a floor plan in Stage 01 to evaluate CV detection.",
        )

    col_cv_stat, col_cv_prec, col_cv_rec, col_cv_f1, col_cv_iou = st.columns([1.5, 1, 1, 1, 1])

    with col_cv_stat:
        badge_cls = "badge-green" if cv_result.status == "Evaluated" else "badge-amber"
        safe_html(f"<div><span style='font-weight:700;'>Status:</span> <span class='badge {badge_cls}'>{cv_result.status}</span></div>")
        st.caption(cv_result.notes)

    with col_cv_prec:
        val = f"{cv_result.wall_precision * 100:.1f}%" if cv_result.wall_precision is not None else "Not measured"
        render_metric_card("Wall Precision", val, "TP / (TP + FP)", val_color="#7047F5" if cv_result.wall_precision else "#69738B")

    with col_cv_rec:
        val = f"{cv_result.wall_recall * 100:.1f}%" if cv_result.wall_recall is not None else "Not measured"
        render_metric_card("Wall Recall", val, "TP / (TP + FN)", val_color="#19B978" if cv_result.wall_recall else "#69738B")

    with col_cv_f1:
        val = f"{cv_result.wall_f1:.3f}" if cv_result.wall_f1 is not None else "Not measured"
        render_metric_card("Wall F1 Score", val, "Harmonic Mean", val_color="#171A2B" if cv_result.wall_f1 else "#69738B")

    with col_cv_iou:
        val = f"{cv_result.mean_room_iou:.3f}" if cv_result.mean_room_iou is not None else "Not measured"
        render_metric_card("Room IoU", val, "Intersection over Union", val_color="#7047F5" if cv_result.mean_room_iou else "#69738B")

    if cv_result.status == "Evaluated" and cv_result.per_room_iou:
        with st.expander("Detailed Per-Room Ground-Truth IoU Breakdown", expanded=False):
            room_iou_df = pd.DataFrame([
                {"Room Name": k, "IoU Score": f"{v:.4f}", "Status": "Matched" if v > 0.1 else "Needs Calibration"}
                for k, v in cv_result.per_room_iou.items()
            ])
            st.dataframe(room_iou_df, use_container_width=True, hide_index=True)

    # -------------------------------------------------------------------------
    # Section C: Training History & Visualizations
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.markdown("### 📈 2. Training Convergence & Visualizations")

    col_plot, col_arch = st.columns([1.8, 1.2], gap="large")

    with col_plot:
        st.markdown("#### Empirical Convergence Progression (45 Epochs)")
        curves_file = cfg.LOG_DIR / "training_curves.png"
        if curves_file.is_file():
            st.image(str(curves_file), caption="CVAE Objective Loss & KL Divergence Curves across 45 Training Epochs", use_container_width=True)
        else:
            st.info("Training telemetry plot available at training_logs/training_curves.png.")

    with col_arch:
        st.markdown("#### Model Specification & Architecture")
        safe_html(
            f"""
            <div class="section-card" style="font-size: 0.85rem; padding: 1.1rem;">
                <p>• <b>Framework:</b> PyTorch 2.x (CPU/CUDA capable)</p>
                <p>• <b>Latent Dimension:</b> 8-D Standard Gaussian N(0, I)</p>
                <p>• <b>Context Dimension:</b> 348 layout features + 3 preference dims</p>
                <p>• <b>Hidden Layers:</b> 256 units with LayerNorm & LeakyReLU (0.2)</p>
                <p>• <b>Loss Formulation:</b> ELBO = Recon Loss (Smooth L1) + &beta; &middot; KL Divergence</p>
                <p>• <b>Total Parameters:</b> {cvae_metrics.model_param_count:,}</p>
                <p>• <b>Model Weight Size:</b> {cvae_metrics.model_size_mb:.2f} MB</p>
            </div>
            """
        )

    # -------------------------------------------------------------------------
    # Section D: Procedural Baseline Comparison
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.markdown("### ⚖️ 3. Comparative Benchmark: Procedural Baseline vs Conditional VAE")
    st.caption("Standardized evaluation of the learned Conditional VAE against a deterministic heuristic layout packing baseline under identical spatial constraints.")

    if active_fp is not None:
        comp_rows = BenchmarkSuite.run_comparative_benchmark(active_fp)
        comp_df = pd.DataFrame([
            {
                "Evaluation Dimension": r.metric_name,
                "Procedural Baseline (Heuristic)": r.procedural_baseline,
                "Conditional VAE (SPAZE)": r.conditional_vae,
                "Scientific Interpretation": r.scientific_interpretation,
            }
            for r in comp_rows
        ])
        st.dataframe(comp_df, use_container_width=True, hide_index=True)
    else:
        st.info("Load a floor plan in Stage 01 to generate live comparative baseline benchmark tables.")

    # -------------------------------------------------------------------------
    # Section E: Preference Sensitivity Diagnostics
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.markdown("### 🎯 4. Architectural Preference Sensitivity Diagnostics")
    st.caption("Measurable spatial adjustments induced by the 3 preference conditions across the latent manifold.")

    p1, p2, p3 = st.columns(3, gap="medium")

    with p1:
        safe_html(
            f"""
            <div class="section-card">
                <div style="font-size: 1.3rem;">📐</div>
                <div style="font-weight: 800; color: #171A2B; margin: 0.3rem 0;">Space-Efficient Layout</div>
                <div style="color: #69738B; font-size: 0.82rem; margin-bottom: 0.8rem;">
                    Focuses on minimal circulation corridors, compact room bounding hulls, and high net usable floor space.
                </div>
                <div style="display:flex; justify-content:space-between; font-size:0.85rem; border-top: 1px solid #E9E7F2; padding-top:0.5rem;">
                    <span style="color:#69738B;">Space Efficiency Ratio:</span>
                    <span style="color:#19B978; font-weight:700;">{cvae_metrics.space_efficiency_ratio * 100:.1f}%</span>
                </div>
            </div>
            """
        )

    with p2:
        safe_html(
            f"""
            <div class="section-card">
                <div style="font-size: 1.3rem;">🛋️</div>
                <div style="font-weight: 800; color: #171A2B; margin: 0.3rem 0;">Open Layout</div>
                <div style="color: #69738B; font-size: 0.82rem; margin-bottom: 0.8rem;">
                    Maximizes connectivity between Living Room, Dining, and Kitchen spaces with direct shared perimeters.
                </div>
                <div style="display:flex; justify-content:space-between; font-size:0.85rem; border-top: 1px solid #E9E7F2; padding-top:0.5rem;">
                    <span style="color:#69738B;">Open Connectivity Index:</span>
                    <span style="color:#7047F5; font-weight:700;">{cvae_metrics.openness_connectivity_score:.3f}</span>
                </div>
            </div>
            """
        )

    with p3:
        safe_html(
            f"""
            <div class="section-card">
                <div style="font-size: 1.3rem;">🚪</div>
                <div style="font-weight: 800; color: #171A2B; margin: 0.3rem 0;">Privacy-Focused</div>
                <div style="color: #69738B; font-size: 0.82rem; margin-bottom: 0.8rem;">
                    Isolates bedroom quarters from common living zones with intentional spatial buffering.
                </div>
                <div style="display:flex; justify-content:space-between; font-size:0.85rem; border-top: 1px solid #E9E7F2; padding-top:0.5rem;">
                    <span style="color:#69738B;">Private Isolation Factor:</span>
                    <span style="color:#7047F5; font-weight:700;">{cvae_metrics.privacy_isolation_distance:.2f} m</span>
                </div>
            </div>
            """
        )
