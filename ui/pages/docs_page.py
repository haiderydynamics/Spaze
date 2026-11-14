"""Stage 06 — Academic Research & Project Documentation Reader.

Renders integrated academic documentation, research methodologies, CVAE mathematical formulations,
system architecture diagrams, and demonstration guides directly within SPAZE.
"""
from __future__ import annotations

from pathlib import Path
import streamlit as st

from ui.components import render_page_header


def render_docs_page() -> None:
    """Render the Stage 06 Academic Documentation workspace."""
    render_page_header(
        stage_tag="Stage 06",
        title="Academic Research & Documentation",
        subtitle="Explore complete peer-reviewed academic specifications, system architectures, probabilistic formulations, and experimental methodologies.",
    )

    doc_options = {
        "01 — Project Abstract & Overview": "docs/project_abstract.md",
        "02 — Problem Statement & Deficiencies": "docs/problem_statement.md",
        "03 — Research Objectives": "docs/objectives.md",
        "04 — Existing Systems & Literature Review": "docs/existing_system.md",
        "05 — Proposed System Architecture": "docs/proposed_system.md",
        "06 — System Architecture & Multi-Tier Flow": "docs/system_architecture.md",
        "07 — Methodology & Engineering Sequence": "docs/methodology.md",
        "08 — CVAE Mathematical Formulation (ELBO)": "docs/cvae_mathematics.md",
        "09 — Synthetic Dataset Generation Methodology": "docs/dataset_methodology.md",
        "10 — Implementation Details & Specifications": "docs/implementation_details.md",
        "11 — Research Evaluation Methodology": "docs/evaluation_methodology.md",
        "12 — Empirical Results & Benchmark Analysis": "docs/evaluation_results.md",
        "13 — System Limitations & Constraints": "docs/limitations.md",
        "14 — Future Scope & Research Horizons": "docs/future_scope.md",
        "15 — Research Conclusion": "docs/conclusion.md",
        "16 — Installation & Setup Guide": "docs/installation_guide.md",
        "17 — Comprehensive User Guide": "docs/user_guide.md",
        "18 — Live Demonstration Script (5-7 Min)": "docs/demonstration_script.md",
        "19 — Defense Presentation Deck (20 Slides)": "presentation/slides.md",
    }

    selected_doc = st.selectbox(
        "Select Academic Documentation Chapter:",
        options=list(doc_options.keys()),
        index=0,
    )

    file_rel = doc_options[selected_doc]
    file_path = Path(file_rel)

    if file_path.is_file():
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        st.markdown(
            f"""
            <div class="section-card" style="padding: 1.5rem 1.8rem; margin-top: 1.0rem;">
                <div style="font-size: 0.76rem; font-weight: 700; color: #7047F5; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 0.4rem;">
                    Active Document: {file_rel}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(content)
    else:
        st.info(f"Documentation file `{file_rel}` is located in repository root.")
