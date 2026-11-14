"""Centralized Reference-Locked Architectural SaaS Theme for SPAZE.

Guarantees 100% light theme across all native Streamlit widgets, headers, toolbars,
sidebars, inputs, dropdowns, expanders, file uploaders, and custom cards.
Prevents any dark-theme leakage or unstyled components.
"""
from __future__ import annotations

import streamlit as st


def apply_spaze_theme() -> None:
    """Inject complete reference-locked light theme CSS into the Streamlit session."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

        /* -------------------------------------------------------------
           1. Base Application Canvas & Root Layout
        ------------------------------------------------------------- */
        :root {
            --spaze-bg: #F8F8FC;
            --spaze-sidebar: #F4F0FF;
            --spaze-card: #FFFFFF;
            --spaze-primary: #7047F5;
            --spaze-primary-light: #EEE8FF;
            --spaze-text: #19213A;
            --spaze-muted: #66718B;
            --spaze-border: #E9E7F2;
            --spaze-success: #19B978;
            --spaze-warning: #F5A623;
            --spaze-danger: #EF5350;
        }

        html, body, [class*="css"], .stApp {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
            background-color: #F8F8FC !important;
            color: #19213A !important;
        }

        [data-testid="stAppViewContainer"] {
            background-color: #F8F8FC !important;
        }

        .block-container {
            padding-top: 1.0rem !important;
            padding-bottom: 2.5rem !important;
            padding-left: 2.0rem !important;
            padding-right: 2.0rem !important;
            max-width: 1440px !important;
        }

        /* -------------------------------------------------------------
           2. Top Streamlit Native Header & Toolbar (Kill All Black Bars!)
        ------------------------------------------------------------- */
        header[data-testid="stHeader"] {
            background-color: rgba(248, 248, 252, 0.98) !important;
            border-bottom: 1px solid #E9E7F2 !important;
            color: #19213A !important;
            backdrop-filter: blur(8px) !important;
        }
        header[data-testid="stHeader"] * {
            color: #19213A !important;
        }
        [data-testid="stToolbar"] {
            background-color: transparent !important;
            color: #19213A !important;
        }
        [data-testid="stToolbar"] * {
            color: #19213A !important;
            fill: #19213A !important;
        }
        [data-testid="stDecoration"] {
            background-image: linear-gradient(90deg, #7047F5, #956BFF) !important;
            height: 3px !important;
        }
        .stAppHeader {
            background-color: transparent !important;
        }
        [data-testid="stDeployButton"] {
            display: none !important;
        }

        /* -------------------------------------------------------------
           3. Left Sidebar Styling & Complete Readability
        ------------------------------------------------------------- */
        section[data-testid="stSidebar"] {
            background-color: #F4F0FF !important;
            border-right: 1px solid #E9E7F2 !important;
            width: 240px !important;
            min-width: 220px !important;
        }
        section[data-testid="stSidebar"] > div {
            background-color: #F4F0FF !important;
        }

        /* Hide the native radio circles in sidebar navigation */
        section[data-testid="stSidebar"] div[role="radiogroup"] label div:first-child {
            display: none !important;
        }

        /* Inactive Navigation Items */
        section[data-testid="stSidebar"] div[role="radiogroup"] > label {
            background: transparent !important;
            padding: 0.65rem 0.9rem !important;
            border-radius: 12px !important;
            margin-bottom: 0.25rem !important;
            border: 1px solid transparent !important;
            transition: all 0.16s ease !important;
            cursor: pointer !important;
        }
        section[data-testid="stSidebar"] div[role="radiogroup"] > label * {
            color: #19213A !important;
            font-weight: 500 !important;
            font-size: 0.88rem !important;
            opacity: 1 !important;
        }
        section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
            background: #EAE3FF !important;
            border-color: #D6D2E8 !important;
        }
        section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover * {
            color: #7047F5 !important;
            font-weight: 600 !important;
        }

        /* Active Navigation Item (Vibrant Purple Gradient) */
        section[data-testid="stSidebar"] div[role="radiogroup"] > label[data-checked="true"],
        section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
            background: linear-gradient(135deg, #7047F5 0%, #956BFF 100%) !important;
            box-shadow: 0 4px 14px rgba(112, 71, 245, 0.3) !important;
            border-color: transparent !important;
        }
        section[data-testid="stSidebar"] div[role="radiogroup"] > label[data-checked="true"] *,
        section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) * {
            color: #FFFFFF !important;
            font-weight: 700 !important;
        }

        /* -------------------------------------------------------------
           4. Top Workspace Header & Workflow Navigation Ribbon
        ------------------------------------------------------------- */
        .spaze-top-bar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 1.25rem;
            gap: 1.0rem;
        }
        .stage-title-wrap {
            display: flex;
            align-items: center;
            gap: 0.85rem;
        }
        .stage-icon-box {
            width: 44px;
            height: 44px;
            min-width: 44px;
            background: #EEE8FF;
            border: 1.5px solid #D6D2E8;
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.3rem;
            color: #7047F5;
        }
        .stage-title-text {
            font-size: 1.45rem;
            font-weight: 800;
            color: #19213A;
            letter-spacing: -0.4px;
            line-height: 1.2;
        }
        .stage-subtitle-text {
            font-size: 0.84rem;
            color: #66718B;
            margin-top: 0.2rem;
            line-height: 1.4;
        }

        .workflow-bar {
            display: flex;
            align-items: center;
            gap: 0.35rem;
            background: #FFFFFF;
            border: 1px solid #E9E7F2;
            border-radius: 12px;
            padding: 0.35rem 0.65rem;
            box-shadow: 0 1px 4px rgba(23, 26, 43, 0.03);
            white-space: nowrap;
        }
        .step-item {
            font-size: 0.76rem;
            font-weight: 600;
            padding: 0.32rem 0.70rem;
            border-radius: 8px;
            color: #66718B;
            letter-spacing: 0.2px;
            transition: all 0.15s ease;
        }
        .step-item.active {
            background: linear-gradient(135deg, #7047F5 0%, #956BFF 100%) !important;
            color: #FFFFFF !important;
            box-shadow: 0 2px 8px rgba(112, 71, 245, 0.25);
            font-weight: 700;
        }
        .step-item.completed {
            background: #E8F8F1;
            color: #19B978;
            border: 1px solid rgba(25, 185, 120, 0.25);
        }
        .step-arrow {
            color: #C7CBD6;
            font-size: 0.75rem;
        }

        /* -------------------------------------------------------------
           5. File Uploader Styling (Kill All Dark Uploader Backgrounds!)
        ------------------------------------------------------------- */
        [data-testid="stFileUploader"] {
            background-color: transparent !important;
        }
        [data-testid="stFileUploader"] section {
            background-color: #FFFFFF !important;
            border: 1.5px dashed #D6D2E8 !important;
            border-radius: 14px !important;
            padding: 1.0rem !important;
            transition: all 0.2s ease !important;
        }
        [data-testid="stFileUploader"] section:hover {
            border-color: #7047F5 !important;
            background-color: #FAF9FE !important;
        }
        [data-testid="stFileUploaderDropzone"] {
            background-color: #FFFFFF !important;
        }
        [data-testid="stFileUploaderDropzone"] svg {
            color: #7047F5 !important;
            fill: #7047F5 !important;
        }
        [data-testid="stFileUploaderDropzoneInstructions"] {
            color: #19213A !important;
        }
        [data-testid="stFileUploaderDropzoneInstructions"] span {
            color: #66718B !important;
        }
        [data-testid="stFileUploader"] button {
            background: #F5F3FF !important;
            color: #7047F5 !important;
            border: 1px solid #D6D2E8 !important;
            border-radius: 10px !important;
            font-weight: 600 !important;
        }
        [data-testid="stFileUploader"] button:hover {
            background: #EEE8FF !important;
            border-color: #7047F5 !important;
        }
        [data-testid="stFileUploader"] small {
            color: #66718B !important;
        }

        /* -------------------------------------------------------------
           6. Inputs, Selectboxes, Dropdowns & Popovers (100% Light)
        ------------------------------------------------------------- */
        [data-baseweb="input"],
        [data-baseweb="input"] > div,
        [data-baseweb="base-input"] {
            background-color: #FFFFFF !important;
            border: 1px solid #E5E2F0 !important;
            border-radius: 10px !important;
            color: #19213A !important;
        }
        [data-baseweb="input"] input,
        [data-baseweb="base-input"] input,
        input[type="text"],
        input[type="number"] {
            background-color: #FFFFFF !important;
            color: #19213A !important;
            -webkit-text-fill-color: #19213A !important;
            font-weight: 500 !important;
        }
        [data-baseweb="input"]:focus-within {
            border-color: #7047F5 !important;
            box-shadow: 0 0 0 2px rgba(112, 71, 245, 0.15) !important;
        }

        /* NumberInput stepper buttons */
        [data-testid="stNumberInput"] button {
            background-color: #F8F8FC !important;
            border: 1px solid #E5E2F0 !important;
            color: #19213A !important;
        }
        [data-testid="stNumberInput"] button:hover {
            background-color: #EEE8FF !important;
            color: #7047F5 !important;
            border-color: #7047F5 !important;
        }

        /* Selectbox & Dropdown menus */
        [data-baseweb="select"] > div {
            background-color: #FFFFFF !important;
            border: 1px solid #E5E2F0 !important;
            border-radius: 10px !important;
            color: #19213A !important;
        }
        [data-baseweb="select"] * {
            color: #19213A !important;
        }

        /* Popover open menu options */
        div[data-baseweb="popover"],
        div[data-baseweb="popover"] > div,
        ul[role="listbox"],
        li[role="option"] {
            background-color: #FFFFFF !important;
            color: #19213A !important;
        }
        li[role="option"]:hover,
        li[role="option"][aria-selected="true"] {
            background-color: #EEE8FF !important;
            color: #7047F5 !important;
        }
        li[role="option"] * {
            color: #19213A !important;
        }
        li[role="option"][aria-selected="true"] * {
            color: #7047F5 !important;
            font-weight: 700 !important;
        }

        [data-testid="stWidgetLabel"] p,
        [data-testid="stWidgetLabel"] label {
            color: #19213A !important;
            font-weight: 600 !important;
            font-size: 0.85rem !important;
        }

        /* -------------------------------------------------------------
           7. Expanders & Collapsible Sections (Kill All Dark Headers!)
        ------------------------------------------------------------- */
        [data-testid="stExpander"] {
            background-color: #FFFFFF !important;
            border: 1px solid #E9E7F2 !important;
            border-radius: 14px !important;
            box-shadow: 0 2px 8px rgba(23, 26, 43, 0.02) !important;
            margin-bottom: 0.85rem !important;
            overflow: hidden !important;
        }
        [data-testid="stExpander"] summary {
            background-color: #FFFFFF !important;
            color: #19213A !important;
            border-radius: 14px !important;
            font-weight: 700 !important;
            padding: 0.85rem 1.1rem !important;
            transition: background 0.15s ease !important;
        }
        [data-testid="stExpander"] summary:hover {
            background-color: #FAF9FE !important;
            color: #7047F5 !important;
        }
        [data-testid="stExpander"] summary * {
            color: #19213A !important;
            fill: #19213A !important;
        }
        [data-testid="stExpander"] [data-testid="stExpanderDetails"] {
            background-color: #FFFFFF !important;
            border-top: 1px solid #E9E7F2 !important;
            padding: 1.1rem 1.25rem !important;
        }

        /* -------------------------------------------------------------
           8. Tables & DataFrames (Light Surface & Headers)
        ------------------------------------------------------------- */
        [data-testid="stDataFrame"],
        [data-testid="stTable"] {
            background-color: #FFFFFF !important;
            border: 1px solid #E9E7F2 !important;
            border-radius: 12px !important;
            overflow: hidden !important;
        }
        [data-testid="stDataFrame"] * {
            color: #19213A !important;
        }

        /* -------------------------------------------------------------
           9. Custom Reference Metric Cards (No Word Splitting or Overflow)
        ------------------------------------------------------------- */
        .ref-metric-card {
            background: #FFFFFF;
            border: 1px solid #E9E7F2;
            border-radius: 14px;
            padding: 0.85rem 1.0rem;
            box-shadow: 0 2px 8px rgba(23, 26, 43, 0.025);
            transition: transform 0.15s ease, box-shadow 0.15s ease;
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }
        .ref-metric-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 16px rgba(112, 71, 245, 0.07);
            border-color: #D6D2E8;
        }
        .ref-metric-icon-box {
            width: 32px;
            height: 32px;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.05rem;
            margin-bottom: 0.45rem;
        }
        .ref-metric-icon-purple { background: #EEE8FF; color: #7047F5; }
        .ref-metric-icon-blue   { background: #EBF3FF; color: #388BFF; }
        .ref-metric-icon-green  { background: #E8F8F1; color: #19B978; }
        .ref-metric-icon-amber  { background: #FEF6E8; color: #F5A623; }
        .ref-metric-icon-cyan   { background: #E6FAFA; color: #0EA5E9; }

        .ref-metric-value {
            font-size: clamp(1.25rem, 1.6vw, 1.55rem);
            font-weight: 800;
            color: #19213A;
            line-height: 1.15;
            letter-spacing: -0.3px;
            white-space: nowrap;
            display: flex;
            align-items: center;
            gap: 0.35rem;
        }
        .ref-metric-label {
            font-size: 0.74rem;
            color: #66718B;
            font-weight: 600;
            margin-top: 0.25rem;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        /* -------------------------------------------------------------
           10. Cards & Layout Containers
        ------------------------------------------------------------- */
        .section-card {
            background: #FFFFFF;
            border: 1px solid #E9E7F2;
            border-radius: 16px;
            padding: 1.25rem 1.4rem;
            margin-bottom: 1.1rem;
            box-shadow: 0 2px 8px rgba(23, 26, 43, 0.025);
            transition: box-shadow 0.2s ease, border-color 0.2s ease;
        }
        .section-card:hover {
            box-shadow: 0 6px 18px rgba(112, 71, 245, 0.05);
            border-color: #D6D2E8;
        }

        /* Horizontal Upload Bar from Reference */
        .upload-bar-card {
            background: #FFFFFF;
            border: 1px solid #E9E7F2;
            border-radius: 16px;
            padding: 0.85rem 1.25rem;
            margin-bottom: 1.15rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            box-shadow: 0 2px 8px rgba(23, 26, 43, 0.02);
            gap: 1.0rem;
        }

        /* Badges */
        .badge {
            display: inline-flex;
            align-items: center;
            gap: 0.3rem;
            padding: 3px 8px;
            border-radius: 6px;
            font-size: 0.72rem;
            font-weight: 700;
            white-space: nowrap;
        }
        .badge-purple { background: #EEE8FF; color: #7047F5; border: 1px solid rgba(112, 71, 245, 0.25); }
        .badge-green  { background: #E8F8F1; color: #19B978; border: 1px solid rgba(25, 185, 120, 0.25); }
        .badge-amber  { background: #FEF6E8; color: #F5A623; border: 1px solid rgba(245, 166, 35, 0.25); }
        .badge-red    { background: #FEECEB; color: #EF5350; border: 1px solid rgba(239, 83, 80, 0.25); }
        .badge-blue   { background: #EBF3FF; color: #388BFF; border: 1px solid rgba(56, 139, 255, 0.25); }

        /* -------------------------------------------------------------
           11. Buttons (Primary Purple Gradient & Secondary White)
        ------------------------------------------------------------- */
        .stButton > button {
            border-radius: 10px !important;
            font-weight: 600 !important;
            font-size: 0.88rem !important;
            padding: 0.55rem 1.2rem !important;
            transition: all 0.18s cubic-bezier(0.16, 1, 0.3, 1) !important;
            border: 1px solid #E5E2F0 !important;
            background: #FFFFFF !important;
            color: #19213A !important;
        }
        .stButton > button:hover {
            border-color: #956BFF !important;
            color: #7047F5 !important;
            background: #F5F3FF !important;
            box-shadow: 0 2px 8px rgba(112, 71, 245, 0.1) !important;
        }
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #7047F5 0%, #956BFF 100%) !important;
            color: #FFFFFF !important;
            border: none !important;
            box-shadow: 0 4px 14px rgba(112, 71, 245, 0.26) !important;
        }
        .stButton > button[kind="primary"]:hover {
            background: linear-gradient(135deg, #6036E4 0%, #8558F4 100%) !important;
            box-shadow: 0 6px 18px rgba(112, 71, 245, 0.38) !important;
            transform: translateY(-1px);
        }

        .stDownloadButton > button {
            border-radius: 10px !important;
            font-weight: 600 !important;
            font-size: 0.88rem !important;
            padding: 0.55rem 1.2rem !important;
            border: 1px solid #E5E2F0 !important;
            background: #FFFFFF !important;
            color: #19213A !important;
            transition: all 0.18s ease !important;
        }
        .stDownloadButton > button:hover {
            border-color: #956BFF !important;
            color: #7047F5 !important;
            background: #F5F3FF !important;
        }
        .stDownloadButton > button[kind="primary"] {
            background: linear-gradient(135deg, #7047F5 0%, #956BFF 100%) !important;
            color: #FFFFFF !important;
            border: none !important;
            box-shadow: 0 4px 14px rgba(112, 71, 245, 0.26) !important;
        }

        /* -------------------------------------------------------------
           12. Tabs & Checkboxes
        ------------------------------------------------------------- */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
            border-bottom: 1px solid #E9E7F2;
            padding-bottom: 0.25rem;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px 8px 0 0;
            padding: 0.55rem 1.0rem;
            font-weight: 600;
            color: #66718B;
            border: none;
            background: transparent;
            font-size: 0.86rem;
        }
        .stTabs [aria-selected="true"] {
            color: #7047F5 !important;
            border-bottom: 2px solid #7047F5 !important;
        }

        [data-testid="stCheckbox"] label {
            color: #19213A !important;
            font-weight: 500 !important;
            font-size: 0.85rem !important;
        }
        [data-testid="stCheckbox"] input:checked + div {
            background-color: #7047F5 !important;
            border-color: #7047F5 !important;
        }

        /* Dividers */
        hr {
            margin: 1.25rem 0 !important;
            border-color: #E9E7F2 !important;
            opacity: 0.7;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
