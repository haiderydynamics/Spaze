"""Centralized Session State Management for SPAZE.

Guarantees data persistence across workflow stages, prevents accidental resets,
and avoids redundant computer vision processing or CVAE model re-execution.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
import streamlit as st

from models.floorplan_schema import FloorPlan, ScaleInfo


STAGE_NAMES = {
    "00_workspace": "00 — Workspace Overview",
    "01_floorplan": "01 — Floor Plan Studio",
    "02_generation": "02 — AI Layout Generator",
    "03_visualization": "03 — 3D Visualization Studio",
    "04_evaluation": "04 — Evaluation Lab",
    "05_export": "05 — Project Export Hub",
    "06_docs": "06 — Academic Research & Docs",
}


class StateManager:
    """Manages application state containers and stage transitions."""

    @staticmethod
    def initialize() -> None:
        """Initialize all session state containers if not present."""
        defaults: Dict[str, Any] = {
            "current_stage": "01_floorplan",
            "loaded_image": None,
            "processed_stages": None,
            "wall_candidates": [],
            "candidate_rooms": [],
            "floorplan": None,
            "scale_info": ScaleInfo(),
            "sample_loaded": False,
            "cvae_alternatives": [],
            "cvae_metrics": {},
            "selected_alt_idx": 0,
            "selected_preference": "space_efficient",
            "latent_temperature": 1.0,
            "random_seed": 42,
            "selected_room_id": None,
            "active_3d_result": None,
            "scene_cache": {},
            "viewer_camera_mode": "isometric",
            "viewer_theme": "light",
            "show_furniture_3d": True,
            "wall_height_3d": 2.80,
            "wall_thickness_3d": 0.20,
        }

        for key, default_val in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_val

    @staticmethod
    def get_stage() -> str:
        """Return the currently active stage identifier."""
        return st.session_state.get("current_stage", "01_floorplan")

    @staticmethod
    def set_stage(stage_id: str) -> None:
        """Update active workflow stage."""
        if stage_id in STAGE_NAMES:
            st.session_state.current_stage = stage_id

    @staticmethod
    def has_floorplan() -> bool:
        """Check if a valid floor plan is active."""
        fp = st.session_state.get("floorplan")
        if fp is None or not isinstance(fp, FloorPlan):
            return False
        rooms = fp.confirmed_rooms + fp.candidate_rooms
        return len(rooms) > 0

    @staticmethod
    def has_alternatives() -> bool:
        """Check if CVAE generated alternatives exist."""
        alts = st.session_state.get("cvae_alternatives", [])
        return len(alts) > 0

    @staticmethod
    def get_selected_alternative() -> Optional[FloorPlan]:
        """Return the currently selected generated alternative, or fallback to source floorplan."""
        alts = st.session_state.get("cvae_alternatives", [])
        idx = st.session_state.get("selected_alt_idx", 0)
        if alts and 0 <= idx < len(alts):
            return alts[idx]
        return st.session_state.get("floorplan")

    @staticmethod
    def get_stage_status(stage_id: str) -> str:
        """Return status badge text for a given stage."""
        if stage_id == "00_workspace":
            return "Active"
        elif stage_id == "01_floorplan":
            if StateManager.has_floorplan():
                return "Completed"
            if st.session_state.get("loaded_image") is not None:
                return "Processing"
            return "Ready"
        elif stage_id == "02_generation":
            if StateManager.has_alternatives():
                return "Generated"
            if StateManager.has_floorplan():
                return "Ready"
            return "Pending Input"
        elif stage_id == "03_visualization":
            if st.session_state.get("active_3d_result") is not None:
                return "Generated"
            if StateManager.has_floorplan() or StateManager.has_alternatives():
                return "Ready"
            return "Pending Input"
        elif stage_id == "04_evaluation":
            if StateManager.has_alternatives() or StateManager.has_floorplan():
                return "Available"
            return "Pending Data"
        elif stage_id == "05_export":
            if StateManager.has_alternatives() or StateManager.has_floorplan():
                return "Ready"
            return "Pending Data"
        elif stage_id == "06_docs":
            return "Available"
        return ""
