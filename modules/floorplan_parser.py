"""Floor-Plan Parser for SPAZE.

Assembles raw computer vision outputs (image attributes, wall lines, candidate rooms)
and user modifications into the structured FloorPlan schema.
Fully decoupled from Streamlit for headless execution, automated testing, and
direct integration with future Phase 2 (CVAE) and Phase 3 (3D generation) engines.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from models.floorplan_schema import (
    FloorPlan,
    ProcessingMetadata,
    Room,
    ScaleInfo,
    WallCandidate,
)
from modules.image_loader import LoadedImage
from modules.image_processor import ProcessedStages


class FloorPlanParser:
    """Parses CV outputs and builds or updates validated FloorPlan objects."""

    @staticmethod
    def build_initial_floorplan(
        loaded_image: LoadedImage,
        processed_stages: ProcessedStages,
        wall_candidates: List[WallCandidate],
        candidate_rooms: List[Room],
        scale_info: Optional[ScaleInfo] = None,
        warnings: Optional[List[str]] = None,
    ) -> FloorPlan:
        """Construct a new structured FloorPlan instance from initial CV pipeline results.

        Args:
            loaded_image: LoadedImage metadata from the loader.
            processed_stages: Intermediate CV outputs and timings.
            wall_candidates: Detected wall lines.
            candidate_rooms: Extracted candidate rooms.
            scale_info: Optional initial scale calibration.
            warnings: Any non-fatal issues detected during processing.

        Returns:
            A fully populated FloorPlan schema instance.
        """
        active_scale = scale_info or ScaleInfo()
        active_warnings = list(warnings or [])

        if not candidate_rooms:
            active_warnings.append(
                "No candidate rooms detected. This may occur if walls are incomplete, "
                "lines are faint, or the drawing does not follow enclosed architectural conventions."
            )

        if not wall_candidates:
            active_warnings.append(
                "No wall candidates detected with current Hough line parameters. "
                "The drawing may consist solely of non-rectilinear or disconnected boundaries."
            )

        timestamp_iso = datetime.now(timezone.utc).isoformat()

        metadata = ProcessingMetadata(
            pipeline_version="phase1-cv-v1",
            timestamp_utc=timestamp_iso,
            processing_time_ms=processed_stages.processing_time_ms,
            image_filename=loaded_image.filename,
            image_sha256=loaded_image.sha256,
            warnings=active_warnings,
            parameters_used=processed_stages.parameters,
        )

        floorplan = FloorPlan(
            image_width=loaded_image.width,
            image_height=loaded_image.height,
            coordinate_system="top_left_origin",
            scale_information=active_scale,
            wall_candidates=list(wall_candidates),
            candidate_rooms=[],
            confirmed_rooms=[],
            processing_metadata=metadata,
        )

        for room in candidate_rooms:
            floorplan.add_candidate_room(room)

        return floorplan

    @staticmethod
    def update_floorplan_with_rooms(
        floorplan: FloorPlan,
        all_rooms: List[Room],
    ) -> FloorPlan:
        """Update a FloorPlan instance with a modified list of rooms.

        Separates rooms into confirmed_rooms vs candidate_rooms based on
        is_user_confirmed flag or detection_status.

        Args:
            floorplan: Existing FloorPlan instance.
            all_rooms: Updated list of Room instances.

        Returns:
            Updated FloorPlan instance.
        """
        candidates: List[Room] = []
        confirmed: List[Room] = []

        for room in all_rooms:
            room.apply_scale(floorplan.scale_information)
            if room.is_user_confirmed or room.detection_status == "User Confirmed":
                room.is_user_confirmed = True
                room.detection_status = "User Confirmed"
                confirmed.append(room)
            else:
                candidates.append(room)

        floorplan.candidate_rooms = candidates
        floorplan.confirmed_rooms = confirmed
        return floorplan
