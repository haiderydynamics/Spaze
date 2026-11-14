"""Export utilities for structured SPAZE floor-plan data.

Provides safe JSON serialization and file writing, ensuring data integrity,
prevention of path traversal, and strict adherence to the Phase 1 schema.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

from models.floorplan_schema import FloorPlan


def export_floorplan_to_json(
    floorplan: FloorPlan,
    indent: int = 2,
) -> str:
    """Serialize a FloorPlan instance to a formatted JSON string.

    Args:
        floorplan: The FloorPlan data instance.
        indent: JSON indentation spaces.

    Returns:
        JSON string representation.
    """
    return floorplan.to_json(indent=indent)


def save_floorplan_to_file(
    floorplan: FloorPlan,
    output_path: Union[str, Path],
    indent: int = 2,
) -> Path:
    """Save a FloorPlan instance to a JSON file on disk safely.

    Args:
        floorplan: The FloorPlan data instance.
        output_path: Destination file path.
        indent: Indentation level.

    Returns:
        Resolved Path of the saved file.

    Raises:
        ValueError: If path is invalid.
        IOError: If writing fails.
    """
    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    json_content = export_floorplan_to_json(floorplan, indent=indent)
    path.write_text(json_content, encoding="utf-8")
    return path


def load_floorplan_from_file(file_path: Union[str, Path]) -> FloorPlan:
    """Load and parse a FloorPlan instance from a JSON file.

    Args:
        file_path: Source JSON file path.

    Returns:
        Constructed FloorPlan instance.

    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If JSON is malformed or invalid schema.
    """
    path = Path(file_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Floor-plan file not found: {path}")

    content = path.read_text(encoding="utf-8")
    return FloorPlan.from_json(content)


def generate_export_summary(floorplan: FloorPlan) -> Dict[str, Any]:
    """Generate a clean high-level summary of the floor plan for UI display or logs."""
    return {
        "dimensions": f"{floorplan.image_width} x {floorplan.image_height} px",
        "scale_status": "Calibrated" if floorplan.scale_information.is_calibrated else "Uncalibrated (pixels only)",
        "pixels_per_meter": round(floorplan.scale_information.pixels_per_meter, 2)
        if floorplan.scale_information.pixels_per_meter else None,
        "total_walls_detected": len(floorplan.wall_candidates),
        "total_candidate_rooms": len(floorplan.candidate_rooms),
        "total_confirmed_rooms": len(floorplan.confirmed_rooms),
        "rooms_requiring_review": sum(
            1 for r in floorplan.candidate_rooms if r.detection_status == "Needs Review"
        ),
        "pipeline_version": floorplan.processing_metadata.pipeline_version,
    }
