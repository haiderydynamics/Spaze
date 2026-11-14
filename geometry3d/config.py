"""Configuration constants and parameters for 3D procedural geometry generation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass(frozen=True)
class Geometry3DConfig:
    """Configurable parameters for architectural 3D geometry synthesis."""

    # 1. Structural Dimensions (in real-world meters)
    default_wall_height: float = 2.80
    exterior_wall_thickness: float = 0.20
    interior_wall_thickness: float = 0.12
    floor_thickness: float = 0.10

    # 2. Opening Dimensions (in real-world meters)
    door_width: float = 0.90
    door_height: float = 2.10
    door_wall_clearance: float = 0.20  # Minimum distance from corners

    window_width: float = 1.20
    window_height: float = 1.20
    window_sill_height: float = 0.90
    window_wall_clearance: float = 0.35

    # 3. Default Scale Conversion (if floor plan has no explicit scale)
    default_pixels_per_meter: float = 50.0  # 50 px = 1.0 m

    # 4. Feature Toggles
    generate_furniture: bool = True
    generate_doors: bool = True
    generate_windows: bool = True
    generate_ceilings: bool = False  # Keep open top for interior inspection

    # 5. Geometric Tolerances
    contact_tolerance_m: float = 0.20       # Maximum gap to consider rooms touching
    min_wall_segment_length_m: float = 0.40 # Filter out degenerate short wall ticks
    furniture_clearance_m: float = 0.08     # Distance between furniture and walls

    # 6. Material Palettes
    palette_name: str = "architectural_clean"
