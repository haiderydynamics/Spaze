"""Validation module for SPAZE floor-plan data and geometric entities.

Provides comprehensive checks for:
- Image dimensions and channels
- Room bounding boxes and containment
- Polygon topological validity via Shapely
- Room ID uniqueness
- Scale parameters and measurement inputs
"""
from typing import Any, Dict, Iterable, List, Optional, Tuple
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.validation import explain_validity

from config.settings import AppConfig


class ValidationError(ValueError):
    """Raised when an architectural floor-plan entity fails validation."""
    pass


def validate_image_dimensions(
    width: int,
    height: int,
    channels: Optional[int] = None,
    min_dim: int = AppConfig.MIN_IMAGE_DIMENSION,
    max_dim: int = AppConfig.MAX_IMAGE_DIMENSION,
) -> None:
    """Validate that image dimensions are within safe processing bounds.

    Raises:
        ValidationError: If dimensions are out of bounds or non-positive.
    """
    if width <= 0 or height <= 0:
        raise ValidationError(f"Image dimensions must be positive integers, got {width}x{height}.")

    if width < min_dim or height < min_dim:
        raise ValidationError(
            f"Image dimensions ({width}x{height}) are smaller than minimum allowed threshold ({min_dim}x{min_dim})."
        )

    if width > max_dim or height > max_dim:
        raise ValidationError(
            f"Image dimensions ({width}x{height}) exceed maximum allowed dimension ({max_dim}x{max_dim})."
        )

    if channels is not None and channels not in (1, 3, 4):
        raise ValidationError(f"Unsupported image channel count: {channels}. Expected 1 (gray), 3 (RGB/BGR), or 4 (RGBA).")


def validate_room_bounds(
    x: float,
    y: float,
    width: float,
    height: float,
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
) -> None:
    """Validate room bounding box coordinates and containment.

    Raises:
        ValidationError: If coordinates or dimensions are invalid.
    """
    if width <= 0 or height <= 0:
        raise ValidationError(f"Room dimensions must be positive, got width={width}, height={height}.")

    if x < 0 or y < 0:
        raise ValidationError(f"Room top-left coordinates cannot be negative, got ({x}, {y}).")

    if image_width is not None and (x + width) > image_width + 1.0:
        raise ValidationError(
            f"Room extends beyond image width: x={x} + width={width} = {x + width} > image_width={image_width}."
        )

    if image_height is not None and (y + height) > image_height + 1.0:
        raise ValidationError(
            f"Room extends beyond image height: y={y} + height={height} = {y + height} > image_height={image_height}."
        )


def validate_polygon_geometry(
    vertices: List[Tuple[float, float]],
    min_area: float = 1.0,
) -> Tuple[bool, str]:
    """Validate a 2D polygon using Shapely for topological soundness.

    Returns:
        Tuple of (is_valid: bool, explanation: str).
    """
    if len(vertices) < 3:
        return False, f"Degenerate polygon: requires at least 3 vertices, got {len(vertices)}."

    try:
        poly = ShapelyPolygon(vertices)
    except Exception as exc:
        return False, f"Failed to construct polygon: {exc}"

    if not poly.is_valid:
        reason = explain_validity(poly)
        return False, f"Invalid polygon topology: {reason}"

    if poly.is_empty:
        return False, "Polygon is geometrically empty."

    if poly.area < min_area:
        return False, f"Polygon area ({poly.area:.2f}) is below minimum threshold ({min_area})."

    return True, "Valid polygon"


def validate_unique_room_ids(room_ids: Iterable[str]) -> None:
    """Ensure that all room identifiers are strictly unique.

    Raises:
        ValidationError: If duplicate IDs are detected.
    """
    seen = set()
    duplicates = []
    for rid in room_ids:
        if rid in seen:
            duplicates.append(rid)
        seen.add(rid)

    if duplicates:
        raise ValidationError(f"Duplicate room IDs detected: {', '.join(set(duplicates))}")


def validate_scale_parameters(pixel_distance: float, real_meters: float) -> None:
    """Validate scale calibration inputs.

    Raises:
        ValidationError: If inputs are invalid or out of realistic range.
    """
    if pixel_distance <= 0:
        raise ValidationError(f"Pixel distance must be strictly positive, got {pixel_distance}.")

    if real_meters <= 0:
        raise ValidationError(f"Real-world length must be strictly positive, got {real_meters} meters.")

    if real_meters > 500.0:
        raise ValidationError(f"Real-world measurement ({real_meters} m) exceeds maximum single-wall limit (500 m).")

    ppm = pixel_distance / real_meters
    if ppm < 0.1 or ppm > 5000.0:
        raise ValidationError(
            f"Calculated scale ({ppm:.2f} px/m) is outside realistic architectural limits (0.1 to 5000 px/m)."
        )
