"""Comprehensive Geometric Validation Engine for 3D Generation.

Validates building boundaries, room polygons, pairwise overlaps,
and architectural code sanity before procedural 3D mesh synthesis.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from shapely.geometry import Polygon, box
from shapely.validation import explain_validity

from geometry3d.config import Geometry3DConfig
from geometry3d.coordinate_system import CoordinateTransformer
from models.floorplan_schema import FloorPlan, Room


@dataclass
class ValidationReport:
    """Structured report assessing geometric feasibility for 3D generation."""
    valid: bool
    overall_score: float
    critical_errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    room_count: int = 0
    overlap_count: int = 0
    boundary_violation_count: int = 0
    adjacency_consistency: float = 1.0
    furniture_collision_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class Layout3DValidator:
    """Validates 2D floor plans prior to 3D geometry generation."""

    def __init__(self, config: Optional[Geometry3DConfig] = None) -> None:
        self.config = config or Geometry3DConfig()

    def validate_layout(
        self,
        floorplan: FloorPlan,
        transformer: Optional[CoordinateTransformer] = None,
    ) -> ValidationReport:
        """Validate layout against physical and computational geometry constraints.

        Args:
            floorplan: FloorPlan schema instance.
            transformer: CoordinateTransformer linking pixels to meters.

        Returns:
            ValidationReport with boolean pass status and error logs.
        """
        tf = transformer or CoordinateTransformer.from_floorplan(floorplan, self.config)
        critical_errors: List[str] = []
        warnings: List[str] = []

        rooms = floorplan.confirmed_rooms + floorplan.candidate_rooms
        num_rooms = len(rooms)

        # 1. Building Boundary Checks
        if floorplan.image_width <= 0 or floorplan.image_height <= 0:
            critical_errors.append(f"Invalid canvas dimensions: {floorplan.image_width}x{floorplan.image_height}")

        if num_rooms == 0:
            critical_errors.append("Layout contains 0 active rooms. Cannot synthesize 3D geometry.")
            return ValidationReport(
                valid=False,
                overall_score=0.0,
                critical_errors=critical_errors,
                warnings=warnings,
                room_count=0,
            )

        canvas_box = box(0, 0, floorplan.image_width, floorplan.image_height)

        # 2. Individual Room Polygons and Dimension Checks
        polygons: List[Polygon] = []
        boundary_violations = 0

        for r in rooms:
            # Check dimensions in meters
            w_m = tf.to_meters(r.width)
            h_m = tf.to_meters(r.height)

            if r.width <= 0 or r.height <= 0:
                critical_errors.append(f"Room '{r.label}' ({r.room_id}) has non-positive dimensions ({r.width}x{r.height}).")

            if w_m < 0.80 or h_m < 0.80:
                warnings.append(f"Room '{r.label}' has narrow physical dimension ({w_m:.2f}m x {h_m:.2f}m).")

            aspect = max(r.width / max(1.0, r.height), r.height / max(1.0, r.width))
            if aspect > 8.0:
                warnings.append(f"Room '{r.label}' has extreme aspect ratio ({aspect:.1f}:1).")

            # Polygon checks
            if not r.polygon or len(r.polygon) < 3:
                critical_errors.append(f"Room '{r.label}' has degenerate polygon ({len(r.polygon)} vertices).")
                poly = box(r.x, r.y, r.x + r.width, r.y + r.height)
            else:
                try:
                    poly = Polygon(r.polygon)
                    if not poly.is_valid:
                        reason = explain_validity(poly)
                        warnings.append(f"Room '{r.label}' self-intersects: {reason}. Attempting buffer(0) repair.")
                        poly = poly.buffer(0)
                except Exception as exc:
                    critical_errors.append(f"Failed to construct polygon for '{r.label}': {exc}")
                    poly = box(r.x, r.y, r.x + r.width, r.y + r.height)

            # Canvas containment
            if not canvas_box.contains(poly):
                boundary_violations += 1
                outside = poly.difference(canvas_box)
                if outside.area > 5.0:
                    warnings.append(f"Room '{r.label}' extends beyond building perimeter by {outside.area:.1f} px².")

            polygons.append(poly)

        # 3. Pairwise Room Overlap Checks
        overlap_count = 0
        total_overlap_area = 0.0
        total_area = sum(p.area for p in polygons if p.is_valid)

        for i in range(num_rooms):
            for j in range(i + 1, num_rooms):
                p1, p2 = polygons[i], polygons[j]
                if p1.is_valid and p2.is_valid and p1.intersects(p2):
                    inter = p1.intersection(p2)
                    if inter.area > 2.0:  # Ignore 1-2px border touching
                        overlap_count += 1
                        total_overlap_area += inter.area
                        warnings.append(
                            f"Room collision: '{rooms[i].label}' and '{rooms[j].label}' "
                            f"overlap by {inter.area:.1f} px² ({tf.to_meters(inter.area ** 0.5):.2f}m equivalent)."
                        )

        overlap_ratio = (total_overlap_area / max(1.0, total_area))

        if overlap_ratio > 0.15:
            critical_errors.append(f"Excessive room interior overlap ({overlap_ratio * 100:.1f}% of total area).")

        # 4. Adjacency Consistency
        adj_claims = 0
        adj_valid = 0
        if floorplan.adjacency_matrix and len(floorplan.adjacency_matrix) == num_rooms:
            mat = floorplan.adjacency_matrix
            for i in range(num_rooms):
                for j in range(i + 1, num_rooms):
                    if mat[i][j] == 1:
                        adj_claims += 1
                        dist_m = tf.to_meters(polygons[i].distance(polygons[j]))
                        if dist_m <= self.config.contact_tolerance_m:
                            adj_valid += 1
                        else:
                            warnings.append(
                                f"Adjacency discrepancy: '{rooms[i].label}' and '{rooms[j].label}' "
                                f"are separated by {dist_m:.2f}m > tolerance ({self.config.contact_tolerance_m}m)."
                            )

        adjacency_consistency = (adj_valid / max(1, adj_claims)) if adj_claims > 0 else 1.0

        # Calculate overall validity score [0.0, 1.0]
        has_critical = len(critical_errors) > 0
        penalty = min(0.6, overlap_ratio * 3.0) + min(0.3, (boundary_violations / num_rooms) * 0.3)
        overall_score = 0.0 if has_critical else max(0.40, min(1.0, 1.0 - penalty))
        overall_score = round(float(overall_score), 3)

        is_valid = (not has_critical) and (overlap_ratio <= 0.10)

        return ValidationReport(
            valid=is_valid,
            overall_score=overall_score,
            critical_errors=critical_errors,
            warnings=warnings,
            room_count=num_rooms,
            overlap_count=overlap_count,
            boundary_violation_count=boundary_violations,
            adjacency_consistency=round(adjacency_consistency, 3),
            furniture_collision_count=0,
        )
