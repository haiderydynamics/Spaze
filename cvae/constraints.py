"""Geometric Constraints and Validation Module for SPAZE Generated Floor Plans.

Implements rigorous computational geometry checks using Shapely:
- Room overlap ratio
- Boundary violation ratio
- Adjacency geometric consistency
- Room type preservation
- Constrained coordinate relaxation / boundary clamping
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from shapely.geometry import Polygon, box
from shapely.validation import explain_validity

from models.floorplan_schema import FloorPlan, Room


@dataclass
class GeometricEvaluationResult:
    """Detailed geometric validation and constraint satisfaction results."""
    is_valid: bool
    overall_validity_score: float  # [0.0, 1.0]
    overlap_ratio: float           # 0.0 = perfect non-overlapping
    boundary_violation_ratio: float # 0.0 = completely within boundary
    adjacency_consistency: float   # [0.0, 1.0] consistency between matrix and geometry
    room_count: int
    rooms_preserved_ratio: float   # [0.0, 1.0]
    violations: List[str] = field(default_factory=list)


class GeometricConstraintsValidator:
    """Validates and refines geometric constraints for architectural layouts."""

    def __init__(
        self,
        max_allowed_overlap_ratio: float = 0.05,
        adjacency_contact_tolerance_px: float = 12.0,
    ) -> None:
        self.max_allowed_overlap_ratio = max_allowed_overlap_ratio
        self.contact_tolerance = adjacency_contact_tolerance_px

    def evaluate_layout(
        self,
        floorplan: FloorPlan,
        source_floorplan: Optional[FloorPlan] = None,
    ) -> GeometricEvaluationResult:
        """Run comprehensive geometric checks on a floor plan.

        Args:
            floorplan: The generated or candidate FloorPlan to validate.
            source_floorplan: Optional source FloorPlan to verify room-program preservation.

        Returns:
            GeometricEvaluationResult with numerical scores and violation logs.
        """
        violations: List[str] = []
        rooms = floorplan.confirmed_rooms + floorplan.candidate_rooms
        num_rooms = len(rooms)

        if num_rooms == 0:
            return GeometricEvaluationResult(
                is_valid=False,
                overall_validity_score=0.0,
                overlap_ratio=1.0,
                boundary_violation_ratio=1.0,
                adjacency_consistency=0.0,
                room_count=0,
                rooms_preserved_ratio=0.0,
                violations=["Layout contains 0 active rooms."],
            )

        img_w = float(floorplan.image_width)
        img_h = float(floorplan.image_height)
        canvas_box = box(0.0, 0.0, img_w, img_h)

        total_room_area = 0.0
        total_overlap_area = 0.0
        total_outside_area = 0.0
        polygons: List[Polygon] = []

        # 1. Individual Room Geometry & Boundary Compliance
        for r in rooms:
            if r.width <= 0 or r.height <= 0:
                violations.append(f"Room {r.room_id} has non-positive dimensions ({r.width}x{r.height}).")

            p = box(r.x, r.y, r.x + r.width, r.y + r.height)
            total_room_area += p.area

            # Boundary containment check
            if not canvas_box.contains(p):
                outside_part = p.difference(canvas_box)
                total_outside_area += outside_part.area
                if outside_part.area > 5.0:
                    violations.append(f"Room {r.room_id} extends outside boundary by {outside_part.area:.1f} px².")

            # Topological check
            if not p.is_valid:
                violations.append(f"Room {r.room_id} has invalid polygon: {explain_validity(p)}")

            polygons.append(p)

        # 2. Pairwise Room Overlap Checks
        for i in range(num_rooms):
            for j in range(i + 1, num_rooms):
                p1 = polygons[i]
                p2 = polygons[j]
                if p1.intersects(p2):
                    intersection = p1.intersection(p2)
                    if intersection.area > 1.0:  # Ignore 1px border contact
                        total_overlap_area += intersection.area
                        violations.append(
                            f"Collision between {rooms[i].room_id} and {rooms[j].room_id}: "
                            f"{intersection.area:.1f} px² overlap."
                        )

        overlap_ratio = (total_overlap_area / total_room_area) if total_room_area > 0 else 1.0
        boundary_violation_ratio = (total_outside_area / total_room_area) if total_room_area > 0 else 1.0

        # 3. Adjacency Consistency Check
        adj_consistent_pairs = 0
        total_adj_claims = 0

        if floorplan.adjacency_matrix and len(floorplan.adjacency_matrix) == num_rooms:
            matrix = floorplan.adjacency_matrix
            for i in range(num_rooms):
                for j in range(i + 1, num_rooms):
                    if matrix[i][j] == 1:
                        total_adj_claims += 1
                        dist = polygons[i].distance(polygons[j])
                        if dist <= self.contact_tolerance:
                            adj_consistent_pairs += 1
                        else:
                            violations.append(
                                f"Adjacency claim failed between {rooms[i].room_id} and {rooms[j].room_id}: "
                                f"distance {dist:.1f} px > {self.contact_tolerance} px."
                            )

        adjacency_consistency = (adj_consistent_pairs / total_adj_claims) if total_adj_claims > 0 else 1.0

        # 4. Room Program Preservation Check
        rooms_preserved_ratio = 1.0
        if source_floorplan is not None:
            src_rooms = source_floorplan.confirmed_rooms + source_floorplan.candidate_rooms
            src_types = sorted([r.room_type for r in src_rooms])
            tgt_types = sorted([r.room_type for r in rooms])

            # Multiset matching
            matched = 0
            temp_tgt = list(tgt_types)
            for st in src_types:
                if st in temp_tgt:
                    matched += 1
                    temp_tgt.remove(st)

            rooms_preserved_ratio = matched / max(1, len(src_types))
            if rooms_preserved_ratio < 0.6:
                violations.append(
                    f"Room program mismatch: only {matched}/{len(src_types)} required room types preserved."
                )

        # 5. Composite Validity Score
        overlap_penalty = min(1.0, overlap_ratio * 5.0)
        boundary_penalty = min(1.0, boundary_violation_ratio * 5.0)
        adjacency_score = adjacency_consistency
        preservation_score = rooms_preserved_ratio

        validity_score = float(
            0.40 * (1.0 - overlap_penalty)
            + 0.25 * (1.0 - boundary_penalty)
            + 0.20 * preservation_score
            + 0.15 * adjacency_score
        )
        validity_score = max(0.0, min(1.0, round(validity_score, 3)))

        is_valid = (
            overlap_ratio <= self.max_allowed_overlap_ratio
            and boundary_violation_ratio <= 0.05
            and rooms_preserved_ratio >= 0.50
        )

        return GeometricEvaluationResult(
            is_valid=is_valid,
            overall_validity_score=validity_score,
            overlap_ratio=round(overlap_ratio, 4),
            boundary_violation_ratio=round(boundary_violation_ratio, 4),
            adjacency_consistency=round(adjacency_consistency, 3),
            room_count=num_rooms,
            rooms_preserved_ratio=round(rooms_preserved_ratio, 3),
            violations=violations,
        )

    def relax_and_clamp_layout(
        self,
        floorplan: FloorPlan,
        max_iterations: int = 15,
        repulsion_step: float = 4.0,
    ) -> FloorPlan:
        """Apply non-destructive geometric relaxation and boundary clamping.

        Pushes overlapping rooms apart along the minimum separating axis and
        clamps rooms strictly inside canvas boundaries.
        """
        rooms = floorplan.confirmed_rooms + floorplan.candidate_rooms
        num_rooms = len(rooms)
        if num_rooms <= 1:
            return floorplan

        img_w = float(floorplan.image_width)
        img_h = float(floorplan.image_height)

        for _ in range(max_iterations):
            had_overlap = False
            for i in range(num_rooms):
                r1 = rooms[i]
                for j in range(i + 1, num_rooms):
                    r2 = rooms[j]

                    # Bounding box collision check
                    overlap_x = min(r1.x + r1.width, r2.x + r2.width) - max(r1.x, r2.x)
                    overlap_y = min(r1.y + r1.height, r2.y + r2.height) - max(r1.y, r2.y)

                    if overlap_x > 0.0 and overlap_y > 0.0:
                        had_overlap = True
                        # Repel along the shallowest penetration axis
                        if overlap_x < overlap_y:
                            delta = min(repulsion_step, overlap_x / 2.0)
                            if (r1.x + r1.width / 2.0) < (r2.x + r2.width / 2.0):
                                r1.x -= delta
                                r2.x += delta
                            else:
                                r1.x += delta
                                r2.x -= delta
                        else:
                            delta = min(repulsion_step, overlap_y / 2.0)
                            if (r1.y + r1.height / 2.0) < (r2.y + r2.height / 2.0):
                                r1.y -= delta
                                r2.y += delta
                            else:
                                r1.y += delta
                                r2.y -= delta

            # Clamp each room strictly to canvas boundary
            for r in rooms:
                r.x = max(0.0, min(img_w - r.width, r.x))
                r.y = max(0.0, min(img_h - r.height, r.y))
                # Update polygon
                r.polygon = [
                    (round(r.x, 1), round(r.y, 1)),
                    (round(r.x + r.width, 1), round(r.y, 1)),
                    (round(r.x + r.width, 1), round(r.y + r.height, 1)),
                    (round(r.x, 1), round(r.y + r.height, 1)),
                ]

            if not had_overlap:
                break

        # Recompute physical dimensions and adjacency
        if floorplan.scale_information.is_calibrated:
            for r in rooms:
                r.apply_scale(floorplan.scale_information)

        floorplan.compute_adjacency()
        return floorplan
