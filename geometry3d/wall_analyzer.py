"""Wall Identification, Classification, and Deduplication Engine.

Extracts boundary segments from 2D room layouts, classifies them into
exterior building walls versus interior room partitions, deduplicates shared
boundaries to prevent overlapping geometry, and records room adjacencies.
"""
from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Point, Polygon, box
from shapely.ops import linemerge, unary_union

from geometry3d.config import Geometry3DConfig
from geometry3d.coordinate_system import CoordinateTransformer
from models.floorplan_schema import FloorPlan, Room


@dataclass
class WallSegment:
    """Represents a discrete structural wall in 3D world meters."""
    id: str
    start: Tuple[float, float]          # (x, y) start point in meters
    end: Tuple[float, float]            # (x, y) end point in meters
    wall_type: str                      # 'exterior' or 'interior'
    thickness: float                    # Wall thickness in meters
    height: float                       # Wall height in meters
    adjacent_room_ids: List[str]        # Room IDs bounding this wall
    length: float = field(init=False)
    normal: Tuple[float, float] = field(init=False)  # 2D unit normal vector
    angle_rad: float = field(init=False)

    def __post_init__(self) -> None:
        dx = self.end[0] - self.start[0]
        dy = self.end[1] - self.start[1]
        self.length = float(math.hypot(dx, dy))
        self.angle_rad = float(math.atan2(dy, dx))
        if self.length > 1e-6:
            # Perpendicular normal (-dy, dx) normalized
            self.normal = (-dy / self.length, dx / self.length)
        else:
            self.normal = (0.0, 0.0)

    @property
    def midpoint(self) -> Tuple[float, float]:
        """Return (x, y) midpoint of the wall segment."""
        return (
            (self.start[0] + self.end[0]) / 2.0,
            (self.start[1] + self.end[1]) / 2.0,
        )


class WallAnalyzer:
    """Analyzes floor-plan geometry to derive deduplicated 3D wall segments."""

    def __init__(self, config: Optional[Geometry3DConfig] = None) -> None:
        self.config = config or Geometry3DConfig()

    def analyze(
        self,
        floorplan: FloorPlan,
        transformer: Optional[CoordinateTransformer] = None,
    ) -> List[WallSegment]:
        """Extract and classify all exterior and interior wall segments.

        Args:
            floorplan: FloorPlan instance containing confirmed or candidate rooms.
            transformer: CoordinateTransformer to map 2D coordinates to 3D meters.

        Returns:
            List of deduplicated, typed WallSegment instances in 3D meters.
        """
        tf = transformer or CoordinateTransformer.from_floorplan(floorplan, self.config)
        rooms = [r for r in (floorplan.confirmed_rooms + floorplan.candidate_rooms) if len(r.polygon) >= 3]

        if not rooms:
            return []

        # Convert all room polygons into 3D world meters (planar x, y)
        room_data: List[Tuple[Room, Polygon]] = []
        for r in rooms:
            poly_3d_pts = tf.to_3d_planar_polygon(r.polygon)
            poly = Polygon(poly_3d_pts)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.is_valid and poly.area > 0.01:
                room_data.append((r, poly))

        if not room_data:
            return []

        # Extract segments from all rooms
        raw_segments: List[Tuple[Tuple[float, float], Tuple[float, float], str]] = []
        # Store for each segment: (start, end, room_id)
        for r, poly in room_data:
            coords = list(poly.exterior.coords)
            for i in range(len(coords) - 1):
                p1 = (round(coords[i][0], 3), round(coords[i][1], 3))
                p2 = (round(coords[i + 1][0], 3), round(coords[i + 1][1], 3))
                if math.hypot(p2[0] - p1[0], p2[1] - p1[1]) >= self.config.min_wall_segment_length_m:
                    raw_segments.append((p1, p2, r.id))

        # Classify and deduplicate segments
        wall_segments = self._classify_and_deduplicate(
            raw_segments=raw_segments,
            room_data=room_data,
        )

        return wall_segments

    def _classify_and_deduplicate(
        self,
        raw_segments: List[Tuple[Tuple[float, float], Tuple[float, float], str]],
        room_data: List[Tuple[Room, Polygon]],
    ) -> List[WallSegment]:
        """Match collinear overlapping segments and classify as exterior or interior."""
        results: List[WallSegment] = []
        used_indices: Set[int] = set()

        num_segs = len(raw_segments)

        for i in range(num_segs):
            if i in used_indices:
                continue

            p1_a, p2_a, room_a = raw_segments[i]
            matched_partner = False

            # Check against subsequent segments to find shared interior partitions
            for j in range(i + 1, num_segs):
                if j in used_indices:
                    continue

                p1_b, p2_b, room_b = raw_segments[j]
                if room_a == room_b:
                    continue

                # Test for collinearity and overlapping range
                if self._are_segments_collinear_and_overlapping(p1_a, p2_a, p1_b, p2_b):
                    used_indices.add(i)
                    used_indices.add(j)
                    matched_partner = True

                    # Merge the two segments into one consolidated interior wall
                    start, end = self._compute_shared_overlap(p1_a, p2_a, p1_b, p2_b)
                    if math.hypot(end[0] - start[0], end[1] - start[1]) >= self.config.min_wall_segment_length_m:
                        wall = WallSegment(
                            id=f"wall_int_{uuid.uuid4().hex[:6]}",
                            start=start,
                            end=end,
                            wall_type="interior",
                            thickness=self.config.interior_wall_thickness,
                            height=self.config.default_wall_height,
                            adjacent_room_ids=[room_a, room_b],
                        )
                        results.append(wall)
                    break

            if not matched_partner:
                # Segment has no direct opposing room edge; determine if it is exterior or interior
                used_indices.add(i)
                # Check if this edge is adjacent to another room polygon within contact tolerance
                adj_room = self._find_adjacent_room(p1_a, p2_a, room_a, room_data)
                if adj_room is not None:
                    wall_type = "interior"
                    thickness = self.config.interior_wall_thickness
                    adj_ids = [room_a, adj_room]
                else:
                    wall_type = "exterior"
                    thickness = self.config.exterior_wall_thickness
                    adj_ids = [room_a]

                wall = WallSegment(
                    id=f"wall_{'ext' if wall_type == 'exterior' else 'int'}_{uuid.uuid4().hex[:6]}",
                    start=p1_a,
                    end=p2_a,
                    wall_type=wall_type,
                    thickness=thickness,
                    height=self.config.default_wall_height,
                    adjacent_room_ids=adj_ids,
                )
                results.append(wall)

        return results

    def _are_segments_collinear_and_overlapping(
        self,
        p1_a: Tuple[float, float],
        p2_a: Tuple[float, float],
        p1_b: Tuple[float, float],
        p2_b: Tuple[float, float],
        tol_dist: float = 0.15,
        tol_angle: float = 0.15,  # ~8.5 degrees
    ) -> bool:
        """Check if two directed 2D line segments are approximately collinear and overlap."""
        line_a = LineString([p1_a, p2_a])
        line_b = LineString([p1_b, p2_b])

        # Distance check
        if line_a.distance(line_b) > tol_dist:
            return False

        dx_a = p2_a[0] - p1_a[0]
        dy_a = p2_a[1] - p1_a[1]
        dx_b = p2_b[0] - p1_b[0]
        dy_b = p2_b[1] - p1_b[1]

        ang_a = math.atan2(dy_a, dx_a)
        ang_b = math.atan2(dy_b, dx_b)

        # Parallel or antiparallel test
        if abs(math.sin(ang_a - ang_b)) > tol_angle:
            return False

        # Projection overlap test along axis of segment A
        len_a = math.hypot(dx_a, dy_a)
        if len_a < 1e-4:
            return False
        ux, uy = dx_a / len_a, dy_a / len_a

        # Project points onto line A axis
        proj_b1 = (p1_b[0] - p1_a[0]) * ux + (p1_b[1] - p1_a[1]) * uy
        proj_b2 = (p2_b[0] - p1_a[0]) * ux + (p2_b[1] - p1_a[1]) * uy

        min_b = min(proj_b1, proj_b2)
        max_b = max(proj_b1, proj_b2)

        # Overlap interval with [0, len_a]
        overlap_start = max(0.0, min_b)
        overlap_end = min(len_a, max_b)

        return (overlap_end - overlap_start) > self.config.min_wall_segment_length_m * 0.5

    def _compute_shared_overlap(
        self,
        p1_a: Tuple[float, float],
        p2_a: Tuple[float, float],
        p1_b: Tuple[float, float],
        p2_b: Tuple[float, float],
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Compute the average overlapping line segment between two parallel segments."""
        dx_a = p2_a[0] - p1_a[0]
        dy_a = p2_a[1] - p1_a[1]
        len_a = math.hypot(dx_a, dy_a)
        ux, uy = dx_a / len_a, dy_a / len_a

        proj_b1 = (p1_b[0] - p1_a[0]) * ux + (p1_b[1] - p1_a[1]) * uy
        proj_b2 = (p2_b[0] - p1_a[0]) * ux + (p2_b[1] - p1_a[1]) * uy

        min_b = min(proj_b1, proj_b2)
        max_b = max(proj_b1, proj_b2)

        u_start = max(0.0, min_b)
        u_end = min(len_a, max_b)

        start = (round(p1_a[0] + u_start * ux, 3), round(p1_a[1] + u_start * uy, 3))
        end = (round(p1_a[0] + u_end * ux, 3), round(p1_a[1] + u_end * uy, 3))
        return start, end

    def _find_adjacent_room(
        self,
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        current_room_id: str,
        room_data: List[Tuple[Room, Polygon]],
    ) -> Optional[str]:
        """Test if segment interior (midpoint) runs along another room's boundary."""
        mid = Point((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)
        for r, poly in room_data:
            if r.id == current_room_id:
                continue
            if poly.boundary.distance(mid) <= self.config.contact_tolerance_m:
                return r.id
        return None
