"""Candidate Room Extraction Module for SPAZE.

Extracts enclosed architectural room candidates from the morphological wall mask
using contour hierarchy analysis, geometric filtering, and polygon simplification.
Calculates an explainable heuristic quality score and assigns an initial transparent status
(Candidate or Needs Review). Never fabricates confirmed rooms.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon

from config.settings import RoomConfig
from models.floorplan_schema import DetectionStatus, MeasurementSource, Room


class RoomExtractor:
    """Extracts, filters, and scores candidate architectural room boundaries."""

    def __init__(self, config: Optional[RoomConfig] = None) -> None:
        self.config = config or RoomConfig()

    def extract_rooms(
        self,
        wall_mask: np.ndarray,
        image_width: int,
        image_height: int,
    ) -> List[Room]:
        """Extract candidate rooms from a binary morphological wall mask.

        In the input wall_mask, walls are non-zero (white: 255) and rooms/floors are 0 (black).
        Inverting this mask produces white enclosed spaces bounded by black walls.

        Args:
            wall_mask: Binary uint8 image (H, W) where walls = 255, spaces = 0.
            image_width: Width of the floor-plan image.
            image_height: Height of the floor-plan image.

        Returns:
            List of detected Room candidate objects.
        """
        if wall_mask.ndim != 2:
            raise ValueError(f"wall_mask must be single-channel 2D, got shape {wall_mask.shape}")

        total_image_area = float(image_width * image_height)
        min_area = max(self.config.MIN_ROOM_AREA_PIXELS, total_image_area * self.config.MIN_ROOM_AREA_RATIO)
        max_area = total_image_area * self.config.MAX_ROOM_AREA_RATIO

        # Invert so rooms/spaces become white (255) foreground, walls become black (0)
        spaces_mask = cv2.bitwise_not(wall_mask)

        # Clear image boundary border (1 px) to avoid exterior framing being merged into interior rooms
        spaces_clean = spaces_mask.copy()
        cv2.rectangle(spaces_clean, (0, 0), (image_width - 1, image_height - 1), 0, thickness=2)

        # Detect contours using two-level hierarchy (CCOMP)
        # CCOMP organizes contours into external boundaries of spaces and holes (e.g. structural pillars)
        contours, hierarchy = cv2.findContours(
            spaces_clean,
            cv2.RETR_CCOMP,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours or hierarchy is None:
            return []

        candidates: List[Room] = []
        candidate_count = 0

        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)

            # Filter 1: Area limits (suppress text, furniture noise, and exterior infinite background)
            if area < min_area or area > max_area:
                continue

            # Bounding box
            x, y, w, h = cv2.boundingRect(cnt)

            # Filter 2: Dimensions sanity
            if w <= 10 or h <= 10:
                continue

            # Filter 3: Aspect ratio limits (reject wall slivers and scanner margins)
            aspect_ratio = float(w) / float(h)
            if aspect_ratio < self.config.MIN_ASPECT_RATIO or aspect_ratio > self.config.MAX_ASPECT_RATIO:
                continue

            # Filter 4: Boundary check (do not allow room spanning entire canvas edge)
            if (x <= 1 and y <= 1 and w >= image_width - 3 and h >= image_height - 3):
                continue

            # Approximate polygon using Douglas-Peucker algorithm
            perimeter = cv2.arcLength(cnt, closed=True)
            epsilon = self.config.POLYGON_APPROX_EPSILON_RATIO * perimeter
            approx = cv2.approxPolyDP(cnt, epsilon, closed=True)

            num_vertices = len(approx)
            if num_vertices < 3 or num_vertices > self.config.MAX_POLYGON_VERTICES:
                continue

            polygon_pts = [(float(pt[0][0]), float(pt[0][1])) for pt in approx]

            # Verify topological validity using Shapely
            try:
                shapely_poly = ShapelyPolygon(polygon_pts)
                if not shapely_poly.is_valid or shapely_poly.is_empty:
                    # Attempt buffer(0) repair for self-touching contour points
                    repaired = shapely_poly.buffer(0)
                    if repaired.is_valid and not repaired.is_empty and repaired.geom_type == "Polygon":
                        polygon_pts = [(float(c[0]), float(c[1])) for c in repaired.exterior.coords[:-1]]
                    else:
                        continue
            except Exception:
                continue

            # Calculate Explainable Heuristic Quality Score
            # 1. Rectangularity: area / bbox_area (typical architectural rooms are near 1.0)
            bbox_area = float(w * h)
            rectangularity = min(1.0, max(0.0, area / bbox_area)) if bbox_area > 0 else 0.0

            # 2. Solidity: area / convex_hull_area
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = min(1.0, max(0.0, area / hull_area)) if hull_area > 0 else 0.0

            # 3. Aspect Ratio Score: penalize extreme deviations from standard room proportions [0.5, 2.0]
            if 0.5 <= aspect_ratio <= 2.0:
                aspect_score = 1.0
            else:
                deviation = min(abs(aspect_ratio - 1.0), 3.0) / 3.0
                aspect_score = max(0.2, 1.0 - deviation)

            confidence = (
                self.config.RECTANGULARITY_WEIGHT * rectangularity
                + self.config.SOLIDITY_WEIGHT * solidity
                + self.config.ASPECT_RATIO_WEIGHT * aspect_score
            )
            confidence = round(float(confidence), 3)

            explanation = (
                f"Heuristic quality: {confidence:.2f} "
                f"(rect={rectangularity:.2f}, solid={solidity:.2f}, aspect={aspect_ratio:.2f}, vertices={num_vertices})."
            )

            # Assign initial status (Needs Review if irregular or borderline confidence)
            if confidence >= 0.70 and 4 <= num_vertices <= 8:
                status = DetectionStatus.CANDIDATE.value
            else:
                status = DetectionStatus.NEEDS_REVIEW.value

            candidate_count += 1
            room_id = f"room_{candidate_count:02d}"

            room = Room(
                room_id=room_id,
                room_type="Unclassified",
                label=f"Candidate Room {candidate_count}",
                x=float(x),
                y=float(y),
                width=float(w),
                height=float(h),
                polygon=polygon_pts,
                coordinate_unit="pixel",
                detection_status=status,
                confidence_score=confidence,
                confidence_explanation=explanation,
                measurement_source=MeasurementSource.UNCALIBRATED_PIXEL.value,
                is_user_confirmed=False,
            )
            candidates.append(room)

        # Sort candidate rooms top-to-bottom, left-to-right
        candidates.sort(key=lambda r: (round(r.y / 50.0), r.x))

        # Re-index room IDs neatly after spatial sorting
        for idx, r in enumerate(candidates, start=1):
            r.room_id = f"room_{idx:02d}"
            r.label = f"Candidate Room {idx}"

        return candidates
