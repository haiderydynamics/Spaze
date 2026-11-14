"""Wall-Line Detection Module for Architectural Floor Plans.

Extracts rectilinear and dominant structural wall candidates using the Progressive
Probabilistic Hough Line Transform (HoughLinesP) applied to morphological edge maps.
Filters out noise, furniture outlines, and small text annotations by length and
architectural orientation heuristics.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import cv2
import numpy as np

from config.settings import ProcessingConfig
from models.floorplan_schema import WallCandidate


class WallDetector:
    """Detects and characterizes architectural wall segments from preprocessed images."""

    def __init__(self, config: Optional[ProcessingConfig] = None) -> None:
        self.config = config or ProcessingConfig()

    def detect_walls(
        self,
        edge_or_morph_map: np.ndarray,
        min_line_length: Optional[int] = None,
        max_line_gap: Optional[int] = None,
        hough_threshold: Optional[int] = None,
    ) -> List[WallCandidate]:
        """Detect wall segments from binary edge or morphological map.

        Args:
            edge_or_morph_map: Single-channel uint8 binary edge or morphology image.
            min_line_length: Minimum line segment length in pixels.
            max_line_gap: Maximum gap between collinear segments to bridge.
            hough_threshold: Hough accumulator threshold.

        Returns:
            List of WallCandidate instances.
        """
        if edge_or_morph_map.ndim != 2:
            raise ValueError(f"Input image must be single-channel 2D, got shape {edge_or_morph_map.shape}")

        # Check if map is empty
        if cv2.countNonZero(edge_or_morph_map) == 0:
            return []

        min_len = min_line_length if min_line_length is not None else self.config.HOUGH_MIN_LINE_LENGTH
        gap = max_line_gap if max_line_gap is not None else self.config.HOUGH_MAX_LINE_GAP
        thresh = hough_threshold if hough_threshold is not None else self.config.HOUGH_THRESHOLD

        lines = cv2.HoughLinesP(
            edge_or_morph_map,
            rho=self.config.HOUGH_RHO,
            theta=np.deg2rad(self.config.HOUGH_THETA_DEGREES),
            threshold=thresh,
            minLineLength=min_len,
            maxLineGap=gap,
        )

        if lines is None or len(lines) == 0:
            return []

        candidates: List[WallCandidate] = []
        tol = self.config.HV_ANGLE_TOLERANCE_DEGREES

        for idx, line in enumerate(lines):
            pts = line.reshape(-1)
            if len(pts) < 4:
                continue
            x1, y1, x2, y2 = int(pts[0]), int(pts[1]), int(pts[2]), int(pts[3])
            dx = float(x2 - x1)
            dy = float(y2 - y1)
            length = math.hypot(dx, dy)

            if length < min_len:
                continue

            # Calculate angle in degrees [0, 180)
            angle = math.degrees(math.atan2(abs(dy), abs(dx)))

            # Determine architectural orientation (horizontal, vertical, or diagonal)
            if abs(angle) <= tol or abs(angle - 180.0) <= tol:
                orientation = "horizontal"
                # Pure horizontal alignment increases confidence
                orientation_score = 0.90
            elif abs(angle - 90.0) <= tol:
                orientation = "vertical"
                orientation_score = 0.90
            else:
                orientation = "diagonal"
                orientation_score = 0.55

            # Length factor: longer lines are more likely to be structural walls than furniture
            length_factor = min(1.0, length / 200.0)
            confidence = round(0.4 * orientation_score + 0.6 * length_factor, 3)

            candidate = WallCandidate(
                wall_id=f"wall_{idx + 1:03d}",
                start_point=(float(x1), float(y1)),
                end_point=(float(x2), float(y2)),
                length_pixels=round(length, 2),
                angle_degrees=round(angle, 2),
                orientation=orientation,
                confidence_score=confidence,
                detection_method="HoughLinesP",
            )
            candidates.append(candidate)

        # Sort by length descending (primary structural walls first)
        candidates.sort(key=lambda w: w.length_pixels, reverse=True)
        return candidates
