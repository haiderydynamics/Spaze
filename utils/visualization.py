"""Visualization utilities for SPAZE floor-plan processing.

Renders high-contrast, professional architectural overlays:
- Stage-by-stage CV inspection (Grayscale, Denoised, Threshold, Edges, Morphology)
- Detected wall line candidates overlay
- Candidate room polygons with semi-transparent fills and status badges
- Fully annotated floor plans with dimensions (meters when calibrated, pixels otherwise)

Always operates on copies to guarantee immutability of the source images.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from models.floorplan_schema import DetectionStatus, Room, ScaleInfo, WallCandidate


# Architectural Status Color Palette (RGB format)
STATUS_COLORS = {
    DetectionStatus.USER_CONFIRMED.value: (46, 204, 113),  # Emerald Green
    DetectionStatus.CANDIDATE.value: (52, 152, 219),       # Electric Cyan / Blue
    DetectionStatus.NEEDS_REVIEW.value: (243, 156, 18),    # Amber Orange
    DetectionStatus.REJECTED.value: (231, 76, 60),         # Coral Red
}

DEFAULT_ROOM_COLOR = (142, 68, 173)  # Amethyst Purple


def overlay_wall_candidates(
    base_image_rgb: np.ndarray,
    walls: List[WallCandidate],
    line_thickness: int = 2,
) -> np.ndarray:
    """Draw detected wall candidates on top of the floor-plan image.

    Args:
        base_image_rgb: Original RGB image (H, W, 3).
        walls: List of detected WallCandidate objects.
        line_thickness: Pixel thickness of drawn lines.

    Returns:
        New RGB image with wall lines rendered.
    """
    canvas = np.copy(base_image_rgb)

    for wall in walls:
        pt1 = (int(round(wall.start_point[0])), int(round(wall.start_point[1])))
        pt2 = (int(round(wall.end_point[0])), int(round(wall.end_point[1])))

        # Color-code by orientation
        if wall.orientation == "horizontal":
            color = (0, 220, 255)  # Cyan
        elif wall.orientation == "vertical":
            color = (255, 105, 180)  # Magenta
        else:
            color = (255, 215, 0)  # Gold

        cv2.line(canvas, pt1, pt2, color, line_thickness, cv2.LINE_AA)

    return canvas


def overlay_candidate_rooms(
    base_image_rgb: np.ndarray,
    rooms: List[Room],
    alpha: float = 0.25,
    border_thickness: int = 2,
    show_labels: bool = True,
    scale_info: Optional[ScaleInfo] = None,
) -> np.ndarray:
    """Render room candidate boundaries with semi-transparent fills and status badges.

    Args:
        base_image_rgb: Original RGB image (H, W, 3).
        rooms: List of Room objects.
        alpha: Alpha transparency of the filled room polygons [0.0, 1.0].
        border_thickness: Thickness of polygon bounding borders.
        show_labels: Whether to render room labels and dimensions.
        scale_info: Scale information for displaying physical meters vs pixels.

    Returns:
        New RGB image with annotated room overlays.
    """
    canvas = np.copy(base_image_rgb)
    overlay = np.copy(base_image_rgb)

    # 1. Draw semi-transparent polygon fills
    for room in rooms:
        if room.detection_status == DetectionStatus.REJECTED.value:
            continue  # Don't shade rejected rooms

        color = STATUS_COLORS.get(room.detection_status, DEFAULT_ROOM_COLOR)

        if room.polygon and len(room.polygon) >= 3:
            pts = np.array(room.polygon, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(overlay, [pts], color)
        else:
            # Fallback to bounding box rectangle
            x1, y1 = int(round(room.x)), int(round(room.y))
            x2, y2 = int(round(room.x + room.width)), int(round(room.y + room.height))
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness=-1)

    # Blend overlay with canvas for transparency
    cv2.addWeighted(overlay, alpha, canvas, 1.0 - alpha, 0, canvas)

    # 2. Draw crisp borders and text badges
    for room in rooms:
        color = STATUS_COLORS.get(room.detection_status, DEFAULT_ROOM_COLOR)

        # Polygon or rectangle border
        if room.polygon and len(room.polygon) >= 3:
            pts = np.array(room.polygon, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=border_thickness, lineType=cv2.LINE_AA)
        else:
            x1, y1 = int(round(room.x)), int(round(room.y))
            x2, y2 = int(round(room.x + room.width)), int(round(room.y + room.height))
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness=border_thickness, lineType=cv2.LINE_AA)

        if not show_labels or room.detection_status == DetectionStatus.REJECTED.value:
            continue

        # Formulate badge text
        label_text = f"{room.label} ({room.room_type})"
        if scale_info is not None and scale_info.is_calibrated and room.physical_width and room.physical_height:
            dim_text = f"{room.physical_width:.1f}m x {room.physical_height:.1f}m ({room.physical_area:.1f}m²)"
        else:
            dim_text = f"{int(room.width)}x{int(room.height)} px"

        badge_x = int(round(room.x + 6))
        badge_y = int(round(room.y + 18))

        # Render subtle dark text background for readability
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.42
        thickness = 1

        (tw1, th1), _ = cv2.getTextSize(label_text, font, font_scale, thickness)
        (tw2, th2), _ = cv2.getTextSize(dim_text, font, font_scale, thickness)
        box_w = max(tw1, tw2) + 12
        box_h = th1 + th2 + 16

        # Ensure text box is within image bounds
        img_h, img_w = canvas.shape[:2]
        if badge_y + box_h > img_h:
            badge_y = max(10, img_h - box_h - 4)
        if badge_x + box_w > img_w:
            badge_x = max(4, img_w - box_w - 4)

        # Background pill
        cv2.rectangle(
            canvas,
            (badge_x - 3, badge_y - th1 - 4),
            (badge_x + box_w, badge_y + th2 + 10),
            (25, 28, 36),
            thickness=-1,
        )
        # Small accent indicator on left of pill
        cv2.rectangle(
            canvas,
            (badge_x - 3, badge_y - th1 - 4),
            (badge_x, badge_y + th2 + 10),
            color,
            thickness=-1,
        )

        # Text strings
        cv2.putText(canvas, label_text, (badge_x + 5, badge_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        cv2.putText(canvas, dim_text, (badge_x + 5, badge_y + th2 + 8), font, font_scale, (180, 195, 210), thickness, cv2.LINE_AA)

    return canvas


def convert_single_channel_to_rgb(img_single: np.ndarray) -> np.ndarray:
    """Convert a single-channel grayscale or binary image to 3-channel RGB for display."""
    if img_single.ndim == 2:
        return cv2.cvtColor(img_single, cv2.COLOR_GRAY2RGB)
    return img_single
