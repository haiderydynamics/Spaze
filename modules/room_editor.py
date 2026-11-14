"""Room Editor Module for SPAZE.

Provides pure functions for user corrections, interactive status updates,
manual room additions, and room geometry editing. Validates all inputs to preserve
architectural integrity and schema conformity.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from config.settings import RoomConfig
from models.floorplan_schema import DetectionStatus, MeasurementSource, Room, ScaleInfo
from utils.validation import (
    ValidationError,
    validate_polygon_geometry,
    validate_room_bounds,
)


class RoomEditor:
    """Manages manual corrections, status changes, and additions of architectural rooms."""

    @staticmethod
    def create_manual_room(
        room_id: str,
        label: str,
        room_type: str,
        x: float,
        y: float,
        width: float,
        height: float,
        image_width: Optional[int] = None,
        image_height: Optional[int] = None,
        scale_info: Optional[ScaleInfo] = None,
    ) -> Room:
        """Create a new user-defined room manually.

        Args:
            room_id: Unique room identifier (e.g. 'room_manual_01').
            label: Human-readable label (e.g. 'Master Bedroom').
            room_type: One of allowed RoomConfig.ROOM_TYPES.
            x: Top-left X coordinate in pixels.
            y: Top-left Y coordinate in pixels.
            width: Width in pixels.
            height: Height in pixels.
            image_width: Optional image width for boundary checks.
            image_height: Optional image height for boundary checks.
            scale_info: Optional scale info for physical dimensions.

        Returns:
            A new Room instance marked as User Confirmed.
        """
        # Validate coordinates and dimensions
        validate_room_bounds(
            x=x,
            y=y,
            width=width,
            height=height,
            image_width=image_width,
            image_height=image_height,
        )

        valid_type = room_type if room_type in RoomConfig.ROOM_TYPES else "Other"

        polygon = [
            (x, y),
            (x + width, y),
            (x + width, y + height),
            (x, y + height),
        ]

        room = Room(
            room_id=room_id,
            room_type=valid_type,
            label=label or room_id,
            x=x,
            y=y,
            width=width,
            height=height,
            polygon=polygon,
            coordinate_unit="pixel",
            detection_status=DetectionStatus.USER_CONFIRMED.value,
            confidence_score=1.0,
            confidence_explanation="Manually defined and confirmed by user.",
            measurement_source=MeasurementSource.MANUAL_ENTRY.value,
            is_user_confirmed=True,
        )

        if scale_info is not None and scale_info.is_calibrated:
            room.apply_scale(scale_info)

        return room

    @staticmethod
    def update_room(
        room: Room,
        label: Optional[str] = None,
        room_type: Optional[str] = None,
        x: Optional[float] = None,
        y: Optional[float] = None,
        width: Optional[float] = None,
        height: Optional[float] = None,
        detection_status: Optional[str] = None,
        image_width: Optional[int] = None,
        image_height: Optional[int] = None,
        scale_info: Optional[ScaleInfo] = None,
    ) -> Room:
        """Update an existing room with modified attributes.

        Returns the updated Room instance (or a validated clone).
        """
        new_x = x if x is not None else room.x
        new_y = y if y is not None else room.y
        new_w = width if width is not None else room.width
        new_h = height if height is not None else room.height

        # Validate new bounds
        validate_room_bounds(
            x=new_x,
            y=new_y,
            width=new_w,
            height=new_h,
            image_width=image_width,
            image_height=image_height,
        )

        if label is not None:
            room.label = label.strip()

        if room_type is not None:
            room.room_type = room_type if room_type in RoomConfig.ROOM_TYPES else room.room_type

        # Update geometry
        coords_changed = (new_x != room.x or new_y != room.y or new_w != room.width or new_h != room.height)
        if coords_changed:
            room.x = new_x
            room.y = new_y
            room.width = new_w
            room.height = new_h
            room.polygon = [
                (new_x, new_y),
                (new_x + new_w, new_y),
                (new_x + new_w, new_y + new_h),
                (new_x, new_y + new_h),
            ]
            room.confidence_explanation = f"{room.confidence_explanation} (Coordinates edited by user)."

        if detection_status is not None and detection_status in RoomConfig.DETECTION_STATUSES:
            room.detection_status = detection_status
            if detection_status == DetectionStatus.USER_CONFIRMED.value:
                room.is_user_confirmed = True
            elif detection_status == DetectionStatus.REJECTED.value:
                room.is_user_confirmed = False

        if scale_info is not None:
            room.apply_scale(scale_info)

        return room

    @staticmethod
    def confirm_room(room: Room, scale_info: Optional[ScaleInfo] = None) -> Room:
        """Mark a room as User Confirmed."""
        room.detection_status = DetectionStatus.USER_CONFIRMED.value
        room.is_user_confirmed = True
        if scale_info is not None:
            room.apply_scale(scale_info)
        return room

    @staticmethod
    def reject_room(room: Room) -> Room:
        """Mark a room as Rejected."""
        room.detection_status = DetectionStatus.REJECTED.value
        room.is_user_confirmed = False
        return room
