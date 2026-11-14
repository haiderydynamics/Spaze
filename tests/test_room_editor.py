"""Unit tests for modules.room_editor."""
import pytest

from models.floorplan_schema import DetectionStatus, Room, ScaleInfo
from modules.room_editor import RoomEditor
from utils.validation import ValidationError


def test_create_manual_room():
    scale = ScaleInfo()
    scale.calibrate(pixel_distance=100.0, real_meters=2.0)  # 50 px/m

    room = RoomEditor.create_manual_room(
        room_id="room_custom_01",
        label="Home Office",
        room_type="Living Room",
        x=20.0,
        y=30.0,
        width=150.0,
        height=100.0,
        image_width=500,
        image_height=500,
        scale_info=scale,
    )

    assert room.room_id == "room_custom_01"
    assert room.label == "Home Office"
    assert room.room_type == "Living Room"
    assert room.is_user_confirmed
    assert room.detection_status == DetectionStatus.USER_CONFIRMED.value
    assert room.physical_width == 3.0  # 150 / 50
    assert room.physical_height == 2.0  # 100 / 50


def test_create_manual_room_out_of_bounds():
    with pytest.raises(ValidationError, match="extends beyond image width"):
        RoomEditor.create_manual_room(
            room_id="r_err",
            label="Error Room",
            room_type="Bedroom",
            x=450.0,
            y=50.0,
            width=100.0,  # 450 + 100 = 550 > 500
            height=80.0,
            image_width=500,
            image_height=500,
        )


def test_update_room_attributes():
    room = Room(
        room_id="r1",
        room_type="Unclassified",
        label="Candidate Room 1",
        x=50.0,
        y=50.0,
        width=100.0,
        height=80.0,
    )

    updated = RoomEditor.update_room(
        room=room,
        label="Master Suite",
        room_type="Bedroom",
        x=60.0,
        y=60.0,
        width=110.0,
        height=90.0,
        detection_status=DetectionStatus.USER_CONFIRMED.value,
    )

    assert updated.label == "Master Suite"
    assert updated.room_type == "Bedroom"
    assert updated.x == 60.0
    assert updated.width == 110.0
    assert updated.is_user_confirmed
    assert updated.detection_status == DetectionStatus.USER_CONFIRMED.value


def test_confirm_and_reject_actions():
    room = Room(
        room_id="r1",
        room_type="Unclassified",
        label="Candidate Room 1",
        x=50.0,
        y=50.0,
        width=100.0,
        height=80.0,
    )

    RoomEditor.confirm_room(room)
    assert room.is_user_confirmed
    assert room.detection_status == DetectionStatus.USER_CONFIRMED.value

    RoomEditor.reject_room(room)
    assert not room.is_user_confirmed
    assert room.detection_status == DetectionStatus.REJECTED.value
