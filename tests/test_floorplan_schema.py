"""Unit tests for models.floorplan_schema."""
import pytest

from models.floorplan_schema import (
    DetectionStatus,
    FloorPlan,
    MeasurementSource,
    Room,
    ScaleInfo,
    WallCandidate,
)


def test_valid_room_instantiation():
    room = Room(
        room_id="room_01",
        room_type="Living Room",
        label="Main Living Area",
        x=50.0,
        y=40.0,
        width=120.0,
        height=80.0,
    )
    assert room.room_id == "room_01"
    assert room.pixel_area == 120.0 * 80.0
    assert room.aspect_ratio == 120.0 / 80.0
    assert len(room.polygon) == 4
    assert room.detection_status == DetectionStatus.CANDIDATE.value
    assert not room.is_user_confirmed


def test_invalid_room_dimensions():
    with pytest.raises(ValueError, match="width must be positive"):
        Room(room_id="r1", room_type="Bedroom", label="Bed", x=10, y=10, width=0, height=50)

    with pytest.raises(ValueError, match="height must be positive"):
        Room(room_id="r1", room_type="Bedroom", label="Bed", x=10, y=10, width=50, height=-10)


def test_invalid_room_coordinates():
    with pytest.raises(ValueError, match="coordinates must be non-negative"):
        Room(room_id="r1", room_type="Bedroom", label="Bed", x=-5, y=10, width=50, height=50)


def test_scale_info_uncalibrated():
    scale = ScaleInfo()
    assert not scale.is_calibrated
    assert scale.pixels_per_meter is None
    assert scale.pixels_to_meters(100.0) is None
    assert scale.square_pixels_to_sqm(10000.0) is None


def test_scale_info_calibration():
    scale = ScaleInfo()
    # 200 pixels = 4.0 meters => 50 px/m
    scale.calibrate(pixel_distance=200.0, real_meters=4.0)

    assert scale.is_calibrated
    assert scale.pixels_per_meter == 50.0
    assert scale.pixels_to_meters(100.0) == 2.0
    assert scale.square_pixels_to_sqm(10000.0) == 4.0  # 10000 / 2500 = 4.0 m²


def test_scale_propagation_to_room():
    room = Room(
        room_id="room_01",
        room_type="Kitchen",
        label="Kitchen",
        x=10.0,
        y=10.0,
        width=100.0,
        height=50.0,
    )
    scale = ScaleInfo()
    room.apply_scale(scale)
    assert room.physical_width is None

    # Calibrate: 50 px = 1 meter => 50 px/m
    scale.calibrate(pixel_distance=50.0, real_meters=1.0)
    room.apply_scale(scale)

    assert room.physical_width == 2.0   # 100 / 50
    assert room.physical_height == 1.0  # 50 / 50
    assert room.physical_area == 2.0    # 5000 / 2500


def test_floorplan_json_roundtrip():
    floorplan = FloorPlan(image_width=800, image_height=600)
    room = Room(
        room_id="room_01",
        room_type="Bedroom",
        label="Guest Bedroom",
        x=100.0,
        y=150.0,
        width=200.0,
        height=150.0,
        confidence_score=0.88,
    )
    wall = WallCandidate(
        wall_id="wall_01",
        start_point=(10.0, 10.0),
        end_point=(200.0, 10.0),
        length_pixels=190.0,
        angle_degrees=0.0,
        orientation="horizontal",
        confidence_score=0.95,
    )
    floorplan.add_candidate_room(room)
    floorplan.wall_candidates.append(wall)

    # Convert to JSON string
    json_str = floorplan.to_json()
    assert "Guest Bedroom" in json_str
    assert "wall_01" in json_str

    # Reload from JSON
    reconstructed = FloorPlan.from_json(json_str)
    assert reconstructed.image_width == 800
    assert reconstructed.image_height == 600
    assert len(reconstructed.candidate_rooms) == 1
    assert reconstructed.candidate_rooms[0].label == "Guest Bedroom"
    assert reconstructed.candidate_rooms[0].width == 200.0
    assert len(reconstructed.wall_candidates) == 1
    assert reconstructed.wall_candidates[0].length_pixels == 190.0


def test_floorplan_confirm_and_reject_room():
    floorplan = FloorPlan(image_width=600, image_height=400)
    room = Room(room_id="r1", room_type="Hallway", label="Hall", x=10, y=10, width=50, height=20)
    floorplan.add_candidate_room(room)

    assert len(floorplan.candidate_rooms) == 1
    assert len(floorplan.confirmed_rooms) == 0

    # Confirm room
    confirmed = floorplan.confirm_room("r1")
    assert confirmed
    assert len(floorplan.candidate_rooms) == 0
    assert len(floorplan.confirmed_rooms) == 1
    assert floorplan.confirmed_rooms[0].detection_status == DetectionStatus.USER_CONFIRMED.value
    assert floorplan.confirmed_rooms[0].is_user_confirmed

    # Reject room
    rejected = floorplan.reject_room("r1")
    assert rejected
    assert floorplan.confirmed_rooms[0].detection_status == DetectionStatus.REJECTED.value
