"""Unit tests for 3D layout geometric feasibility validation."""
import pytest
from geometry3d.layout_validator import Layout3DValidator
from models.floorplan_schema import FloorPlan, Room


def test_validator_empty_floorplan():
    """Empty floor plan should fail validation."""
    val = Layout3DValidator()
    fp = FloorPlan(image_width=800, image_height=600)
    report = val.validate_layout(fp)

    assert not report.valid
    assert any("0 active rooms" in err for err in report.critical_errors)


def test_validator_valid_layout():
    """Well-formed floor plan should pass validation with high score."""
    val = Layout3DValidator()
    fp = FloorPlan(
        image_width=800,
        image_height=600,
        confirmed_rooms=[
            Room(room_id="r1", room_type="living_room", label="Living Room", x=50, y=50, width=300, height=250),
            Room(room_id="r2", room_type="bedroom", label="Master Bedroom", x=360, y=50, width=250, height=250),
        ]
    )
    report = val.validate_layout(fp)
    assert report.valid
    assert report.overall_score >= 0.8
    assert len(report.critical_errors) == 0


def test_validator_detects_severe_overlap():
    """Significantly overlapping rooms should trigger warning or reduced score."""
    val = Layout3DValidator()
    fp = FloorPlan(
        image_width=800,
        image_height=600,
        confirmed_rooms=[
            Room(room_id="r1", room_type="living_room", label="Room 1", x=100, y=100, width=300, height=300),
            Room(room_id="r2", room_type="bedroom", label="Room 2", x=150, y=150, width=300, height=300),
        ]
    )
    report = val.validate_layout(fp)
    assert report.overlap_count > 0
