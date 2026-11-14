"""Unit tests for wall classification and shared boundary deduplication."""
import pytest
from geometry3d.coordinate_system import CoordinateTransformer
from geometry3d.wall_analyzer import WallAnalyzer
from models.floorplan_schema import FloorPlan, Room


def test_single_room_exterior_walls():
    """A single isolated room should generate four exterior walls."""
    analyzer = WallAnalyzer()
    fp = FloorPlan(
        image_width=800,
        image_height=600,
        confirmed_rooms=[
            Room(room_id="r1", room_type="living_room", label="Living Room", x=100, y=100, width=200, height=200),
        ]
    )
    walls = analyzer.analyze(fp)
    assert len(walls) == 4
    for w in walls:
        assert w.wall_type == "exterior"
        assert w.thickness == 0.20
        assert w.length > 0


def test_shared_interior_wall_deduplication():
    """Two side-by-side touching rooms should produce exterior walls plus ONE shared interior wall."""
    analyzer = WallAnalyzer()
    # Room 1: x from 100 to 300, y from 100 to 300 (width=200, height=200)
    # Room 2: x from 300 to 500, y from 100 to 300 (width=200, height=200)
    # Shared wall is at x=300, spanning y=100 to y=300
    fp = FloorPlan(
        image_width=800,
        image_height=600,
        confirmed_rooms=[
            Room(room_id="r1", room_type="living_room", label="Living Room", x=100, y=100, width=200, height=200),
            Room(room_id="r2", room_type="bedroom", label="Bedroom", x=300, y=100, width=200, height=200),
        ]
    )
    walls = analyzer.analyze(fp)

    interior_walls = [w for w in walls if w.wall_type == "interior"]
    exterior_walls = [w for w in walls if w.wall_type == "exterior"]

    # Exactly one deduplicated partition separating the two rooms
    assert len(interior_walls) == 1
    assert "r1" in interior_walls[0].adjacent_room_ids
    assert "r2" in interior_walls[0].adjacent_room_ids
    assert interior_walls[0].thickness == 0.12

    # Exterior perimeter walls should be present
    assert len(exterior_walls) >= 4
