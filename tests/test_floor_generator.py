"""Unit tests for floor and foundation mesh synthesis."""
import pytest
import trimesh
from geometry3d.floor_generator import FloorGenerator
from models.floorplan_schema import FloorPlan, Room


def test_floor_generator_creates_floor_and_foundation():
    """Verify extrusion of room floor slabs and foundation plinth."""
    gen = FloorGenerator()
    fp = FloorPlan(
        image_width=800,
        image_height=600,
        confirmed_rooms=[
            Room(room_id="r1", room_type="living_room", label="Living Room", x=100, y=100, width=250, height=250),
            Room(room_id="r2", room_type="bedroom", label="Bedroom", x=350, y=100, width=200, height=250),
        ]
    )
    floor_meshes, foundation = gen.generate_floors(fp)

    assert len(floor_meshes) == 2
    assert foundation is not None

    for m in floor_meshes:
        assert isinstance(m, trimesh.Trimesh)
        assert len(m.vertices) > 0
        assert len(m.faces) > 0
        # Elevation: top walk surface should be at z=0.0
        max_z = float(m.vertices[:, 2].max())
        assert abs(max_z - 0.0) < 1e-3

    assert isinstance(foundation, trimesh.Trimesh)
    assert len(foundation.vertices) > 0
