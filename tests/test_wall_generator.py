"""Unit tests for 3D wall mesh synthesis with volumetric apertures."""
import pytest
import trimesh
from geometry3d.opening_generator import Opening3D
from geometry3d.wall_analyzer import WallSegment
from geometry3d.wall_generator import WallGenerator


def test_solid_wall_prism_generation():
    """Wall without openings should generate a single solid box mesh."""
    gen = WallGenerator()
    wall = WallSegment(
        id="w_solid",
        start=(0.0, 0.0),
        end=(5.0, 0.0),
        wall_type="exterior",
        thickness=0.20,
        height=2.80,
        adjacent_room_ids=["r1"],
    )
    wall_meshes, accents = gen.generate_walls([wall], openings=[])

    assert len(wall_meshes) == 1
    assert len(accents) == 0

    m = wall_meshes[0]
    assert isinstance(m, trimesh.Trimesh)
    assert len(m.faces) == 12  # Box mesh has 12 triangular faces
    assert abs(m.vertices[:, 2].max() - 2.80) < 1e-3


def test_wall_with_door_creates_pillars_and_lintel():
    """Wall with a door should create two pillars and one lintel (3 wall meshes + 1 door trim)."""
    gen = WallGenerator()
    wall = WallSegment(
        id="w_door",
        start=(0.0, 0.0),
        end=(4.0, 0.0),
        wall_type="interior",
        thickness=0.12,
        height=2.80,
        adjacent_room_ids=["r1", "r2"],
    )
    door = Opening3D(
        id="d1",
        opening_type="door",
        wall_id="w_door",
        width=0.90,
        height=2.10,
        sill_height=0.0,
        wall_start_offset=1.55,
        center_3d=(2.0, 0.0, 1.05),
    )
    wall_meshes, accents = gen.generate_walls([wall], openings=[door])

    # Left pillar, lintel, right pillar
    assert len(wall_meshes) == 3
    assert len(accents) == 1  # Decorative header frame
