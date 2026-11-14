"""Unit tests for door and window opening generation."""
import pytest
from geometry3d.config import Geometry3DConfig
from geometry3d.opening_generator import OpeningGenerator
from geometry3d.wall_analyzer import WallSegment


def test_opening_generator_interior_door():
    """Interior wall long enough should receive an interior door opening."""
    gen = OpeningGenerator()
    # Wall length = 4.0m
    wall = WallSegment(
        id="w_int_1",
        start=(0.0, 0.0),
        end=(4.0, 0.0),
        wall_type="interior",
        thickness=0.12,
        height=2.80,
        adjacent_room_ids=["r1", "r2"],
    )
    openings = gen.generate_openings([wall])
    assert len(openings) == 1
    op = openings[0]
    assert op.opening_type == "door"
    assert op.width == 0.90
    assert op.height == 2.10
    assert op.sill_height == 0.0
    assert op.wall_id == "w_int_1"


def test_opening_generator_exterior_window():
    """Exterior wall should receive window(s) if sufficiently long."""
    cfg = Geometry3DConfig(generate_doors=False, generate_windows=True)
    gen = OpeningGenerator(cfg)
    wall = WallSegment(
        id="w_ext_1",
        start=(0.0, 0.0),
        end=(3.0, 0.0),
        wall_type="exterior",
        thickness=0.20,
        height=2.80,
        adjacent_room_ids=["r1"],
    )
    openings = gen.generate_openings([wall])
    assert len(openings) == 1
    op = openings[0]
    assert op.opening_type == "window"
    assert op.width == 1.20
    assert op.height == 1.20
    assert op.sill_height == 0.90


def test_opening_generator_short_wall_no_aperture():
    """Short wall below clearance threshold should receive no door."""
    gen = OpeningGenerator()
    wall = WallSegment(
        id="w_int_short",
        start=(0.0, 0.0),
        end=(0.8, 0.0),  # Less than door_width + clearance
        wall_type="interior",
        thickness=0.12,
        height=2.80,
        adjacent_room_ids=["r1", "r2"],
    )
    openings = gen.generate_openings([wall])
    assert len(openings) == 0
