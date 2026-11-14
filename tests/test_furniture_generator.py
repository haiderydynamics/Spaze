"""Unit tests for procedural architectural furniture synthesis."""
import pytest
import trimesh
from geometry3d.furniture_generator import FurnitureGenerator
from models.floorplan_schema import FloorPlan, Room


def test_furniture_generated_for_standard_rooms():
    """Bedroom and living room should receive appropriate furniture items."""
    gen = FurnitureGenerator()
    fp = FloorPlan(
        image_width=800,
        image_height=600,
        confirmed_rooms=[
            Room(room_id="r_bed", room_type="bedroom", label="Master Bedroom", x=100, y=100, width=250, height=250),
            Room(room_id="r_liv", room_type="living_room", label="Living Room", x=360, y=100, width=300, height=300),
        ]
    )
    items = gen.generate_furniture(fp)

    assert len(items) > 0
    for m in items:
        assert isinstance(m, trimesh.Trimesh)
        assert len(m.vertices) > 0
        assert m.metadata.get("semantic_category") == "furniture"


def test_furniture_skipped_when_disabled():
    """Furniture generator should return empty list when disabled in config."""
    from geometry3d.config import Geometry3DConfig
    cfg = Geometry3DConfig(generate_furniture=False)
    gen = FurnitureGenerator(cfg)
    fp = FloorPlan(
        image_width=800,
        image_height=600,
        confirmed_rooms=[
            Room(room_id="r_bed", room_type="bedroom", label="Bedroom", x=100, y=100, width=250, height=250),
        ]
    )
    items = gen.generate_furniture(fp)
    assert len(items) == 0
