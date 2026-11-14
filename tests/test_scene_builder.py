"""Unit tests for scene assembly and hierarchical metadata."""
import pytest
import trimesh
from geometry3d.config import Geometry3DConfig
from geometry3d.coordinate_system import CoordinateTransformer
from geometry3d.scene_builder import SceneBuilder
from models.floorplan_schema import FloorPlan, Room


def test_scene_builder_hierarchy():
    """Verify scene node creation, geometry counts, and metadata tags."""
    builder = SceneBuilder()
    fp = FloorPlan(
        image_width=800,
        image_height=600,
        confirmed_rooms=[
            Room(room_id="r1", room_type="living_room", label="Living Room", x=100, y=100, width=200, height=200),
        ]
    )
    tf = CoordinateTransformer.from_floorplan(fp)

    dummy_box = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
    scene, meta = builder.build_scene(
        floorplan=fp,
        transformer=tf,
        walls=[],
        openings=[],
        floor_meshes=[dummy_box],
        wall_meshes=[],
        accent_meshes=[],
        furniture_meshes=[],
    )

    assert isinstance(scene, trimesh.Scene)
    assert meta.total_triangles > 0
    assert meta.total_vertices > 0
    assert "spaze_metadata" in scene.metadata
