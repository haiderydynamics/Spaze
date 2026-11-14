"""Unit tests for multi-format architectural export (GLB, JSON manifest, PNG preview)."""
import io
import json
import pytest
import trimesh
from geometry3d.config import Geometry3DConfig
from geometry3d.exporter import ModelExporter
from geometry3d.model_generator import generate_3d_model
from models.floorplan_schema import FloorPlan, Room


def test_glb_export_and_reload():
    """Verify GLB binary generation, file integrity, and re-loading capability."""
    fp = FloorPlan(
        image_width=800,
        image_height=600,
        confirmed_rooms=[
            Room(room_id="r1", room_type="living_room", label="Living Room", x=100, y=100, width=250, height=250),
        ]
    )
    result = generate_3d_model(fp)
    glb_bytes = result.get_glb_bytes()

    assert isinstance(glb_bytes, bytes)
    assert len(glb_bytes) > 0

    # Verify structural validity
    is_valid = ModelExporter.verify_glb(glb_bytes)
    assert is_valid

    # Reload into trimesh
    buf = io.BytesIO(glb_bytes)
    reloaded = trimesh.load(buf, file_type="glb")
    assert isinstance(reloaded, trimesh.Scene)
    assert len(reloaded.geometry) > 0


def test_manifest_export():
    """Verify architectural JSON metadata manifest structure and contents."""
    fp = FloorPlan(
        image_width=800,
        image_height=600,
        confirmed_rooms=[
            Room(room_id="r1", room_type="living_room", label="Living Room", x=100, y=100, width=250, height=250),
        ]
    )
    result = generate_3d_model(fp)
    manifest = result.manifest

    assert manifest["format"] == "SPAZE_3D_MANIFEST_v1"
    assert manifest["units"] == "meters"
    assert "statistics" in manifest
    assert "rooms" in manifest
    assert len(manifest["rooms"]) == 1


def test_blueprint_preview_export():
    """Verify 2D blueprint preview image export."""
    fp = FloorPlan(
        image_width=800,
        image_height=600,
        confirmed_rooms=[
            Room(room_id="r1", room_type="bedroom", label="Bedroom", x=100, y=100, width=200, height=200),
        ]
    )
    png_bytes = ModelExporter.export_floorplan_preview(fp)
    assert isinstance(png_bytes, bytes)
    assert len(png_bytes) > 100
    assert png_bytes.startswith(b"\x89PNG")
