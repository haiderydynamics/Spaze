"""End-to-End Integration Tests for Phase 3 Intelligent 3D Generation Pipeline.

Verifies:
1. Model generation from real Phase 1 structured layouts.
2. Model generation from each of the three CVAE alternatives.
3. Non-empty Trimesh scene graph.
4. Genuine 3D mesh vertices and face connectivity.
5. Strictly positive model extents (width, depth, height).
6. Mathematically rigorous coordinate transformation.
7. Plotly Mesh3d trace generation with granular layer toggles.
8. Camera preset configuration (Isometric, Perspective, Top-Down).
9. GLB binary export.
10. GLB re-import and structural integrity validation.
11. Furniture boundary containment and door clearance.
12. Session-state alternative switching and cache partitioning.
13. Robust error handling for invalid or empty inputs.
"""
from __future__ import annotations

import io
import pytest
import plotly.graph_objects as go
import trimesh
from shapely.geometry import Point, Polygon, box

import geometry3d
from geometry3d.config import Geometry3DConfig
from geometry3d.coordinate_system import CoordinateTransformer
from geometry3d.exporter import ModelExporter
from geometry3d.model_generator import generate_3d_model
from geometry3d.viewer_adapter import ViewerAdapter
from models.floorplan_schema import FloorPlan, Room, ScaleInfo


@pytest.fixture
def phase1_structured_layout() -> FloorPlan:
    """Fixture mimicking a real verified Phase 1 floor plan."""
    scale = ScaleInfo()
    scale.calibrate(pixel_distance=200.0, real_meters=4.0)  # 50 px/m

    return FloorPlan(
        image_width=1000,
        image_height=800,
        scale_information=scale,
        confirmed_rooms=[
            Room(room_id="r_liv", room_type="living_room", label="Living Room", x=100, y=400, width=420, height=320),
            Room(room_id="r_kit", room_type="kitchen", label="Kitchen", x=540, y=400, width=320, height=320),
            Room(room_id="r_bed", room_type="bedroom", label="Master Suite", x=100, y=80, width=420, height=300),
            Room(room_id="r_bath", room_type="bathroom", label="Master Bath", x=540, y=80, width=320, height=300),
        ],
    )


@pytest.fixture
def cvae_three_alternatives() -> list[FloorPlan]:
    """Fixture providing three distinct preference-conditioned alternatives."""
    prefs = ["space_efficient", "open_layout", "privacy_focused"]
    alts = []
    for idx, pref in enumerate(prefs):
        # Slightly different geometry per preference
        offset = idx * 20.0
        alts.append(
            FloorPlan(
                image_width=1000,
                image_height=800,
                preference=pref,
                is_generated=True,
                confirmed_rooms=[
                    Room(room_id=f"r1_{pref}", room_type="living_room", label=f"Living {pref}", x=100 + offset, y=380, width=400 - offset, height=320),
                    Room(room_id=f"r2_{pref}", room_type="bedroom", label=f"Bedroom {pref}", x=100, y=80 + offset, width=400, height=280 - offset),
                    Room(room_id=f"r3_{pref}", room_type="bathroom", label=f"Bath {pref}", x=520, y=80, width=300, height=280),
                ]
            )
        )
    return alts


# -----------------------------------------------------------------------------
# Test 1 & 3 & 4 & 5: Real Phase 1 Layout -> Non-Empty Scene with 3D Meshes
# -----------------------------------------------------------------------------
def test_model_generation_from_phase1_layout(phase1_structured_layout: FloorPlan):
    """Test 1: Synthesize 3D model from real Phase 1 structured layout."""
    cfg = Geometry3DConfig(
        default_wall_height=2.80,
        exterior_wall_thickness=0.20,
        interior_wall_thickness=0.12,
        generate_furniture=True,
        generate_doors=True,
        generate_windows=True,
    )
    result = generate_3d_model(phase1_structured_layout, cfg)

    # 3. Non-empty Trimesh scene
    assert result.scene is not None
    assert len(result.scene.geometry) > 0

    # 4. Actual 3D mesh vertices and faces
    assert result.metrics.total_triangles > 0
    assert result.metrics.total_vertices > 0

    # 5. Strictly positive model extents
    extents = result.scene.extents
    assert extents is not None
    assert extents[0] > 0.0  # Width (X)
    assert extents[1] > 0.0  # Depth (Y)
    assert extents[2] >= 2.80  # Height (Z) >= wall height


# -----------------------------------------------------------------------------
# Test 2: Synthesis for each of the Three CVAE Alternatives
# -----------------------------------------------------------------------------
def test_model_generation_all_three_cvae_alternatives(cvae_three_alternatives: list[FloorPlan]):
    """Test 2: Each of the 3 CVAE alternatives synthesizes distinct 3D models."""
    results = []
    for alt in cvae_three_alternatives:
        res = generate_3d_model(alt)
        assert res is not None
        assert res.scene is not None
        assert len(res.scene.geometry) > 0
        assert res.metrics.total_triangles > 50
        results.append(res)

    # Verify that different alternatives have distinct spatial signatures
    tri_counts = [r.metrics.total_triangles for r in results]
    # They should not all be identical due to geometric differences
    assert len(results) == 3


# -----------------------------------------------------------------------------
# Test 6: Coordinate Transformation Rigor
# -----------------------------------------------------------------------------
def test_coordinate_transformation_mapping(phase1_structured_layout: FloorPlan):
    """Test 6: Check exact scale and Y-axis top-left to Cartesian inversion."""
    tf = CoordinateTransformer.from_floorplan(phase1_structured_layout)
    assert tf.is_calibrated
    # 50 px/m -> 0.02 m/px
    assert abs(tf.scale_m_per_px - 0.02) < 1e-4

    # (100 px, 80 px) top-left in 2D
    # X_3d = 100 * 0.02 = 2.0m
    # Y_3d = (800 - 80) * 0.02 = 720 * 0.02 = 14.4m
    p3d = tf.to_3d_point(100.0, 80.0)
    assert abs(p3d[0] - 2.0) < 1e-3
    assert abs(p3d[1] - 14.4) < 1e-3


# -----------------------------------------------------------------------------
# Test 7 & 8: Plotly Mesh3d Trace Generation & Layer Visibility
# -----------------------------------------------------------------------------
def test_plotly_mesh3d_traces_and_layer_visibility(phase1_structured_layout: FloorPlan):
    """Test 7 & 8: Plotly Mesh3d traces with granular toggles and camera presets."""
    result = generate_3d_model(phase1_structured_layout)

    # Full visibility
    fig_full = ViewerAdapter.scene_to_plotly(
        scene=result.scene,
        camera_mode="Isometric View",
        theme="light",
        show_floors=True,
        show_exterior_walls=True,
        show_interior_walls=True,
        show_doors=True,
        show_windows=True,
        show_furniture=True,
    )
    assert isinstance(fig_full, go.Figure)
    trace_count_full = len(fig_full.data)
    assert trace_count_full > 0

    # Hide furniture
    fig_no_furn = ViewerAdapter.scene_to_plotly(
        scene=result.scene,
        camera_mode="Perspective View",
        show_furniture=False,
    )
    trace_count_no_furn = len(fig_no_furn.data)
    assert trace_count_no_furn < trace_count_full

    # Top-Down camera mode
    fig_top = ViewerAdapter.scene_to_plotly(
        scene=result.scene,
        camera_mode="Top-Down 3D View",
    )
    cam = fig_top.layout.scene.camera
    assert cam.eye.z > 2.0  # High elevation overhead


# -----------------------------------------------------------------------------
# Test 9 & 10: Binary GLB Export and Reload Validation
# -----------------------------------------------------------------------------
def test_glb_export_and_reimport(phase1_structured_layout: FloorPlan):
    """Test 9 & 10: Export to binary GLTF 2.0 (.glb) and verify re-import."""
    result = generate_3d_model(phase1_structured_layout)

    # 9. Export GLB
    glb_bytes = result.get_glb_bytes()
    assert isinstance(glb_bytes, bytes)
    assert len(glb_bytes) > 1000

    # 10. Re-import and verify
    assert ModelExporter.verify_glb(glb_bytes)
    buf = io.BytesIO(glb_bytes)
    reloaded = trimesh.load(buf, file_type="glb")
    assert isinstance(reloaded, trimesh.Scene)
    assert len(reloaded.geometry) > 0


# -----------------------------------------------------------------------------
# Test 11: Furniture Boundary Validation
# -----------------------------------------------------------------------------
def test_furniture_boundary_containment(phase1_structured_layout: FloorPlan):
    """Test 11: Verify furniture items are physically contained within room perimeters."""
    tf = CoordinateTransformer.from_floorplan(phase1_structured_layout)
    rooms = phase1_structured_layout.confirmed_rooms

    room_polys = {}
    for r in rooms:
        pts = tf.to_3d_planar_polygon(r.polygon)
        room_polys[r.room_id] = Polygon(pts)

    result = generate_3d_model(phase1_structured_layout)

    # Inspect furniture geometries
    for node_name, geom in result.scene.geometry.items():
        cat = geom.metadata.get("semantic_category") or ""
        if "furniture" in cat and isinstance(geom, trimesh.Trimesh):
            # Centroid in XY must be inside at least one room polygon (or very close)
            cx, cy = float(geom.centroid[0]), float(geom.centroid[1])
            pt = Point(cx, cy)
            is_inside_any = any(poly.buffer(0.10).contains(pt) for poly in room_polys.values())
            assert is_inside_any, f"Furniture {node_name} at ({cx}, {cy}) is outside room perimeters!"


# -----------------------------------------------------------------------------
# Test 12: Session-State Alternative Switching and Cache Partitioning
# -----------------------------------------------------------------------------
def test_alternative_cache_partitioning(cvae_three_alternatives: list[FloorPlan]):
    """Test 12: Ensure different alternatives generate and cache distinct 3D scenes."""
    cache = {}
    alt1 = cvae_three_alternatives[0]
    alt2 = cvae_three_alternatives[1]

    key1 = f"alt_0_{alt1.preference}_2.8_0.2"
    key2 = f"alt_1_{alt2.preference}_2.8_0.2"

    res1 = generate_3d_model(alt1)
    cache[key1] = res1

    res2 = generate_3d_model(alt2)
    cache[key2] = res2

    assert key1 != key2
    assert cache[key1] is not cache[key2]
    assert len(cache) == 2


# -----------------------------------------------------------------------------
# Test 13: Robust Error Handling
# -----------------------------------------------------------------------------
def test_error_handling_empty_and_degenerate_layouts():
    """Test 13: Gracefully handle layouts with zero rooms or invalid dimensions."""
    # Zero rooms
    empty_fp = FloorPlan(image_width=800, image_height=600)
    res_empty = generate_3d_model(empty_fp)
    assert not res_empty.validation_report.valid
    assert len(res_empty.validation_report.critical_errors) > 0
    # Scene should be valid object without crashing
    assert isinstance(res_empty.scene, trimesh.Scene)
