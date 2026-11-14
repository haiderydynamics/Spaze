"""Unit tests for 3D coordinate system transformation engine."""
import pytest
from geometry3d.config import Geometry3DConfig
from geometry3d.coordinate_system import CoordinateTransformer
from models.floorplan_schema import FloorPlan, Room, ScaleInfo


def test_uncalibrated_coordinate_transformation():
    """Verify default 50 px/m scale transformation."""
    tf = CoordinateTransformer(image_width=1000, image_height=800)
    assert not tf.is_calibrated
    assert tf.scale_m_per_px == 1.0 / 50.0  # 0.02 m/px

    # (0, 0) top-left in 2D maps to (0, 16.0, 0) in 3D (Y inverted: 800 * 0.02 = 16m)
    p3d = tf.to_3d_point(0, 0)
    assert p3d == (0.0, 16.0, 0.0)

    # (500, 400) center maps to (10.0, 8.0, 0)
    p3d_center = tf.to_3d_point(500, 400)
    assert p3d_center == (10.0, 8.0, 0.0)

    # Reversible roundtrip mapping
    x2d, y2d = tf.to_2d_point(p3d_center[0], p3d_center[1])
    assert abs(x2d - 500.0) < 0.01
    assert abs(y2d - 400.0) < 0.01


def test_calibrated_coordinate_transformation():
    """Verify calibrated scale info (e.g., 100 px/m)."""
    scale = ScaleInfo()
    scale.calibrate(pixel_distance=200.0, real_meters=2.0)  # 100 px/m -> 0.01 m/px

    tf = CoordinateTransformer(image_width=1200, image_height=800, scale_info=scale)
    assert tf.is_calibrated
    assert tf.scale_m_per_px == 0.01

    p3d = tf.to_3d_point(100, 100, z_elevation_m=2.8)
    assert p3d == (1.0, 7.0, 2.8)

    # Roundtrip
    p2d = tf.to_2d_point(1.0, 7.0)
    assert p2d == (100.0, 100.0)


def test_polygon_coordinate_conversion():
    """Verify conversion of 2D polygon vertices to 3D planar points."""
    tf = CoordinateTransformer(image_width=1000, image_height=1000)
    poly_2d = [(100, 100), (300, 100), (300, 300), (100, 300)]
    poly_3d = tf.to_3d_planar_polygon(poly_2d)

    assert len(poly_3d) == 4
    # Check that Y-order is flipped (Cartesian Y-up)
    assert poly_3d[0] == (2.0, 18.0)
    assert poly_3d[2] == (6.0, 14.0)
