"""Unit tests for modules.wall_detector."""
import cv2
import numpy as np
import pytest

from modules.wall_detector import WallDetector


@pytest.fixture
def binary_lines_map() -> np.ndarray:
    """Create a 200x200 binary edge map with clean horizontal and vertical lines."""
    img = np.zeros((200, 200), dtype=np.uint8)
    # Horizontal line
    cv2.line(img, (20, 50), (180, 50), 255, thickness=2)
    # Vertical line
    cv2.line(img, (100, 20), (100, 180), 255, thickness=2)
    return img


def test_detect_walls_finds_lines(binary_lines_map):
    detector = WallDetector()
    walls = detector.detect_walls(binary_lines_map, min_line_length=30)

    assert len(walls) >= 2
    orientations = {w.orientation for w in walls}
    assert "horizontal" in orientations
    assert "vertical" in orientations

    for w in walls:
        assert 0.0 <= w.confidence_score <= 1.0
        assert w.length_pixels >= 30.0
        assert w.wall_id.startswith("wall_")


def test_detect_walls_empty_map():
    detector = WallDetector()
    empty_map = np.zeros((100, 100), dtype=np.uint8)
    walls = detector.detect_walls(empty_map)
    assert walls == []


def test_detect_walls_short_lines_filtered():
    detector = WallDetector()
    img = np.zeros((100, 100), dtype=np.uint8)
    # Draw tiny tick line of 5px length
    cv2.line(img, (10, 10), (15, 10), 255, thickness=1)

    walls = detector.detect_walls(img, min_line_length=30)
    assert len(walls) == 0


def test_invalid_input_dimensions():
    detector = WallDetector()
    invalid_3d = np.zeros((100, 100, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="single-channel 2D"):
        detector.detect_walls(invalid_3d)
