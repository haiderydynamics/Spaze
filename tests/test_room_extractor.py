"""Unit tests for modules.room_extractor."""
import cv2
import numpy as np
import pytest

from modules.room_extractor import RoomExtractor
from models.floorplan_schema import DetectionStatus


@pytest.fixture
def enclosed_room_wall_mask() -> np.ndarray:
    """Create a 300x300 wall mask (walls=255, interior space=0) with an enclosed room."""
    mask = np.zeros((300, 300), dtype=np.uint8)
    # Draw walls surrounding a 100x100 room at (50, 50)
    cv2.rectangle(mask, (50, 50), (150, 150), 255, thickness=4)
    return mask


def test_extract_rooms_finds_enclosed_region(enclosed_room_wall_mask):
    extractor = RoomExtractor()
    rooms = extractor.extract_rooms(enclosed_room_wall_mask, image_width=300, image_height=300)

    assert len(rooms) >= 1
    room = rooms[0]
    assert room.room_id.startswith("room_")
    assert not room.is_user_confirmed  # Never automatically confirmed!
    assert room.detection_status in (DetectionStatus.CANDIDATE.value, DetectionStatus.NEEDS_REVIEW.value)
    assert 0.0 <= room.confidence_score <= 1.0
    assert "Heuristic quality" in room.confidence_explanation
    assert room.width > 0
    assert room.height > 0
    assert len(room.polygon) >= 3


def test_extract_rooms_empty_mask():
    extractor = RoomExtractor()
    empty = np.zeros((200, 200), dtype=np.uint8)
    rooms = extractor.extract_rooms(empty, image_width=200, image_height=200)
    # An entirely black wall mask means no walls, so no enclosed rooms
    assert len(rooms) == 0


def test_extract_rooms_tiny_noise_filtered():
    extractor = RoomExtractor()
    mask = np.zeros((200, 200), dtype=np.uint8)
    # Tiny box of 4x4 px enclosed space
    cv2.rectangle(mask, (20, 20), (26, 26), 255, thickness=1)

    rooms = extractor.extract_rooms(mask, image_width=200, image_height=200)
    assert len(rooms) == 0


def test_invalid_mask_dimensions():
    extractor = RoomExtractor()
    invalid_mask = np.zeros((100, 100, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="single-channel 2D"):
        extractor.extract_rooms(invalid_mask, image_width=100, image_height=100)
