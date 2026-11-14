"""End-to-end integration tests for the SPAZE Phase 1 processing pipeline."""
from pathlib import Path
import pytest

from models.floorplan_schema import DetectionStatus, FloorPlan
from modules.floorplan_parser import FloorPlanParser
from modules.image_loader import load_image_from_path
from modules.image_processor import ImageProcessor
from modules.room_editor import RoomEditor
from modules.room_extractor import RoomExtractor
from modules.wall_detector import WallDetector
from utils.export import export_floorplan_to_json, save_floorplan_to_file, load_floorplan_from_file


def test_full_pipeline_on_sample_floorplan(tmp_path):
    sample_path = Path("sample_data/sample_floorplan.png")
    assert sample_path.exists(), "Sample test fixture must exist"

    # Step 1: Load Image
    loaded = load_image_from_path(sample_path)
    assert loaded.width == 800
    assert loaded.height == 600
    assert loaded.channels == 3
    assert loaded.image_rgb is not None

    # Step 2: Computer Vision Processing Stages
    processor = ImageProcessor()
    stages = processor.process(loaded.image_rgb)
    assert stages.grayscale.shape == (600, 800)
    assert stages.threshold.shape == (600, 800)
    assert stages.edges.shape == (600, 800)
    assert stages.morphology.shape == (600, 800)
    assert stages.processing_time_ms > 0

    # Step 3: Wall Detection
    wall_detector = WallDetector()
    walls = wall_detector.detect_walls(stages.morphology)
    assert len(walls) > 0, "Should detect actual wall line candidates in sample"

    # Step 4: Room Candidate Extraction
    room_extractor = RoomExtractor()
    candidates = room_extractor.extract_rooms(
        stages.morphology,
        image_width=loaded.width,
        image_height=loaded.height,
    )
    # The sample fixture contains distinct enclosed rooms (bedrooms, kitchen, bath, living)
    assert len(candidates) >= 3, f"Expected at least 3 detected rooms, got {len(candidates)}"

    # Verify no room is falsely auto-confirmed
    for r in candidates:
        assert not r.is_user_confirmed
        assert r.detection_status in (DetectionStatus.CANDIDATE.value, DetectionStatus.NEEDS_REVIEW.value)
        assert r.physical_width is None  # Initially uncalibrated!

    # Step 5: Floor-plan assembly
    floorplan = FloorPlanParser.build_initial_floorplan(
        loaded_image=loaded,
        processed_stages=stages,
        wall_candidates=walls,
        candidate_rooms=candidates,
    )
    assert floorplan.image_width == 800
    assert floorplan.image_height == 600
    assert len(floorplan.candidate_rooms) == len(candidates)
    assert len(floorplan.confirmed_rooms) == 0

    # Step 6: Scale Calibration
    # Calibrate: exterior width between x=60 and x=740 is 680 px = 13.6 meters (50 px/m)
    floorplan.update_scale(pixel_distance=680.0, real_meters=13.6)
    assert floorplan.scale_information.is_calibrated
    assert floorplan.scale_information.pixels_per_meter == pytest.approx(50.0, rel=1e-2)

    # Check physical measurements are now established for candidate rooms
    for r in floorplan.candidate_rooms:
        assert r.physical_width is not None
        assert r.physical_height is not None
        assert r.physical_area is not None

    # Step 7: Human Confirmation & Editing
    # User confirms the first room as 'Master Bedroom'
    first_room = floorplan.candidate_rooms[0]
    first_room_id = first_room.room_id

    RoomEditor.update_room(
        room=first_room,
        label="Master Bedroom",
        room_type="Bedroom",
        detection_status=DetectionStatus.USER_CONFIRMED.value,
        scale_info=floorplan.scale_information,
    )
    floorplan.confirm_room(first_room_id)

    assert len(floorplan.confirmed_rooms) == 1
    assert floorplan.confirmed_rooms[0].label == "Master Bedroom"
    assert floorplan.confirmed_rooms[0].room_type == "Bedroom"
    assert floorplan.confirmed_rooms[0].is_user_confirmed

    # User adds a manual room (e.g. Balcony missed by auto-detection)
    manual_balcony = RoomEditor.create_manual_room(
        room_id="room_balcony_01",
        label="Outdoor Balcony",
        room_type="Balcony",
        x=60.0,
        y=450.0,
        width=100.0,
        height=80.0,
        image_width=800,
        image_height=600,
        scale_info=floorplan.scale_information,
    )
    floorplan.confirmed_rooms.append(manual_balcony)
    assert len(floorplan.confirmed_rooms) == 2

    # Step 8: JSON Export and Persistence
    export_json = export_floorplan_to_json(floorplan)
    assert "Master Bedroom" in export_json
    assert "Outdoor Balcony" in export_json
    assert "13.6" in export_json

    out_file = tmp_path / "exported_plan.json"
    save_floorplan_to_file(floorplan, out_file)
    assert out_file.exists()

    # Step 9: Re-import and verify parity
    loaded_plan = load_floorplan_from_file(out_file)
    assert loaded_plan.image_width == 800
    assert loaded_plan.scale_information.is_calibrated
    assert len(loaded_plan.confirmed_rooms) == 2
    assert loaded_plan.confirmed_rooms[0].label == "Master Bedroom"
    assert loaded_plan.confirmed_rooms[1].label == "Outdoor Balcony"
