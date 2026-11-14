"""Unit tests for cvae.dataset_generator."""
import pytest
from cvae.config import CVAEConfig
from cvae.dataset_generator import SyntheticFloorPlanGenerator, SyntheticPair


def test_generator_single_layout():
    config = CVAEConfig()
    gen = SyntheticFloorPlanGenerator(config, seed=123)

    program = ["Living Room", "Kitchen", "Bedroom", "Bathroom"]
    fp = gen.generate_single_layout(program, preference="open_layout")

    assert fp is not None
    assert fp.image_width == 800
    assert fp.image_height == 600
    rooms = fp.confirmed_rooms + fp.candidate_rooms
    assert len(rooms) == 4

    types = {r.room_type for r in rooms}
    assert "Living Room" in types
    assert "Kitchen" in types
    assert "Bedroom" in types
    assert "Bathroom" in types

    # Check adjacency matrix exists and is symmetric
    assert fp.adjacency_matrix is not None
    n = len(rooms)
    assert len(fp.adjacency_matrix) == n
    for i in range(n):
        for j in range(n):
            assert fp.adjacency_matrix[i][j] == fp.adjacency_matrix[j][i]


def test_generator_dataset_pairs(tmp_path):
    config = CVAEConfig()
    gen = SyntheticFloorPlanGenerator(config, seed=456)

    pairs = gen.generate_dataset_pairs(num_pairs=10, output_dir=tmp_path)
    assert len(pairs) == 10

    first_pair = pairs[0]
    assert isinstance(first_pair, SyntheticPair)
    assert first_pair.preference in config.PREFERENCES
    assert len(first_pair.preference_vector) == 3
    assert sum(first_pair.preference_vector) == 1.0

    # Source and target share same room program
    src_types = sorted([r["room_type"] for r in first_pair.source_floorplan["confirmed_rooms"]])
    tgt_types = sorted([r["room_type"] for r in first_pair.target_floorplan["confirmed_rooms"]])
    assert src_types == tgt_types
