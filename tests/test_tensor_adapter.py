"""Unit tests for cvae.tensor_adapter."""
import pytest
import torch

from cvae.config import CVAEConfig
from cvae.dataset_generator import SyntheticFloorPlanGenerator
from cvae.tensor_adapter import FloorPlanTensorAdapter
from models.floorplan_schema import FloorPlan


def test_tensor_adapter_roundtrip():
    config = CVAEConfig()
    adapter = FloorPlanTensorAdapter(config)
    gen = SyntheticFloorPlanGenerator(config, seed=789)

    program = ["Living Room", "Kitchen", "Dining Room", "Master Bedroom", "Bathroom"]
    fp = gen.generate_single_layout(program, preference="space_efficient")
    assert fp is not None

    # 1. Encode
    room_tensor, adj_tensor = adapter.encode_floorplan(fp)
    assert room_tensor.shape == (config.MAX_ROOMS, config.ROOM_FEATURE_DIM)
    assert adj_tensor.shape == (config.MAX_ROOMS, config.MAX_ROOMS)

    flat_vec = adapter.encode_to_flat_vector(fp)
    assert flat_vec.shape == (config.total_layout_dim,)

    # 2. Decode
    reconstructed = adapter.decode_flat_vector(
        flat_vec,
        image_width=fp.image_width,
        image_height=fp.image_height,
        scale_info=fp.scale_information,
        preference="space_efficient",
    )

    assert isinstance(reconstructed, FloorPlan)
    rec_rooms = reconstructed.confirmed_rooms + reconstructed.candidate_rooms
    assert len(rec_rooms) == len(program)

    # Check room types preserved
    orig_types = sorted([r.room_type for r in fp.confirmed_rooms])
    rec_types = sorted([r.room_type for r in rec_rooms])
    assert orig_types == rec_types


def test_preference_encoding():
    config = CVAEConfig()
    adapter = FloorPlanTensorAdapter(config)

    vec_space = adapter.encode_preference("space_efficient")
    assert torch.equal(vec_space, torch.tensor([1.0, 0.0, 0.0]))

    vec_open = adapter.encode_preference("open_layout")
    assert torch.equal(vec_open, torch.tensor([0.0, 1.0, 0.0]))

    vec_priv = adapter.encode_preference("privacy_focused")
    assert torch.equal(vec_priv, torch.tensor([0.0, 0.0, 1.0]))
