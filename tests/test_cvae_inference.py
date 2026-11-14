"""Unit tests for cvae.infer."""
import pytest
from cvae.config import CVAEConfig
from cvae.dataset_generator import SyntheticFloorPlanGenerator
from cvae.infer import LayoutGenerationEngine
from cvae.train import train_cvae
from models.floorplan_schema import FloorPlan


@pytest.fixture(scope="module")
def trained_checkpoint(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("infer_test")
    config = CVAEConfig()
    gen = SyntheticFloorPlanGenerator(config, seed=101)
    gen.generate_dataset_pairs(num_pairs=25, output_dir=tmp_path / "data")

    ckpt_dir = tmp_path / "checkpoints"
    train_cvae(
        epochs=3,
        batch_size=8,
        lr=1e-3,
        device_name="cpu",
        checkpoint_dir=ckpt_dir,
        log_dir=tmp_path / "logs",
    )
    return ckpt_dir / config.BEST_MODEL_FILENAME


def test_inference_generates_three_alternatives(trained_checkpoint):
    config = CVAEConfig()
    engine = LayoutGenerationEngine(checkpoint_path=trained_checkpoint, config=config, device="cpu")
    assert engine.is_model_loaded()

    # Create input floor plan
    gen = SyntheticFloorPlanGenerator(config, seed=202)
    src_fp = gen.generate_single_layout(["Living Room", "Kitchen", "Bedroom", "Bathroom"])
    assert src_fp is not None

    alternatives, metrics = engine.generate_alternatives(
        source_floorplan=src_fp,
        preference="open_layout",
        num_alternatives=3,
        random_seed=42,
    )

    assert len(alternatives) == 3
    for alt in alternatives:
        assert isinstance(alt, FloorPlan)
        assert alt.is_generated
        assert alt.preference == "open_layout"
        rooms = alt.confirmed_rooms + alt.candidate_rooms
        assert len(rooms) >= 1

    assert "layout_diversity_score" in metrics
    assert "mean_validity_score" in metrics
    assert metrics["num_alternatives"] == 3
