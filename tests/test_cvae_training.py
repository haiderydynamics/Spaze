"""Unit tests for cvae.train."""
import pytest
from cvae.config import CVAEConfig
from cvae.dataset_generator import SyntheticFloorPlanGenerator
from cvae.train import train_cvae


def test_short_cvae_training_pipeline(tmp_path):
    config = CVAEConfig()
    gen = SyntheticFloorPlanGenerator(config, seed=999)

    # Generate small dataset of 20 pairs
    pairs = gen.generate_dataset_pairs(num_pairs=20, output_dir=tmp_path / "data")

    # Run quick 2-epoch training
    ckpt_dir = tmp_path / "checkpoints"
    log_dir = tmp_path / "logs"

    model, summary = train_cvae(
        epochs=2,
        batch_size=4,
        lr=1e-3,
        device_name="cpu",
        checkpoint_dir=ckpt_dir,
        log_dir=log_dir,
    )

    assert model is not None
    assert summary["epochs_trained"] == 2
    assert (ckpt_dir / config.BEST_MODEL_FILENAME).is_file()
    assert (log_dir / "training_history.csv").is_file()
