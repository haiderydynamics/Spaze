"""Configuration and constants for the SPAZE Conditional VAE engine."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class CVAEConfig:
    """Hyperparameters, feature dimensions, and vocabulary constants for CVAE."""

    # 1. Room Type Vocabulary (10 standard residential types)
    ROOM_VOCABULARY: Tuple[str, ...] = (
        "Living Room",
        "Master Bedroom",
        "Bedroom",
        "Kitchen",
        "Bathroom",
        "Dining Room",
        "Hallway",
        "Study Room",
        "Balcony",
        "Utility Room",
    )

    # 2. Preference Conditioning Labels
    PREFERENCES: Tuple[str, ...] = (
        "space_efficient",  # [1, 0, 0]
        "open_layout",      # [0, 1, 0]
        "privacy_focused",  # [0, 0, 1]
    )

    # 3. Dimensionality Constants
    MAX_ROOMS: int = 12
    NUM_ROOM_TYPES: int = 10
    NUM_PREFERENCES: int = 3

    # Room feature vector decomposition:
    # - one-hot room type: 10 dims
    # - center_x (normalized [0, 1]): 1 dim
    # - center_y (normalized [0, 1]): 1 dim
    # - width (normalized [0, 1]): 1 dim
    # - height (normalized [0, 1]): 1 dim
    # - orientation sin(theta): 1 dim
    # - orientation cos(theta): 1 dim
    # - presence flag (0 or 1): 1 dim
    ROOM_FEATURE_DIM: int = 17

    # Total flattened room features = 12 * 17 = 204
    @property
    def total_room_features(self) -> int:
        return self.MAX_ROOMS * self.ROOM_FEATURE_DIM

    # Adjacency matrix features = 12 * 12 = 144
    @property
    def total_adjacency_features(self) -> int:
        return self.MAX_ROOMS * self.MAX_ROOMS

    # Full layout vector size = 204 + 144 = 348
    @property
    def total_layout_dim(self) -> int:
        return self.total_room_features + self.total_adjacency_features

    # 4. Model Architecture Hyperparameters
    LATENT_DIM: int = 32
    HIDDEN_DIM: int = 256
    ENCODER_LAYERS: int = 3
    DECODER_LAYERS: int = 3
    DROPOUT: float = 0.1

    # 5. Training Defaults
    DEFAULT_BATCH_SIZE: int = 32
    DEFAULT_EPOCHS: int = 60
    DEFAULT_LR: float = 1e-3
    DEFAULT_WEIGHT_DECAY: float = 1e-5
    KL_WEIGHT: float = 0.005
    KL_WARMUP_EPOCHS: int = 15
    GEOMETRY_WEIGHT: float = 0.05
    RANDOM_SEED: int = 42

    # 6. Paths
    CHECKPOINT_DIR: Path = Path("checkpoints")
    BEST_MODEL_FILENAME: str = "spaze_cvae_best.pth"
    LOG_DIR: Path = Path("training_logs")
    DATASET_DIR: Path = Path("datasets/generated")
    PROCESSED_DATASET_DIR: Path = Path("datasets/processed")

    # 7. Helper mappings
    @property
    def type_to_idx(self) -> Dict[str, int]:
        return {rt: idx for idx, rt in enumerate(self.ROOM_VOCABULARY)}

    @property
    def idx_to_type(self) -> Dict[int, str]:
        return {idx: rt for idx, rt in enumerate(self.ROOM_VOCABULARY)}

    @property
    def pref_to_idx(self) -> Dict[str, int]:
        return {pref: idx for idx, pref in enumerate(self.PREFERENCES)}

    @property
    def idx_to_pref(self) -> Dict[int, str]:
        return {idx: pref for idx, pref in enumerate(self.PREFERENCES)}
