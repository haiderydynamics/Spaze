"""PyTorch Dataset and DataLoader for SPAZE Conditional VAE.

Loads paired source-target floor plans, extracts feature vectors,
and enforces family-aware train/val/test splits to eliminate data leakage.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from cvae.config import CVAEConfig
from cvae.dataset_generator import SyntheticFloorPlanGenerator, SyntheticPair
from cvae.tensor_adapter import FloorPlanTensorAdapter
from models.floorplan_schema import FloorPlan


class FloorPlanPairDataset(Dataset):
    """PyTorch Dataset yielding (source_vector, target_vector, preference_vector, metadata)."""

    def __init__(
        self,
        pairs: List[SyntheticPair],
        config: Optional[CVAEConfig] = None,
    ) -> None:
        self.pairs = pairs
        self.config = config or CVAEConfig()
        self.adapter = FloorPlanTensorAdapter(self.config)

        # Pre-encode all tensors in memory for fast training
        self.src_vectors: List[torch.Tensor] = []
        self.tgt_vectors: List[torch.Tensor] = []
        self.pref_vectors: List[torch.Tensor] = []
        self.pair_ids: List[str] = []
        self.family_ids: List[str] = []

        for p in self.pairs:
            src_fp = FloorPlan.from_dict(p.source_floorplan)
            tgt_fp = FloorPlan.from_dict(p.target_floorplan)

            src_vec = self.adapter.encode_to_flat_vector(src_fp)
            tgt_vec = self.adapter.encode_to_flat_vector(tgt_fp)
            pref_vec = torch.tensor(p.preference_vector, dtype=torch.float32)

            self.src_vectors.append(src_vec)
            self.tgt_vectors.append(tgt_vec)
            self.pref_vectors.append(pref_vec)
            self.pair_ids.append(p.pair_id)
            self.family_ids.append(p.family_id)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "source_vector": self.src_vectors[idx],
            "target_vector": self.tgt_vectors[idx],
            "preference_vector": self.pref_vectors[idx],
            "pair_id": self.pair_ids[idx],
            "family_id": self.family_ids[idx],
        }


def create_dataloaders(
    dataset_source: Union[List[SyntheticPair], str, Path],
    config: Optional[CVAEConfig] = None,
    batch_size: int = 32,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test DataLoaders with family-aware splitting.

    Args:
        dataset_source: List of SyntheticPair objects or path to synthetic_pairs.json.
        config: CVAEConfig instance.
        batch_size: Mini-batch size.
        train_ratio: Proportion of families for training (e.g. 0.80).
        val_ratio: Proportion of families for validation (e.g. 0.10).
        seed: Random seed for reproducibility.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    cfg = config or CVAEConfig()

    if isinstance(dataset_source, (str, Path)):
        p = Path(dataset_source)
        if not p.is_file():
            # Generate dataset if file doesn't exist
            print(f"Dataset not found at {p}. Generating 3,000 synthetic pairs...")
            gen = SyntheticFloorPlanGenerator(cfg, seed=seed)
            pairs = gen.generate_dataset_pairs(num_pairs=3000, output_dir=p.parent)
        else:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            pairs = [SyntheticPair(**item) for item in data]
    else:
        pairs = dataset_source

    if not pairs:
        raise ValueError("Cannot create DataLoaders from empty dataset.")

    # Group pairs by family_id to eliminate data leakage
    family_map: Dict[str, List[SyntheticPair]] = {}
    for pair in pairs:
        family_map.setdefault(pair.family_id, []).append(pair)

    family_ids = sorted(list(family_map.keys()))
    rng = np.random.default_rng(seed)
    rng.shuffle(family_ids)

    n_families = len(family_ids)
    n_train = int(n_families * train_ratio)
    n_val = int(n_families * val_ratio)

    train_fams = set(family_ids[:n_train])
    val_fams = set(family_ids[n_train:n_train + n_val])
    test_fams = set(family_ids[n_train + n_val:])

    train_pairs = [p for p in pairs if p.family_id in train_fams]
    val_pairs = [p for p in pairs if p.family_id in val_fams]
    test_pairs = [p for p in pairs if p.family_id in test_fams]

    train_dataset = FloorPlanPairDataset(train_pairs, cfg)
    val_dataset = FloorPlanPairDataset(val_pairs, cfg)
    test_dataset = FloorPlanPairDataset(test_pairs, cfg)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    print(
        f"DataLoaders prepared: {len(train_dataset)} train, {len(val_dataset)} val, "
        f"{len(test_dataset)} test across {n_families} layout families."
    )
    return train_loader, val_loader, test_loader
