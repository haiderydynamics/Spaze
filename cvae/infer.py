"""Generative Layout Inference Service for SPAZE CVAE.

Accepts an input FloorPlan and architectural design preference,
samples diverse latent vectors from the trained CVAE, validates and refines
geometry, filters out duplicates, and produces 3 distinct, geometrically sound alternatives.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from cvae.config import CVAEConfig
from cvae.constraints import GeometricConstraintsValidator
from cvae.metrics import LayoutMetricsEvaluator
from cvae.model import SpazeCVAE
from cvae.tensor_adapter import FloorPlanTensorAdapter
from models.floorplan_schema import FloorPlan


class LayoutGenerationEngine:
    """Manages inference, latent sampling, ranking, and validation of layout alternatives."""

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        config: Optional[CVAEConfig] = None,
        device: Optional[str] = None,
    ) -> None:
        self.config = config or CVAEConfig()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.adapter = FloorPlanTensorAdapter(self.config)
        self.validator = GeometricConstraintsValidator()
        self.evaluator = LayoutMetricsEvaluator(self.config)

        self.model: Optional[SpazeCVAE] = None
        self.checkpoint_metadata: Dict[str, Any] = {}

        ckpt_path = Path(checkpoint_path or (self.config.CHECKPOINT_DIR / self.config.BEST_MODEL_FILENAME))
        if ckpt_path.is_file():
            self.load_checkpoint(ckpt_path)

    def is_model_loaded(self) -> bool:
        """Check if model checkpoint is loaded and ready for inference."""
        return self.model is not None

    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> None:
        """Safely load model weights and configuration from a .pth checkpoint."""
        path = Path(checkpoint_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found at: {path}")

        state_dict = torch.load(path, map_location=self.device, weights_only=False)

        # Initialize model
        self.model = SpazeCVAE(self.config).to(self.device)

        if "model_state_dict" in state_dict:
            self.model.load_state_dict(state_dict["model_state_dict"])
            self.checkpoint_metadata = {
                k: v for k, v in state_dict.items() if k != "model_state_dict"
            }
        else:
            self.model.load_state_dict(state_dict)

        self.model.eval()
        print(f"CVAE model successfully loaded from {path} on {self.device}")

    def generate_alternatives(
        self,
        source_floorplan: FloorPlan,
        preference: str = "space_efficient",
        num_alternatives: int = 3,
        temperature: float = 1.0,
        random_seed: Optional[int] = None,
        num_pool_samples: int = 12,
    ) -> Tuple[List[FloorPlan], Dict[str, Any]]:
        """Generate num_alternatives distinct, validated layouts from one input FloorPlan.

        Args:
            source_floorplan: Phase 1 FloorPlan instance.
            preference: 'space_efficient', 'open_layout', or 'privacy_focused'.
            num_alternatives: Number of alternative layouts to return (default 3).
            temperature: Sampling temperature scaling latent Gaussian spread.
            random_seed: Optional seed for reproducibility.
            num_pool_samples: Number of latent samples drawn for ranking & diversity filtering.

        Returns:
            Tuple of (List[FloorPlan], metrics_summary_dict).
        """
        if not self.is_model_loaded():
            raise RuntimeError(
                "Cannot generate layouts: CVAE model checkpoint has not been trained or loaded. "
                "Run 'python cvae/train.py' first."
            )

        start_time = time.perf_counter()

        if random_seed is not None:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)

        # 1. Encode source floor plan and preference
        src_vec = self.adapter.encode_to_flat_vector(source_floorplan).unsqueeze(0).to(self.device)
        pref_vec = self.adapter.encode_preference(preference).unsqueeze(0).to(self.device)

        # 2. Draw multiple candidate latent samples from decoder pool
        pool_samples: List[Tuple[FloorPlan, float]] = []

        with torch.no_grad():
            for i in range(num_pool_samples):
                # Scale latent noise by temperature
                z = torch.randn(1, self.config.LATENT_DIM, device=self.device) * temperature
                out = self.model.decode(z, src_vec, pref_vec)
                pred_layout = out["flat_layout"].squeeze(0)

                # Decode into structured FloorPlan
                cand_fp = self.adapter.decode_flat_vector(
                    flat_vector=pred_layout,
                    image_width=source_floorplan.image_width,
                    image_height=source_floorplan.image_height,
                    scale_info=source_floorplan.scale_information,
                    preference=preference,
                )

                # Geometric relaxation and boundary clamping
                cand_fp = self.validator.relax_and_clamp_layout(cand_fp)

                # Score candidate geometry
                eval_res = self.validator.evaluate_layout(cand_fp, source_floorplan=source_floorplan)
                pool_samples.append((cand_fp, eval_res.overall_validity_score))

        # 3. Sort pool by validity score descending
        pool_samples.sort(key=lambda item: item[1], reverse=True)

        # 4. Filter for diversity and distinctness (avoid duplicate layouts)
        selected_alternatives: List[FloorPlan] = []

        for cand_fp, score in pool_samples:
            if len(selected_alternatives) >= num_alternatives:
                break

            # Check duplication with already selected candidates
            is_duplicate = False
            for existing in selected_alternatives:
                sim = 1.0 - self.evaluator.compute_pairwise_iou_diversity([cand_fp, existing])
                if sim >= 0.88:  # 88% overlap threshold
                    is_duplicate = True
                    break

            if not is_duplicate:
                selected_alternatives.append(cand_fp)

        # Fallback if pool diversity was too narrow
        if len(selected_alternatives) < num_alternatives and pool_samples:
            for cand_fp, _ in pool_samples:
                if cand_fp not in selected_alternatives:
                    selected_alternatives.append(cand_fp)
                if len(selected_alternatives) >= num_alternatives:
                    break

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # 5. Measure metrics across selected alternatives
        metrics = self.evaluator.evaluate_generated_batch(
            source_floorplan=source_floorplan,
            alternatives=selected_alternatives,
            preference=preference,
            inference_time_ms=elapsed_ms,
        )

        # Attach diversity score to alternatives
        for alt in selected_alternatives:
            alt.diversity_score = metrics["layout_diversity_score"]

        return selected_alternatives, metrics
