"""Evaluation and Benchmarking Metrics for SPAZE Generated Floor Plans."""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from shapely.geometry import Polygon, box

from cvae.config import CVAEConfig
from cvae.constraints import GeometricConstraintsValidator
from models.floorplan_schema import FloorPlan


class LayoutMetricsEvaluator:
    """Computes comprehensive quantitative evaluation metrics for generated layouts."""

    def __init__(self, config: Optional[CVAEConfig] = None) -> None:
        self.config = config or CVAEConfig()
        self.validator = GeometricConstraintsValidator()

    def compute_pairwise_iou_diversity(self, alternatives: List[FloorPlan]) -> float:
        """Calculate pairwise layout diversity based on spatial intersection-over-union.

        Diversity = 1.0 - mean_pairwise_iou.
        Higher score indicates greater spatial variation among generated alternatives.
        """
        if len(alternatives) < 2:
            return 0.0

        n = len(alternatives)
        pair_ious: List[float] = []

        for i in range(n):
            fp1 = alternatives[i]
            rooms1 = fp1.confirmed_rooms + fp1.candidate_rooms
            for j in range(i + 1, n):
                fp2 = alternatives[j]
                rooms2 = fp2.confirmed_rooms + fp2.candidate_rooms

                # Calculate union and intersection across all rooms
                total_inter = 0.0
                total_union = 0.0

                for r1 in rooms1:
                    p1 = box(r1.x, r1.y, r1.x + r1.width, r1.y + r1.height)
                    for r2 in rooms2:
                        p2 = box(r2.x, r2.y, r2.x + r2.width, r2.y + r2.height)
                        if p1.intersects(p2):
                            inter = p1.intersection(p2).area
                            total_inter += inter

                area1 = sum(r.pixel_area for r in rooms1)
                area2 = sum(r.pixel_area for r in rooms2)
                total_union = max(1.0, area1 + area2 - total_inter)

                iou = total_inter / total_union
                pair_ious.append(iou)

        mean_iou = float(np.mean(pair_ious)) if pair_ious else 0.0
        diversity_score = max(0.0, min(1.0, 1.0 - mean_iou))
        return round(diversity_score, 4)

    def evaluate_generated_batch(
        self,
        source_floorplan: FloorPlan,
        alternatives: List[FloorPlan],
        preference: str,
        inference_time_ms: float = 0.0,
    ) -> Dict[str, Any]:
        """Evaluate a set of 3 generated alternatives against the source layout."""
        validity_results = [
            self.validator.evaluate_layout(alt, source_floorplan=source_floorplan)
            for alt in alternatives
        ]

        valid_count = sum(1 for res in validity_results if res.is_valid)
        validity_rate = (valid_count / len(alternatives)) if alternatives else 0.0
        mean_validity_score = float(np.mean([res.overall_validity_score for res in validity_results])) if validity_results else 0.0
        mean_overlap_ratio = float(np.mean([res.overlap_ratio for res in validity_results])) if validity_results else 0.0

        diversity = self.compute_pairwise_iou_diversity(alternatives)

        # Room count and type preservation
        src_rooms = source_floorplan.confirmed_rooms + source_floorplan.candidate_rooms
        src_count = len(src_rooms)
        count_matches = sum(
            1 for alt in alternatives
            if len(alt.confirmed_rooms + alt.candidate_rooms) == src_count
        )
        count_consistency = count_matches / max(1, len(alternatives))

        mean_preservation = float(np.mean([res.rooms_preserved_ratio for res in validity_results])) if validity_results else 0.0

        return {
            "num_alternatives": len(alternatives),
            "valid_alternatives": valid_count,
            "geometric_validity_rate": round(validity_rate, 3),
            "mean_validity_score": round(mean_validity_score, 3),
            "mean_overlap_ratio": round(mean_overlap_ratio, 4),
            "layout_diversity_score": diversity,
            "room_count_consistency": round(count_consistency, 3),
            "room_type_preservation": round(mean_preservation, 3),
            "inference_time_ms": round(inference_time_ms, 2),
            "preference": preference,
        }
