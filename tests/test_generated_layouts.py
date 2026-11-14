"""Unit tests for cvae.constraints and cvae.metrics."""
import pytest
from cvae.config import CVAEConfig
from cvae.constraints import GeometricConstraintsValidator
from cvae.dataset_generator import SyntheticFloorPlanGenerator
from cvae.metrics import LayoutMetricsEvaluator


def test_geometric_constraints_evaluation():
    config = CVAEConfig()
    gen = SyntheticFloorPlanGenerator(config, seed=303)
    validator = GeometricConstraintsValidator()

    program = ["Living Room", "Kitchen", "Dining Room", "Master Bedroom", "Bathroom"]
    fp = gen.generate_single_layout(program, preference="privacy_focused")
    assert fp is not None

    res = validator.evaluate_layout(fp)
    assert res.is_valid
    assert res.overall_validity_score >= 0.70
    assert res.overlap_ratio <= 0.05
    assert res.boundary_violation_ratio <= 0.05
    assert res.room_count == len(program)


def test_layout_metrics_evaluator():
    config = CVAEConfig()
    gen = SyntheticFloorPlanGenerator(config, seed=404)
    evaluator = LayoutMetricsEvaluator(config)

    program = ["Living Room", "Kitchen", "Bedroom", "Bathroom"]
    fp1 = gen.generate_single_layout(program, preference="space_efficient")
    fp2 = gen.generate_single_layout(program, preference="open_layout")
    fp3 = gen.generate_single_layout(program, preference="privacy_focused")

    diversity = evaluator.compute_pairwise_iou_diversity([fp1, fp2, fp3])
    assert 0.0 <= diversity <= 1.0

    batch_metrics = evaluator.evaluate_generated_batch(
        source_floorplan=fp1,
        alternatives=[fp1, fp2, fp3],
        preference="space_efficient",
        inference_time_ms=12.5,
    )
    assert batch_metrics["num_alternatives"] == 3
    assert batch_metrics["layout_diversity_score"] == diversity
