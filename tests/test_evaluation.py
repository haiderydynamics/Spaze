"""Unit tests for SPAZE Phase 4 Research Evaluation Engine."""
from __future__ import annotations

import json
from pathlib import Path
import pytest

from evaluation.baseline import ProceduralBaselineGenerator
from evaluation.benchmark_suite import BenchmarkSuite
from evaluation.cvae_evaluator import CVAEEvaluator, CVAEEvaluationMetrics
from evaluation.cv_evaluator import CVEvaluator, CVEvaluationResult
from models.floorplan_schema import FloorPlan, Room, WallCandidate


@pytest.fixture
def dummy_floorplan() -> FloorPlan:
    """Construct a clean test floor plan with known rooms and walls."""
    fp = FloorPlan(
        image_width=800,
        image_height=600,
        preference="space_efficient",
        is_generated=False,
    )
    fp.wall_candidates = [
        WallCandidate(wall_id="w1", start_point=(50.0, 50.0), end_point=(750.0, 50.0), length_pixels=700.0, angle_degrees=0.0, orientation="horizontal", confidence_score=0.9),
        WallCandidate(wall_id="w2", start_point=(50.0, 550.0), end_point=(750.0, 550.0), length_pixels=700.0, angle_degrees=0.0, orientation="horizontal", confidence_score=0.9),
        WallCandidate(wall_id="w3", start_point=(50.0, 50.0), end_point=(50.0, 550.0), length_pixels=500.0, angle_degrees=90.0, orientation="vertical", confidence_score=0.9),
        WallCandidate(wall_id="w4", start_point=(750.0, 50.0), end_point=(750.0, 550.0), length_pixels=500.0, angle_degrees=90.0, orientation="vertical", confidence_score=0.9),
    ]
    fp.confirmed_rooms = [
        Room(room_id="living_01", room_type="Living Room", label="Living Room", x=50, y=50, width=350, height=250),
        Room(room_id="kitchen_01", room_type="Kitchen", label="Kitchen", x=400, y=50, width=350, height=250),
        Room(room_id="bed_01", room_type="Master Bedroom", label="Master Bedroom", x=50, y=300, width=350, height=250),
        Room(room_id="bath_01", room_type="Bathroom", label="Bathroom", x=400, y=300, width=350, height=250),
    ]
    return fp


class TestCVEvaluator:
    """Test Computer Vision evaluation against ground truth."""

    def test_evaluate_without_ground_truth(self, dummy_floorplan: FloorPlan) -> None:
        evaluator = CVEvaluator()
        result = evaluator.evaluate(dummy_floorplan, ground_truth=None)
        assert isinstance(result, CVEvaluationResult)
        assert result.status == "Not Evaluated — Ground Truth Unavailable"
        assert result.wall_precision is None
        assert result.mean_room_iou is None

    def test_evaluate_with_ground_truth_dict(self, dummy_floorplan: FloorPlan) -> None:
        evaluator = CVEvaluator()
        gt_data = {
            "ground_truth_walls": [
                {"start": [50, 50], "end": [750, 50]},
                {"start": [50, 550], "end": [750, 550]},
                {"start": [50, 50], "end": [50, 550]},
                {"start": [750, 50], "end": [750, 550]},
            ],
            "ground_truth_rooms": [
                {"name": "Living Room", "room_type": "Living Room", "bbox": [50, 50, 400, 300]},
                {"name": "Kitchen", "room_type": "Kitchen", "bbox": [400, 50, 750, 300]},
            ],
        }
        result = evaluator.evaluate(dummy_floorplan, ground_truth=gt_data)
        assert result.status == "Evaluated"
        assert result.wall_precision is not None
        assert result.wall_precision > 0.0
        assert result.wall_recall is not None
        assert result.mean_room_iou is not None
        assert result.mean_room_iou > 0.0

    def test_evaluate_with_sample_ground_truth_file(self, dummy_floorplan: FloorPlan) -> None:
        evaluator = CVEvaluator()
        gt_path = Path("sample_data/sample_ground_truth.json")
        if gt_path.exists():
            result = evaluator.evaluate(dummy_floorplan, ground_truth=gt_path)
            assert result.status == "Evaluated"
            assert result.detected_wall_count == 4


class TestProceduralBaseline:
    """Test deterministic procedural heuristic layout baseline."""

    def test_generate_space_efficient(self, dummy_floorplan: FloorPlan) -> None:
        baseline = ProceduralBaselineGenerator.generate(dummy_floorplan, "space_efficient")
        assert isinstance(baseline, FloorPlan)
        assert baseline.preference == "space_efficient"
        assert baseline.is_generated is True
        assert len(baseline.confirmed_rooms) == len(dummy_floorplan.confirmed_rooms)
        for r in baseline.confirmed_rooms:
            assert r.x >= 0
            assert r.y >= 0
            assert r.x + r.width <= dummy_floorplan.image_width + 100
            assert r.y + r.height <= dummy_floorplan.image_height + 100

    def test_generate_open_layout(self, dummy_floorplan: FloorPlan) -> None:
        baseline = ProceduralBaselineGenerator.generate(dummy_floorplan, "open_layout")
        assert baseline.preference == "open_layout"
        assert len(baseline.confirmed_rooms) == len(dummy_floorplan.confirmed_rooms)

    def test_generate_privacy_focused(self, dummy_floorplan: FloorPlan) -> None:
        baseline = ProceduralBaselineGenerator.generate(dummy_floorplan, "privacy_focused")
        assert baseline.preference == "privacy_focused"
        assert len(baseline.confirmed_rooms) == len(dummy_floorplan.confirmed_rooms)


class TestCVAEEvaluator:
    """Test CVAE model performance metrics and evaluation."""

    def test_evaluator_checkpoint_evaluation(self) -> None:
        evaluator = CVAEEvaluator()
        metrics = evaluator.evaluate_model_checkpoint()
        assert isinstance(metrics, CVAEEvaluationMetrics)
        assert metrics.reconstruction_loss > 0.0
        assert metrics.kl_divergence >= 0.0
        assert metrics.model_param_count > 0
        assert metrics.layout_diversity_score > 0.0
        assert metrics.geometric_validity_rate >= 0.0

    def test_evaluate_layout_alternatives(self, dummy_floorplan: FloorPlan) -> None:
        alt1 = ProceduralBaselineGenerator.generate(dummy_floorplan, "space_efficient")
        alt2 = ProceduralBaselineGenerator.generate(dummy_floorplan, "open_layout")
        summary = CVAEEvaluator.evaluate_layout_alternatives([alt1, alt2])
        assert "diversity_score" in summary
        assert "mean_validity_rate" in summary
        assert "mean_overlap_rate" in summary

    def test_metrics_to_dict(self) -> None:
        evaluator = CVAEEvaluator()
        metrics = evaluator.evaluate_model_checkpoint()
        m_dict = metrics.to_dict()
        assert "reconstruction_loss" in m_dict
        assert "kl_divergence" in m_dict
        assert "layout_diversity_score" in m_dict


class TestBenchmarkSuite:
    """Test comparative benchmark orchestrator."""

    def test_run_comparative_benchmark(self, dummy_floorplan: FloorPlan) -> None:
        rows = BenchmarkSuite.run_comparative_benchmark(dummy_floorplan)
        assert len(rows) >= 5
        metric_names = [r.metric_name for r in rows]
        assert "Geometric Validity Rate" in metric_names
        assert "Layout Diversity (1 - IoU)" in metric_names
        assert "Room Overlap Rate" in metric_names
