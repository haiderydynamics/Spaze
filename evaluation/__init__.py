"""SPAZE Research Evaluation & Benchmarking Package (Phase 4)."""
from evaluation.baseline import ProceduralBaselineGenerator
from evaluation.benchmark_suite import BenchmarkSuite, ComparisonMetricRow
from evaluation.cvae_evaluator import CVAEEvaluationMetrics, CVAEEvaluator
from evaluation.cv_evaluator import CVEvaluationResult, CVEvaluator

__all__ = [
    "ProceduralBaselineGenerator",
    "BenchmarkSuite",
    "ComparisonMetricRow",
    "CVAEEvaluationMetrics",
    "CVAEEvaluator",
    "CVEvaluationResult",
    "CVEvaluator",
]
