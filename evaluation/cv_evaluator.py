"""Computer Vision Ground-Truth Evaluation Engine.

Calculates Precision, Recall, and F1 Score for detected wall lines, and
Intersection-over-Union (IoU) for extracted room regions against verified
architectural ground-truth annotations.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from shapely.geometry import box, LineString, Polygon

from models.floorplan_schema import FloorPlan, Room, WallCandidate


@dataclass
class CVEvaluationResult:
    """Quantitative evaluation report for computer vision extraction."""
    status: str  # "Evaluated" | "Not Evaluated — Ground Truth Unavailable"
    wall_precision: Optional[float] = None
    wall_recall: Optional[float] = None
    wall_f1: Optional[float] = None
    mean_room_iou: Optional[float] = None
    matched_wall_count: int = 0
    ground_truth_wall_count: int = 0
    detected_wall_count: int = 0
    evaluated_room_count: int = 0
    per_room_iou: Dict[str, float] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "wall_precision": round(self.wall_precision, 4) if self.wall_precision is not None else None,
            "wall_recall": round(self.wall_recall, 4) if self.wall_recall is not None else None,
            "wall_f1": round(self.wall_f1, 4) if self.wall_f1 is not None else None,
            "mean_room_iou": round(self.mean_room_iou, 4) if self.mean_room_iou is not None else None,
            "matched_wall_count": self.matched_wall_count,
            "ground_truth_wall_count": self.ground_truth_wall_count,
            "detected_wall_count": self.detected_wall_count,
            "evaluated_room_count": self.evaluated_room_count,
            "per_room_iou": {k: round(v, 4) for k, v in self.per_room_iou.items()},
            "notes": self.notes,
        }


class CVEvaluator:
    """Evaluates wall line detection and room extraction against ground truth."""

    def __init__(
        self,
        line_distance_tol: float = 18.0,
        line_angle_tol_deg: float = 10.0,
    ) -> None:
        self.line_distance_tol = line_distance_tol
        self.line_angle_tol_deg = line_angle_tol_deg

    def evaluate(
        self,
        floorplan: FloorPlan,
        ground_truth: Optional[Union[Path, str, Dict[str, Any]]] = None,
    ) -> CVEvaluationResult:
        """Run evaluation against ground truth file or dictionary if available."""
        gt_data: Optional[Dict[str, Any]] = None
        source_name = "in-memory"

        if isinstance(ground_truth, dict):
            gt_data = ground_truth
        elif ground_truth is not None:
            gt_file = Path(ground_truth)
            if gt_file.is_file():
                source_name = gt_file.name
                try:
                    with open(gt_file, "r", encoding="utf-8") as f:
                        gt_data = json.load(f)
                except Exception as e:
                    return CVEvaluationResult(
                        status="Not Evaluated — Error Loading Ground Truth",
                        notes=f"Error reading {gt_file}: {e}",
                    )
        else:
            # Check default sample ground truth
            cand = Path("sample_data/sample_ground_truth.json")
            if cand.is_file() and (floorplan.processing_metadata.image_filename == "sample_floorplan.png" or "sample" in floorplan.image_path):
                source_name = cand.name
                try:
                    with open(cand, "r", encoding="utf-8") as f:
                        gt_data = json.load(f)
                except Exception:
                    gt_data = None

        if gt_data is None:
            return CVEvaluationResult(
                status="Not Evaluated — Ground Truth Unavailable",
                notes="Evaluation requires manually verified ground-truth CAD annotations.",
            )

        gt_walls = gt_data.get("ground_truth_walls", [])
        gt_rooms = gt_data.get("ground_truth_rooms", [])

        # 1. Wall Line Detection Evaluation
        detected_walls = floorplan.wall_candidates
        tp_walls = 0
        matched_gt_indices = set()

        for d_wall in detected_walls:
            d_line = LineString([d_wall.start_point, d_wall.end_point])
            d_angle = d_wall.angle_degrees

            for idx, gt_w in enumerate(gt_walls):
                if idx in matched_gt_indices:
                    continue

                g_p1, g_p2 = gt_w["start"], gt_w["end"]
                g_line = LineString([g_p1, g_p2])
                g_dx = g_p2[0] - g_p1[0]
                g_dy = g_p2[1] - g_p1[1]
                g_angle = (math.degrees(math.atan2(g_dy, g_dx)) + 180.0) % 180.0
                norm_d_angle = (d_angle + 180.0) % 180.0

                angle_diff = abs(norm_d_angle - g_angle)
                if angle_diff > 90:
                    angle_diff = 180.0 - angle_diff

                if angle_diff <= self.line_angle_tol_deg:
                    if d_line.distance(g_line) <= self.line_distance_tol:
                        tp_walls += 1
                        matched_gt_indices.add(idx)
                        break

        total_detected = len(detected_walls)
        total_gt_walls = len(gt_walls)

        precision = (tp_walls / total_detected) if total_detected > 0 else 0.0
        recall = (tp_walls / total_gt_walls) if total_gt_walls > 0 else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        # 2. Room IoU Evaluation
        all_rooms = floorplan.confirmed_rooms + floorplan.candidate_rooms
        per_room_iou: Dict[str, float] = {}
        matched_room_ious = []

        for gt_r in gt_rooms:
            gt_bbox = gt_r["bbox"]  # [x0, y0, x1, y1]
            gt_poly = box(gt_bbox[0], gt_bbox[1], gt_bbox[2], gt_bbox[3])
            best_iou = 0.0

            for pred_r in all_rooms:
                pred_poly = Polygon(pred_r.polygon) if len(pred_r.polygon) >= 3 else box(pred_r.x, pred_r.y, pred_r.x + pred_r.width, pred_r.y + pred_r.height)
                if pred_poly.is_valid and pred_poly.area > 0:
                    inter_area = pred_poly.intersection(gt_poly).area
                    union_area = pred_poly.union(gt_poly).area
                    iou = inter_area / union_area if union_area > 0 else 0.0
                    if iou > best_iou:
                        best_iou = iou

            per_room_iou[gt_r["name"]] = best_iou
            matched_room_ious.append(best_iou)

        mean_iou = float(np.mean(matched_room_ious)) if matched_room_ious else 0.0

        return CVEvaluationResult(
            status="Evaluated",
            wall_precision=precision,
            wall_recall=recall,
            wall_f1=f1,
            mean_room_iou=mean_iou,
            matched_wall_count=tp_walls,
            ground_truth_wall_count=total_gt_walls,
            detected_wall_count=total_detected,
            evaluated_room_count=len(gt_rooms),
            per_room_iou=per_room_iou,
            notes=f"Evaluated against ground-truth fixture: {source_name}",
        )
