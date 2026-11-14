"""3D Model Synthesis Metrics and Architectural Performance Diagnostics.

Measures computational mesh geometry complexity, triangle/vertex budgets,
spatial dimensions, usable floor areas, and execution runtime.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import trimesh

from geometry3d.opening_generator import Opening3D
from geometry3d.wall_analyzer import WallSegment
from models.floorplan_schema import FloorPlan


@dataclass
class Model3DMetrics:
    """Quantitative performance and geometric metrics for a generated 3D model."""
    total_triangles: int
    total_vertices: int
    mesh_count: int
    wall_count: int
    total_wall_length_m: float
    usable_floor_area_sqm: float
    exterior_wall_area_sqm: float
    interior_wall_area_sqm: float
    door_count: int
    window_count: int
    furniture_count: int
    generation_time_ms: float
    validation_score: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MetricsCalculator:
    """Computes architectural metrics from 3D scene elements."""

    @staticmethod
    def calculate(
        scene: trimesh.Scene,
        walls: List[WallSegment],
        openings: List[Opening3D],
        floorplan: FloorPlan,
        generation_time_ms: float = 0.0,
        validation_score: float = 1.0,
        furniture_count: int = 0,
    ) -> Model3DMetrics:
        """Calculate comprehensive 3D metrics."""
        total_tris = 0
        total_verts = 0
        for geom in scene.geometry.values():
            if isinstance(geom, trimesh.Trimesh):
                total_tris += len(geom.faces)
                total_verts += len(geom.vertices)

        total_wall_len = sum(w.length for w in walls)
        ext_wall_area = sum(w.length * w.height for w in walls if w.wall_type == "exterior")
        int_wall_area = sum(w.length * w.height for w in walls if w.wall_type == "interior")

        door_count = sum(1 for o in openings if o.opening_type == "door")
        win_count = sum(1 for o in openings if o.opening_type == "window")

        usable_area = sum(
            r.area_sqm for r in (floorplan.confirmed_rooms + floorplan.candidate_rooms)
            if r.area_sqm is not None
        )

        return Model3DMetrics(
            total_triangles=total_tris,
            total_vertices=total_verts,
            mesh_count=len(scene.geometry),
            wall_count=len(walls),
            total_wall_length_m=round(total_wall_len, 2),
            usable_floor_area_sqm=round(usable_area, 2),
            exterior_wall_area_sqm=round(ext_wall_area, 2),
            interior_wall_area_sqm=round(int_wall_area, 2),
            door_count=door_count,
            window_count=win_count,
            furniture_count=furniture_count,
            generation_time_ms=round(generation_time_ms, 2),
            validation_score=round(validation_score, 2),
        )
