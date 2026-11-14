"""Headless Procedural 3D Model Generation Pipeline.

Orchestrates 2D floor-plan parsing, geometric validation, wall deduplication,
aperture creation, surface extrusion, furniture placement, scene assembly,
and performance metric measurement into a unified, reproducible result object.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import trimesh

from geometry3d.config import Geometry3DConfig
from geometry3d.coordinate_system import CoordinateTransformer
from geometry3d.exporter import ModelExporter
from geometry3d.floor_generator import FloorGenerator
from geometry3d.furniture_generator import FurnitureGenerator
from geometry3d.layout_validator import Layout3DValidator, ValidationReport
from geometry3d.material_manager import MaterialManager
from geometry3d.metrics import MetricsCalculator, Model3DMetrics
from geometry3d.opening_generator import Opening3D, OpeningGenerator
from geometry3d.scene_builder import SceneAssemblyMetadata, SceneBuilder
from geometry3d.wall_analyzer import WallAnalyzer, WallSegment
from geometry3d.wall_generator import WallGenerator
from models.floorplan_schema import FloorPlan


@dataclass
class Model3DResult:
    """Comprehensive output container for procedural 3D model synthesis."""
    scene: trimesh.Scene
    metrics: Model3DMetrics
    validation_report: ValidationReport
    walls: List[WallSegment]
    openings: List[Opening3D]
    transformer: CoordinateTransformer
    manifest: Dict[str, Any]
    _glb_bytes: Optional[bytes] = None

    def get_glb_bytes(self) -> bytes:
        """Fetch or lazily generate binary GLB bytes."""
        if self._glb_bytes is None:
            self._glb_bytes = ModelExporter.export_glb(self.scene)
        return self._glb_bytes


def generate_3d_model(
    floorplan: FloorPlan,
    config: Optional[Geometry3DConfig] = None,
    include_furniture: bool = True,
) -> Model3DResult:
    """Orchestrate complete procedural 3D generation from a 2D floor plan.

    Args:
        floorplan: Verified FloorPlan schema instance.
        config: Geometry3DConfig parameter overrides.
        include_furniture: Whether to populate procedural furniture.

    Returns:
        Model3DResult with assembled scene, metrics, validation, and export data.
    """
    t0 = time.perf_counter()

    base_config = config or Geometry3DConfig()
    if not include_furniture and base_config.generate_furniture:
        # Create an updated config copy
        from dataclasses import replace
        cfg = replace(base_config, generate_furniture=False)
    else:
        cfg = base_config

    materials = MaterialManager(palette_name=cfg.palette_name)
    transformer = CoordinateTransformer.from_floorplan(floorplan, cfg)

    # 1. Geometric Validation
    validator = Layout3DValidator(cfg)
    val_report = validator.validate_layout(floorplan, transformer)

    # 2. Wall Analysis and Deduplication
    wall_analyzer = WallAnalyzer(cfg)
    walls = wall_analyzer.analyze(floorplan, transformer)

    # 3. Opening Generation (Doors & Windows)
    opening_gen = OpeningGenerator(cfg)
    openings = opening_gen.generate_openings(walls, floorplan)

    # 4. Floor & Foundation Mesh Synthesis
    floor_gen = FloorGenerator(cfg, materials)
    room_floor_meshes, foundation_mesh = floor_gen.generate_floors(floorplan, transformer)

    # 5. Wall & Aperture Mesh Synthesis
    wall_gen = WallGenerator(cfg, materials)
    wall_meshes, accent_meshes = wall_gen.generate_walls(walls, openings)

    # 6. Procedural Furniture Synthesis
    furniture_meshes: List[trimesh.Trimesh] = []
    if cfg.generate_furniture:
        furn_gen = FurnitureGenerator(cfg, materials)
        furniture_meshes = furn_gen.generate_furniture(floorplan, transformer, openings)

    # 7. Scene Assembly
    builder = SceneBuilder(cfg)
    scene, scene_meta = builder.build_scene(
        floorplan=floorplan,
        transformer=transformer,
        walls=walls,
        openings=openings,
        floor_meshes=room_floor_meshes,
        wall_meshes=wall_meshes,
        accent_meshes=accent_meshes,
        furniture_meshes=furniture_meshes,
        foundation_mesh=foundation_mesh,
    )

    t1 = time.perf_counter()
    duration_ms = (t1 - t0) * 1000.0

    # 8. Diagnostics & Metrics
    metrics = MetricsCalculator.calculate(
        scene=scene,
        walls=walls,
        openings=openings,
        floorplan=floorplan,
        generation_time_ms=duration_ms,
        validation_score=val_report.overall_score,
        furniture_count=len(furniture_meshes),
    )

    # 9. JSON Manifest
    manifest = ModelExporter.export_manifest(scene, floorplan, cfg)

    return Model3DResult(
        scene=scene,
        metrics=metrics,
        validation_report=val_report,
        walls=walls,
        openings=openings,
        transformer=transformer,
        manifest=manifest,
    )
