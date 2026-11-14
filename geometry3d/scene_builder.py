"""Procedural 3D Scene Assembly Engine.

Consolidates architectural structural meshes (foundation, floors, walls, openings,
and furniture) into a hierarchical, semantically tagged trimesh.Scene graph.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import trimesh

from geometry3d.config import Geometry3DConfig
from geometry3d.coordinate_system import CoordinateMetadata, CoordinateTransformer
from geometry3d.opening_generator import Opening3D
from geometry3d.wall_analyzer import WallSegment
from models.floorplan_schema import FloorPlan


@dataclass
class SceneAssemblyMetadata:
    """Metadata describing the synthesized 3D architectural model."""
    total_triangles: int
    total_vertices: int
    room_count: int
    wall_count: int
    opening_count: int
    furniture_count: int
    building_width_m: float
    building_depth_m: float
    wall_height_m: float
    scale_meters_per_pixel: float
    is_calibrated: bool


class SceneBuilder:
    """Assembles discrete architectural meshes into a unified trimesh.Scene."""

    def __init__(self, config: Optional[Geometry3DConfig] = None) -> None:
        self.config = config or Geometry3DConfig()

    def build_scene(
        self,
        floorplan: FloorPlan,
        transformer: CoordinateTransformer,
        walls: List[WallSegment],
        openings: List[Opening3D],
        floor_meshes: List[trimesh.Trimesh],
        wall_meshes: List[trimesh.Trimesh],
        accent_meshes: List[trimesh.Trimesh],
        furniture_meshes: List[trimesh.Trimesh],
        foundation_mesh: Optional[trimesh.Trimesh] = None,
    ) -> Tuple[trimesh.Scene, SceneAssemblyMetadata]:
        """Construct the complete 3D scene graph with semantic nodes and metadata.

        Returns:
            Tuple of (trimesh.Scene, SceneAssemblyMetadata).
        """
        scene = trimesh.Scene()

        # 1. Foundation Slab
        if foundation_mesh is not None and len(foundation_mesh.vertices) > 0:
            foundation_mesh.metadata["semantic_node"] = "foundation"
            scene.add_geometry(foundation_mesh, node_name="foundation_slab")

        # 2. Room Floors
        for idx, floor_m in enumerate(floor_meshes):
            if len(floor_m.vertices) > 0:
                name = f"floor_room_{idx}_{floor_m.metadata.get('room_id', 'unknown')}"
                floor_m.metadata["semantic_node"] = "room_floor"
                scene.add_geometry(floor_m, node_name=name)

        # 3. Walls
        for idx, wall_m in enumerate(wall_meshes):
            if len(wall_m.vertices) > 0:
                name = f"wall_segment_{idx}"
                wall_m.metadata["semantic_node"] = "wall"
                scene.add_geometry(wall_m, node_name=name)

        # 4. Opening Accents (Door frames, Window glass)
        for idx, acc_m in enumerate(accent_meshes):
            if len(acc_m.vertices) > 0:
                name = f"accent_{idx}"
                acc_m.metadata["semantic_node"] = "opening_accent"
                scene.add_geometry(acc_m, node_name=name)

        # 5. Furniture Items
        for idx, furn_m in enumerate(furniture_meshes):
            if len(furn_m.vertices) > 0:
                name = f"furniture_{idx}"
                furn_m.metadata["semantic_node"] = "furniture"
                scene.add_geometry(furn_m, node_name=name)

        # Compute scene assembly stats
        total_tris = 0
        total_verts = 0
        for geom in scene.geometry.values():
            if isinstance(geom, trimesh.Trimesh):
                total_tris += len(geom.faces)
                total_verts += len(geom.vertices)

        meta = SceneAssemblyMetadata(
            total_triangles=total_tris,
            total_vertices=total_verts,
            room_count=len(floorplan.confirmed_rooms + floorplan.candidate_rooms),
            wall_count=len(walls),
            opening_count=len(openings),
            furniture_count=len(furniture_meshes),
            building_width_m=transformer.building_width_m,
            building_depth_m=transformer.building_depth_m,
            wall_height_m=self.config.default_wall_height,
            scale_meters_per_pixel=round(transformer.scale_m_per_px, 4),
            is_calibrated=transformer.is_calibrated,
        )

        scene.metadata["spaze_metadata"] = {
            "floorplan_id": floorplan.image_path,
            "total_triangles": total_tris,
            "total_vertices": total_verts,
            "room_count": meta.room_count,
            "wall_count": meta.wall_count,
            "opening_count": meta.opening_count,
            "furniture_count": meta.furniture_count,
            "building_width_m": meta.building_width_m,
            "building_depth_m": meta.building_depth_m,
        }

        return scene, meta
