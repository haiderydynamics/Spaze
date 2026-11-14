"""Procedural Architectural Floor and Foundation Slab Mesh Generation Engine.

Extrudes 2D room polygons into 3D floor slabs, attaches semantic room metadata,
and applies room-specific architectural surface materials.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import trimesh
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

from geometry3d.config import Geometry3DConfig
from geometry3d.coordinate_system import CoordinateTransformer
from geometry3d.material_manager import MaterialManager
from models.floorplan_schema import FloorPlan, Room


class FloorGenerator:
    """Generates 3D meshes for room floors and exterior foundation plinth."""

    def __init__(
        self,
        config: Optional[Geometry3DConfig] = None,
        material_manager: Optional[MaterialManager] = None,
    ) -> None:
        self.config = config or Geometry3DConfig()
        self.materials = material_manager or MaterialManager(self.config.palette_name)

    def generate_floors(
        self,
        floorplan: FloorPlan,
        transformer: Optional[CoordinateTransformer] = None,
    ) -> Tuple[List[trimesh.Trimesh], Optional[trimesh.Trimesh]]:
        """Synthesize room floor meshes and foundation slab.

        Args:
            floorplan: FloorPlan instance.
            transformer: CoordinateTransformer mapping pixels to world meters.

        Returns:
            Tuple of:
            - room_floor_meshes: List of Trimesh meshes for each room floor slab.
            - foundation_mesh: Optional Trimesh for exterior foundation plinth.
        """
        tf = transformer or CoordinateTransformer.from_floorplan(floorplan, self.config)
        rooms = [r for r in (floorplan.confirmed_rooms + floorplan.candidate_rooms) if len(r.polygon) >= 3]

        room_meshes: List[trimesh.Trimesh] = []
        shapely_polys: List[Polygon] = []

        for room in rooms:
            planar_pts = tf.to_3d_planar_polygon(room.polygon)
            poly = Polygon(planar_pts)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if not poly.is_valid or poly.area < 0.05:
                continue

            shapely_polys.append(poly)

            # Extrude room floor
            try:
                mesh = trimesh.creation.extrude_polygon(
                    poly,
                    height=self.config.floor_thickness,
                )
                # Translate downward so top walking surface is at z = 0.0
                mesh.apply_translation([0.0, 0.0, -self.config.floor_thickness])

                # Color matching room type
                mat = self.materials.get_floor_material_for_room(room.room_type)
                mesh.visual.vertex_colors = mat.color_rgba_uint8

                mesh.metadata["semantic_category"] = "floor"
                mesh.metadata["room_id"] = room.id
                mesh.metadata["room_name"] = room.label or room.room_type
                mesh.metadata["room_type"] = room.room_type
                mesh.metadata["area_sqm"] = round(poly.area, 2)

                room_meshes.append(mesh)
            except Exception:
                continue

        # Foundation Plinth
        foundation_mesh: Optional[trimesh.Trimesh] = None
        if shapely_polys:
            try:
                footprint = unary_union(shapely_polys)
                # Buffer slightly for plinth ledge (0.25m)
                plinth_poly = footprint.buffer(0.25)
                if isinstance(plinth_poly, MultiPolygon):
                    plinth_poly = plinth_poly.geoms[0]

                if plinth_poly.is_valid and plinth_poly.area > 0.1:
                    foundation_thickness = 0.15
                    plinth = trimesh.creation.extrude_polygon(
                        plinth_poly,
                        height=foundation_thickness,
                    )
                    # Translate below floor slab
                    plinth.apply_translation([0.0, 0.0, -(self.config.floor_thickness + foundation_thickness)])
                    plinth.visual.vertex_colors = self.materials.get_rgba_uint8("foundation")
                    plinth.metadata["semantic_category"] = "foundation"
                    foundation_mesh = plinth
            except Exception:
                foundation_mesh = None

        return room_meshes, foundation_mesh
