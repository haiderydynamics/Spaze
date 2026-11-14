"""Procedural 3D Wall Mesh Generation Engine with Geometric Apertures.

Constructs 3D volumetric wall meshes for exterior and interior walls.
Features real physical door and window openings by decomposing walls into
pillars, sills, and lintels, guaranteeing watertight manifold geometries.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import trimesh

from geometry3d.config import Geometry3DConfig
from geometry3d.material_manager import MaterialManager
from geometry3d.opening_generator import Opening3D
from geometry3d.wall_analyzer import WallSegment


class WallGenerator:
    """Generates 3D meshes for architectural walls with precise door and window cutouts."""

    def __init__(
        self,
        config: Optional[Geometry3DConfig] = None,
        material_manager: Optional[MaterialManager] = None,
    ) -> None:
        self.config = config or Geometry3DConfig()
        self.materials = material_manager or MaterialManager(self.config.palette_name)

    def generate_walls(
        self,
        walls: List[WallSegment],
        openings: Optional[List[Opening3D]] = None,
    ) -> Tuple[List[trimesh.Trimesh], List[trimesh.Trimesh]]:
        """Synthesize 3D wall meshes and opening accent meshes (frames/glass).

        Args:
            walls: List of WallSegment instances.
            openings: Optional list of Opening3D instances.

        Returns:
            Tuple of:
            - wall_meshes: List of Trimesh objects representing solid wall structures.
            - accent_meshes: List of Trimesh objects for door frames and window glass panes.
        """
        wall_meshes: List[trimesh.Trimesh] = []
        accent_meshes: List[trimesh.Trimesh] = []

        openings_by_wall: Dict[str, List[Opening3D]] = {}
        if openings:
            for op in openings:
                openings_by_wall.setdefault(op.wall_id, []).append(op)

        for wall in walls:
            if wall.length < 1e-4:
                continue

            wall_openings = openings_by_wall.get(wall.id, [])
            # Sort openings along wall by start offset
            wall_openings.sort(key=lambda o: o.wall_start_offset)

            mat_key = "wall_exterior" if wall.wall_type == "exterior" else "wall_interior"
            wall_color = self.materials.get_rgba_uint8(mat_key)
            lintel_color = self.materials.get_rgba_uint8("wall_lintel")

            if not wall_openings:
                # Solid wall without openings
                box_mesh = self._create_wall_prism(
                    wall=wall,
                    u_start=0.0,
                    u_end=wall.length,
                    z_bottom=0.0,
                    z_top=wall.height,
                    color=wall_color,
                )
                if box_mesh is not None:
                    box_mesh.metadata["wall_id"] = wall.id
                    box_mesh.metadata["wall_type"] = wall.wall_type
                    box_mesh.metadata["semantic_category"] = f"{wall.wall_type}_wall"
                    wall_meshes.append(box_mesh)
            else:
                # Segmented wall around openings
                curr_u = 0.0
                for op in wall_openings:
                    op_start = max(0.0, op.wall_start_offset)
                    op_end = min(wall.length, op.wall_start_offset + op.width)

                    # 1. Solid pillar before this opening
                    if op_start > curr_u + 1e-4:
                        pillar = self._create_wall_prism(
                            wall=wall,
                            u_start=curr_u,
                            u_end=op_start,
                            z_bottom=0.0,
                            z_top=wall.height,
                            color=wall_color,
                        )
                        if pillar is not None:
                            pillar.metadata["wall_id"] = wall.id
                            pillar.metadata["wall_type"] = wall.wall_type
                            pillar.metadata["semantic_category"] = f"{wall.wall_type}_wall"
                            wall_meshes.append(pillar)

                    # 2. Aperture volume
                    if op.opening_type == "door":
                        # Door opening: void below lintel (z from 0 to op.height)
                        # Lintel above door
                        if wall.height > op.height + 1e-4:
                            lintel = self._create_wall_prism(
                                wall=wall,
                                u_start=op_start,
                                u_end=op_end,
                                z_bottom=op.height,
                                z_top=wall.height,
                                color=lintel_color,
                            )
                            if lintel is not None:
                                lintel.metadata["wall_id"] = wall.id
                                lintel.metadata["wall_type"] = wall.wall_type
                                lintel.metadata["semantic_category"] = f"{wall.wall_type}_wall"
                                wall_meshes.append(lintel)

                        # Decorative door frame trim
                        frame = self._create_door_frame(wall, op_start, op_end, op.height)
                        if frame is not None:
                            frame.metadata["semantic_category"] = "door"
                            accent_meshes.append(frame)

                    elif op.opening_type == "window":
                        # Window bottom sill
                        if op.sill_height > 1e-4:
                            sill = self._create_wall_prism(
                                wall=wall,
                                u_start=op_start,
                                u_end=op_end,
                                z_bottom=0.0,
                                z_top=op.sill_height,
                                color=wall_color,
                            )
                            if sill is not None:
                                sill.metadata["wall_id"] = wall.id
                                sill.metadata["wall_type"] = wall.wall_type
                                sill.metadata["semantic_category"] = f"{wall.wall_type}_wall"
                                wall_meshes.append(sill)

                        # Window top lintel
                        top_z = op.sill_height + op.height
                        if wall.height > top_z + 1e-4:
                            lintel = self._create_wall_prism(
                                wall=wall,
                                u_start=op_start,
                                u_end=op_end,
                                z_bottom=top_z,
                                z_top=wall.height,
                                color=lintel_color,
                            )
                            if lintel is not None:
                                lintel.metadata["wall_id"] = wall.id
                                lintel.metadata["wall_type"] = wall.wall_type
                                lintel.metadata["semantic_category"] = f"{wall.wall_type}_wall"
                                wall_meshes.append(lintel)

                        # Window glass pane
                        glass = self._create_window_glass(wall, op_start, op_end, op.sill_height, op.height)
                        if glass is not None:
                            glass.metadata["semantic_category"] = "window"
                            accent_meshes.append(glass)

                    curr_u = op_end

                # Trailing solid pillar after final opening
                if curr_u < wall.length - 1e-4:
                    pillar = self._create_wall_prism(
                        wall=wall,
                        u_start=curr_u,
                        u_end=wall.length,
                        z_bottom=0.0,
                        z_top=wall.height,
                        color=wall_color,
                    )
                    if pillar is not None:
                        pillar.metadata["wall_id"] = wall.id
                        pillar.metadata["wall_type"] = wall.wall_type
                        pillar.metadata["semantic_category"] = f"{wall.wall_type}_wall"
                        wall_meshes.append(pillar)

        return wall_meshes, accent_meshes

    def _create_wall_prism(
        self,
        wall: WallSegment,
        u_start: float,
        u_end: float,
        z_bottom: float,
        z_top: float,
        color: Tuple[int, int, int, int],
    ) -> Optional[trimesh.Trimesh]:
        """Create an oriented rectangular 3D box mesh aligned with the wall direction."""
        length = u_end - u_start
        height = z_top - z_bottom
        thickness = wall.thickness

        if length <= 1e-4 or height <= 1e-4 or thickness <= 1e-4:
            return None

        # Unit vectors
        ux = (wall.end[0] - wall.start[0]) / wall.length
        uy = (wall.end[1] - wall.start[1]) / wall.length
        # Normal vector (perpendicular to wall line)
        nx = -uy
        ny = ux

        # Center point in XY
        u_mid = (u_start + u_end) / 2.0
        cx = wall.start[0] + u_mid * ux
        cy = wall.start[1] + u_mid * uy
        cz = (z_bottom + z_top) / 2.0

        # Construct box via trimesh
        box = trimesh.creation.box(extents=[length, thickness, height])

        # Rotation matrix aligning box X-axis with (ux, uy, 0)
        angle = math.atan2(uy, ux)
        rot_matrix = trimesh.transformations.rotation_matrix(angle, [0, 0, 1])
        trans_matrix = trimesh.transformations.translation_matrix([cx, cy, cz])
        transform = trimesh.transformations.concatenate_matrices(trans_matrix, rot_matrix)

        box.apply_transform(transform)
        box.visual.vertex_colors = color
        return box

    def _create_door_frame(
        self,
        wall: WallSegment,
        u_start: float,
        u_end: float,
        door_height: float,
    ) -> Optional[trimesh.Trimesh]:
        """Generate a minimal trim frame around a door opening."""
        frame_thick = wall.thickness + 0.02
        frame_color = self.materials.get_rgba_uint8("door_frame")
        ux = (wall.end[0] - wall.start[0]) / wall.length
        uy = (wall.end[1] - wall.start[1]) / wall.length
        angle = math.atan2(uy, ux)

        # Lintel header trim
        u_mid = (u_start + u_end) / 2.0
        cx = wall.start[0] + u_mid * ux
        cy = wall.start[1] + u_mid * uy
        cz = door_height

        header = trimesh.creation.box(extents=[u_end - u_start + 0.06, frame_thick, 0.05])
        rot = trimesh.transformations.rotation_matrix(angle, [0, 0, 1])
        trans = trimesh.transformations.translation_matrix([cx, cy, cz])
        header.apply_transform(trimesh.transformations.concatenate_matrices(trans, rot))
        header.visual.vertex_colors = frame_color
        return header

    def _create_window_glass(
        self,
        wall: WallSegment,
        u_start: float,
        u_end: float,
        sill_height: float,
        window_height: float,
    ) -> Optional[trimesh.Trimesh]:
        """Generate a translucent glass pane inside the window cutout."""
        length = u_end - u_start
        glass_thick = 0.02
        glass_color = self.materials.get_rgba_uint8("glass")

        ux = (wall.end[0] - wall.start[0]) / wall.length
        uy = (wall.end[1] - wall.start[1]) / wall.length
        angle = math.atan2(uy, ux)

        u_mid = (u_start + u_end) / 2.0
        cx = wall.start[0] + u_mid * ux
        cy = wall.start[1] + u_mid * uy
        cz = sill_height + window_height / 2.0

        pane = trimesh.creation.box(extents=[length - 0.02, glass_thick, window_height - 0.02])
        rot = trimesh.transformations.rotation_matrix(angle, [0, 0, 1])
        trans = trimesh.transformations.translation_matrix([cx, cy, cz])
        pane.apply_transform(trimesh.transformations.concatenate_matrices(trans, rot))
        pane.visual.vertex_colors = glass_color
        pane.metadata["semantic_category"] = "glass"
        return pane
