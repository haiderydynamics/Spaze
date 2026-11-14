"""Procedural Architectural Furniture Generation Engine.

Synthesizes stylized, lightweight 3D furniture models tailored to room types
(beds, nightstands, sofas, coffee tables, dining sets, kitchen counters, sanitaryware)
with strict collision checking against room perimeters and door swings.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import trimesh
from shapely.geometry import Point, Polygon, box

from geometry3d.config import Geometry3DConfig
from geometry3d.coordinate_system import CoordinateTransformer
from geometry3d.material_manager import MaterialManager
from geometry3d.opening_generator import Opening3D
from models.floorplan_schema import FloorPlan, Room


class FurnitureGenerator:
    """Procedurally synthesizes and places 3D furniture assemblies."""

    def __init__(
        self,
        config: Optional[Geometry3DConfig] = None,
        material_manager: Optional[MaterialManager] = None,
    ) -> None:
        self.config = config or Geometry3DConfig()
        self.materials = material_manager or MaterialManager(self.config.palette_name)

    def generate_furniture(
        self,
        floorplan: FloorPlan,
        transformer: Optional[CoordinateTransformer] = None,
        openings: Optional[List[Opening3D]] = None,
    ) -> List[trimesh.Trimesh]:
        """Generate placed furniture meshes across all rooms.

        Args:
            floorplan: FloorPlan instance.
            transformer: CoordinateTransformer.
            openings: Optional list of doors/windows to avoid blocking.

        Returns:
            List of Trimesh instances representing furniture items.
        """
        if not self.config.generate_furniture:
            return []

        tf = transformer or CoordinateTransformer.from_floorplan(floorplan, self.config)
        rooms = [r for r in (floorplan.confirmed_rooms + floorplan.candidate_rooms) if len(r.polygon) >= 3]

        door_points_2d = [
            Point(op.center_3d[0], op.center_3d[1])
            for op in (openings or [])
            if op.opening_type == "door"
        ]

        furniture_meshes: List[trimesh.Trimesh] = []

        for room in rooms:
            planar_pts = tf.to_3d_planar_polygon(room.polygon)
            poly = Polygon(planar_pts)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if not poly.is_valid or poly.area < 1.5:
                continue

            rtype = (room.room_type or "").lower().replace(" ", "_")
            items = self._populate_room(room, poly, rtype, door_points_2d)
            furniture_meshes.extend(items)

        return furniture_meshes

    def _populate_room(
        self,
        room: Room,
        poly: Polygon,
        room_type: str,
        door_points: List[Point],
    ) -> List[trimesh.Trimesh]:
        """Place appropriate furniture pieces for a room type within its polygon."""
        minx, miny, maxx, maxy = poly.bounds
        width = maxx - minx
        depth = maxy - miny
        center_x = (minx + maxx) / 2.0
        center_y = (miny + maxy) / 2.0

        items: List[trimesh.Trimesh] = []

        if room_type == "bedroom":
            # Bed + Nightstand
            bed_w, bed_d = 1.6, 2.0
            if width >= 2.2 and depth >= 2.6:
                # Place bed against top or bottom wall
                bx = center_x
                by = miny + bed_d / 2.0 + 0.15
                if self._can_place_box(bx, by, bed_w, bed_d, poly, door_points):
                    items.extend(self._create_bed(bx, by, bed_w, bed_d))
                    # Nightstand next to bed
                    nx = bx + bed_w / 2.0 + 0.35
                    ny = by - bed_d / 2.0 + 0.3
                    if self._can_place_box(nx, ny, 0.45, 0.45, poly, door_points):
                        items.append(self._create_nightstand(nx, ny))

        elif room_type == "living_room":
            # Sofa + Coffee Table + Media Console
            sofa_w, sofa_d = 2.1, 0.85
            if width >= 2.8 and depth >= 2.8:
                sx = center_x
                sy = miny + sofa_d / 2.0 + 0.2
                if self._can_place_box(sx, sy, sofa_w, sofa_d, poly, door_points):
                    items.extend(self._create_sofa(sx, sy, sofa_w, sofa_d))

                    # Coffee table in front of sofa
                    tx = sx
                    ty = sy + sofa_d / 2.0 + 0.55
                    if self._can_place_box(tx, ty, 1.1, 0.6, poly, door_points):
                        items.append(self._create_coffee_table(tx, ty))

                    # Media console against opposite wall
                    cx = sx
                    cy = maxy - 0.25
                    if self._can_place_box(cx, cy, 1.6, 0.35, poly, door_points):
                        items.append(self._create_media_console(cx, cy))

        elif room_type == "dining_room":
            # Dining table + chairs
            tbl_w, tbl_d = 1.5, 0.9
            if width >= 2.2 and depth >= 2.2:
                if self._can_place_box(center_x, center_y, tbl_w, tbl_d, poly, door_points):
                    items.extend(self._create_dining_set(center_x, center_y, tbl_w, tbl_d))

        elif room_type == "kitchen":
            # Kitchen counter run
            cw, cd = min(width - 0.4, 2.0), 0.6
            if cw >= 1.2:
                kx = minx + cw / 2.0 + 0.2
                ky = maxy - cd / 2.0 - 0.1
                if self._can_place_box(kx, ky, cw, cd, poly, door_points):
                    items.append(self._create_kitchen_counter(kx, ky, cw, cd))

        elif room_type == "bathroom":
            # Vanity + Toilet
            if width >= 1.4 and depth >= 1.4:
                # Vanity
                vx = minx + 0.5
                vy = maxy - 0.35
                if self._can_place_box(vx, vy, 0.8, 0.5, poly, door_points):
                    items.append(self._create_bathroom_vanity(vx, vy))

                # Toilet
                tx = maxx - 0.4
                ty = miny + 0.45
                if self._can_place_box(tx, ty, 0.45, 0.65, poly, door_points):
                    items.extend(self._create_toilet(tx, ty))

        return items

    def _can_place_box(
        self,
        cx: float,
        cy: float,
        width: float,
        depth: float,
        room_poly: Polygon,
        door_points: List[Point],
        min_door_clearance: float = 0.85,
    ) -> bool:
        """Check if bounding box fits inside room polygon and is clear of door swings."""
        box_2d = box(cx - width / 2.0, cy - depth / 2.0, cx + width / 2.0, cy + depth / 2.0)
        # Check containment with 5cm margin
        if not room_poly.buffer(0.02).contains(box_2d):
            return False

        # Check door clearance
        for dp in door_points:
            if box_2d.distance(dp) < min_door_clearance:
                return False

        return True

    # ---- Furniture Component Builders ----

    def _create_bed(self, cx: float, cy: float, w: float, d: float) -> List[trimesh.Trimesh]:
        """Procedural bed assembly (frame, mattress, headboard, pillows)."""
        wood_color = self.materials.get_rgba_uint8("furniture_wood")
        sheet_color = self.materials.get_rgba_uint8("bed_sheet")
        pillow_color = self.materials.get_rgba_uint8("bed_pillow")

        # 1. Base frame
        frame = trimesh.creation.box(extents=[w, d, 0.25])
        frame.apply_translation([cx, cy, 0.125])
        frame.visual.vertex_colors = wood_color

        # 2. Mattress
        mattress = trimesh.creation.box(extents=[w - 0.08, d - 0.08, 0.22])
        mattress.apply_translation([cx, cy, 0.25 + 0.11])
        mattress.visual.vertex_colors = sheet_color

        # 3. Headboard (against rear of bed: -Y)
        headboard = trimesh.creation.box(extents=[w + 0.08, 0.10, 0.85])
        headboard.apply_translation([cx, cy - d / 2.0 + 0.05, 0.425])
        headboard.visual.vertex_colors = wood_color

        # 4. Two pillows
        p1 = trimesh.creation.box(extents=[0.55, 0.38, 0.10])
        p1.apply_translation([cx - w / 4.0, cy - d / 2.0 + 0.32, 0.52])
        p1.visual.vertex_colors = pillow_color

        p2 = trimesh.creation.box(extents=[0.55, 0.38, 0.10])
        p2.apply_translation([cx + w / 4.0, cy - d / 2.0 + 0.32, 0.52])
        p2.visual.vertex_colors = pillow_color

        for m in (frame, mattress, headboard, p1, p2):
            m.metadata["semantic_category"] = "furniture"

        return [frame, mattress, headboard, p1, p2]

    def _create_nightstand(self, cx: float, cy: float) -> trimesh.Trimesh:
        """Procedural nightstand bedside table."""
        stand = trimesh.creation.box(extents=[0.45, 0.45, 0.48])
        stand.apply_translation([cx, cy, 0.24])
        stand.visual.vertex_colors = self.materials.get_rgba_uint8("furniture_wood")
        stand.metadata["semantic_category"] = "furniture"
        return stand

    def _create_sofa(self, cx: float, cy: float, w: float, d: float) -> List[trimesh.Trimesh]:
        """Procedural comfortable sofa assembly (seat cushion, backrest, armrests)."""
        fabric_color = self.materials.get_rgba_uint8("furniture_fabric_blue")

        # Seat base
        seat = trimesh.creation.box(extents=[w, d, 0.40])
        seat.apply_translation([cx, cy, 0.20])
        seat.visual.vertex_colors = fabric_color

        # Backrest (along -Y side)
        back = trimesh.creation.box(extents=[w, 0.22, 0.42])
        back.apply_translation([cx, cy - d / 2.0 + 0.11, 0.40 + 0.21])
        back.visual.vertex_colors = fabric_color

        # Left armrest
        arm_l = trimesh.creation.box(extents=[0.20, d, 0.25])
        arm_l.apply_translation([cx - w / 2.0 + 0.10, cy, 0.40 + 0.125])
        arm_l.visual.vertex_colors = fabric_color

        # Right armrest
        arm_r = trimesh.creation.box(extents=[0.20, d, 0.25])
        arm_r.apply_translation([cx + w / 2.0 - 0.10, cy, 0.40 + 0.125])
        arm_r.visual.vertex_colors = fabric_color

        for m in (seat, back, arm_l, arm_r):
            m.metadata["semantic_category"] = "furniture"

        return [seat, back, arm_l, arm_r]

    def _create_coffee_table(self, cx: float, cy: float) -> trimesh.Trimesh:
        """Procedural low coffee table."""
        tbl = trimesh.creation.box(extents=[1.10, 0.60, 0.38])
        tbl.apply_translation([cx, cy, 0.19])
        tbl.visual.vertex_colors = self.materials.get_rgba_uint8("furniture_wood")
        tbl.metadata["semantic_category"] = "furniture"
        return tbl

    def _create_media_console(self, cx: float, cy: float) -> trimesh.Trimesh:
        """Procedural TV entertainment console."""
        console = trimesh.creation.box(extents=[1.60, 0.35, 0.45])
        console.apply_translation([cx, cy, 0.225])
        console.visual.vertex_colors = self.materials.get_rgba_uint8("furniture_wood")
        console.metadata["semantic_category"] = "furniture"
        return console

    def _create_dining_set(self, cx: float, cy: float, w: float, d: float) -> List[trimesh.Trimesh]:
        """Dining table and four surrounding chairs."""
        wood_color = self.materials.get_rgba_uint8("furniture_wood")
        chair_color = self.materials.get_rgba_uint8("furniture_fabric_gray")

        # Table top
        top = trimesh.creation.box(extents=[w, d, 0.05])
        top.apply_translation([cx, cy, 0.73])
        top.visual.vertex_colors = wood_color

        # Table pedestal / base
        base = trimesh.creation.box(extents=[0.3, 0.3, 0.70])
        base.apply_translation([cx, cy, 0.35])
        base.visual.vertex_colors = wood_color

        # 4 chairs
        chairs: List[trimesh.Trimesh] = []
        chair_offsets = [
            (cx - w / 2.0 - 0.35, cy),  # Left
            (cx + w / 2.0 + 0.35, cy),  # Right
            (cx, cy - d / 2.0 - 0.35),  # Bottom
            (cx, cy + d / 2.0 + 0.35),  # Top
        ]
        for chx, chy in chair_offsets:
            ch_seat = trimesh.creation.box(extents=[0.42, 0.42, 0.44])
            ch_seat.apply_translation([chx, chy, 0.22])
            ch_seat.visual.vertex_colors = chair_color
            chairs.append(ch_seat)

        all_items = [top, base] + chairs
        for m in all_items:
            m.metadata["semantic_category"] = "furniture"
        return all_items

    def _create_kitchen_counter(self, cx: float, cy: float, w: float, d: float) -> trimesh.Trimesh:
        """Kitchen cabinetry counter with cooktop accent."""
        counter = trimesh.creation.box(extents=[w, d, 0.90])
        counter.apply_translation([cx, cy, 0.45])
        counter.visual.vertex_colors = self.materials.get_rgba_uint8("furniture_wood")
        counter.metadata["semantic_category"] = "furniture"
        return counter

    def _create_bathroom_vanity(self, cx: float, cy: float) -> trimesh.Trimesh:
        """Bathroom vanity counter with sink basin."""
        vanity = trimesh.creation.box(extents=[0.80, 0.50, 0.85])
        vanity.apply_translation([cx, cy, 0.425])
        vanity.visual.vertex_colors = self.materials.get_rgba_uint8("furniture_ceramic")
        vanity.metadata["semantic_category"] = "furniture"
        return vanity

    def _create_toilet(self, cx: float, cy: float) -> List[trimesh.Trimesh]:
        """Modern porcelain toilet unit (bowl and water tank)."""
        white_ceramic = self.materials.get_rgba_uint8("furniture_ceramic")

        # Bowl
        bowl = trimesh.creation.box(extents=[0.38, 0.52, 0.42])
        bowl.apply_translation([cx, cy, 0.21])
        bowl.visual.vertex_colors = white_ceramic

        # Cistern tank
        tank = trimesh.creation.box(extents=[0.38, 0.20, 0.36])
        tank.apply_translation([cx, cy - 0.20, 0.42 + 0.18])
        tank.visual.vertex_colors = white_ceramic

        for m in (bowl, tank):
            m.metadata["semantic_category"] = "furniture"

        return [bowl, tank]
