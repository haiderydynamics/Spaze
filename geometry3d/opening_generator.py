"""Procedural Door and Window Opening Generation Engine.

Synthesizes architectural apertures (doors between adjacent rooms and
windows along exterior facades) based on spatial classification and clearance rules.
"""
from __future__ import annotations

import math
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from geometry3d.config import Geometry3DConfig
from geometry3d.wall_analyzer import WallSegment
from models.floorplan_schema import FloorPlan, Room


@dataclass
class Opening3D:
    """Represents a physical architectural aperture embedded within a wall."""
    id: str
    opening_type: str              # 'door' or 'window'
    wall_id: str                   # Identifier of host WallSegment
    width: float                   # Opening width in meters
    height: float                  # Opening height in meters
    sill_height: float             # Elevation above floor level in meters (0 for doors)
    wall_start_offset: float       # Linear offset along wall from wall.start (meters)
    center_3d: Tuple[float, float, float]  # (x, y, z) midpoint of opening in 3D meters

    @property
    def top_height(self) -> float:
        """Elevation of the opening's top lintel."""
        return self.sill_height + self.height


class OpeningGenerator:
    """Calculates valid door and window placements along wall segments."""

    def __init__(self, config: Optional[Geometry3DConfig] = None) -> None:
        self.config = config or Geometry3DConfig()

    def generate_openings(
        self,
        walls: List[WallSegment],
        floorplan: Optional[FloorPlan] = None,
    ) -> List[Opening3D]:
        """Generate doors and windows with strict clearance and collision avoidance.

        Args:
            walls: List of classified WallSegment objects.
            floorplan: Optional FloorPlan to inspect room semantic types.

        Returns:
            List of Opening3D instances.
        """
        openings: List[Opening3D] = []
        room_types: Dict[str, str] = {}

        if floorplan:
            for r in floorplan.confirmed_rooms + floorplan.candidate_rooms:
                room_types[r.id] = (r.room_type or "").lower().replace(" ", "_")

        placed_doors_between_pairs: set = set()
        exterior_entrance_placed = False

        for wall in walls:
            if wall.length < self.config.min_wall_segment_length_m:
                continue

            # 1. Interior Walls -> Doors
            if wall.wall_type == "interior" and self.config.generate_doors:
                # Check if we already connected this pair of rooms
                adj_pair = tuple(sorted(wall.adjacent_room_ids)) if len(wall.adjacent_room_ids) == 2 else None
                if adj_pair and adj_pair in placed_doors_between_pairs:
                    continue

                min_door_wall_len = self.config.door_width + 2 * self.config.door_wall_clearance
                if wall.length >= min_door_wall_len:
                    # Place door at center of wall segment
                    offset = (wall.length - self.config.door_width) / 2.0
                    center_u = offset + self.config.door_width / 2.0
                    cx, cy = self._get_point_along_wall(wall, center_u)
                    cz = self.config.door_height / 2.0

                    door = Opening3D(
                        id=f"door_{uuid.uuid4().hex[:6]}",
                        opening_type="door",
                        wall_id=wall.id,
                        width=self.config.door_width,
                        height=self.config.door_height,
                        sill_height=0.0,
                        wall_start_offset=round(offset, 3),
                        center_3d=(round(cx, 3), round(cy, 3), round(cz, 3)),
                    )
                    openings.append(door)
                    if adj_pair:
                        placed_doors_between_pairs.add(adj_pair)

            # 2. Exterior Walls -> Windows and optional Main Entrance Door
            elif wall.wall_type == "exterior":
                # Check if this exterior wall belongs to living room or corridor for main entrance
                room_id = wall.adjacent_room_ids[0] if wall.adjacent_room_ids else ""
                rtype = room_types.get(room_id, "")

                # Main entrance placement
                if not exterior_entrance_placed and self.config.generate_doors and rtype in ("living_room", "corridor"):
                    min_door_wall_len = self.config.door_width + 2 * self.config.door_wall_clearance
                    if wall.length >= min_door_wall_len:
                        offset = 0.5 * (wall.length - self.config.door_width)
                        center_u = offset + self.config.door_width / 2.0
                        cx, cy = self._get_point_along_wall(wall, center_u)
                        cz = self.config.door_height / 2.0

                        entrance_door = Opening3D(
                            id=f"door_entrance_{uuid.uuid4().hex[:6]}",
                            opening_type="door",
                            wall_id=wall.id,
                            width=self.config.door_width,
                            height=self.config.door_height,
                            sill_height=0.0,
                            wall_start_offset=round(offset, 3),
                            center_3d=(round(cx, 3), round(cy, 3), round(cz, 3)),
                        )
                        openings.append(entrance_door)
                        exterior_entrance_placed = True
                        continue

                # Windows on exterior walls
                if self.config.generate_windows:
                    min_win_wall_len = self.config.window_width + 2 * self.config.window_wall_clearance
                    if wall.length >= min_win_wall_len:
                        # Decide 1 or 2 windows based on length
                        if wall.length >= 4.5:
                            # Two windows
                            offsets = [
                                wall.length * 0.25 - self.config.window_width / 2.0,
                                wall.length * 0.75 - self.config.window_width / 2.0,
                            ]
                        else:
                            # Single centered window
                            offsets = [(wall.length - self.config.window_width) / 2.0]

                        for off in offsets:
                            if off < self.config.window_wall_clearance or (off + self.config.window_width) > (wall.length - self.config.window_wall_clearance):
                                continue
                            center_u = off + self.config.window_width / 2.0
                            cx, cy = self._get_point_along_wall(wall, center_u)
                            cz = self.config.window_sill_height + self.config.window_height / 2.0

                            win = Opening3D(
                                id=f"win_{uuid.uuid4().hex[:6]}",
                                opening_type="window",
                                wall_id=wall.id,
                                width=self.config.window_width,
                                height=self.config.window_height,
                                sill_height=self.config.window_sill_height,
                                wall_start_offset=round(off, 3),
                                center_3d=(round(cx, 3), round(cy, 3), round(cz, 3)),
                            )
                            openings.append(win)

        return openings

    def _get_point_along_wall(self, wall: WallSegment, distance_m: float) -> Tuple[float, float]:
        """Interpolate (x, y) point at a specified distance along the wall direction."""
        if wall.length < 1e-5:
            return wall.start
        t = distance_m / wall.length
        x = wall.start[0] + t * (wall.end[0] - wall.start[0])
        y = wall.start[1] + t * (wall.end[1] - wall.start[1])
        return (x, y)
