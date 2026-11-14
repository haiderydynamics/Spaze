"""Deterministic Procedural Layout Generation Baseline for Architectural Benchmarking.

Provides a rule-based heuristic packing baseline to benchmark against
the Conditional Variational Autoencoder (CVAE) under identical spatial constraints.
"""
from __future__ import annotations

import math
import uuid
from typing import Dict, List, Optional, Tuple

from models.floorplan_schema import FloorPlan, Room


class ProceduralBaselineGenerator:
    """Generates procedural architectural floor plans via heuristic layout rules."""

    ROOM_PRIORITIES = {
        "living_room": 1,
        "dining_room": 2,
        "kitchen": 3,
        "bedroom": 4,
        "bathroom": 5,
        "hallway": 6,
        "corridor": 6,
        "balcony": 7,
    }

    @classmethod
    def generate(
        cls,
        source_floorplan: FloorPlan,
        preference: str = "space_efficient",
    ) -> FloorPlan:
        """Synthesize a baseline layout from the source room inventory.

        Args:
            source_floorplan: Input FloorPlan supplying canvas and room inventory.
            preference: One of 'space_efficient', 'open_layout', 'privacy_focused'.

        Returns:
            FloorPlan with procedurally arranged rooms.
        """
        src_rooms = source_floorplan.confirmed_rooms + source_floorplan.candidate_rooms
        if not src_rooms:
            return FloorPlan(
                image_width=source_floorplan.image_width,
                image_height=source_floorplan.image_height,
                preference=preference,
                is_generated=True,
            )

        width = source_floorplan.image_width
        height = source_floorplan.image_height
        margin = 40.0
        usable_w = width - 2 * margin
        usable_h = height - 2 * margin

        # Sort rooms by priority
        sorted_rooms = sorted(
            src_rooms,
            key=lambda r: cls.ROOM_PRIORITIES.get(r.room_type.lower().replace(" ", "_"), 99),
        )

        n = len(sorted_rooms)
        placed_rooms: List[Room] = []

        if preference == "open_layout":
            # Large open living/dining zone on one half, private rooms clustered on the other
            split_x = margin + usable_w * 0.52
            
            # Left side: Public zones (Living, Dining, Kitchen)
            public_rooms = [r for r in sorted_rooms if r.room_type.lower() in ["living_room", "dining_room", "kitchen"]]
            private_rooms = [r for r in sorted_rooms if r not in public_rooms]

            # Place public rooms on left
            if public_rooms:
                h_step = usable_h / len(public_rooms)
                for idx, r in enumerate(public_rooms):
                    rx = margin
                    ry = margin + idx * h_step
                    rw = split_x - margin - 10.0
                    rh = h_step - 10.0
                    placed_rooms.append(cls._make_room(r, rx, ry, rw, rh))

            # Place private rooms on right
            if private_rooms:
                h_step = usable_h / len(private_rooms)
                for idx, r in enumerate(private_rooms):
                    rx = split_x
                    ry = margin + idx * h_step
                    rw = margin + usable_w - split_x
                    rh = h_step - 10.0
                    placed_rooms.append(cls._make_room(r, rx, ry, rw, rh))

        elif preference == "privacy_focused":
            # Living room central/front, bedrooms placed at opposite diagonal extremes
            cols = 2
            rows = math.ceil(n / cols)
            cw = usable_w / cols
            rh = usable_h / rows

            for idx, r in enumerate(sorted_rooms):
                c = idx % cols
                row_idx = idx // cols
                # Separate private rooms by buffering
                rx = margin + c * cw + (5.0 if c == 0 else 10.0)
                ry = margin + row_idx * rh + 5.0
                rw = cw - 15.0
                room_h = rh - 10.0
                placed_rooms.append(cls._make_room(r, rx, ry, rw, room_h))

        else:
            # Space-Efficient: Compact grid packing with minimal gaps
            cols = math.ceil(math.sqrt(n))
            rows = math.ceil(n / cols)
            cw = usable_w / cols
            ch = usable_h / rows

            for idx, r in enumerate(sorted_rooms):
                c = idx % cols
                row_idx = idx // cols
                rx = margin + c * cw
                ry = margin + row_idx * ch
                rw = cw - 6.0
                rh = ch - 6.0
                placed_rooms.append(cls._make_room(r, rx, ry, rw, rh))

        return FloorPlan(
            image_width=width,
            image_height=height,
            scale_information=source_floorplan.scale_information,
            confirmed_rooms=placed_rooms,
            preference=preference,
            is_generated=True,
        )

    @staticmethod
    def _make_room(src: Room, x: float, y: float, w: float, h: float) -> Room:
        """Helper to create a room with bounding box and polygon."""
        w = max(40.0, w)
        h = max(40.0, h)
        poly = [
            (round(x, 1), round(y, 1)),
            (round(x + w, 1), round(y, 1)),
            (round(x + w, 1), round(y + h, 1)),
            (round(x, 1), round(y + h, 1)),
        ]
        return Room(
            room_id=f"base_{src.room_id}_{uuid.uuid4().hex[:4]}",
            room_type=src.room_type,
            label=src.label or src.room_type,
            x=round(x, 1),
            y=round(y, 1),
            width=round(w, 1),
            height=round(h, 1),
            polygon=poly,
            confidence_score=1.0,
            is_user_confirmed=True,
        )
