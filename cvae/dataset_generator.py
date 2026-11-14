"""Synthetic Architectural Floor-Plan Dataset Generator.

Generates topology-first residential floor-plan layouts using Binary Space Partitioning (BSP),
functional room area allocation, architectural connectivity heuristics, and preference labeling.
Produces source-target pairs sharing identical room programs for Conditional VAE training.
"""
from __future__ import annotations

import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from cvae.config import CVAEConfig
from cvae.constraints import GeometricConstraintsValidator
from models.floorplan_schema import DetectionStatus, FloorPlan, MeasurementSource, ProcessingMetadata, Room, ScaleInfo


# Predefined architectural residential room programs
RESIDENTIAL_PROGRAMS = [
    # 1. Compact 1-BHK (4 rooms)
    ["Living Room", "Kitchen", "Bedroom", "Bathroom"],
    # 2. Standard 2-BHK (5 rooms)
    ["Living Room", "Kitchen", "Dining Room", "Master Bedroom", "Bathroom"],
    # 3. Family 2-BHK with Hallway (6 rooms)
    ["Living Room", "Kitchen", "Dining Room", "Master Bedroom", "Bedroom", "Bathroom"],
    # 4. Full 2-BHK + Balcony (7 rooms)
    ["Living Room", "Kitchen", "Dining Room", "Master Bedroom", "Bedroom", "Bathroom", "Balcony"],
    # 5. Spacious 3-BHK (8 rooms)
    ["Living Room", "Kitchen", "Dining Room", "Master Bedroom", "Bedroom", "Bathroom", "Hallway", "Utility Room"],
    # 6. Premium 3-BHK + Study (9 rooms)
    ["Living Room", "Kitchen", "Dining Room", "Master Bedroom", "Bedroom", "Bathroom", "Study Room", "Hallway", "Balcony"],
]

# Relative room area budget weights
ROOM_AREA_WEIGHTS = {
    "Living Room": 3.2,
    "Master Bedroom": 2.5,
    "Bedroom": 2.0,
    "Kitchen": 1.6,
    "Dining Room": 1.8,
    "Hallway": 1.0,
    "Study Room": 1.5,
    "Bathroom": 1.0,
    "Balcony": 0.9,
    "Utility Room": 0.7,
}


@dataclass
class SyntheticPair:
    """A source-target floor-plan pair sharing identical room programs for CVAE training."""
    pair_id: str
    family_id: str
    preference: str
    preference_vector: List[float]
    source_floorplan: Dict[str, Any]
    target_floorplan: Dict[str, Any]
    room_program: List[str]


class BSPNode:
    """Binary Space Partitioning Node representing a rectangular partition of space."""

    def __init__(self, x: float, y: float, w: float, h: float) -> None:
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.left: Optional[BSPNode] = None
        self.right: Optional[BSPNode] = None
        self.assigned_room_type: Optional[str] = None

    def is_leaf(self) -> bool:
        return self.left is None and self.right is None

    def split(self, split_ratio: float, split_vertical: bool) -> bool:
        """Split node into two child rectangles."""
        if not self.is_leaf():
            return False

        if split_vertical:
            w1 = self.w * split_ratio
            w2 = self.w - w1
            if w1 < 30.0 or w2 < 30.0:
                return False
            self.left = BSPNode(self.x, self.y, w1, self.h)
            self.right = BSPNode(self.x + w1, self.y, w2, self.h)
        else:
            h1 = self.h * split_ratio
            h2 = self.h - h1
            if h1 < 30.0 or h2 < 30.0:
                return False
            self.left = BSPNode(self.x, self.y, self.w, h1)
            self.right = BSPNode(self.x, self.y + h1, self.w, h2)

        return True

    def get_leaves(self) -> List[BSPNode]:
        """Collect all leaf partitions."""
        if self.is_leaf():
            return [self]
        leaves = []
        if self.left:
            leaves.extend(self.left.get_leaves())
        if self.right:
            leaves.extend(self.right.get_leaves())
        return leaves


class SyntheticFloorPlanGenerator:
    """Generates procedural architectural floor plans and source-target CVAE training pairs."""

    def __init__(self, config: Optional[CVAEConfig] = None, seed: int = 42) -> None:
        self.config = config or CVAEConfig()
        self.validator = GeometricConstraintsValidator()
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

    def _partition_space(
        self,
        width: float,
        height: float,
        num_rooms: int,
        preference: str,
    ) -> List[BSPNode]:
        """Perform recursive rectangular BSP subdivision conditioned on architectural preference."""
        root = BSPNode(0.0, 0.0, width, height)
        nodes_to_split = [root]

        while len(nodes_to_split) < num_rooms:
            # Pick node with largest area
            nodes_to_split.sort(key=lambda n: n.w * n.h, reverse=True)
            node = nodes_to_split.pop(0)

            # Determine split direction
            aspect = node.w / max(1.0, node.h)
            if aspect > 1.3:
                split_vertical = True
            elif aspect < 0.77:
                split_vertical = False
            else:
                split_vertical = random.random() < 0.5

            # Controlled split ratios depending on preference
            if preference == "open_layout":
                # Wider main room partitions
                split_ratio = random.uniform(0.42, 0.58)
            elif preference == "space_efficient":
                # High-efficiency balanced splits
                split_ratio = random.uniform(0.35, 0.65)
            else:  # privacy_focused
                # Deeper partitions to create separate corridors/corners
                split_ratio = random.uniform(0.30, 0.70)

            if node.split(split_ratio, split_vertical):
                nodes_to_split.append(node.left)
                nodes_to_split.append(node.right)
            else:
                # Cannot split further
                nodes_to_split.append(node)
                break

        return root.get_leaves()

    def generate_single_layout(
        self,
        room_program: List[str],
        preference: str = "space_efficient",
        canvas_width: int = 800,
        canvas_height: int = 600,
        margin: float = 40.0,
        layout_id: str = "layout_0001",
    ) -> Optional[FloorPlan]:
        """Generate a single geometrically valid procedural FloorPlan instance."""
        num_rooms = len(room_program)
        usable_w = canvas_width - 2 * margin
        usable_h = canvas_height - 2 * margin

        leaves = self._partition_space(usable_w, usable_h, num_rooms, preference)
        if len(leaves) < num_rooms:
            return None

        # Sort room program by area weight descending
        sorted_types = sorted(room_program, key=lambda t: ROOM_AREA_WEIGHTS.get(t, 1.0), reverse=True)
        # Sort partition leaves by area descending
        leaves.sort(key=lambda l: l.w * l.h, reverse=True)

        rooms: List[Room] = []
        for idx, (node, r_type) in enumerate(zip(leaves, sorted_types)):
            # Translate coordinates by margin
            rx = round(node.x + margin, 1)
            ry = round(node.y + margin, 1)
            rw = round(node.w, 1)
            rh = round(node.h, 1)

            # Subtle architectural inset for realistic walls (2-3px gap)
            inset = 2.0
            rx_in = rx + inset
            ry_in = ry + inset
            rw_in = max(20.0, rw - 2 * inset)
            rh_in = max(20.0, rh - 2 * inset)

            polygon = [
                (rx_in, ry_in),
                (rx_in + rw_in, ry_in),
                (rx_in + rw_in, ry_in + rh_in),
                (rx_in, ry_in + rh_in),
            ]

            room = Room(
                room_id=f"room_{idx + 1:02d}",
                room_type=r_type,
                label=f"{r_type} {idx + 1}",
                x=rx_in,
                y=ry_in,
                width=rw_in,
                height=rh_in,
                polygon=polygon,
                coordinate_unit="pixel",
                detection_status=DetectionStatus.USER_CONFIRMED.value,
                confidence_score=1.0,
                confidence_explanation=f"Procedurally generated for preference '{preference}'.",
                measurement_source=MeasurementSource.UNCALIBRATED_PIXEL.value,
                is_user_confirmed=True,
                orientation_degrees=0.0 if rw_in >= rh_in else 90.0,
            )
            rooms.append(room)

        meta = ProcessingMetadata(
            pipeline_version="spaze-synth-bsp-v1",
            image_filename=f"{layout_id}.png",
            parameters_used={"preference": preference, "room_count": num_rooms},
        )

        floorplan = FloorPlan(
            image_width=canvas_width,
            image_height=canvas_height,
            coordinate_system="top_left_origin",
            scale_information=ScaleInfo(
                is_calibrated=True,
                pixels_per_meter=50.0,
                reference_length_meters=16.0,
                calibration_method="synthetic_standard",
            ),
            wall_candidates=[],
            candidate_rooms=[],
            confirmed_rooms=rooms,
            processing_metadata=meta,
            preference=preference,
        )

        for r in floorplan.confirmed_rooms:
            r.apply_scale(floorplan.scale_information)

        floorplan.compute_adjacency(tolerance_px=10.0)

        # Validate geometry
        eval_res = self.validator.evaluate_layout(floorplan)
        if not eval_res.is_valid:
            # Attempt relaxation
            floorplan = self.validator.relax_and_clamp_layout(floorplan)
            eval_res = self.validator.evaluate_layout(floorplan)
            if not eval_res.is_valid:
                return None

        return floorplan

    def generate_dataset_pairs(
        self,
        num_pairs: int = 3000,
        output_dir: Optional[Path] = None,
    ) -> List[SyntheticPair]:
        """Generate num_pairs source-target training pairs grouped by room-program families.

        Args:
            num_pairs: Total training pairs to generate (2,000–5,000).
            output_dir: Directory where JSON datasets will be persisted.

        Returns:
            List of SyntheticPair objects.
        """
        out_path = Path(output_dir or self.config.DATASET_DIR)
        out_path.mkdir(parents=True, exist_ok=True)

        pairs: List[SyntheticPair] = []
        preferences = list(self.config.PREFERENCES)
        pref_to_idx = self.config.pref_to_idx

        # Generate layout families: each family shares a fixed room program
        family_counter = 0
        attempts = 0
        max_attempts = num_pairs * 5

        while len(pairs) < num_pairs and attempts < max_attempts:
            attempts += 1
            family_counter += 1
            family_id = f"fam_{family_counter:04d}"

            # 1. Pick a room program
            program = random.choice(RESIDENTIAL_PROGRAMS)

            # 2. Pick target preference
            target_pref = random.choice(preferences)
            source_pref = random.choice(preferences)

            # 3. Randomize aspect ratio slightly (800x600, 750x600, 850x650)
            cw = random.choice([750, 800, 850])
            ch = random.choice([550, 600, 650])

            # Generate source layout
            src_fp = self.generate_single_layout(
                room_program=program,
                preference=source_pref,
                canvas_width=cw,
                canvas_height=ch,
                layout_id=f"{family_id}_src",
            )
            if src_fp is None:
                continue

            # Generate target layout with different random seed partition & target preference
            tgt_fp = self.generate_single_layout(
                room_program=program,
                preference=target_pref,
                canvas_width=cw,
                canvas_height=ch,
                layout_id=f"{family_id}_tgt_{target_pref}",
            )
            if tgt_fp is None:
                continue

            # Preference vector
            pref_vec = [0.0] * self.config.NUM_PREFERENCES
            pref_vec[pref_to_idx[target_pref]] = 1.0

            pair = SyntheticPair(
                pair_id=f"pair_{len(pairs) + 1:05d}",
                family_id=family_id,
                preference=target_pref,
                preference_vector=pref_vec,
                source_floorplan=src_fp.to_dict(),
                target_floorplan=tgt_fp.to_dict(),
                room_program=list(program),
            )
            pairs.append(pair)

        # Save dataset to disk
        dataset_file = out_path / "synthetic_pairs.json"
        with open(dataset_file, "w", encoding="utf-8") as f:
            json.dump([asdict(p) for p in pairs], f, indent=2)

        # Save summary metadata
        summary_file = out_path / "dataset_summary.json"
        summary = {
            "total_pairs_generated": len(pairs),
            "num_families": family_counter,
            "preference_distribution": {
                p: sum(1 for pair in pairs if pair.preference == p) for p in preferences
            },
            "room_count_distribution": {
                f"{cnt}_rooms": sum(1 for pair in pairs if len(pair.room_program) == cnt)
                for cnt in sorted(list({len(p.room_program) for p in pairs}))
            },
            "saved_file": str(dataset_file),
        }
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"Generated {len(pairs)} synthetic source-target pairs in {dataset_file}")
        return pairs
