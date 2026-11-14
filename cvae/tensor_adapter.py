"""Tensor Adapter for SPAZE Conditional VAE.

Provides bidirectional conversion between the structured FloorPlan schema
and fixed-dimension PyTorch tensors required for deep learning training and inference.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from cvae.config import CVAEConfig
from models.floorplan_schema import DetectionStatus, FloorPlan, MeasurementSource, ProcessingMetadata, Room, ScaleInfo


class FloorPlanTensorAdapter:
    """Encodes FloorPlan instances into numerical tensors and decodes tensors back to FloorPlan."""

    def __init__(self, config: Optional[CVAEConfig] = None) -> None:
        self.config = config or CVAEConfig()

    def encode_floorplan(
        self,
        floorplan: FloorPlan,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert a FloorPlan instance to (room_features, adjacency_matrix) tensors.

        Args:
            floorplan: FloorPlan instance.

        Returns:
            room_tensor: Tensor of shape (MAX_ROOMS, ROOM_FEATURE_DIM)
            adj_tensor: Tensor of shape (MAX_ROOMS, MAX_ROOMS)
        """
        max_rooms = self.config.MAX_ROOMS
        feat_dim = self.config.ROOM_FEATURE_DIM
        type_to_idx = self.config.type_to_idx

        # Active rooms (confirmed first, then candidates)
        active_rooms = floorplan.confirmed_rooms + floorplan.candidate_rooms
        num_rooms = min(len(active_rooms), max_rooms)

        room_features = np.zeros((max_rooms, feat_dim), dtype=np.float32)
        adj_matrix = np.zeros((max_rooms, max_rooms), dtype=np.float32)

        w_img = float(max(1, floorplan.image_width))
        h_img = float(max(1, floorplan.image_height))

        room_id_to_idx = {}

        for i in range(num_rooms):
            room = active_rooms[i]
            room_id_to_idx[room.room_id] = i

            # 1. One-hot room type (indices 0..9)
            t_idx = type_to_idx.get(room.room_type, type_to_idx.get("Bedroom", 2))
            room_features[i, t_idx] = 1.0

            # 2. Normalized bounding box center and size (indices 10..13)
            cx = (room.x + room.width / 2.0) / w_img
            cy = (room.y + room.height / 2.0) / h_img
            rw = room.width / w_img
            rh = room.height / h_img

            room_features[i, 10] = float(np.clip(cx, 0.0, 1.0))
            room_features[i, 11] = float(np.clip(cy, 0.0, 1.0))
            room_features[i, 12] = float(np.clip(rw, 0.01, 1.0))
            room_features[i, 13] = float(np.clip(rh, 0.01, 1.0))

            # 3. Orientation sin and cos (indices 14..15)
            theta_rad = math.radians(getattr(room, "orientation_degrees", 0.0))
            room_features[i, 14] = float(math.sin(theta_rad))
            room_features[i, 15] = float(math.cos(theta_rad))

            # 4. Presence flag (index 16)
            room_features[i, 16] = 1.0

        # Adjacency matrix construction
        if floorplan.adjacency_matrix and len(floorplan.adjacency_matrix) > 0:
            raw_adj = floorplan.adjacency_matrix
            n_raw = min(len(raw_adj), num_rooms)
            for r1 in range(n_raw):
                for r2 in range(n_raw):
                    adj_matrix[r1, r2] = float(raw_adj[r1][r2])
        else:
            # Reconstruct from room.adjacent_room_ids or compute on the fly
            for i in range(num_rooms):
                room = active_rooms[i]
                for adj_id in getattr(room, "adjacent_room_ids", []):
                    if adj_id in room_id_to_idx:
                        j = room_id_to_idx[adj_id]
                        adj_matrix[i, j] = 1.0
                        adj_matrix[j, i] = 1.0

        # Enforce exact symmetry on adjacency
        adj_matrix = np.maximum(adj_matrix, adj_matrix.T)
        np.fill_diagonal(adj_matrix, 0.0)

        return torch.from_numpy(room_features), torch.from_numpy(adj_matrix)

    def encode_to_flat_vector(self, floorplan: FloorPlan) -> torch.Tensor:
        """Encode a FloorPlan into a single 1D vector of shape (total_layout_dim,)."""
        room_tensor, adj_tensor = self.encode_floorplan(floorplan)
        flat_rooms = room_tensor.reshape(-1)
        flat_adj = adj_tensor.reshape(-1)
        return torch.cat([flat_rooms, flat_adj], dim=0)

    def encode_preference(self, preference_name: str) -> torch.Tensor:
        """Encode preference string into a 1-hot condition vector of shape (NUM_PREFERENCES,)."""
        pref_vec = torch.zeros(self.config.NUM_PREFERENCES, dtype=torch.float32)
        idx = self.config.pref_to_idx.get(preference_name.lower(), 0)
        pref_vec[idx] = 1.0
        return pref_vec

    def decode_flat_vector(
        self,
        flat_vector: torch.Tensor,
        image_width: int = 800,
        image_height: int = 600,
        scale_info: Optional[ScaleInfo] = None,
        preference: Optional[str] = None,
        presence_threshold: float = 0.5,
    ) -> FloorPlan:
        """Decode a 1D layout vector of shape (total_layout_dim,) into a FloorPlan instance."""
        if flat_vector.ndim > 1:
            flat_vector = flat_vector.squeeze(0)

        vec = flat_vector.detach().cpu().numpy()
        max_rooms = self.config.MAX_ROOMS
        feat_dim = self.config.ROOM_FEATURE_DIM
        total_room_feats = self.config.total_room_features

        raw_room_feats = vec[:total_room_feats].reshape(max_rooms, feat_dim)
        raw_adj = vec[total_room_feats:].reshape(max_rooms, max_rooms)

        # Symmetrize predicted adjacency
        pred_adj = 0.5 * (raw_adj + raw_adj.T)
        np.fill_diagonal(pred_adj, 0.0)

        rooms: List[Room] = []
        active_indices: List[int] = []

        w_img = float(image_width)
        h_img = float(image_height)

        for i in range(max_rooms):
            presence_prob = float(raw_room_feats[i, 16])
            if presence_prob < presence_threshold:
                continue

            # Identify room type
            type_logits = raw_room_feats[i, :self.config.NUM_ROOM_TYPES]
            t_idx = int(np.argmax(type_logits))
            room_type = self.config.idx_to_type.get(t_idx, "Bedroom")

            # Bounding box coordinates
            norm_cx = float(np.clip(raw_room_feats[i, 10], 0.05, 0.95))
            norm_cy = float(np.clip(raw_room_feats[i, 11], 0.05, 0.95))
            norm_w = float(np.clip(raw_room_feats[i, 12], 0.05, 0.90))
            norm_h = float(np.clip(raw_room_feats[i, 13], 0.05, 0.90))

            px_w = max(20.0, norm_w * w_img)
            px_h = max(20.0, norm_h * h_img)
            px_x = max(0.0, min(w_img - px_w, (norm_cx * w_img) - (px_w / 2.0)))
            px_y = max(0.0, min(h_img - px_h, (norm_cy * h_img) - (px_h / 2.0)))

            # Orientation
            sin_t = float(raw_room_feats[i, 14])
            cos_t = float(raw_room_feats[i, 15])
            angle_deg = float(math.degrees(math.atan2(sin_t, cos_t))) % 360.0

            room_id = f"gen_room_{len(rooms) + 1:02d}"
            polygon = [
                (round(px_x, 1), round(px_y, 1)),
                (round(px_x + px_w, 1), round(px_y, 1)),
                (round(px_x + px_w, 1), round(px_y + px_h, 1)),
                (round(px_x, 1), round(px_y + px_h, 1)),
            ]

            room = Room(
                room_id=room_id,
                room_type=room_type,
                label=f"{room_type} {len(rooms) + 1}",
                x=round(px_x, 1),
                y=round(px_y, 1),
                width=round(px_w, 1),
                height=round(px_h, 1),
                polygon=polygon,
                coordinate_unit="pixel",
                detection_status=DetectionStatus.USER_CONFIRMED.value,
                confidence_score=round(float(np.clip(presence_prob, 0.0, 1.0)), 3),
                confidence_explanation=f"Generated via CVAE conditioned on {preference or 'default'}.",
                measurement_source=MeasurementSource.USER_SCALE_CALIBRATED.value if scale_info and scale_info.is_calibrated else MeasurementSource.UNCALIBRATED_PIXEL.value,
                is_user_confirmed=True,
                orientation_degrees=round(angle_deg, 1),
            )
            rooms.append(room)
            active_indices.append(i)

        # Build active adjacency matrix and update adjacent_room_ids
        n_act = len(rooms)
        final_adj_matrix = [[0] * n_act for _ in range(n_act)]

        for a_idx, orig_i in enumerate(active_indices):
            for b_idx, orig_j in enumerate(active_indices):
                if a_idx != b_idx and pred_adj[orig_i, orig_j] >= 0.5:
                    final_adj_matrix[a_idx][b_idx] = 1
                    rooms[a_idx].adjacent_room_ids.append(rooms[b_idx].room_id)

        meta = ProcessingMetadata(
            pipeline_version="spaze-cvae-v1",
            image_filename=f"generated_{preference or 'layout'}",
            warnings=[],
            parameters_used={"preference": preference, "presence_threshold": presence_threshold},
        )

        active_scale = scale_info or ScaleInfo()
        floorplan = FloorPlan(
            image_width=image_width,
            image_height=image_height,
            coordinate_system="top_left_origin",
            scale_information=active_scale,
            wall_candidates=[],
            candidate_rooms=[],
            confirmed_rooms=rooms,
            processing_metadata=meta,
            adjacency_matrix=final_adj_matrix,
            room_order=[r.room_id for r in rooms],
            preference=preference,
            is_generated=True,
        )

        for r in floorplan.confirmed_rooms:
            r.apply_scale(active_scale)

        return floorplan
