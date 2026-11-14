"""Loss Functions for SPAZE Conditional Variational Autoencoder.

Implements multi-component loss formulation:
- Room presence binary cross-entropy
- Room type cross-entropy (presence-masked)
- Bounding box coordinates & dimensions Smooth L1 (presence-masked)
- Orientation angular alignment loss
- Symmetric adjacency binary cross-entropy
- Analytical Kullback-Leibler divergence with warmup scheduling
- Differentiable geometric room overlap penalty
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from cvae.config import CVAEConfig


class CVAELoss(nn.Module):
    """Calculates composite objective function for CVAE layout training."""

    def __init__(self, config: Optional[CVAEConfig] = None) -> None:
        super().__init__()
        self.config = config or CVAEConfig()

    def compute_geometric_overlap_loss(
        self,
        coords_norm: torch.Tensor,
        presence_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Differentiable pairwise bounding-box overlap penalty.

        Args:
            coords_norm: (B, MAX_ROOMS, 4) in [cx, cy, w, h]
            presence_mask: (B, MAX_ROOMS) in {0, 1}
        """
        B, N, _ = coords_norm.shape
        cx = coords_norm[..., 0]
        cy = coords_norm[..., 1]
        w = coords_norm[..., 2]
        h = coords_norm[..., 3]

        # Convert to x1, y1, x2, y2
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0

        # Broadcast for pairwise differences: (B, N, 1) vs (B, 1, N)
        x1_a, x1_b = x1.unsqueeze(2), x1.unsqueeze(1)
        y1_a, y1_b = y1.unsqueeze(2), y1.unsqueeze(1)
        x2_a, x2_b = x2.unsqueeze(2), x2.unsqueeze(1)
        y2_a, y2_b = y2.unsqueeze(2), y2.unsqueeze(1)

        # Pairwise overlap dimensions
        inter_w = torch.clamp(torch.min(x2_a, x2_b) - torch.max(x1_a, x1_b), min=0.0)
        inter_h = torch.clamp(torch.min(y2_a, y2_b) - torch.max(y1_a, y1_b), min=0.0)
        inter_area = inter_w * inter_h

        # Zero out self-overlap on diagonal
        eye_mask = torch.eye(N, device=coords_norm.device).unsqueeze(0).expand(B, N, N)
        inter_area = inter_area * (1.0 - eye_mask)

        # Mask by pairwise presence (both rooms must exist)
        pair_presence = presence_mask.unsqueeze(2) * presence_mask.unsqueeze(1)
        masked_overlap = inter_area * pair_presence

        # Normalize by batch
        return torch.sum(masked_overlap) / max(1.0, B * N)

    def forward(
        self,
        model_output: Dict[str, torch.Tensor],
        target_vector: torch.Tensor,
        beta: float = 0.005,
        lambda_geo: float = 0.05,
    ) -> Dict[str, torch.Tensor]:
        """Compute all loss components.

        Args:
            model_output: Dictionary from SpazeCVAE.forward().
            target_vector: Ground truth target layout vector (B, 348).
            beta: Current KL divergence weight (from warmup scheduler).
            lambda_geo: Geometric overlap penalty weight.

        Returns:
            Dictionary containing total_loss and individual metric components.
        """
        B = target_vector.shape[0]
        max_rooms = self.config.MAX_ROOMS
        feat_dim = self.config.ROOM_FEATURE_DIM
        num_types = self.config.NUM_ROOM_TYPES
        total_room_feats = self.config.total_room_features

        # 1. Unpack target tensor
        tgt_rooms = target_vector[:, :total_room_feats].reshape(B, max_rooms, feat_dim)
        tgt_adj = target_vector[:, total_room_feats:].reshape(B, max_rooms, max_rooms)

        tgt_types_onehot = tgt_rooms[:, :, :num_types]
        tgt_coords = tgt_rooms[:, :, 10:14]
        tgt_orient = tgt_rooms[:, :, 14:16]
        tgt_presence = tgt_rooms[:, :, 16]

        presence_mask = tgt_presence.bool()
        num_active_rooms = torch.clamp(presence_mask.sum(), min=1.0)

        # 2. Presence Loss (BCE)
        presence_logits = model_output["presence_logits"].squeeze(-1)
        loss_presence = F.binary_cross_entropy_with_logits(
            presence_logits,
            tgt_presence,
            reduction="mean",
        )

        # 3. Room Type Loss (Cross Entropy on active rooms)
        type_logits = model_output["type_logits"]
        tgt_type_labels = torch.argmax(tgt_types_onehot, dim=-1)

        loss_type = F.cross_entropy(
            type_logits[presence_mask],
            tgt_type_labels[presence_mask],
            reduction="mean",
        ) if presence_mask.any() else torch.tensor(0.0, device=target_vector.device)

        # 4. Coordinates & Dimensions Loss (Smooth L1 on active rooms)
        pred_coords = model_output["coords_norm"]
        loss_coords = F.smooth_l1_loss(
            pred_coords[presence_mask],
            tgt_coords[presence_mask],
            beta=0.05,
            reduction="sum",
        ) / num_active_rooms

        # 5. Orientation Loss (Cosine alignment)
        pred_orient = model_output["orientation_norm"]
        loss_orient = F.mse_loss(
            pred_orient[presence_mask],
            tgt_orient[presence_mask],
            reduction="sum",
        ) / num_active_rooms if presence_mask.any() else torch.tensor(0.0, device=target_vector.device)

        # 6. Adjacency Loss (Symmetric BCE on active pairs)
        adj_logits = model_output["adj_logits"]
        pair_mask = (tgt_presence.unsqueeze(2) * tgt_presence.unsqueeze(1)).bool()
        loss_adj = F.binary_cross_entropy_with_logits(
            adj_logits[pair_mask],
            tgt_adj[pair_mask],
            reduction="mean",
        ) if pair_mask.any() else torch.tensor(0.0, device=target_vector.device)

        # Composite Reconstruction Loss
        loss_reconstruction = (
            2.0 * loss_presence
            + 1.5 * loss_type
            + 3.0 * loss_coords
            + 0.5 * loss_orient
            + 1.0 * loss_adj
        )

        # 7. KL Divergence: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        mu = model_output["mu"]
        logvar = model_output["logvar"]
        kl_div = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - torch.exp(logvar), dim=-1)
        loss_kl = torch.mean(kl_div)

        # 8. Differentiable Geometric Overlap Penalty
        loss_geo = self.compute_geometric_overlap_loss(pred_coords, tgt_presence)

        # 9. Total Weighted Loss
        total_loss = loss_reconstruction + (beta * loss_kl) + (lambda_geo * loss_geo)

        return {
            "loss": total_loss,
            "reconstruction_loss": loss_reconstruction.detach(),
            "kl_loss": loss_kl.detach(),
            "presence_loss": loss_presence.detach(),
            "type_loss": loss_type.detach(),
            "coords_loss": loss_coords.detach(),
            "adj_loss": loss_adj.detach(),
            "geo_loss": loss_geo.detach(),
            "beta": beta,
        }
