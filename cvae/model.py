"""PyTorch Conditional Variational Autoencoder (CVAE) for Floor-Plan Layout Generation.

Implements from-scratch encoder, reparameterization trick, and conditional decoder
to generate alternative floor-plan layouts conditioned on source features and user preferences.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from cvae.config import CVAEConfig


class SpazeCVAE(nn.Module):
    """Conditional Variational Autoencoder for architectural layout transformation."""

    def __init__(self, config: Optional[CVAEConfig] = None) -> None:
        super().__init__()
        self.config = config or CVAEConfig()

        layout_dim = self.config.total_layout_dim     # 348
        pref_dim = self.config.NUM_PREFERENCES        # 3
        latent_dim = self.config.LATENT_DIM           # 32
        hidden_dim = self.config.HIDDEN_DIM           # 256

        # =====================================================================
        # 1. Context Embedder for Source Floor Plan + Preference
        # =====================================================================
        self.source_encoder = nn.Sequential(
            nn.Linear(layout_dim + pref_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(self.config.DROPOUT),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # =====================================================================
        # 2. Recognition Encoder: q_phi(z | x_target, x_source, c)
        # =====================================================================
        # Receives target layout (348) + source embedding (256)
        encoder_in_dim = layout_dim + hidden_dim

        self.encoder_backbone = nn.Sequential(
            nn.Linear(encoder_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(self.config.DROPOUT),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Latent Gaussian parameters
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # =====================================================================
        # 3. Conditional Decoder: p_theta(x_target | z, x_source, c)
        # =====================================================================
        # Receives latent sample z (32) + source embedding (256)
        decoder_in_dim = latent_dim + hidden_dim

        self.decoder_backbone = nn.Sequential(
            nn.Linear(decoder_in_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(self.config.DROPOUT),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Output projection heads for individual feature groups
        max_rooms = self.config.MAX_ROOMS
        num_types = self.config.NUM_ROOM_TYPES

        # Room prediction heads
        self.head_room_types = nn.Linear(hidden_dim, max_rooms * num_types)       # 12 * 10 = 120
        self.head_room_coords = nn.Linear(hidden_dim, max_rooms * 4)              # 12 * 4 (cx, cy, w, h)
        self.head_room_orientation = nn.Linear(hidden_dim, max_rooms * 2)         # 12 * 2 (sin, cos)
        self.head_room_presence = nn.Linear(hidden_dim, max_rooms * 1)            # 12 * 1

        # Adjacency matrix head: predicts upper-triangular logits and mirrors
        self.head_adjacency = nn.Linear(hidden_dim, max_rooms * max_rooms)

    def encode(
        self,
        x_target: torch.Tensor,
        x_source: torch.Tensor,
        c_pref: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode target, source, and condition into latent mean and log-variance."""
        # 1. Embed source context
        src_context = torch.cat([x_source, c_pref], dim=-1)
        src_emb = self.source_encoder(src_context)

        # 2. Encode target conditioned on source embedding
        enc_in = torch.cat([x_target, src_emb], dim=-1)
        hidden = self.encoder_backbone(enc_in)

        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        # Clamp logvar for numerical stability [-10, 10]
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)

        return mu, logvar

    def reparameterize(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Sample z = mu + sigma * eps."""
        if deterministic or not self.training:
            return mu

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(
        self,
        z: torch.Tensor,
        x_source: torch.Tensor,
        c_pref: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Decode latent vector and source context into structured layout predictions."""
        # 1. Embed source context
        src_context = torch.cat([x_source, c_pref], dim=-1)
        src_emb = self.source_encoder(src_context)

        # 2. Decode latent vector
        dec_in = torch.cat([z, src_emb], dim=-1)
        features = self.decoder_backbone(dec_in)

        max_rooms = self.config.MAX_ROOMS
        num_types = self.config.NUM_ROOM_TYPES

        # Raw outputs
        type_logits = self.head_room_types(features).reshape(-1, max_rooms, num_types)
        coords_raw = self.head_room_coords(features).reshape(-1, max_rooms, 4)
        orientation_raw = self.head_room_orientation(features).reshape(-1, max_rooms, 2)
        presence_logits = self.head_room_presence(features).reshape(-1, max_rooms, 1)
        adj_logits = self.head_adjacency(features).reshape(-1, max_rooms, max_rooms)

        # Bounded activations
        coords_norm = torch.sigmoid(coords_raw)
        presence_prob = torch.sigmoid(presence_logits)
        orientation_norm = F.normalize(orientation_raw, p=2, dim=-1)

        # Symmetrize adjacency matrix
        adj_sym_logits = 0.5 * (adj_logits + adj_logits.transpose(-1, -2))
        adj_prob = torch.sigmoid(adj_sym_logits)

        # Construct full flat predicted vector matching layout_dim (348)
        # Room features: (B, MAX_ROOMS, 17)
        type_prob = F.softmax(type_logits, dim=-1)
        room_feats = torch.cat([
            type_prob,
            coords_norm,
            orientation_norm,
            presence_prob,
        ], dim=-1)

        flat_rooms = room_feats.reshape(-1, self.config.total_room_features)
        flat_adj = adj_prob.reshape(-1, self.config.total_adjacency_features)
        flat_layout = torch.cat([flat_rooms, flat_adj], dim=-1)

        return {
            "type_logits": type_logits,
            "coords_norm": coords_norm,
            "orientation_norm": orientation_norm,
            "presence_logits": presence_logits,
            "presence_prob": presence_prob,
            "adj_logits": adj_sym_logits,
            "adj_prob": adj_prob,
            "flat_layout": flat_layout,
        }

    def forward(
        self,
        x_target: torch.Tensor,
        x_source: torch.Tensor,
        c_pref: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Full forward pass for training."""
        mu, logvar = self.encode(x_target, x_source, c_pref)
        z = self.reparameterize(mu, logvar)
        out = self.decode(z, x_source, c_pref)
        out["mu"] = mu
        out["logvar"] = logvar
        out["z"] = z
        return out

    @torch.no_grad()
    def sample_alternatives(
        self,
        x_source: torch.Tensor,
        c_pref: torch.Tensor,
        num_alternatives: int = 3,
        temperature: float = 1.0,
        random_seed: Optional[int] = None,
    ) -> List[torch.Tensor]:
        """Sample distinct candidate layout vectors from the conditional latent distribution."""
        self.eval()
        if random_seed is not None:
            torch.manual_seed(random_seed)

        if x_source.ndim == 1:
            x_source = x_source.unsqueeze(0)
        if c_pref.ndim == 1:
            c_pref = c_pref.unsqueeze(0)

        device = x_source.device
        latent_dim = self.config.LATENT_DIM

        # Generate candidates using distinct latent samples
        results: List[torch.Tensor] = []
        for i in range(num_alternatives):
            # Sample latent noise with temperature scaling
            z = torch.randn(1, latent_dim, device=device) * temperature
            out = self.decode(z, x_source, c_pref)
            results.append(out["flat_layout"].squeeze(0))

        return results
