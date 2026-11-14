"""Unit tests for cvae.model and cvae.losses."""
import pytest
import torch

from cvae.config import CVAEConfig
from cvae.losses import CVAELoss
from cvae.model import SpazeCVAE


def test_cvae_forward_pass():
    config = CVAEConfig()
    model = SpazeCVAE(config)

    batch_size = 4
    layout_dim = config.total_layout_dim
    pref_dim = config.NUM_PREFERENCES

    x_tgt = torch.randn(batch_size, layout_dim)
    x_src = torch.randn(batch_size, layout_dim)
    c_pref = torch.zeros(batch_size, pref_dim)
    c_pref[:, 0] = 1.0

    out = model(x_tgt, x_src, c_pref)

    assert "flat_layout" in out
    assert out["flat_layout"].shape == (batch_size, layout_dim)
    assert out["mu"].shape == (batch_size, config.LATENT_DIM)
    assert out["logvar"].shape == (batch_size, config.LATENT_DIM)
    assert out["z"].shape == (batch_size, config.LATENT_DIM)


def test_cvae_loss_computation():
    config = CVAEConfig()
    model = SpazeCVAE(config)
    loss_fn = CVAELoss(config)

    batch_size = 4
    layout_dim = config.total_layout_dim
    pref_dim = config.NUM_PREFERENCES

    x_tgt = torch.abs(torch.randn(batch_size, layout_dim)) % 1.0
    x_src = torch.abs(torch.randn(batch_size, layout_dim)) % 1.0
    c_pref = torch.zeros(batch_size, pref_dim)
    c_pref[:, 1] = 1.0

    out = model(x_tgt, x_src, c_pref)
    losses = loss_fn(out, x_tgt, beta=0.01, lambda_geo=0.05)

    assert "loss" in losses
    assert torch.isfinite(losses["loss"])
    assert losses["loss"].item() > 0.0

    # Verify gradients backward
    losses["loss"].backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} missing gradient"
