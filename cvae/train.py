"""Training Pipeline for SPAZE Conditional Variational Autoencoder.

Executes dataset loading or generation, family-aware data splitting, AdamW optimization,
KL divergence warmup, early stopping, and checkpoint management.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim

from cvae.config import CVAEConfig
from cvae.dataset import create_dataloaders
from cvae.losses import CVAELoss
from cvae.model import SpazeCVAE


def plot_training_curves(history: List[Dict[str, float]], save_path: Path) -> None:
    """Plot and save training and validation loss curves."""
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    recon_loss = [h["val_recon_loss"] for h in history]
    kl_loss = [h["val_kl_loss"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor="#0b0f17")

    # Panel 1: Overall Train vs Val Loss
    ax1.set_facecolor("#0f172a")
    ax1.plot(epochs, train_loss, label="Train Total Loss", color="#38bdf8", linewidth=2)
    ax1.plot(epochs, val_loss, label="Val Total Loss", color="#f43f5e", linewidth=2, linestyle="--")
    ax1.set_title("Total Loss Progression", color="#e2e8f0", fontweight="bold")
    ax1.set_xlabel("Epoch", color="#94a3b8")
    ax1.set_ylabel("Loss", color="#94a3b8")
    ax1.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0")
    ax1.grid(True, color="#334155", alpha=0.5)
    ax1.tick_params(colors="#94a3b8")

    # Panel 2: Reconstruction vs KL Component
    ax2.set_facecolor("#0f172a")
    ax2.plot(epochs, recon_loss, label="Reconstruction Loss", color="#10b981", linewidth=2)
    ax2.plot(epochs, kl_loss, label="KL Divergence", color="#a855f7", linewidth=2)
    ax2.set_title("Reconstruction vs KL Divergence", color="#e2e8f0", fontweight="bold")
    ax2.set_xlabel("Epoch", color="#94a3b8")
    ax2.set_ylabel("Metric", color="#94a3b8")
    ax2.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0")
    ax2.grid(True, color="#334155", alpha=0.5)
    ax2.tick_params(colors="#94a3b8")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), dpi=150, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print(f"Training curves saved to {save_path}")


def train_cvae(
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    dataset_size: int = 3000,
    device_name: Optional[str] = None,
    checkpoint_dir: Optional[Path] = None,
    log_dir: Optional[Path] = None,
    seed: int = 42,
) -> Tuple[SpazeCVAE, Dict[str, Any]]:
    """Execute complete CVAE training loop."""
    config = CVAEConfig()
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device(
        device_name or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"=== SPAZE CVAE Training Pipeline (Device: {device}) ===")

    ckpt_dir = Path(checkpoint_dir or config.CHECKPOINT_DIR)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = ckpt_dir / config.BEST_MODEL_FILENAME

    logger_dir = Path(log_dir or config.LOG_DIR)
    logger_dir.mkdir(parents=True, exist_ok=True)
    history_csv = logger_dir / "training_history.csv"
    curves_png = logger_dir / "training_curves.png"

    # 1. Prepare DataLoaders
    dataset_file = config.DATASET_DIR / "synthetic_pairs.json"
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset_source=dataset_file,
        config=config,
        batch_size=batch_size,
        seed=seed,
    )

    # 2. Instantiate Model, Loss & Optimizer
    model = SpazeCVAE(config).to(device)
    loss_fn = CVAELoss(config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=config.DEFAULT_WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)

    best_val_loss = float("inf")
    patience = 12
    patience_counter = 0
    history: List[Dict[str, float]] = []

    start_train_time = time.perf_counter()

    for epoch in range(1, epochs + 1):
        # Linear KL warmup
        if epoch <= config.KL_WARMUP_EPOCHS:
            current_beta = (epoch / config.KL_WARMUP_EPOCHS) * config.KL_WEIGHT
        else:
            current_beta = config.KL_WEIGHT

        # ----------------- Training Step -----------------
        model.train()
        train_loss_accum = 0.0
        train_recon_accum = 0.0
        train_kl_accum = 0.0
        train_batches = 0

        for batch in train_loader:
            x_src = batch["source_vector"].to(device)
            x_tgt = batch["target_vector"].to(device)
            c_pref = batch["preference_vector"].to(device)

            optimizer.zero_grad()
            out = model(x_tgt, x_src, c_pref)
            losses = loss_fn(out, x_tgt, beta=current_beta, lambda_geo=config.GEOMETRY_WEIGHT)

            loss = losses["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss_accum += float(loss.item())
            train_recon_accum += float(losses["reconstruction_loss"].item())
            train_kl_accum += float(losses["kl_loss"].item())
            train_batches += 1

        avg_train_loss = train_loss_accum / max(1, train_batches)
        avg_train_recon = train_recon_accum / max(1, train_batches)
        avg_train_kl = train_kl_accum / max(1, train_batches)

        # ----------------- Validation Step -----------------
        model.eval()
        val_loss_accum = 0.0
        val_recon_accum = 0.0
        val_kl_accum = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                x_src = batch["source_vector"].to(device)
                x_tgt = batch["target_vector"].to(device)
                c_pref = batch["preference_vector"].to(device)

                out = model(x_tgt, x_src, c_pref)
                losses = loss_fn(out, x_tgt, beta=current_beta, lambda_geo=config.GEOMETRY_WEIGHT)

                val_loss_accum += float(losses["loss"].item())
                val_recon_accum += float(losses["reconstruction_loss"].item())
                val_kl_accum += float(losses["kl_loss"].item())
                val_batches += 1

        avg_val_loss = val_loss_accum / max(1, val_batches)
        avg_val_recon = val_recon_accum / max(1, val_batches)
        avg_val_kl = val_kl_accum / max(1, val_batches)

        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "train_recon_loss": round(avg_train_recon, 4),
            "train_kl_loss": round(avg_train_kl, 4),
            "val_loss": round(avg_val_loss, 4),
            "val_recon_loss": round(avg_val_recon, 4),
            "val_kl_loss": round(avg_val_kl, 4),
            "beta": round(current_beta, 5),
            "learning_rate": current_lr,
        }
        history.append(epoch_record)

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(
                f"Epoch [{epoch:03d}/{epochs:03d}] | "
                f"Train: {avg_train_loss:.4f} (Recon: {avg_train_recon:.4f}, KL: {avg_train_kl:.4f}) | "
                f"Val: {avg_val_loss:.4f} | LR: {current_lr:.6f}"
            )

        # Checkpoint saving
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "best_val_loss": round(best_val_loss, 4),
                "config": {
                    "latent_dim": config.LATENT_DIM,
                    "hidden_dim": config.HIDDEN_DIM,
                    "max_rooms": config.MAX_ROOMS,
                    "room_vocab": list(config.ROOM_VOCABULARY),
                },
                "history": history,
            }
            torch.save(checkpoint, best_ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch} (Best Val Loss: {best_val_loss:.4f}).")
                break

    total_training_sec = time.perf_counter() - start_train_time

    # 3. Held-out Test Evaluation
    model.eval()
    test_loss_accum = 0.0
    test_recon_accum = 0.0
    test_batches = 0

    with torch.no_grad():
        for batch in test_loader:
            x_src = batch["source_vector"].to(device)
            x_tgt = batch["target_vector"].to(device)
            c_pref = batch["preference_vector"].to(device)

            out = model(x_tgt, x_src, c_pref)
            losses = loss_fn(out, x_tgt, beta=config.KL_WEIGHT, lambda_geo=config.GEOMETRY_WEIGHT)
            test_loss_accum += float(losses["loss"].item())
            test_recon_accum += float(losses["reconstruction_loss"].item())
            test_batches += 1

    avg_test_loss = test_loss_accum / max(1, test_batches)
    avg_test_recon = test_recon_accum / max(1, test_batches)

    # 4. Save history CSV and Plot Curves
    if history:
        with open(history_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)
        plot_training_curves(history, curves_png)

    summary = {
        "epochs_trained": len(history),
        "total_training_seconds": round(total_training_sec, 2),
        "best_val_loss": round(best_val_loss, 4),
        "test_total_loss": round(avg_test_loss, 4),
        "test_recon_loss": round(avg_test_recon, 4),
        "checkpoint_path": str(best_ckpt_path),
        "history_path": str(history_csv),
        "curves_path": str(curves_png),
    }

    print(
        f"\nTraining Complete in {total_training_sec:.1f}s | "
        f"Best Val Loss: {best_val_loss:.4f} | Test Loss: {avg_test_loss:.4f} | "
        f"Saved: {best_ckpt_path}"
    )
    return model, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SPAZE Conditional VAE")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--dataset_size", type=int, default=3000, help="Synthetic dataset size")
    parser.add_argument("--device", type=str, default=None, help="Target device (cpu, cuda)")
    args = parser.parse_args()

    train_cvae(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        dataset_size=args.dataset_size,
        device_name=args.device,
    )
