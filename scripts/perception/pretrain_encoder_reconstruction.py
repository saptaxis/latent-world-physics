#!/usr/bin/env python3
"""Pre-train ImpalaCNN encoder on frame reconstruction (autoencoder or VAE).

Same encoder as pretrain_encoder.py (kinematic prediction), but trained to
reconstruct input frames instead of predicting kinematic state. This is the
ablation control: does the benefit of pre-training come from physics-aligned
features (kinematic) or just good visual features (reconstruction)?

Supports three modes:
  --objective autoencoder --reconstruct last   (simplest: reconstruct last frame)
  --objective autoencoder --reconstruct all    (reconstruct all 4 stacked frames)
  --objective vae --reconstruct last           (VAE with KL regularization)

The encoder architecture is identical to the kinematic script — same
StandaloneImpalaCNN, same state_dict keys, same weight transfer to SB3 PPO.
Only the decoder (throwaway) and loss function differ.

Usage:
    python lunar_lander/scripts/pretrain_encoder_reconstruction.py \
        --prepared-dataset /path/to/prepared.npz \
        --output-dir /path/to/output \
        --objective autoencoder --reconstruct all

The script saves:
    - best_encoder.pt: encoder state_dict at best val loss
    - encoder.pt: final encoder state_dict
    - checkpoints/: full training state per epoch
    - training_log.json: per-epoch metrics
    - tb/: TensorBoard events
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add project root to path for imports.

from lwp.agents.encoder_dataset import EncoderPretrainDataset
from scripts.perception.pretrain_encoder import StandaloneImpalaCNN


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class ConvDecoder(nn.Module):
    """Transpose-conv decoder: 256D bottleneck → (out_channels, 128, 128).

    Mirrors the encoder's spatial path in reverse. Five ConvTranspose2d
    layers double spatial resolution each step: 4→8→16→32→64→128.

    Args:
        features_dim: Encoder output dimension (256).
        out_channels: 1 for single-frame reconstruction, 4 for all frames.
    """

    def __init__(self, features_dim: int = 256, out_channels: int = 1):
        super().__init__()
        # Project from bottleneck to spatial feature map.
        # 32 channels × 4×4 spatial = 512 values.
        self.project = nn.Linear(features_dim, 32 * 4 * 4)

        # 5 upsample stages: 4→8→16→32→64→128
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),  # 4→8
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),  # 8→16
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # 16→32
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),   # 32→64
            nn.ReLU(),
            nn.ConvTranspose2d(8, out_channels, kernel_size=4, stride=2, padding=1),  # 64→128
            nn.Sigmoid(),  # output [0, 1] matching normalized input frames
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.project(z)
        x = x.view(-1, 32, 4, 4)
        return self.deconv(x)


# ---------------------------------------------------------------------------
# VAE heads (throwaway — only used during pre-training)
# ---------------------------------------------------------------------------

class VAEHeads(nn.Module):
    """Maps encoder features to μ and log_σ² for VAE reparameterization.

    The encoder itself stays deterministic (same architecture as SB3).
    These heads are external and throwaway — they add VAE structure on
    top of the encoder's 256D output without modifying the encoder.
    """

    def __init__(self, features_dim: int = 256, latent_dim: int = 256):
        super().__init__()
        self.mu = nn.Linear(features_dim, latent_dim)
        self.log_var = nn.Linear(features_dim, latent_dim)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.mu(features), self.log_var(features)

    @staticmethod
    def reparameterize(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Sample z = μ + σ * ε, where ε ~ N(0, I)."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + std * eps

    @staticmethod
    def kl_divergence(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """KL(q(z|x) || p(z)) where p(z) = N(0, I). Per-batch mean."""
        return -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())


# ---------------------------------------------------------------------------
# SSIM (simple implementation, no external deps)
# ---------------------------------------------------------------------------

def _ssim_batch(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute mean SSIM over a batch. Both inputs: (B, C, H, W) in [0, 1].

    Uses 11×11 uniform window (simplified — not Gaussian, but good enough
    for tracking reconstruction quality during training).
    """
    C1 = 0.01 ** 2  # stability constants
    C2 = 0.03 ** 2
    # Flatten channels into batch for per-channel SSIM
    B, C, H, W = pred.shape
    pred = pred.reshape(B * C, 1, H, W)
    target = target.reshape(B * C, 1, H, W)

    # Uniform 11×11 averaging kernel
    window = torch.ones(1, 1, 11, 11, device=pred.device) / 121.0

    mu_p = F.conv2d(pred, window, padding=5)
    mu_t = F.conv2d(target, window, padding=5)
    mu_pp = mu_p * mu_p
    mu_tt = mu_t * mu_t
    mu_pt = mu_p * mu_t

    sigma_pp = F.conv2d(pred * pred, window, padding=5) - mu_pp
    sigma_tt = F.conv2d(target * target, window, padding=5) - mu_tt
    sigma_pt = F.conv2d(pred * target, window, padding=5) - mu_pt

    ssim_map = ((2 * mu_pt + C1) * (2 * sigma_pt + C2)) / \
               ((mu_pp + mu_tt + C1) * (sigma_pp + sigma_tt + C2))

    return float(ssim_map.mean())


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------

def train_one_epoch(
    encoder: nn.Module,
    decoder: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    objective: str,
    reconstruct: str,
    vae_heads: VAEHeads | None = None,
    beta: float = 1.0,
) -> dict:
    """Train for one epoch. Returns dict with loss, mse, kl (if VAE)."""
    encoder.train()
    decoder.train()
    if vae_heads is not None:
        vae_heads.train()

    total_loss = 0.0
    total_mse = 0.0
    total_kl = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc="Train", leave=False)
    for frames, _kinematic_targets in pbar:
        frames = frames.to(device)  # (B, 4, H, W)

        # Reconstruction target: last frame or all frames.
        if reconstruct == "last":
            target = frames[:, -1:, :, :]  # (B, 1, H, W)
        else:
            target = frames  # (B, 4, H, W)

        # Forward through encoder.
        features = encoder(frames)  # (B, 256)

        # VAE: reparameterize.
        if objective == "vae" and vae_heads is not None:
            mu, log_var = vae_heads(features)
            z = VAEHeads.reparameterize(mu, log_var)
            kl = VAEHeads.kl_divergence(mu, log_var)
        else:
            z = features
            kl = torch.tensor(0.0)

        # Decode and compute loss.
        reconstruction = decoder(z)
        mse = F.mse_loss(reconstruction, target)

        if objective == "vae":
            loss = mse + beta * kl
        else:
            loss = mse

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_mse += mse.item()
        total_kl += kl.item()
        n_batches += 1
        pbar.set_postfix(loss=f"{total_loss / n_batches:.4f}")

    n = max(n_batches, 1)
    result = {"loss": total_loss / n, "mse": total_mse / n}
    if objective == "vae":
        result["kl"] = total_kl / n
    return result


def evaluate(
    encoder: nn.Module,
    decoder: nn.Module,
    loader: DataLoader,
    device: str,
    objective: str,
    reconstruct: str,
    vae_heads: VAEHeads | None = None,
    beta: float = 1.0,
) -> dict:
    """Evaluate on validation set. Returns loss, mse, kl (if VAE), ssim."""
    encoder.eval()
    decoder.eval()
    if vae_heads is not None:
        vae_heads.eval()

    total_loss = 0.0
    total_mse = 0.0
    total_kl = 0.0
    total_ssim = 0.0
    n_batches = 0

    with torch.no_grad():
        for frames, _kinematic_targets in tqdm(loader, desc="Val", leave=False):
            frames = frames.to(device)

            if reconstruct == "last":
                target = frames[:, -1:, :, :]
            else:
                target = frames

            features = encoder(frames)

            if objective == "vae" and vae_heads is not None:
                mu, log_var = vae_heads(features)
                z = VAEHeads.reparameterize(mu, log_var)
                kl = VAEHeads.kl_divergence(mu, log_var)
            else:
                z = features
                kl = torch.tensor(0.0)

            reconstruction = decoder(z)
            mse = F.mse_loss(reconstruction, target)

            if objective == "vae":
                loss = mse + beta * kl
            else:
                loss = mse

            total_loss += loss.item()
            total_mse += mse.item()
            total_kl += kl.item()
            total_ssim += _ssim_batch(reconstruction, target)
            n_batches += 1

    n = max(n_batches, 1)
    result = {
        "loss": total_loss / n,
        "mse": total_mse / n,
        "ssim": total_ssim / n,
    }
    if objective == "vae":
        result["kl"] = total_kl / n
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Pre-train ImpalaCNN encoder on frame reconstruction."
    )
    # Data source: exactly one of --data-dir or --prepared-dataset.
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument(
        "--data-dir", type=str, nargs="+",
        help="Directory(ies) containing raw .npz episodes (slower).",
    )
    data_group.add_argument(
        "--prepared-dataset", type=str,
        help="Path to prepared .npz from prepare_encoder_dataset.py (faster).",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Directory for encoder.pt, training_log.json, etc.",
    )
    parser.add_argument(
        "--objective", type=str, required=True, choices=["autoencoder", "vae"],
        help="autoencoder = MSE only. vae = MSE + beta*KL.",
    )
    parser.add_argument(
        "--reconstruct", type=str, default="all", choices=["last", "all"],
        help="last = reconstruct final frame. all = reconstruct all 4 stacked frames.",
    )
    parser.add_argument("--beta", type=float, default=1.0,
                        help="VAE KL weight (beta-VAE). Ignored for autoencoder.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n-stack", type=int, default=4)
    parser.add_argument("--frame-size", type=int, default=128)
    parser.add_argument("--features-dim", type=int, default=256)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=50,
                        help="Early stopping patience.")
    parser.add_argument("--lr-patience", type=int, default=5,
                        help="ReduceLROnPlateau patience.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest checkpoint in output-dir.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Objective: {args.objective}, Reconstruct: {args.reconstruct}")
    if args.objective == "vae":
        print(f"Beta: {args.beta}")

    # --- Data ---
    if args.prepared_dataset:
        from lwp.agents.encoder_dataset import PreparedEncoderDataset
        print(f"Loading prepared dataset from {args.prepared_dataset}...")
        train_ds = PreparedEncoderDataset(
            args.prepared_dataset, n_stack=args.n_stack,
            split="train", val_fraction=args.val_fraction, seed=args.seed,
        )
        val_ds = PreparedEncoderDataset(
            args.prepared_dataset, n_stack=args.n_stack,
            split="val", val_fraction=args.val_fraction, seed=args.seed,
        )
    else:
        print(f"Loading raw episodes from {args.data_dir}...")
        train_ds = EncoderPretrainDataset(
            args.data_dir, n_stack=args.n_stack, frame_size=args.frame_size,
            split="train", val_fraction=args.val_fraction, seed=args.seed,
        )
        val_ds = EncoderPretrainDataset(
            args.data_dir, n_stack=args.n_stack, frame_size=args.frame_size,
            split="val", val_fraction=args.val_fraction, seed=args.seed,
        )
    print(f"Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")

    # num_workers=0: see pretrain_encoder.py for rationale (OOM with workers).
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    # --- Model ---
    out_channels = 1 if args.reconstruct == "last" else args.n_stack

    encoder = StandaloneImpalaCNN(
        in_channels=args.n_stack, features_dim=args.features_dim,
    ).to(device)

    decoder = ConvDecoder(
        features_dim=args.features_dim, out_channels=out_channels,
    ).to(device)

    vae_heads = None
    if args.objective == "vae":
        vae_heads = VAEHeads(
            features_dim=args.features_dim, latent_dim=args.features_dim,
        ).to(device)

    n_enc = sum(p.numel() for p in encoder.parameters())
    n_dec = sum(p.numel() for p in decoder.parameters())
    n_vae = sum(p.numel() for p in vae_heads.parameters()) if vae_heads else 0
    print(f"Encoder: {n_enc:,} params, Decoder: {n_dec:,} params"
          + (f", VAE heads: {n_vae:,} params" if vae_heads else ""))

    # Collect all trainable params.
    all_params = list(encoder.parameters()) + list(decoder.parameters())
    if vae_heads is not None:
        all_params += list(vae_heads.parameters())

    optimizer = torch.optim.Adam(all_params, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=args.lr_patience,
    )

    # --- TensorBoard ---
    writer = SummaryWriter(log_dir=str(output_dir / "tb"))

    # --- Checkpoints ---
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # --- Resume ---
    start_epoch = 1
    best_val_loss = float("inf")
    patience_counter = 0
    log = []

    if args.resume:
        ckpt_files = sorted(ckpt_dir.glob("checkpoint_epoch*.pt"))
        if ckpt_files:
            latest_ckpt = ckpt_files[-1]
            print(f"Resuming from {latest_ckpt}...")
            ckpt = torch.load(str(latest_ckpt), map_location=device, weights_only=False)
            encoder.load_state_dict(ckpt["encoder"])
            decoder.load_state_dict(ckpt["decoder"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            if vae_heads is not None and "vae_heads" in ckpt:
                vae_heads.load_state_dict(ckpt["vae_heads"])
            start_epoch = ckpt["epoch"] + 1
            best_val_loss = ckpt["best_val_loss"]
            patience_counter = ckpt["patience_counter"]
            print(f"  Resumed at epoch {start_epoch}, best_val_loss={best_val_loss:.4f}, "
                  f"lr={optimizer.param_groups[0]['lr']:.1e}")

            log_path = output_dir / "training_log.json"
            if log_path.exists():
                with open(log_path) as f:
                    log = json.load(f)
                log = [e for e in log if e["epoch"] < start_epoch]
        else:
            print("WARNING: --resume but no checkpoints found. Starting from scratch.")

    # --- Training loop ---
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        train_metrics = train_one_epoch(
            encoder, decoder, train_loader, optimizer, device,
            args.objective, args.reconstruct, vae_heads, args.beta,
        )
        val_metrics = evaluate(
            encoder, decoder, val_loader, device,
            args.objective, args.reconstruct, vae_heads, args.beta,
        )
        elapsed = time.time() - t0

        scheduler.step(val_metrics["loss"])

        entry = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_mse": train_metrics["mse"],
            "val_loss": val_metrics["loss"],
            "val_mse": val_metrics["mse"],
            "val_ssim": val_metrics["ssim"],
            "lr": current_lr,
            "elapsed_s": round(elapsed, 1),
        }
        if args.objective == "vae":
            entry["train_kl"] = train_metrics["kl"]
            entry["val_kl"] = val_metrics["kl"]
        log.append(entry)

        # TensorBoard
        writer.add_scalar("loss/train", train_metrics["loss"], epoch)
        writer.add_scalar("loss/val", val_metrics["loss"], epoch)
        writer.add_scalar("mse/train", train_metrics["mse"], epoch)
        writer.add_scalar("mse/val", val_metrics["mse"], epoch)
        writer.add_scalar("ssim/val", val_metrics["ssim"], epoch)
        writer.add_scalar("lr", current_lr, epoch)
        writer.add_scalar("timing/epoch_s", elapsed, epoch)
        if args.objective == "vae":
            writer.add_scalar("kl/train", train_metrics["kl"], epoch)
            writer.add_scalar("kl/val", val_metrics["kl"], epoch)

        # Console
        extra = ""
        if args.objective == "vae":
            extra = f" kl={val_metrics['kl']:.4f} |"
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"lr={current_lr:.1e} | "
            f"train={train_metrics['loss']:.4f} | "
            f"val={val_metrics['loss']:.4f} |{extra} "
            f"ssim={val_metrics['ssim']:.3f} | "
            f"{elapsed:.1f}s"
        )

        # Checkpoint
        ckpt_state = {
            "epoch": epoch,
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "patience_counter": patience_counter,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
        }
        if vae_heads is not None:
            ckpt_state["vae_heads"] = vae_heads.state_dict()
        torch.save(ckpt_state, ckpt_dir / f"checkpoint_epoch{epoch:03d}.pt")

        # Best model tracking
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            patience_counter = 0
            torch.save(encoder.state_dict(), output_dir / "best_encoder.pt")
            print(f"  → New best val_loss={best_val_loss:.4f}, saved best_encoder.pt")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"  → Early stopping after {args.patience} epochs without improvement.")
                break

        with open(output_dir / "training_log.json", "w") as f:
            json.dump(log, f, indent=2)

    writer.close()

    # Save final encoder
    torch.save(encoder.state_dict(), output_dir / "encoder.pt")
    print(f"\nDone. Best val_loss={best_val_loss:.4f}")
    print(f"Encoder saved to {output_dir / 'encoder.pt'}")
    print(f"Best encoder saved to {output_dir / 'best_encoder.pt'}")

    # Save config
    config = {
        "objective": args.objective,
        "reconstruct": args.reconstruct,
        "beta": args.beta if args.objective == "vae" else None,
        "data_dirs": args.data_dir,
        "prepared_dataset": args.prepared_dataset,
        "n_stack": args.n_stack,
        "frame_size": args.frame_size,
        "features_dim": args.features_dim,
        "epochs_completed": len(log),
        "best_val_loss": best_val_loss,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "seed": args.seed,
    }
    with open(output_dir / "pretrain_config.json", "w") as f:
        json.dump(config, f, indent=2)


if __name__ == "__main__":
    main()
