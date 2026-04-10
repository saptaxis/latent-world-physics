#!/usr/bin/env python3
"""Pre-train ImpalaCNN encoder on kinematic prediction from frame stacks.

Pure PyTorch — no SB3. Trains the encoder to predict [x, y, vx, vy, angle, ang_vel]
from stacked grayscale frames. The trained encoder weights can then be loaded into
SB3 PPO for policy training with frozen or fine-tuned features.

Usage (raw mode — slower, works at any scale):
    python lunar_lander/scripts/pretrain_encoder.py \
        --data-dir /path/to/random /path/to/heuristic \
        --output-dir /tmp/encoder-pretrain-run

Usage (prepared mode — faster, requires prepare_encoder_dataset.py first):
    python lunar_lander/scripts/pretrain_encoder.py \
        --prepared-dataset /path/to/prepared.npz \
        --output-dir /tmp/encoder-pretrain-run

The script saves:
    - encoder.pt: ImpalaCNN state_dict (no prediction head)
    - training_log.json: per-epoch loss and R² metrics
    - best_encoder.pt: best validation loss checkpoint
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
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add project root to path for imports.

from lwp.agents.encoder_dataset import EncoderPretrainDataset

# Names for the 6 kinematic targets, in order matching states[:6].
ALL_TARGET_NAMES = ["x", "y", "vx", "vy", "angle", "ang_vel"]

def parse_targets(target_str: str | None) -> tuple[list[str], list[int]]:
    """Parse --targets flag into names and indices.

    Args:
        target_str: Comma-separated target names (e.g. "x,y,vx,vy") or None for all.

    Returns:
        (names, indices) — e.g. (["x", "y"], [0, 1])
    """
    if target_str is None:
        return ALL_TARGET_NAMES, list(range(len(ALL_TARGET_NAMES)))
    names = [n.strip() for n in target_str.split(",")]
    indices = []
    for name in names:
        if name not in ALL_TARGET_NAMES:
            raise ValueError(f"Unknown target '{name}'. Valid: {ALL_TARGET_NAMES}")
        indices.append(ALL_TARGET_NAMES.index(name))
    return names, indices


class _ResidualBlock(nn.Module):
    """Residual block matching ImpalaCNN's _ResidualBlock exactly."""

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class _ConvSequence(nn.Module):
    """Conv sequence matching ImpalaCNN's _ConvSequence exactly."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.res1 = _ResidualBlock(out_channels)
        self.res2 = _ResidualBlock(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.pool(x)
        x = self.res1(x)
        x = self.res2(x)
        return x


class StandaloneImpalaCNN(nn.Module):
    """Standalone ImpalaCNN matching SB3 ImpalaCNN's architecture and state_dict keys.

    This is architecturally identical to lunar_lander.src.visual_backbones.ImpalaCNN
    but without the SB3 BaseFeaturesExtractor base class (which requires an
    observation_space). The module names (conv_sequences, relu, pool, flatten, linear)
    match exactly so state_dict keys are compatible for weight transfer.

    Architecture:
        Input: (batch, n_stack, frame_size, frame_size) float32
        → 3 conv sequences (channels [16, 32, 32] by default)
        → ReLU → AdaptiveAvgPool(pool_size) → Flatten
        → Linear(flat_size, features_dim) → ReLU
        Output: (batch, features_dim) float32
    """

    def __init__(
        self,
        in_channels: int = 4,
        features_dim: int = 256,
        channels: list[int] | None = None,
        pool_size: int = 4,
    ):
        super().__init__()
        channels = channels or [16, 32, 32]

        sequences = []
        in_ch = in_channels
        for out_ch in channels:
            sequences.append(_ConvSequence(in_ch, out_ch))
            in_ch = out_ch

        self.conv_sequences = nn.Sequential(*sequences)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        self.flatten = nn.Flatten()

        # Compute flattened size with dummy forward.
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 64, 64)  # any spatial size works
            n_flatten = self.flatten(self.pool(self.relu(self.conv_sequences(dummy)))).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

        self._features_dim = features_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_sequences(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        return self.linear(x)


class PredictionHead(nn.Module):
    """Simple MLP head: features_dim → 64 → n_targets.

    This head is disposable — only the encoder weights are saved.
    """

    def __init__(self, features_dim: int = 256, n_targets: int = 6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(features_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_targets),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_model(
    n_stack: int = 4,
    frame_size: int = 128,
    features_dim: int = 256,
    n_targets: int = 6,
    channels: list[int] | None = None,
    pool_size: int = 4,
) -> tuple[StandaloneImpalaCNN, PredictionHead]:
    """Create encoder + prediction head.

    Args:
        n_stack: Number of stacked frames (input channels).
        frame_size: Pixel resolution (not used in model, but documents intent).
        features_dim: Encoder output dimension.
        n_targets: Number of prediction targets (6 = kinematic state).
        channels: IMPALA channel widths (default [16, 32, 32]).
        pool_size: Adaptive pool spatial size (default 4).

    Returns:
        (encoder, head) tuple. Encoder outputs (batch, features_dim),
        head maps that to (batch, n_targets).
    """
    encoder = StandaloneImpalaCNN(
        in_channels=n_stack,
        features_dim=features_dim,
        channels=channels,
        pool_size=pool_size,
    )
    head = PredictionHead(features_dim=features_dim, n_targets=n_targets)
    return encoder, head


def train_one_epoch(
    encoder: nn.Module,
    head: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str = "cuda",
    target_indices: list[int] | None = None,
) -> float:
    """Train for one epoch. Returns mean MSE loss."""
    encoder.train()
    head.train()
    total_loss = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc="Train", leave=False)
    for frames, targets in pbar:
        frames = frames.to(device)
        if target_indices is not None:
            targets = targets[:, target_indices]
        targets = targets.to(device)

        features = encoder(frames)
        predictions = head(features)
        loss = nn.functional.mse_loss(predictions, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
        pbar.set_postfix(loss=f"{total_loss / n_batches:.4f}")

    return total_loss / max(n_batches, 1)


def evaluate(
    encoder: nn.Module,
    head: nn.Module,
    loader: DataLoader,
    device: str = "cuda",
    target_names: list[str] | None = None,
    target_indices: list[int] | None = None,
) -> dict:
    """Evaluate on a dataset. Returns loss and per-target R².

    R² (coefficient of determination) measures how well each target is
    predicted relative to just predicting the mean. R²=1.0 is perfect,
    R²=0.0 is no better than mean, R²<0 is worse than mean.

    Returns:
        {"loss": float, "r2": {"x": float, "y": float, ...}}
    """
    encoder.eval()
    head.eval()
    all_preds = []
    all_targets = []

    if target_names is None:
        target_names = ALL_TARGET_NAMES

    with torch.no_grad():
        for frames, targets in tqdm(loader, desc="Val", leave=False):
            frames = frames.to(device)
            if target_indices is not None:
                targets = targets[:, target_indices]
            features = encoder(frames)
            predictions = head(features)
            all_preds.append(predictions.cpu())
            all_targets.append(targets)

    preds = torch.cat(all_preds, dim=0).numpy()
    targets = torch.cat(all_targets, dim=0).numpy()

    # MSE loss
    mse = float(np.mean((preds - targets) ** 2))

    # Per-target R²
    r2 = {}
    for i, name in enumerate(target_names):
        ss_res = np.sum((targets[:, i] - preds[:, i]) ** 2)
        ss_tot = np.sum((targets[:, i] - np.mean(targets[:, i])) ** 2)
        r2[name] = float(1.0 - ss_res / max(ss_tot, 1e-8))

    return {"loss": mse, "r2": r2}


def main():
    parser = argparse.ArgumentParser(
        description="Pre-train ImpalaCNN encoder on kinematic prediction."
    )
    # Data source: exactly one of --data-dir or --prepared-dataset required.
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument(
        "--data-dir", type=str, nargs="+",
        help="Directory(ies) containing raw .npz episodes with rgb_frames (slower).",
    )
    data_group.add_argument(
        "--prepared-dataset", type=str,
        help="Path to prepared .npz from prepare_encoder_dataset.py (faster).",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Directory for encoder.pt, training_log.json, etc.",
    )
    parser.add_argument("--targets", type=str, default=None,
                        help="Comma-separated kinematic targets to predict. "
                             "Default: all 6 (x,y,vx,vy,angle,ang_vel). "
                             "E.g. --targets x,y for position-only.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n-stack", type=int, default=4)
    parser.add_argument("--frame-size", type=int, default=128)
    parser.add_argument("--features-dim", type=int, default=256)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=50,
                        help="Early stopping patience (epochs without val improvement).")
    parser.add_argument("--lr-patience", type=int, default=5,
                        help="ReduceLROnPlateau patience (epochs before halving LR).")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader workers. 0=main process only (safe for prepared mode). "
                             "Use 4+ for raw mode to parallelize disk I/O.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest checkpoint in output-dir. "
                             "Restores encoder, head, optimizer, scheduler, epoch count.")
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

    # --- Parse target selection ---
    target_names, target_indices = parse_targets(args.targets)
    n_targets = len(target_names)
    print(f"Targets ({n_targets}): {target_names}")

    # --- Data ---
    # Two modes: raw (on-the-fly resize from per-episode files) or
    # prepared (single pre-processed file, loaded into RAM).
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

    # num_workers guard:
    # - .npz prepared mode: all frames in RAM (~19 GB), forking duplicates them.
    # - Raw mode: each worker forks its own LRU cache. 64 cached episodes ×
    #   ~144 MB/episode = ~9 GB per worker. 4 workers = 36+ GB in caches.
    # - mmap'd .npy directory mode: fork shares mmap pages read-only, no
    #   duplication. Workers help parallelize HDD random I/O. Safe to use > 0.
    is_mmap_mode = (
        args.prepared_dataset is not None
        and Path(args.prepared_dataset).is_dir()
    )
    if args.num_workers > 0 and not is_mmap_mode:
        print(f"WARNING: num_workers={args.num_workers} can cause OOM. "
              f"Each worker caches episodes independently (~9 GB/worker for raw mode, "
              f"~19 GB/worker for prepared .npz mode). Using num_workers=0 instead.")
        args.num_workers = 0
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # --- Model ---
    encoder, head = build_model(
        n_stack=args.n_stack, frame_size=args.frame_size,
        features_dim=args.features_dim, n_targets=n_targets,
    )
    encoder = encoder.to(device)
    head = head.to(device)

    n_params_enc = sum(p.numel() for p in encoder.parameters())
    n_params_head = sum(p.numel() for p in head.parameters())
    print(f"Encoder: {n_params_enc:,} params, Head: {n_params_head:,} params")

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(head.parameters()), lr=args.lr,
    )

    # LR scheduler: halve LR when val_loss plateaus for lr_patience epochs.
    # This dampens oscillations and lets the model converge tighter.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=args.lr_patience,
    )

    # --- TensorBoard ---
    writer = SummaryWriter(log_dir=str(output_dir / "tb"))

    # --- Checkpoints dir for per-epoch saves ---
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # --- Resume from checkpoint ---
    start_epoch = 1
    best_val_loss = float("inf")
    patience_counter = 0
    log = []

    if args.resume:
        # Find latest checkpoint in checkpoints/ dir.
        ckpt_files = sorted(ckpt_dir.glob("checkpoint_epoch*.pt"))
        if ckpt_files:
            latest_ckpt = ckpt_files[-1]
            print(f"Resuming from {latest_ckpt}...")
            ckpt = torch.load(str(latest_ckpt), map_location=device, weights_only=False)
            encoder.load_state_dict(ckpt["encoder"])
            head.load_state_dict(ckpt["head"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            start_epoch = ckpt["epoch"] + 1
            best_val_loss = ckpt["best_val_loss"]
            patience_counter = ckpt["patience_counter"]
            print(f"  Resumed at epoch {start_epoch}, best_val_loss={best_val_loss:.4f}, "
                  f"lr={optimizer.param_groups[0]['lr']:.1e}")

            # Reload existing log if present.
            log_path = output_dir / "training_log.json"
            if log_path.exists():
                with open(log_path) as f:
                    log = json.load(f)
                # Trim log to match resume point (in case of partial writes).
                log = [e for e in log if e["epoch"] < start_epoch]
        else:
            print("WARNING: --resume but no checkpoints found. Starting from scratch.")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        current_lr = optimizer.param_groups[0]["lr"]
        train_loss = train_one_epoch(encoder, head, train_loader, optimizer, device,
                                     target_indices=target_indices)
        val_metrics = evaluate(encoder, head, val_loader, device,
                               target_names=target_names, target_indices=target_indices)
        elapsed = time.time() - t0

        # Step the LR scheduler based on val loss.
        scheduler.step(val_metrics["loss"])

        entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_r2": val_metrics["r2"],
            "lr": current_lr,
            "elapsed_s": round(elapsed, 1),
        }
        log.append(entry)

        # TensorBoard logging
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_metrics["loss"], epoch)
        for name, val in val_metrics["r2"].items():
            writer.add_scalar(f"r2/{name}", val, epoch)
        writer.add_scalar("lr", current_lr, epoch)
        writer.add_scalar("timing/epoch_s", elapsed, epoch)

        # Pretty-print R² for key targets
        r2 = val_metrics["r2"]
        r2_str = " ".join(f"{k}={v:.3f}" for k, v in r2.items())
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"lr={current_lr:.1e} | "
            f"train={train_loss:.4f} | "
            f"val={val_metrics['loss']:.4f} | "
            f"R²: {r2_str} | "
            f"{elapsed:.1f}s"
        )

        # Save full checkpoint (encoder, head, optimizer, scheduler state)
        # so we can resume training OR pick any epoch's encoder later.
        torch.save({
            "epoch": epoch,
            "encoder": encoder.state_dict(),
            "head": head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "patience_counter": patience_counter,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
        }, ckpt_dir / f"checkpoint_epoch{epoch:03d}.pt")

        # Track best model
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

        # Save log after each epoch (overwrite) so progress is visible.
        with open(output_dir / "training_log.json", "w") as f:
            json.dump(log, f, indent=2)

    writer.close()

    # Save final encoder (may differ from best)
    torch.save(encoder.state_dict(), output_dir / "encoder.pt")
    print(f"\nDone. Best val_loss={best_val_loss:.4f}")
    print(f"Encoder saved to {output_dir / 'encoder.pt'}")
    print(f"Best encoder saved to {output_dir / 'best_encoder.pt'}")

    # Save training config for reproducibility
    config = {
        "targets": target_names,
        "target_indices": target_indices,
        "data_dirs": args.data_dir,
        "n_stack": args.n_stack,
        "frame_size": args.frame_size,
        "features_dim": args.features_dim,
        "epochs_completed": len(log),
        "best_val_loss": best_val_loss,
        "final_val_r2": log[-1]["val_r2"],
        "lr": args.lr,
        "batch_size": args.batch_size,
        "seed": args.seed,
    }
    with open(output_dir / "pretrain_config.json", "w") as f:
        json.dump(config, f, indent=2)


if __name__ == "__main__":
    main()
