#!/usr/bin/env python3
"""Run all kinematic target ablation pretraining in one process.

Loads the prepared dataset ONCE into RAM, then trains 5 encoders
sequentially with different target subsets. Avoids loading 19 GB
multiple times.

Usage:
    python lunar_lander/scripts/run_kinematic_ablation.py \
        --prepared-dataset /path/to/prepared.npz \
        --output-base /path/to/encoder-pretrain \
        --device cuda:1
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


from lwp.agents.encoder_dataset import PreparedEncoderDataset
from scripts.perception.pretrain_encoder import (
    StandaloneImpalaCNN,
    PredictionHead,
    build_model,
    train_one_epoch,
    evaluate,
    parse_targets,
)


# The 5 ablation conditions (full kinematics already done as run1).
ABLATIONS = [
    ("pos-only",    "x,y"),
    ("vel-only",    "vx,vy"),
    ("angle-only",  "angle,ang_vel"),
    ("pos-vel",     "x,y,vx,vy"),
    ("no-angvel",   "x,y,vx,vy,angle"),
]


def train_ablation(
    train_ds: PreparedEncoderDataset,
    val_ds: PreparedEncoderDataset,
    output_dir: Path,
    target_names: list[str],
    target_indices: list[int],
    epochs: int,
    patience: int,
    lr_patience: int,
    batch_size: int,
    lr: float,
    device: str,
    n_stack: int,
    features_dim: int,
):
    """Train one ablation condition. Saves encoder to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    n_targets = len(target_names)

    print(f"\n{'='*60}")
    print(f"Training: {output_dir.name}")
    print(f"Targets ({n_targets}): {target_names}")
    print(f"{'='*60}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)

    encoder, head = build_model(
        n_stack=n_stack, frame_size=128,
        features_dim=features_dim, n_targets=n_targets,
    )
    encoder = encoder.to(device)
    head = head.to(device)

    n_enc = sum(p.numel() for p in encoder.parameters())
    n_head = sum(p.numel() for p in head.parameters())
    print(f"Encoder: {n_enc:,} params, Head: {n_head:,} params")

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(head.parameters()), lr=lr,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=lr_patience,
    )

    writer = SummaryWriter(log_dir=str(output_dir / "tb"))
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    patience_counter = 0
    log = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        current_lr = optimizer.param_groups[0]["lr"]
        train_loss = train_one_epoch(encoder, head, train_loader, optimizer,
                                     device, target_indices=target_indices)
        val_metrics = evaluate(encoder, head, val_loader, device,
                               target_names=target_names,
                               target_indices=target_indices)
        elapsed = time.time() - t0

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

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_metrics["loss"], epoch)
        for name, val in val_metrics["r2"].items():
            writer.add_scalar(f"r2/{name}", val, epoch)
        writer.add_scalar("lr", current_lr, epoch)

        r2_str = " ".join(f"{k}={v:.3f}" for k, v in val_metrics["r2"].items())
        print(f"  Epoch {epoch:3d}/{epochs} | lr={current_lr:.1e} | "
              f"train={train_loss:.4f} | val={val_metrics['loss']:.4f} | "
              f"R²: {r2_str} | {elapsed:.1f}s")

        # Checkpoint
        torch.save({
            "epoch": epoch,
            "encoder": encoder.state_dict(),
            "head": head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "patience_counter": patience_counter,
        }, ckpt_dir / f"checkpoint_epoch{epoch:03d}.pt")

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            patience_counter = 0
            torch.save(encoder.state_dict(), output_dir / "best_encoder.pt")
            print(f"    → New best val_loss={best_val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"    → Early stopping after {patience} epochs.")
                break

        with open(output_dir / "training_log.json", "w") as f:
            json.dump(log, f, indent=2)

    writer.close()
    torch.save(encoder.state_dict(), output_dir / "encoder.pt")

    config = {
        "targets": target_names,
        "target_indices": target_indices,
        "epochs_completed": len(log),
        "best_val_loss": best_val_loss,
        "final_val_r2": log[-1]["val_r2"],
        "lr": lr,
        "batch_size": batch_size,
        "features_dim": features_dim,
    }
    with open(output_dir / "pretrain_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"  Done. Best val_loss={best_val_loss:.4f}")
    return best_val_loss


def main():
    parser = argparse.ArgumentParser(
        description="Run all kinematic target ablation pretraining in one process."
    )
    parser.add_argument("--prepared-dataset", type=str, required=True,
                        help="Path to prepared .npz (loaded once, shared across all ablations).")
    parser.add_argument("--output-base", type=str, required=True,
                        help="Base dir. Each ablation creates a subdir (e.g. ablation-pos-only/).")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--lr-patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n-stack", type=int, default=4)
    parser.add_argument("--features-dim", type=int, default=256)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--only", type=str, default=None,
                        help="Run only this ablation (e.g. 'pos-only'). Default: all 5.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = args.device if torch.cuda.is_available() else "cpu"
    output_base = Path(args.output_base)

    # Load dataset ONCE.
    print(f"Loading prepared dataset from {args.prepared_dataset}...")
    train_ds = PreparedEncoderDataset(
        args.prepared_dataset, n_stack=args.n_stack,
        split="train", val_fraction=args.val_fraction, seed=args.seed,
    )
    val_ds = PreparedEncoderDataset(
        args.prepared_dataset, n_stack=args.n_stack,
        split="val", val_fraction=args.val_fraction, seed=args.seed,
    )
    print(f"Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")

    # Filter ablations if --only is specified.
    ablations = ABLATIONS
    if args.only:
        ablations = [(n, t) for n, t in ABLATIONS if n == args.only]
        if not ablations:
            valid = [n for n, _ in ABLATIONS]
            print(f"ERROR: Unknown ablation '{args.only}'. Valid: {valid}")
            sys.exit(1)

    # Run each ablation sequentially.
    results = {}
    for name, target_str in ablations:
        target_names, target_indices = parse_targets(target_str)
        best = train_ablation(
            train_ds=train_ds,
            val_ds=val_ds,
            output_dir=output_base / f"ablation-{name}",
            target_names=target_names,
            target_indices=target_indices,
            epochs=args.epochs,
            patience=args.patience,
            lr_patience=args.lr_patience,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
            n_stack=args.n_stack,
            features_dim=args.features_dim,
        )
        results[name] = best

    print(f"\n{'='*60}")
    print("All ablations complete:")
    for name, best_loss in results.items():
        print(f"  {name:<15s}: best_val_loss={best_loss:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
