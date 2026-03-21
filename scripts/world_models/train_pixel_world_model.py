#!/usr/bin/env python3
"""Train a complete pixel world model (VAE + dynamics).

Supports three training modes:
  staged:   Train VAE (Phase 1), then dynamics with frozen VAE (Phase 2)
  joint:    Train VAE + dynamics end-to-end simultaneously
  finetune: Staged first, then unfreeze VAE and fine-tune jointly

Usage:
    python scripts/train_pixel_world_model.py \
        --data-path /path/to/episodes \
        --run-dir runs/pixel-wm \
        --mode staged
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import torch


def parse_args():
    p = argparse.ArgumentParser(description="Train pixel world model")
    p.add_argument("--data-path", type=str, required=True)
    p.add_argument("--run-dir", type=str, default="runs/pixel-wm")
    p.add_argument("--mode", type=str, default="staged",
                   choices=["staged", "joint", "finetune"])
    # VAE args
    p.add_argument("--frame-size", type=int, default=84)
    p.add_argument("--latent-dim", type=int, default=64)
    p.add_argument("--in-channels", type=int, default=1)
    p.add_argument("--channels", type=int, nargs="+", default=[32, 64, 128, 256])
    p.add_argument("--beta", type=float, default=0.0001)
    p.add_argument("--vae-epochs", type=int, default=50)
    p.add_argument("--vae-lr", type=float, default=3e-4)
    # Dynamics args
    p.add_argument("--hidden-size", type=int, default=256)
    p.add_argument("--action-dim", type=int, default=2)
    p.add_argument("--seq-len", type=int, default=20)
    p.add_argument("--frame-stack", type=int, default=1)
    p.add_argument("--dynamics-epochs", type=int, default=100)
    p.add_argument("--dynamics-lr", type=float, default=3e-4)
    p.add_argument("--sampling-start", type=float, default=0.0)
    p.add_argument("--sampling-end", type=float, default=0.5)
    p.add_argument("--sampling-warmup-frac", type=float, default=0.5)
    # Fine-tune args
    p.add_argument("--finetune-epochs", type=int, default=25)
    p.add_argument("--finetune-lr", type=float, default=3e-5)
    # Common
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def run_staged(args):
    """Run Phase 1 (VAE) then Phase 2 (dynamics)."""
    scripts_dir = Path(__file__).parent
    python = sys.executable

    # Phase 1: VAE
    print("=" * 60)
    print("PHASE 1: Training VAE")
    print("=" * 60)
    vae_cmd = [
        python, str(scripts_dir / "train_pixel_vae.py"),
        "--data-path", args.data_path,
        "--run-dir", args.run_dir,
        "--frame-size", str(args.frame_size),
        "--latent-dim", str(args.latent_dim),
        "--in-channels", str(args.in_channels),
        "--channels", *[str(c) for c in args.channels],
        "--beta", str(args.beta),
        "--lr", str(args.vae_lr),
        "--batch-size", str(args.batch_size),
        "--epochs", str(args.vae_epochs),
        "--device", args.device,
    ]
    result = subprocess.run(vae_cmd)
    if result.returncode != 0:
        print("VAE training failed!")
        sys.exit(1)

    # Phase 2: Dynamics
    vae_ckpt = Path(args.run_dir) / "vae" / "best.pt"
    print("\n" + "=" * 60)
    print("PHASE 2: Training Dynamics (VAE frozen)")
    print("=" * 60)
    dyn_cmd = [
        python, str(scripts_dir / "train_pixel_dynamics.py"),
        "--vae-checkpoint", str(vae_ckpt),
        "--data-path", args.data_path,
        "--run-dir", args.run_dir,
        "--hidden-size", str(args.hidden_size),
        "--action-dim", str(args.action_dim),
        "--seq-len", str(args.seq_len),
        "--frame-stack", str(args.frame_stack),
        "--sampling-start", str(args.sampling_start),
        "--sampling-end", str(args.sampling_end),
        "--sampling-warmup-frac", str(args.sampling_warmup_frac),
        "--lr", str(args.dynamics_lr),
        "--batch-size", str(args.batch_size),
        "--epochs", str(args.dynamics_epochs),
        "--grad-clip", str(args.grad_clip),
        "--device", args.device,
    ]
    result = subprocess.run(dyn_cmd)
    if result.returncode != 0:
        print("Dynamics training failed!")
        sys.exit(1)

    print("\nStaged training complete!")


def main():
    args = parse_args()

    if args.mode == "staged":
        run_staged(args)
    elif args.mode == "joint":
        print("Joint training mode — not yet implemented.")
        print("Use staged mode for now.")
        sys.exit(1)
    elif args.mode == "finetune":
        print("Running staged first, then fine-tune...")
        run_staged(args)
        print("\nFine-tune phase — not yet implemented.")
        print("Staged training completed. Fine-tune deferred.")
    else:
        print(f"Unknown mode: {args.mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
