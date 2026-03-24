#!/usr/bin/env python3
"""Train latent dynamics model (Phase 2 of pixel world model).

Loads a frozen VAE from checkpoint, trains either a GRU or RSSM dynamics
model in latent space.  Supports multiple training modes:
  - latent_mse: single-step teacher-forced MSE (GRU only)
  - multi_step_latent: k-step autoregressive rollout + MSE (GRU or RSSM)
  - latent_elbo: posterior-guided ELBO with KL (RSSM only)

Usage:
    # GRU with single-step MSE (original default)
    python scripts/train_pixel_dynamics.py \
        --vae-checkpoint path/to/vae/best.pt \
        --data-path /path/to/episodes \
        --run-dir runs/pixel-wm

    # RSSM with ELBO training
    python scripts/train_pixel_dynamics.py \
        --model-type rssm --training-mode latent_elbo \
        --vae-checkpoint path/to/vae/best.pt \
        --data-path /path/to/episodes \
        --run-dir runs/pixel-rssm
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from lwp.data.pixel_dataset import PixelEpisodeDataset
from lwp.models.pixel_vae import PixelVAE
from lwp.models.pixel_dynamics import LatentDynamicsModel, FiLMDynamicsModel
from lwp.models.pixel_rssm import LatentRSSM
from lwp.training.callbacks import (
    CallbackContext,
    CheckpointCallback,
    GradNormCallback,
    NaNDetectionCallback,
    ProgressCallback,
)
from lwp.training.pixel_callbacks import (
    PixelDynamicsValidationCallback,
    DreamGridCallback,
    KinematicsValidationCallback,
    DreamComparisonVideoCallback,
    RSSMDiagnosticCallback,
)
from lwp.training.pixel_loop import pixel_dynamics_train_epoch

# Valid (model_type, training_mode) combinations.  Not every loss function
# makes sense for every architecture:
#   - latent_mse requires teacher-forced predict_sequence (GRU only)
#   - multi_step_latent works with any model that supports rollout()
#   - latent_elbo requires posterior inference, which only RSSM provides
VALID_COMBOS = {
    ("gru", "latent_mse"),
    ("gru", "multi_step_latent"),
    ("film", "latent_mse"),
    ("film", "multi_step_latent"),
    ("rssm", "multi_step_latent"),
    ("rssm", "latent_elbo"),
    ("factored-dyn", "latent_mse"),
    ("factored-dyn", "multi_step_latent"),
}


def load_vae(checkpoint_path: str, device: str) -> PixelVAE:
    """Load VAE from checkpoint, dispatching on model_type for FactoredPixelVAE.

    Reads the 'config' dict from the checkpoint to determine architecture.
    Standard VAE: creates PixelVAE. Factored VAE: creates FactoredPixelVAE
    with kin_targets and decoder_type from config.
    """
    # weights_only=False because we stored config as a plain dict alongside
    # the state_dict — torch.load needs to unpickle it.
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model_type = cfg.get("model_type", "standard")

    if model_type == "factored":
        # Factored VAE with [z_kin, z_ctx] split — needs kin_targets and
        # decoder_type from config to reconstruct the architecture.
        from lwp.models.factored_pixel_vae import FactoredPixelVAE
        vae = FactoredPixelVAE(
            in_channels=cfg["in_channels"],
            latent_dim=cfg["latent_dim"],
            frame_size=cfg["frame_size"],
            channels=cfg.get("channels", [32, 64, 128, 256]),
            kin_targets=cfg.get("kin_targets", [0, 1, 2, 3, 4, 5]),
            decoder_type=cfg.get("decoder_type", "concat"),
            coord_conv=cfg.get("coord_conv", False),
        )
    elif model_type == "compositional":
        from lwp.models.compositional_vae import CompositionalPixelVAE
        vae = CompositionalPixelVAE(
            in_channels=cfg.get("in_channels", 1),
            latent_dim=cfg["latent_dim"],
            frame_size=cfg["frame_size"],
            latent_mode=cfg.get("latent_mode", "flat"),
            bg_dim=cfg.get("bg_dim", 8),
            obj_dim=cfg.get("obj_dim", 32),
            canonical_size=cfg.get("canonical_size", 16),
            beta=cfg.get("beta", 0.0001),
        )
    else:
        # Standard PixelVAE with optional MLP state head
        vae = PixelVAE(
            in_channels=cfg["in_channels"],
            latent_dim=cfg["latent_dim"],
            frame_size=cfg["frame_size"],
            channels=cfg.get("channels", [32, 64, 128, 256]),
            state_dim=cfg.get("state_dim", 0),
            coord_conv=cfg.get("coord_conv", False),
        )

    vae.load_state_dict(ckpt["model_state_dict"])
    vae.to(device)
    # Freeze the VAE entirely — Phase 2 only trains the dynamics model.
    # eval() disables dropout/batchnorm updates; requires_grad_(False)
    # excludes VAE params from the optimizer and saves backward-pass memory.
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae


def get_sampling_prob(epoch: int, total_epochs: int,
                      start: float, end: float, warmup_frac: float) -> float:
    """Linear annealing of scheduled sampling probability.

    During warmup, use only teacher forcing (sampling_prob = start, typically 0).
    After warmup, linearly increase to `end` — this gradually forces the
    dynamics model to consume its own predictions rather than ground-truth
    latents, reducing the train/dream distribution gap.
    """
    warmup_epochs = int(total_epochs * warmup_frac)
    if epoch < warmup_epochs:
        return start
    progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
    return start + (end - start) * min(progress, 1.0)


def _collect_val_episode_paths(data_paths: list[str], n_detail: int = 5) -> list[str]:
    """Select a few representative val-split npz files for detailed callbacks.

    Uses the same deterministic RNG split logic as PixelEpisodeDataset so we
    pick from the validation portion.  Returns up to `n_detail` paths.
    """
    # Gather all raw npz files (skip any preprocessed cache files)
    all_npz: list[Path] = []
    for dp in data_paths:
        for root, _dirs, files in os.walk(str(Path(dp)), followlinks=True):
            for f in sorted(files):
                if f.endswith(".npz") and "prepared" not in f:
                    all_npz.append(Path(root) / f)
    all_npz.sort()

    if not all_npz:
        return []

    # Mirror the 90/10 train/val split from PixelEpisodeDataset (seed=0).
    # Must use the same RNG + seed so we pick episodes that are actually
    # in the validation set — otherwise we'd leak training data into the
    # per-episode diagnostic callbacks.
    rng = np.random.RandomState(0)
    n_val = max(1, int(len(all_npz) * 0.1))
    val_indices = rng.permutation(len(all_npz))[:n_val]
    val_episode_paths = [str(all_npz[i]) for i in sorted(val_indices)]

    # Return a manageable subset — running detailed callbacks (dream videos,
    # kinematics plots) on all val episodes would be too slow.
    return val_episode_paths[:min(n_detail, len(val_episode_paths))]


def parse_args():
    p = argparse.ArgumentParser(description="Train latent dynamics (GRU or RSSM)")
    p.add_argument("--vae-checkpoint", type=str, required=True)
    p.add_argument("--data-path", type=str, nargs="+", required=True,
                   help="One or more directories with .npz episode files")
    p.add_argument("--run-dir", type=str, default="runs/pixel-wm")

    # --- Model architecture selection ---
    p.add_argument("--model-type", type=str, default="gru",
                   choices=["gru", "film", "rssm", "factored-dyn"],
                   help="Dynamics model architecture: gru (concat), film (FiLM conditioning), rssm")
    p.add_argument("--training-mode", type=str, default="latent_mse",
                   choices=["latent_mse", "multi_step_latent", "latent_elbo"],
                   help="Loss function / training regime (default: latent_mse)")

    # --- Model hyperparameters ---
    p.add_argument("--hidden-size", type=int, default=256,
                   help="GRU hidden size or RSSM hidden_dim")
    p.add_argument("--action-dim", type=int, default=2)
    p.add_argument("--deter-dim", type=int, default=200,
                   help="RSSM deterministic state dimension (ignored for GRU)")
    p.add_argument("--stoch-dim", type=int, default=30,
                   help="RSSM stochastic state dimension (ignored for GRU)")

    # --- Training-mode specific hyperparameters ---
    p.add_argument("--rollout-k", type=int, default=1,
                   help="Multi-step rollout horizon for multi_step_latent mode")
    p.add_argument("--multi-step-weight", type=float, default=1.0,
                   help="Weight on multi-step loss term")
    p.add_argument("--kl-weight", type=float, default=1.0,
                   help="KL divergence weight for latent_elbo mode")
    p.add_argument("--free-bits", type=float, default=1.0,
                   help="Minimum KL per stochastic dim (nats). Prevents posterior "
                        "collapse. Dreamer uses 1.0. Set 0 to disable.")
    p.add_argument("--kin-weight", type=float, default=1.0,
                   help="Upweight factor for z[0:kin_dims] in dynamics loss. "
                        "Only valid with factored VAE (z_kin dims are known). "
                        "1.0 = uniform (default). 10 = 10x weight on kinematics.")
    p.add_argument("--kin-dims", type=int, default=6,
                   help="Number of leading z dims to upweight. Must match "
                        "factored VAE's len(kin_targets). Ignored if kin_weight=1.")

    # --- Sequence / data ---
    p.add_argument("--seq-len", type=int, default=20)
    p.add_argument("--frame-stack", type=int, default=1)

    # --- Optimisation ---
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--val-every", type=int, default=500)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--ckpt-every", type=int, default=2000)
    p.add_argument("--sampling-start", type=float, default=0.0)
    p.add_argument("--sampling-end", type=float, default=0.5)
    p.add_argument("--sampling-warmup-frac", type=float, default=0.5)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num-workers", type=int, default=4,
                   help="DataLoader workers for batch prefetching")
    p.add_argument("--load-workers", type=int, default=8,
                   help="Parallel workers for initial npz loading")
    p.add_argument("--cache-dir", type=str, default=None,
                   help="Dir to cache preprocessed episodes. Instant load on reruns.")
    p.add_argument("--lr-patience", type=int, default=0,
                   help="ReduceLROnPlateau patience (0 = no scheduler)")
    p.add_argument("--lr-factor", type=float, default=0.5,
                   help="LR reduction factor when plateau detected")
    p.add_argument("--lr-min", type=float, default=1e-6,
                   help="Minimum LR for scheduler")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to checkpoint (.pt) to resume from. Restores model weights, "
                        "optimizer state, epoch, and global_step.")

    args = p.parse_args()

    # --- Validate model/mode combination ---
    # Not every loss function works with every architecture (see VALID_COMBOS).
    if (args.model_type, args.training_mode) not in VALID_COMBOS:
        p.error(f"Invalid combination: --model-type {args.model_type} "
                f"with --training-mode {args.training_mode}")

    return args


def main():
    args = parse_args()

    # --- Load frozen VAE ---
    # The VAE was trained in Phase 1 (train_pixel_vae.py). We load it frozen
    # to provide encode/decode for the dynamics model. Extracting frame_size,
    # latent_dim, and grayscale from the VAE config ensures consistency
    # between phases — no risk of a mismatch if the user forgets a flag.
    print(f"Loading VAE from {args.vae_checkpoint} ...")
    vae = load_vae(args.vae_checkpoint, args.device)
    vae_ckpt = torch.load(args.vae_checkpoint, map_location=args.device, weights_only=False)
    vae_cfg = vae_ckpt["config"]
    frame_size = vae_cfg["frame_size"]
    latent_dim = vae_cfg["latent_dim"]
    grayscale = vae_cfg["in_channels"] == 1
    print(f"  VAE: latent_dim={latent_dim}, frame_size={frame_size}")

    # Compositional VAE constraints on dynamics model type
    if hasattr(vae, 'latent_mode'):
        if vae.latent_mode == 'split' and args.model_type != 'factored-dyn':
            print(f"ERROR: Split-latent compositional VAE requires --model-type factored-dyn, "
                  f"got '{args.model_type}'")
            sys.exit(1)
        if vae.latent_mode == 'split':
            # Override kin_dims to match pose_dim (5)
            args.kin_dims = vae.pose_dim
            print(f"  Compositional split mode: kin_dims={args.kin_dims} (pose)")

    # Warn if kin-weight is used without a factored VAE — the first kin_dims
    # z dimensions may not correspond to kinematics, making the weighting
    # unprincipled (but not an error — user may know what they're doing).
    if args.kin_weight != 1.0:
        vae_model_type = vae_cfg.get("model_type", "standard")
        if vae_model_type != "factored":
            print(f"WARNING: --kin-weight={args.kin_weight} but VAE is not factored "
                  f"(model_type={vae_model_type}). First {args.kin_dims} z dims "
                  f"may not be kinematics — weighting is unprincipled.")

    # Directories -- run_dir IS the dynamics run dir (not a parent)
    dyn_dir = Path(args.run_dir)
    dyn_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = str(dyn_dir)

    # Dump full config so we know exactly what this run was trained with.
    # Includes model_type and training_mode so downstream tools (eval, export)
    # know which architecture to instantiate.
    run_config = {
        "model_type": args.model_type,
        "training_mode": args.training_mode,
        "vae_checkpoint": str(Path(args.vae_checkpoint).resolve()),
        "vae_config": vae_cfg,
        "data_path": args.data_path,
        "hidden_size": args.hidden_size,
        "action_dim": args.action_dim,
        "deter_dim": args.deter_dim,
        "stoch_dim": args.stoch_dim,
        "seq_len": args.seq_len,
        "frame_stack": args.frame_stack,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "grad_clip": args.grad_clip,
        "rollout_k": args.rollout_k,
        "multi_step_weight": args.multi_step_weight,
        "kl_weight": args.kl_weight,
        "free_bits": args.free_bits,
        "kin_weight": args.kin_weight,
        "kin_dims": args.kin_dims,
        "sampling_start": args.sampling_start,
        "sampling_end": args.sampling_end,
        "sampling_warmup_frac": args.sampling_warmup_frac,
        "device": args.device,
    }
    with open(dyn_dir / "config.json", "w") as f:
        json.dump(run_config, f, indent=2)
    print(f"Config saved to {dyn_dir / 'config.json'}")

    # Data
    print(f"Loading data from {args.data_path} ...")
    cache_train = None
    cache_val = None
    if args.cache_dir:
        cache_dir = Path(args.cache_dir)
        gs_tag = "gray" if grayscale else "rgb"
        cache_train = cache_dir / f"episodes_train_{frame_size}_{gs_tag}.npz"
        cache_val = cache_dir / f"episodes_val_{frame_size}_{gs_tag}.npz"

    train_ds = PixelEpisodeDataset(
        args.data_path, frame_size=frame_size, grayscale=grayscale,
        seq_len=args.seq_len, frame_stack=args.frame_stack, split="train",
        n_workers=args.load_workers, cache_path=cache_train,
    )
    val_ds = PixelEpisodeDataset(
        args.data_path, frame_size=frame_size, grayscale=grayscale,
        seq_len=args.seq_len, frame_stack=args.frame_stack, split="val",
        n_workers=args.load_workers, cache_path=cache_val,
    )
    print(f"  Train: {len(train_ds)} windows, Val: {len(val_ds)} windows")

    # shuffle=True for training ensures the model sees windows from different
    # episodes in each batch — prevents temporal correlation within batches
    # that could bias gradient estimates.
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    # --- Model construction: dispatch on --model-type ---
    if args.model_type == "gru":
        dynamics = LatentDynamicsModel(
            latent_dim=latent_dim,
            action_dim=args.action_dim,
            hidden_size=args.hidden_size,
        ).to(args.device)
        model_label = "LatentDynamicsModel (GRU)"
    elif args.model_type == "film":
        dynamics = FiLMDynamicsModel(
            latent_dim=latent_dim,
            action_dim=args.action_dim,
            hidden_size=args.hidden_size,
        ).to(args.device)
        model_label = "FiLMDynamicsModel (GRU+FiLM)"
    elif args.model_type == "rssm":
        dynamics = LatentRSSM(
            latent_dim=latent_dim,
            action_dim=args.action_dim,
            deter_dim=args.deter_dim,
            stoch_dim=args.stoch_dim,
            hidden_dim=args.hidden_size,
        ).to(args.device)
        model_label = "LatentRSSM"
    elif args.model_type == "factored-dyn":
        from lwp.models.factored_dynamics import FactoredDynamicsModel
        dynamics = FactoredDynamicsModel(
            latent_dim=latent_dim,
            action_dim=args.action_dim,
            hidden_size=args.hidden_size,
            kin_dims=args.kin_dims,
        ).to(args.device)
        model_label = "FactoredDynamicsModel (shared GRU, split heads)"
    else:
        # Should be unreachable due to argparse choices, but be safe
        raise ValueError(f"Unknown model type: {args.model_type}")

    param_count = sum(p.numel() for p in dynamics.parameters())
    print(f"{model_label}: {param_count:,} parameters")
    print(f"  training_mode={args.training_mode}, rollout_k={args.rollout_k}, "
          f"kl_weight={args.kl_weight}, ms_weight={args.multi_step_weight}")

    optimizer = torch.optim.Adam(dynamics.parameters(), lr=args.lr)

    # --- Resume from checkpoint ---
    # Restores model weights, optimizer state (including Adam momentum buffers
    # and per-param LR), epoch counter, and global_step.
    start_epoch = 0
    start_step = 0
    if args.resume:
        print(f"Resuming from {args.resume} ...")
        ckpt = torch.load(args.resume, map_location=args.device, weights_only=False)
        dynamics.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0)
        start_step = ckpt.get("global_step", 0)
        print(f"  Restored epoch={start_epoch}, global_step={start_step}")

    # LR scheduler (optional -- only if --lr-patience > 0)
    # Created AFTER resume so it wraps the restored optimizer state.
    scheduler = None
    if args.lr_patience > 0:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=args.lr_factor,
            patience=args.lr_patience, min_lr=args.lr_min,
        )
        print(f"LR scheduler: ReduceLROnPlateau(patience={args.lr_patience}, "
              f"factor={args.lr_factor}, min_lr={args.lr_min})")

    writer = SummaryWriter(log_dir=str(dyn_dir / "tb"))

    # Config dict stored in checkpoints -- includes model_type and training_mode
    # so we can reconstruct the correct architecture at load time
    config = {
        "model_type": args.model_type,
        "training_mode": args.training_mode,
        "hidden_size": args.hidden_size,
        "action_dim": args.action_dim,
        "deter_dim": args.deter_dim,
        "stoch_dim": args.stoch_dim,
        "seq_len": args.seq_len,
        "frame_stack": args.frame_stack,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "rollout_k": args.rollout_k,
        "multi_step_weight": args.multi_step_weight,
        "kl_weight": args.kl_weight,
        "kin_weight": args.kin_weight,
        "kin_dims": args.kin_dims,
        "vae_checkpoint": args.vae_checkpoint,
    }

    # --- Select a few val episodes for detailed per-episode callbacks ---
    # These callbacks (KinematicsValidation, DreamComparisonVideo, RSSMDiagnostic)
    # need raw npz file paths, not DataLoader batches.
    detail_paths = _collect_val_episode_paths(args.data_path, n_detail=5)
    if detail_paths:
        print(f"  Selected {len(detail_paths)} val episodes for detailed callbacks")

    # --- Construct callbacks ---
    callbacks = [
        # Core validation + early stopping -- uses the same loss function as
        # the training loop so val loss is directly comparable
        PixelDynamicsValidationCallback(
            val_loader=val_loader, vae=vae,
            every_n_steps=args.val_every, patience=args.patience,
            checkpoint_dir=ckpt_dir,
            training_mode=args.training_mode,
            rollout_k=args.rollout_k,
            kl_weight=args.kl_weight,
        ),
        CheckpointCallback(checkpoint_dir=ckpt_dir, every_n_steps=args.ckpt_every),
        # DreamGridCallback now takes val_dataset for random episode sampling
        DreamGridCallback(vae=vae, val_dataset=val_ds,
                          every_n_steps=args.val_every * 2),
        GradNormCallback(every_n_steps=50),
        NaNDetectionCallback(),
        ProgressCallback(every_n_steps=100, total_epochs=args.epochs),
    ]

    # Kinematics validation -- only fires if VAE has a state head (state_dim > 0).
    # The callback no-ops silently otherwise, so we wire it unconditionally.
    if detail_paths:
        callbacks.append(KinematicsValidationCallback(
            vae=vae,
            episode_paths=detail_paths,
            every_n_steps=args.val_every * 4,
            frame_size=frame_size,
        ))

    # Dream comparison video -- exports GT|Dream side-by-side MP4s for
    # qualitative evaluation of dream quality over training
    if detail_paths:
        video_dir = str(dyn_dir / "dream_videos")
        callbacks.append(DreamComparisonVideoCallback(
            vae=vae,
            episode_paths=detail_paths,
            video_dir=video_dir,
            every_n_steps=args.val_every * 10,
            frame_size=frame_size,
        ))

    # RSSM diagnostics -- tracks KL divergence, prior vs posterior MSE.
    # No-ops for GRU models, so we wire it unconditionally when we have
    # episode paths available.
    if detail_paths:
        callbacks.append(RSSMDiagnosticCallback(
            dynamics=dynamics,
            episode_paths=detail_paths,
            vae=vae,
            every_n_steps=args.val_every * 4,
            frame_size=frame_size,
        ))

    ctx = CallbackContext(
        model=dynamics, optimizer=optimizer, writer=writer,
        global_step=start_step, epoch=start_epoch, run_dir=str(dyn_dir),
        device=args.device,
        extras={
            "config": config,
            "vae_checkpoint": args.vae_checkpoint,
            "model_type": args.model_type,
            "scheduler": scheduler,
        },
    )

    # SIGINT handler -- save an emergency checkpoint so training can resume
    interrupted = False
    def handle_sigint(sig, frame):
        nonlocal interrupted
        if interrupted:
            sys.exit(1)
        interrupted = True
        print("\nInterrupted! Saving checkpoint and exiting...")
        torch.save({
            "model_state_dict": dynamics.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": ctx.epoch,
            "global_step": ctx.global_step,
            "config": config,
        }, Path(ckpt_dir) / "interrupted.pt")
        sys.exit(0)
    signal.signal(signal.SIGINT, handle_sigint)

    # Train
    for cb in callbacks:
        cb.on_train_start(ctx)

    remaining = args.epochs - start_epoch
    print(f"\nTraining {model_label} ({args.training_mode}) "
          f"for {remaining} remaining epochs "
          f"(start_epoch={start_epoch}, total={args.epochs}) on {args.device}")
    for epoch in range(start_epoch, args.epochs):
        ctx.epoch = epoch

        sampling_prob = get_sampling_prob(
            epoch, args.epochs,
            args.sampling_start, args.sampling_end,
            args.sampling_warmup_frac,
        )

        # Pass training_mode params so the loop dispatches to the right loss
        result = pixel_dynamics_train_epoch(
            dynamics, vae, train_loader, optimizer,
            sampling_prob=sampling_prob,
            device=args.device,
            max_grad_norm=args.grad_clip,
            ctx=ctx, callbacks=callbacks,
            training_mode=args.training_mode,
            rollout_k=args.rollout_k,
            kl_weight=args.kl_weight,
            ms_weight=args.multi_step_weight,
            free_bits=args.free_bits,
            kin_weight=args.kin_weight,
            kin_dims=args.kin_dims,
        )

        print(f"Epoch {epoch}: train_loss={result['train_loss']:.6f} "
              f"sampling_prob={sampling_prob:.3f}")

        if ctx.writer:
            ctx.writer.add_scalar("train/sampling_prob", sampling_prob, ctx.global_step)
            ctx.writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], ctx.global_step)

        # NOTE: LR scheduler now steps inside the validation callback
        # via ctx.extras["scheduler"], not here at epoch level. This ensures
        # lr_patience=N means N val checks, not N epochs.

        for cb in callbacks:
            if cb.on_epoch_end(ctx) is False:
                print("Early stopping triggered.")
                break

        if result.get("stop_requested"):
            print("Stop requested by callback.")
            break

    for cb in callbacks:
        cb.on_train_end(ctx)

    writer.close()
    print(f"\nDynamics training complete. Best checkpoint: {ckpt_dir}/best.pt")


if __name__ == "__main__":
    main()
