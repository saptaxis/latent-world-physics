#!/usr/bin/env python3
"""Train a world model.

Usage:
    python scripts/train.py --config configs/examples/mlp-single-step.yaml
    python scripts/train.py --config configs/examples/mlp-single-step.yaml --rollout_k 10 --suffix k10
    python scripts/train.py --resume runs/mlp-delta-single_step_k1-policy/best.pt
"""
from __future__ import annotations

import argparse
import signal
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from lwp.data.loader import EpisodeDataset
from lwp.data.normalization import compute_norm_stats
from lwp.models.factory import build_model
from lwp.training.callbacks import (
    CallbackContext,
    CheckpointCallback,
    GradNormCallback,
    HiddenStateHealthCallback,
    NaNDetectionCallback,
    PerDimLossCallback,
    PerTimestepLossCallback,
    PlotExportCallback,
    ProgressCallback,
    RolloutMetricsCallback,
    ValidationCallback,
    WarmupRolloutCallback,
)
from lwp.training.loop import train_epoch
from lwp.training.profiler import ProfileLogger
from lwp.training.scheduling import curriculum_schedule, sampling_schedule
from lwp.training.torch_profiler import make_torch_profiler
from lwp.utils.checkpoint import load_checkpoint, get_git_hash
from lwp.utils.config import RunConfig, load_config, generate_run_name, validate_config
from lwp.utils.logging import get_dim_names


def parse_args():
    parser = argparse.ArgumentParser(description="Train a world model")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    # Allow any RunConfig field as an override
    parser.add_argument("--rollout_k", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--suffix", type=str, default=None)
    parser.add_argument("--kl_weight", type=float, default=None)
    parser.add_argument("--val_every", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--ckpt_every", type=int, default=None)
    parser.add_argument("--plot_every", type=int, default=None)
    parser.add_argument("--grad_clip", type=float, default=None)
    parser.add_argument("--dim_weights", type=str, default=None)
    parser.add_argument("--subsample", type=int, default=None)
    # Pure CLI flags (not in config)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--no_callbacks", action="store_true",
                        help="Disable callbacks (minimal epoch-based loop)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Load config + data, print banner, exit before training")
    parser.add_argument("--profile", action="store_true",
                        help="Enable step profiler (writes profile.jsonl to run dir)")
    parser.add_argument("--torch-profile", action="store_true",
                        help="Enable torch.profiler Chrome traces (writes to torch_trace/)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Build overrides dict from non-None CLI args
    overrides = {}
    for field in ["rollout_k", "lr", "batch_size", "epochs", "suffix", "kl_weight",
                   "val_every", "patience", "ckpt_every", "plot_every", "grad_clip",
                   "dim_weights", "subsample"]:
        val = getattr(args, field)
        if val is not None:
            overrides[field] = val

    config = load_config(args.config, overrides=overrides if overrides else None)

    # Auto-detect dims from data if not explicitly set
    if config.state_dim == 0 or config.action_dim == 0:
        from lwp.data.loader import detect_dims
        detected_state, detected_action = detect_dims(config.data_path)
        if config.state_dim == 0:
            config.state_dim = detected_state
            print(f"Auto-detected state_dim={config.state_dim}")
        if config.action_dim == 0:
            config.action_dim = detected_action
            print(f"Auto-detected action_dim={config.action_dim}")

    validate_config(config)

    run_name = generate_run_name(config)
    run_dir = Path(config.run_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    config.save(run_dir / "config.yaml")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    git_hash = get_git_hash()
    print(f"Run: {run_name}")
    print(f"Device: {device}")
    print(f"Output: {run_dir}")
    if git_hash:
        print(f"Git: {git_hash[:8]}")

    # Data
    data_mode = "single_step" if config.training_mode == "single_step" else "sequence"
    seq_len = config.seq_len if data_mode == "sequence" else None

    train_ds = EpisodeDataset(config.data_path, state_dim=config.state_dim,
                              mode=data_mode, seq_len=seq_len, split="train",
                              val_fraction=config.val_fraction,
                              subsample=config.subsample)
    val_ds = EpisodeDataset(config.data_path, state_dim=config.state_dim,
                            mode=data_mode, seq_len=seq_len, split="val",
                            val_fraction=config.val_fraction,
                            subsample=config.subsample)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size,
                              shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size)

    # Data summary
    paths = config.data_path if isinstance(config.data_path, list) else [config.data_path]
    for dp in paths:
        print(f"Data: {dp}")
    if config.subsample > 1:
        print(f"Subsample: {config.subsample}x ({50 // config.subsample} FPS effective)")
    print(f"Train: {train_ds.n_episodes} episodes, {len(train_ds)} samples")
    print(f"Val: {val_ds.n_episodes} episodes, {len(val_ds)} samples")

    # Also need single_step val loader for per-dim MSE (regardless of training mode)
    val_ds_ss = EpisodeDataset(config.data_path, state_dim=config.state_dim,
                               mode="single_step", split="val",
                               val_fraction=config.val_fraction,
                               subsample=config.subsample)
    val_loader_ss = DataLoader(val_ds_ss, batch_size=config.batch_size)

    # Normalization stats from training data
    norm_stats = compute_norm_stats(train_ds.episode_dicts())

    # Model
    model = build_model(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {config.arch} ({n_params:,} params)")

    if args.dry_run:
        print("Dry run — exiting before training.")
        return

    # Profiling
    profiler = ProfileLogger(run_dir / "profile.jsonl" if args.profile else None)
    torch_prof = make_torch_profiler(
        enabled=args.torch_profile,
        trace_dir=str(run_dir / "torch_trace"),
    )
    if torch_prof is not None:
        torch_prof.start()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    # Resume from checkpoint
    start_epoch = 0
    start_step = 0
    if args.resume:
        ckpt = load_checkpoint(args.resume)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        start_step = ckpt.get("global_step", 0)
        print(f"Resumed from {args.resume} (epoch={start_epoch}, step={start_step})")
        ckpt_hash = ckpt.get("git_hash")
        if ckpt_hash and git_hash and ckpt_hash != git_hash:
            print(f"WARNING: checkpoint saved at {ckpt_hash[:8]}, current code is {git_hash[:8]}")

    # Logging
    tb_dir = str(run_dir / "tb")
    writer = SummaryWriter(log_dir=tb_dir)
    # Build callbacks
    callbacks = []
    if not args.no_callbacks:
        callbacks.append(ValidationCallback(
            val_loader=val_loader,
            norm_stats=norm_stats,
            training_mode=config.training_mode,
            every_n_steps=config.val_every,
            patience=config.patience,
            checkpoint_dir=str(run_dir),
            rollout_k=config.rollout_k,
            kl_weight=config.kl_weight,
            dim_weights=config.dim_weights,
        ))
        callbacks.append(CheckpointCallback(
            checkpoint_dir=str(run_dir),
            every_n_steps=config.ckpt_every,
        ))
        callbacks.append(PerDimLossCallback(
            val_loader=val_loader_ss,
            norm_stats=norm_stats,
            every_n_steps=config.rollout_every,
            dim_names=get_dim_names(config.dim_names, config.state_dim),
        ))
        callbacks.append(RolloutMetricsCallback(
            dataset=val_ds,
            norm_stats=norm_stats,
            every_n_steps=config.rollout_every,
            n_rollouts=config.rollout_n_rollouts,
        ))
        callbacks.append(GradNormCallback(every_n_steps=config.grad_norm_every))
        callbacks.append(NaNDetectionCallback())
        callbacks.append(ProgressCallback(
            every_n_steps=config.val_every,
            total_epochs=config.epochs,
        ))
        if data_mode == "sequence":
            callbacks.append(PerTimestepLossCallback(
                val_loader=val_loader,
                norm_stats=norm_stats,
                every_n_steps=config.rollout_every,
            ))
            callbacks.append(HiddenStateHealthCallback(
                dataset=val_ds,
                norm_stats=norm_stats,
                every_n_steps=config.rollout_every,
            ))
            callbacks.append(WarmupRolloutCallback(
                dataset=val_ds,
                norm_stats=norm_stats,
                every_n_steps=config.rollout_every,
                n_rollouts=config.rollout_n_rollouts,
            ))
        callbacks.append(PlotExportCallback(
            tb_dir=tb_dir,
            plot_dir=str(run_dir / "plots"),
            every_n_steps=config.plot_every,
        ))

    # Callback context
    ctx = CallbackContext(
        model=model, optimizer=optimizer, writer=writer,
        global_step=start_step, epoch=start_epoch,
        run_dir=str(run_dir), device=device,
    )
    ctx.extras["config"] = config
    ctx.extras["norm_stats"] = norm_stats

    # Graceful Ctrl-C: set flag so finally block saves checkpoint
    stop_requested = False

    def _sigint_handler(signum, frame):
        nonlocal stop_requested
        if stop_requested:
            sys.exit(1)
        print("\nCtrl-C received — finishing current step and saving checkpoint...")
        stop_requested = True

    signal.signal(signal.SIGINT, _sigint_handler)

    # Dispatch on_train_start
    for cb in callbacks:
        cb.on_train_start(ctx)

    # Training loop
    try:
        for epoch in range(start_epoch, config.epochs):
            with profiler.phase("epoch/schedule", step=ctx.global_step, epoch=epoch):
                current_k = config.rollout_k
                current_sampling_prob = 0.0
                if config.curriculum:
                    current_k = curriculum_schedule(epoch, config.epochs,
                                                    k_min=1, k_max=config.rollout_k)
                if config.training_mode == "scheduled_sampling":
                    current_sampling_prob = sampling_schedule(
                        epoch, config.epochs,
                        start=config.sampling_start, end=config.sampling_end)

            ctx.epoch = epoch
            train_metrics = train_epoch(
                model, train_loader, optimizer, norm_stats,
                training_mode=config.training_mode, rollout_k=current_k,
                device=device, max_grad_norm=config.grad_clip,
                sampling_prob=current_sampling_prob, kl_weight=config.kl_weight,
                dim_weights=config.dim_weights,
                ctx=ctx, callbacks=callbacks,
                profiler=profiler, torch_profiler=torch_prof,
            )

            stop_requested = stop_requested or train_metrics.get("stop_requested", False)
            if not stop_requested:
                with profiler.phase("epoch/on_epoch_end", step=ctx.global_step, epoch=epoch):
                    for cb in callbacks:
                        if cb.on_epoch_end(ctx) is False:
                            stop_requested = True
                            break

            if stop_requested:
                break

    finally:
        # Always dispatch on_train_end
        for cb in callbacks:
            cb.on_train_end(ctx)
        writer.close()
        profiler.close()
        if torch_prof is not None:
            torch_prof.stop()
            print(f"Torch traces written to {run_dir / 'torch_trace'}")
        if args.profile:
            import subprocess
            summary_script = Path(__file__).parent / "profile_summary.py"
            subprocess.run([sys.executable, str(summary_script),
                            str(run_dir / "profile.jsonl")])

    best_val = ctx.extras.get("val_loss", float("nan"))
    print(f"Done. Final val loss: {best_val:.6f}")
    print(f"Output: {run_dir}")


if __name__ == "__main__":
    main()
