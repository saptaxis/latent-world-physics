#!/usr/bin/env python3
"""Evaluate a trained world model.

Usage:
    python scripts/eval.py --checkpoint runs/mlp-delta-single_step_k1-policy/best.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from lwp.data.loader import EpisodeDataset
from lwp.data.normalization import NormStats
from lwp.models.factory import build_model
from lwp.evaluation.metrics.core import (
    per_dim_mse,
    horizon_error_curve,
    cumulative_trajectory_mse,
    divergence_exponent,
    horizon_to_failure,
)
from lwp.utils.checkpoint import load_checkpoint
from lwp.utils.logging import get_dim_names
from lwp.utils.plotting import plot_horizon_curve, plot_per_dim_bars
from lwp.utils.reporting import generate_eval_report


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a world model checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt")
    parser.add_argument("--device", default=None)
    parser.add_argument("--n_rollouts", type=int, default=50,
                        help="Number of episodes for horizon curves")
    parser.add_argument("--failure_threshold", type=float, default=1.0,
                        help="MSE threshold for horizon-to-failure")
    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    ckpt = load_checkpoint(args.checkpoint, device=device)
    config = ckpt["config"]
    norm_stats = NormStats.from_dict(ckpt["norm_stats"]).to(device)

    model = build_model(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"Model: {config.arch} (epoch {ckpt['epoch']})")
    print(f"Device: {device}")

    # Warmup for recurrent models
    arch = config.arch
    warmup = 50 if arch in ("gru", "rssm") else 0
    if warmup > 0:
        print(f"Warmup: {warmup} steps (recurrent model)")

    # Output directory
    run_dir = Path(args.checkpoint).parent
    eval_dir = run_dir / "eval"
    eval_dir.mkdir(exist_ok=True)

    # Load validation data
    val_ds = EpisodeDataset(config.data_path, state_dim=config.state_dim,
                            mode="single_step", split="val",
                            val_fraction=config.val_fraction)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size)

    dim_names = get_dim_names(getattr(config, 'dim_names', None), config.state_dim)
    results = {}

    # Eval A: per-dim MSE
    print("Running Eval A: per-dim MSE...")
    pdm = per_dim_mse(model, val_loader, norm_stats, device=device)
    results["per_dim_mse"] = {name: float(pdm[i]) for i, name in enumerate(dim_names)}
    print(f"  Mean 1-step MSE: {pdm.mean():.6f}")
    for i, name in enumerate(dim_names):
        print(f"    {name}: {pdm[i]:.6f}")

    # Eval B: horizon curves
    print("Running Eval B: horizon error curves...")
    horizons = [1, 5, 10, 20, 50]
    # Filter horizons to what the data supports
    max_T = max(len(val_ds.actions[i]) for i in range(val_ds.n_episodes))
    horizons = [h for h in horizons if h <= max_T]

    curves = horizon_error_curve(model, val_ds, norm_stats, horizons=horizons,
                                 n_rollouts=args.n_rollouts, device=device,
                                 warmup_steps=warmup)
    results["horizon_curves"] = {
        h: {name: float(curves[h][i]) for i, name in enumerate(dim_names)}
        for h in horizons
    }
    mean_curves = {h: float(curves[h].mean()) for h in horizons}
    results["horizon_mean_mse"] = mean_curves
    print(f"  Horizon mean MSE: {mean_curves}")

    # Divergence exponent
    lam = divergence_exponent(mean_curves)
    results["divergence_exponent"] = lam
    print(f"  Divergence exponent lambda: {lam:.4f}")

    # Horizon to failure
    htf = horizon_to_failure(mean_curves, threshold=args.failure_threshold)
    results["horizon_to_failure"] = htf
    print(f"  Horizon to failure (threshold={args.failure_threshold}): {htf}")

    # Eval B': cumulative trajectory MSE
    print("Running Eval B': cumulative trajectory MSE...")
    cumul_curves = cumulative_trajectory_mse(model, val_ds, norm_stats,
                                              horizons=horizons,
                                              n_rollouts=args.n_rollouts,
                                              device=device,
                                              warmup_steps=warmup)
    results["cumul_horizon_curves"] = {
        h: {name: float(cumul_curves[h][i]) for i, name in enumerate(dim_names)}
        for h in horizons if h in cumul_curves
    }
    cumul_mean = {h: float(cumul_curves[h].mean()) for h in horizons if h in cumul_curves}
    results["cumul_horizon_mean_mse"] = cumul_mean
    print(f"  Cumulative mean MSE: {cumul_mean}")

    # Save results
    results_path = eval_dir / "metrics.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # Generate markdown report
    report_path = eval_dir / "report.md"
    generate_eval_report(
        run_name=Path(args.checkpoint).parent.name,
        results=results,
        output_path=str(report_path),
    )
    print(f"Report saved to: {report_path}")

    # Generate plots
    plot_dir = eval_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    run_label = Path(args.checkpoint).parent.name
    plot_per_dim_bars(
        results["per_dim_mse"],
        str(plot_dir / "per_dim_mse.png"),
        title=f"Per-Dimension MSE: {run_label}",
    )
    print(f"Plot: {plot_dir / 'per_dim_mse.png'}")

    if mean_curves:
        plot_horizon_curve(
            mean_curves,
            str(plot_dir / "horizon_curve.png"),
            title=f"Horizon Error: {run_label}",
        )
        plot_horizon_curve(
            mean_curves,
            str(plot_dir / "horizon_curve_log.png"),
            title=f"Horizon Error (log): {run_label}",
            log_scale=True,
        )
        print(f"Plot: {plot_dir / 'horizon_curve.png'}")


if __name__ == "__main__":
    main()
