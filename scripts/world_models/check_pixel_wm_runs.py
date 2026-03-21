#!/usr/bin/env python
"""Check status of pixel world model training runs (VAE + dynamics).

Reads TensorBoard events and config.json from run directories to produce
a summary table and per-run trend snapshots. Designed for the pixel WM
training pipeline (train_pixel_vae.py / train_pixel_dynamics.py).

Usage:
    # All runs under default pixel WM dir
    python lunar_lander/scripts/check_pixel_wm_runs.py

    # Specific run directory
    python lunar_lander/scripts/check_pixel_wm_runs.py \
        --run-dir /path/to/dynamics-state6prims-film-multistep

    # All runs under a custom parent dir
    python lunar_lander/scripts/check_pixel_wm_runs.py \
        --wm-dir /path/to/pixel-world-model

    # Compact table only
    python lunar_lander/scripts/check_pixel_wm_runs.py --table-only

    # Show last N points for trends (default: 8)
    python lunar_lander/scripts/check_pixel_wm_runs.py --trend 12
"""
import argparse
import json
import os
import sys
from pathlib import Path

# Lazy import — tensorboard is heavy
EventAccumulator = None

# Default location for pixel WM runs
DEFAULT_WM_DIR = "/media/hdd1/physics-priors-latent-space/lunar-lander-networks/pixel-world-model"


def _load_ea():
    """Lazy-load EventAccumulator to avoid slow import on --help."""
    global EventAccumulator
    if EventAccumulator is None:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator as EA,
        )
        EventAccumulator = EA


def _find_runs(wm_dir):
    """Find all run directories with config.json under the WM parent dir."""
    runs = []
    root = Path(wm_dir)
    for config_path in sorted(root.rglob("config.json")):
        run_dir = config_path.parent
        # Skip nested dirs (e.g., tb/ subdirs that might have configs)
        # A valid run dir has config.json at the top level
        if run_dir.parent == root or run_dir == root:
            runs.append(str(run_dir))
    return runs


def _load_events(run_dir):
    """Load TensorBoard events from run_dir/tb/."""
    _load_ea()
    tb_dir = os.path.join(run_dir, "tb")
    if not os.path.exists(tb_dir):
        return None
    try:
        # Only load scalars — skip images/histograms/etc. which are huge
        # and slow to parse. We only need scalar tags for the status report.
        # size_guidance: 0 = load all for scalars, 0 = load none for others
        # Actually: for non-scalars, set to 1 (minimum) to avoid loading bulk data
        ea = EventAccumulator(tb_dir, size_guidance={
            "scalars": 0,        # 0 = load all scalar events
            "images": 1,         # 1 = load only 1 (minimum, effectively skip)
            "histograms": 1,
            "tensors": 1,
            "compressed_histograms": 1,
        })
        ea.Reload()
        return ea
    except Exception:
        return None


def _get_scalar_history(ea, tag, n=None, step_range=None):
    """Get (step, value) pairs for a scalar tag.

    Args:
        n: return only last n events (after step_range filtering)
        step_range: (min_step, max_step) tuple to filter events by step.
            Useful for comparing runs at the same training stage.
    """
    tags = ea.Tags().get("scalars", [])
    if tag not in tags:
        return []
    events = ea.Scalars(tag)
    pairs = [(e.step, e.value) for e in events]
    if step_range is not None:
        lo, hi = step_range
        pairs = [(s, v) for s, v in pairs if lo <= s <= hi]
    if n is not None:
        return pairs[-n:]
    return pairs


def _format_trend(values, fmt=".4f"):
    """Format a list of values as a compact trend string."""
    formatted = [f"{v:{fmt}}" for v in values]
    return " → ".join(formatted)


def _detect_run_type(cfg):
    """Detect whether this is a VAE run or dynamics run from config."""
    if "vae_checkpoint" in cfg or "model_type" in cfg:
        return "dynamics"
    if "beta" in cfg and "fg_weight" in cfg:
        return "vae"
    return "unknown"


def check_single_run(run_dir, n_trend=8, step_range=None):
    """Generate a status report for a single pixel WM run."""
    config_path = os.path.join(run_dir, "config.json")
    if not os.path.exists(config_path):
        return f"\n  {run_dir}: No config.json found"

    with open(config_path) as f:
        cfg = json.load(f)

    run_name = os.path.basename(run_dir)
    run_type = _detect_run_type(cfg)

    report = []
    report.append(f"\n{'='*70}")
    report.append(f"  {run_name}  [{run_type.upper()}]")
    report.append(f"{'='*70}")

    # Config summary
    if run_type == "vae":
        latent_dim = cfg.get("latent_dim", "?")
        frame_size = cfg.get("frame_size", "?")
        fg_weight = cfg.get("fg_weight", 1.0)
        state_dim = cfg.get("state_dim", 0)
        beta = cfg.get("beta", "?")
        state_str = f" | state_dim={state_dim}" if state_dim > 0 else ""
        report.append(f"  Config: z={latent_dim} | {frame_size}px | fg_weight={fg_weight} | beta={beta}{state_str}")
    elif run_type == "dynamics":
        model_type = cfg.get("model_type", "gru")
        training_mode = cfg.get("training_mode", "latent_mse")
        rollout_k = cfg.get("rollout_k", 1)
        hidden = cfg.get("hidden_size", "?")
        kl_weight = cfg.get("kl_weight", 1.0)
        free_bits = cfg.get("free_bits", 0)
        report.append(f"  Config: {model_type} | {training_mode} | k={rollout_k} | hidden={hidden}")
        if model_type == "rssm":
            deter = cfg.get("deter_dim", 200)
            stoch = cfg.get("stoch_dim", 30)
            report.append(f"          deter={deter} | stoch={stoch} | kl_weight={kl_weight} | free_bits={free_bits}")

    ea = _load_events(run_dir)
    if ea is None:
        report.append("  No TensorBoard events found.")
        return "\n".join(report)

    # Local wrapper that threads step_range through all scalar queries
    def _hist(tag, n=None):
        return _get_scalar_history(ea, tag, n=n, step_range=step_range)

    if step_range:
        report.append(f"  Step range: {step_range[0]:,} — {step_range[1]:,}")

    # --- Current progress ---
    train_hist = _hist("train/loss")
    val_hist = _hist("val/loss")
    current_step = train_hist[-1][0] if train_hist else 0

    # Detect status from patience / early stopping
    epochs = cfg.get("epochs", 100)
    patience = cfg.get("patience", 10)
    has_best = os.path.exists(os.path.join(run_dir, "best.pt"))

    if val_hist:
        # Check if val loss has been flat (early stopped or done)
        recent_val = [v for _, v in val_hist[-patience:]]
        if len(recent_val) >= patience and min(recent_val) >= recent_val[0] * 0.999:
            status = "EARLY-STOPPED" if has_best else "PLATEAU"
        else:
            status = "TRAINING"
    else:
        status = "STARTING"

    report.append(f"  Step: {current_step:,}  |  Status: {status}")

    # --- Loss trends ---
    train_recent = _hist("train/loss", n=n_trend)
    if train_recent:
        vals = [v for _, v in train_recent]
        report.append(f"  Train loss:  {_format_trend(vals, '.6f')}")

    val_recent = _hist("val/loss", n=n_trend)
    if val_recent:
        vals = [v for _, v in val_recent]
        best_val = min(v for _, v in val_hist) if val_hist else float("inf")
        report.append(f"  Val loss:    {_format_trend(vals, '.6f')}  (best: {best_val:.6f})")

    # --- VAE-specific metrics ---
    if run_type == "vae":
        recon_recent = _hist("train/recon_loss", n=n_trend)
        if recon_recent:
            vals = [v for _, v in recon_recent]
            report.append(f"  Recon loss:  {_format_trend(vals, '.6f')}")

        kl_recent = _hist("train/kl_loss", n=n_trend)
        if kl_recent:
            vals = [v for _, v in kl_recent]
            report.append(f"  KL loss:     {_format_trend(vals, '.2f')}")

        state_recent = _hist("train/state_loss", n=n_trend)
        if state_recent:
            vals = [v for _, v in state_recent]
            report.append(f"  State loss:  {_format_trend(vals, '.6f')}")

    # --- Dynamics-specific metrics ---
    if run_type == "dynamics":
        # LR trend
        lr_recent = _hist("train/lr", n=n_trend)
        if lr_recent:
            vals = [v for _, v in lr_recent]
            report.append(f"  LR:          {_format_trend(vals, '.2e')}")

        # ELBO breakdown
        recon_recent = _hist("train/recon_loss", n=n_trend)
        if recon_recent:
            vals = [v for _, v in recon_recent]
            report.append(f"  Recon loss:  {_format_trend(vals, '.6f')}")

        kl_recent = _hist("train/kl_loss", n=n_trend)
        if kl_recent:
            vals = [v for _, v in kl_recent]
            report.append(f"  KL loss:     {_format_trend(vals, '.4f')}")

        # RSSM diagnostics
        prior_mse = _hist("rssm/prior_mse", n=n_trend)
        if prior_mse:
            vals = [v for _, v in prior_mse]
            report.append(f"  Prior MSE:   {_format_trend(vals, '.6f')}")

        post_mse = _hist("rssm/posterior_mse", n=n_trend)
        if post_mse:
            vals = [v for _, v in post_mse]
            report.append(f"  Post MSE:    {_format_trend(vals, '.6f')}")

        kl_step = _hist("rssm/kl_per_step", n=n_trend)
        if kl_step:
            vals = [v for _, v in kl_step]
            last_kl = vals[-1]
            flag = " << COLLAPSED" if last_kl < 0.1 else ""
            report.append(f"  KL/step:     {_format_trend(vals, '.4f')}{flag}")

        # Prior-posterior gap
        if prior_mse and post_mse:
            p_last = prior_mse[-1][1]
            q_last = post_mse[-1][1]
            gap = p_last - q_last
            gap_pct = gap / max(p_last, 1e-8) * 100
            report.append(f"  Prior-Post gap: {gap:.6f} ({gap_pct:.1f}%)")

        # Kinematics validation (from KinematicsValidationCallback)
        kin_tags = sorted([t for t in ea.Tags().get("scalars", []) if "kinematics/" in t])
        if kin_tags:
            report.append(f"  --- Kinematics (last eval) ---")
            # Group by horizon
            horizons = {}
            for tag in kin_tags:
                # tag format: kinematics/mse_h{h}_{dim}
                parts = tag.split("/")[1].split("_")  # mse, h{h}, {dim}
                h = parts[1]  # e.g., "h1", "h5", "h10"
                dim = "_".join(parts[2:])  # e.g., "x", "ang_vel"
                if h not in horizons:
                    horizons[h] = {}
                events = ea.Scalars(tag)
                if events:
                    horizons[h][dim] = events[-1].value

            for h in sorted(horizons.keys(), key=lambda x: int(x[1:])):
                dims = horizons[h]
                dim_strs = [f"{d}={v:.4f}" for d, v in sorted(dims.items())]
                report.append(f"    {h}: {', '.join(dim_strs)}")

    # --- Checkpoint info ---
    ckpts = sorted(Path(run_dir).glob("step_*.pt"))
    n_ckpts = len(ckpts)
    latest_ckpt = ckpts[-1].name if ckpts else "none"
    report.append(f"  Checkpoints: {n_ckpts} saved (latest: {latest_ckpt})")

    return "\n".join(report)


def summary_table(run_dirs):
    """Generate a compact summary table across all runs."""
    _load_ea()

    rows = []
    for run_dir in run_dirs:
        config_path = os.path.join(run_dir, "config.json")
        if not os.path.exists(config_path):
            continue

        with open(config_path) as f:
            cfg = json.load(f)

        run_name = os.path.basename(run_dir)
        run_type = _detect_run_type(cfg)

        ea = _load_events(run_dir)
        if ea is None:
            rows.append({
                "name": run_name, "type": run_type,
                "step": 0, "train": float("nan"), "val": float("nan"),
                "status": "NO_TB",
            })
            continue

        train_hist = _get_scalar_history(ea, "train/loss")
        val_hist = _get_scalar_history(ea, "val/loss")
        step = train_hist[-1][0] if train_hist else 0
        train_loss = train_hist[-1][1] if train_hist else float("nan")
        val_loss = val_hist[-1][1] if val_hist else float("nan")
        best_val = min(v for _, v in val_hist) if val_hist else float("nan")

        # Extra info based on type
        extra = ""
        if run_type == "dynamics":
            model_type = cfg.get("model_type", "gru")
            mode = cfg.get("training_mode", "?")
            extra = f"{model_type}/{mode}"

            # Kinematics h10 avg if available
            kin_h10_tags = [t for t in ea.Tags().get("scalars", [])
                           if "kinematics/mse_h10_" in t]
            if kin_h10_tags:
                kin_vals = []
                for t in kin_h10_tags:
                    events = ea.Scalars(t)
                    if events:
                        kin_vals.append(events[-1].value)
                if kin_vals:
                    extra += f" | kin_h10={sum(kin_vals)/len(kin_vals):.4f}"

            # RSSM KL
            kl_events = _get_scalar_history(ea, "rssm/kl_per_step")
            if kl_events:
                extra += f" | kl={kl_events[-1][1]:.2f}"
        elif run_type == "vae":
            state_dim = cfg.get("state_dim", 0)
            fg_w = cfg.get("fg_weight", 1.0)
            extra = f"fg={fg_w}"
            if state_dim > 0:
                extra += f" | state={state_dim}"
                state_events = _get_scalar_history(ea, "train/state_loss")
                if state_events:
                    extra += f" | sloss={state_events[-1][1]:.4f}"

        # Status
        has_best = os.path.exists(os.path.join(run_dir, "best.pt"))
        status = "OK" if has_best else "..."

        rows.append({
            "name": run_name, "type": run_type,
            "step": step, "train": train_loss, "val": val_loss,
            "best_val": best_val, "extra": extra, "status": status,
        })

    if not rows:
        return "No runs found."

    # Format table
    lines = []
    lines.append(f"{'Run':<45} {'Type':<8} {'Step':>8} {'Train':>10} {'Val':>10} {'Best':>10} {'Info'}")
    lines.append("-" * 120)
    for r in rows:
        lines.append(
            f"{r['name']:<45} {r['type']:<8} {r['step']:>8,} "
            f"{r['train']:>10.6f} {r['val']:>10.6f} {r.get('best_val', float('nan')):>10.6f} "
            f"{r.get('extra', '')}"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Check status of pixel world model training runs")
    parser.add_argument("--run-dir", type=str, default=None,
                        help="Single run directory to inspect")
    parser.add_argument("--wm-dir", type=str, default=DEFAULT_WM_DIR,
                        help="Parent directory containing all WM runs")
    parser.add_argument("--table-only", action="store_true",
                        help="Show only the compact summary table")
    parser.add_argument("--trend", type=int, default=8,
                        help="Number of recent points to show in trends")
    parser.add_argument("--step-range", type=str, default=None,
                        help="Filter to step range, e.g. '0-20000' or '10000-50000'. "
                             "Useful for comparing runs at the same training stage.")
    args = parser.parse_args()

    # Parse step range
    step_range = None
    if args.step_range:
        lo, hi = args.step_range.split("-")
        step_range = (int(lo), int(hi))

    if args.run_dir:
        # Single run mode
        print(check_single_run(args.run_dir, n_trend=args.trend, step_range=step_range))
    else:
        # All runs mode
        runs = _find_runs(args.wm_dir)
        if not runs:
            print(f"No runs found in {args.wm_dir}")
            sys.exit(1)

        print(f"\n  Pixel WM Runs: {args.wm_dir}")
        print(f"  Found {len(runs)} runs\n")

        # Summary table first
        print("Loading TensorBoard events...", end="", flush=True)
        print(f"\r{' '*40}\r", end="")  # clear loading msg
        print(summary_table(runs))

        # Detailed reports unless table-only
        if not args.table_only:
            for i, run_dir in enumerate(runs):
                print(f"\r  Loading {i+1}/{len(runs)}: {os.path.basename(run_dir)}...", end="", flush=True)
                report = check_single_run(run_dir, n_trend=args.trend, step_range=step_range)
                print(f"\r{' '*60}\r", end="")  # clear progress
                print(report)


if __name__ == "__main__":
    main()
