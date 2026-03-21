#!/usr/bin/env python
"""Check status of world model ladder training runs.

Reads TensorBoard events and checkpoint metadata from run directories
to produce a summary table and per-run detail reports.

Usage:
    # All runs under default WM ladder dir
    python lunar_lander/scripts/check_wm_runs.py

    # Only primitives runs
    python lunar_lander/scripts/check_wm_runs.py --filter primitives

    # Only v3 runs
    python lunar_lander/scripts/check_wm_runs.py --filter v3

    # Specific run directory
    python lunar_lander/scripts/check_wm_runs.py --run-dir /path/to/run

    # Table only (no per-run details)
    python lunar_lander/scripts/check_wm_runs.py --table-only
"""
import argparse
import os
import sys
from pathlib import Path

import yaml

# Lazy import — tensorboard is heavy
EventAccumulator = None


def _load_ea():
    global EventAccumulator
    if EventAccumulator is None:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator as EA,
        )
        EventAccumulator = EA


def _find_runs(runs_dir, name_filter=None):
    """Find all run directories with a tb/ subdirectory."""
    runs = []
    for name in sorted(os.listdir(runs_dir)):
        run_path = os.path.join(runs_dir, name)
        if not os.path.isdir(run_path):
            continue
        if name_filter and name_filter not in name:
            continue
        tb_path = os.path.join(run_path, "tb")
        if os.path.isdir(tb_path):
            runs.append(run_path)
    return runs


def _load_events(run_dir):
    """Load TensorBoard EventAccumulator for a run's tb/ subdirectory."""
    _load_ea()
    tb_dir = os.path.join(run_dir, "tb")
    if not os.path.isdir(tb_dir):
        return None
    tf_files = [f for f in os.listdir(tb_dir) if f.startswith("events.out")]
    if not tf_files:
        return None
    ea = EventAccumulator(os.path.join(tb_dir, tf_files[0]))
    ea.Reload()
    return ea


def _get_scalar(ea, tag, n=None):
    """Get scalar history. Returns list of (step, value) tuples."""
    if tag not in ea.Tags()["scalars"]:
        return []
    events = ea.Scalars(tag)
    pairs = [(e.step, e.value) for e in events]
    if n is not None:
        pairs = pairs[-n:]
    return pairs


def _format_trend(values, fmt=".6f"):
    """Format a list of values as a compact trend string."""
    if not values:
        return "-"
    return " > ".join(f"{v:{fmt}}" for v in values)


def _load_config(run_dir):
    """Load run config from config.yaml."""
    cfg_path = os.path.join(run_dir, "config.yaml")
    if not os.path.exists(cfg_path):
        return {}
    with open(cfg_path) as f:
        return yaml.safe_load(f) or {}


def _run_status(ea, cfg):
    """Determine run status: DONE, STOPPED (early), or RUNNING."""
    epochs = cfg.get("epochs", 100)
    patience = cfg.get("patience", 50)

    train_loss = _get_scalar(ea, "train/loss")
    val_loss = _get_scalar(ea, "val/loss")

    if not train_loss:
        return "NO_DATA", 0, 0

    last_step = train_loss[-1][0]

    # Check if best.pt exists (training completed or early-stopped)
    # We infer from the data: if val loss has many points and the last
    # improvement was far from the end, it early-stopped.
    if val_loss:
        # Find best val loss and when it occurred
        best_val = min(v for _, v in val_loss)
        best_step = [s for s, v in val_loss if v == best_val][-1]

        val_every = cfg.get("val_every", 500)
        steps_since_best = last_step - best_step
        patience_steps = patience * val_every

        if steps_since_best >= patience_steps:
            return "EARLY_STOP", last_step, best_step

    return "DONE", last_step, 0


def check_single_run(run_dir, n_trend=5):
    """Generate a detailed status report for a single run."""
    run_name = os.path.basename(run_dir)
    cfg = _load_config(run_dir)

    report = []
    report.append(f"\n{'='*70}")
    report.append(f"  {run_name}")
    report.append(f"{'='*70}")

    # Config summary
    arch = cfg.get("arch", "?")
    mode = cfg.get("training_mode", "?")
    data_mix = cfg.get("data_mix", "?")
    rollout_k = cfg.get("rollout_k", 1)
    lr = cfg.get("lr", "?")
    lr_str = f"{lr:.1e}" if isinstance(lr, (int, float)) else str(lr)
    batch_size = cfg.get("batch_size", "?")
    subsample = cfg.get("subsample", 1)
    patience = cfg.get("patience", "?")
    epochs = cfg.get("epochs", "?")
    seq_len = cfg.get("seq_len", "-")
    report.append(f"  Config: {arch} | {mode} | k={rollout_k} | data={data_mix}")
    report.append(f"  Hypers: LR={lr_str} | batch={batch_size} | sub={subsample} | seq={seq_len} | patience={patience} | epochs={epochs}")

    ea = _load_events(run_dir)
    if ea is None:
        report.append("  No TensorBoard events found.")
        return "\n".join(report)

    status, last_step, best_step = _run_status(ea, cfg)
    report.append(f"  Status: {status} at step {last_step:,}" +
                  (f" (best @ {best_step:,})" if best_step > 0 else ""))

    # Train/val loss trend
    train = _get_scalar(ea, "train/loss", n=n_trend)
    val = _get_scalar(ea, "val/loss", n=n_trend)
    if train:
        vals = [v for _, v in train]
        report.append(f"  Train:   {_format_trend(vals)}")
    if val:
        vals = [v for _, v in val]
        all_val = _get_scalar(ea, "val/loss")
        best = min(v for _, v in all_val)
        report.append(f"  Val:     {_format_trend(vals)}")
        report.append(f"           best={best:.6f}")

    # Per-dim losses
    dim_names = cfg.get("dim_names", [])
    if dim_names:
        dim_losses = []
        for name in dim_names:
            tag = f"loss_dim/{name}"
            hist = _get_scalar(ea, tag)
            if hist:
                dim_losses.append((name, hist[-1][1]))
        if dim_losses:
            parts = [f"{n}={v:.4f}" for n, v in dim_losses]
            report.append(f"  Dims:    {', '.join(parts)}")

    # Rollout MSE at different horizons
    report.append(f"  --- Rollout MSE ---")
    for h in [1, 5, 10, 20, 50]:
        tag = f"rollout/mse_h{h:02d}"
        hist = _get_scalar(ea, tag)
        if hist:
            last = hist[-1][1]
            report.append(f"    h={h:<3d}  {last:.6f}")

    # Performance
    perf = _get_scalar(ea, "perf/steps_per_sec", n=3)
    if perf:
        avg = sum(v for _, v in perf) / len(perf)
        report.append(f"  Speed:   {avg:.0f} steps/s")

    return "\n".join(report)


def summary_table(run_dirs):
    """Generate a compact summary table across all runs."""
    rows = []
    for run_dir in run_dirs:
        run_name = os.path.basename(run_dir)
        cfg = _load_config(run_dir)

        arch = cfg.get("arch", "?")
        mode = cfg.get("training_mode", "?")[:5]
        data_mix = cfg.get("data_mix", "?")
        k = cfg.get("rollout_k", 1)

        ea = _load_events(run_dir)
        last_step = 0
        best_val = "-"
        last_val = "-"
        h10 = "-"
        h50 = "-"
        status = "?"

        if ea:
            st, last_step, best_step = _run_status(ea, cfg)
            status = st[:5]

            val = _get_scalar(ea, "val/loss")
            if val:
                best_v = min(v for _, v in val)
                best_val = f"{best_v:.5f}"
                last_val = f"{val[-1][1]:.5f}"

            r10 = _get_scalar(ea, "rollout/mse_h10")
            if r10:
                h10 = f"{r10[-1][1]:.4f}"

            r50 = _get_scalar(ea, "rollout/mse_h50")
            if r50:
                h50 = f"{r50[-1][1]:.2f}"

        # Shorten run name for display
        short = run_name
        rows.append((short, arch, mode, k, data_mix, f"{last_step/1e3:.0f}K",
                      best_val, last_val, h10, h50, status))

    header = (f"{'Run':<45s} {'Arch':<6s} {'Mode':<6s} {'K':<3s} {'Data':<12s} "
              f"{'Steps':<7s} {'BestVal':<9s} {'LastVal':<9s} {'h10':<8s} {'h50':<8s} {'Status'}")
    sep = "-" * len(header)
    lines = ["\n" + header, sep]
    for r in rows:
        lines.append(
            f"{r[0]:<45s} {r[1]:<6s} {r[2]:<6s} {r[3]:<3d} {r[4]:<12s} "
            f"{r[5]:<7s} {r[6]:<9s} {r[7]:<9s} {r[8]:<8s} {r[9]:<8s} {r[10]}"
        )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Check status of world model ladder training runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--run-dir",
                       help="Single run directory")
    group.add_argument("--runs-dir",
                       help="Parent dir containing run subdirs")

    parser.add_argument("--filter", type=str, default=None,
                        help="Only show runs whose name contains this string")
    parser.add_argument("--trend", type=int, default=5,
                        help="Number of recent points for trend display (default: 5)")
    parser.add_argument("--table-only", action="store_true",
                        help="Only show summary table, skip per-run details")

    args = parser.parse_args()

    default_dir = "/media/hdd1/physics-priors-latent-space/lunar-lander-networks/world-model-ladder-runs"

    if args.run_dir:
        run_dirs = [args.run_dir]
    else:
        runs_dir = args.runs_dir or default_dir
        run_dirs = _find_runs(runs_dir, name_filter=args.filter)

    if not run_dirs:
        print("No runs found.")
        sys.exit(1)

    # Summary table
    print(summary_table(run_dirs))

    # Per-run details
    if not args.table_only:
        for run_dir in run_dirs:
            print(check_single_run(run_dir, n_trend=args.trend))

    print()


if __name__ == "__main__":
    main()
