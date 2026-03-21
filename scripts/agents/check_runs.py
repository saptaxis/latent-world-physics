#!/usr/bin/env python
"""Check status of visual RL training runs.

Reads TensorBoard events and config.json from run directories to produce
a summary table and per-run trend snapshots.

Usage:
    # All runs under default visual agents dir
    python lunar_lander/scripts/check_runs.py

    # Specific run directory
    python lunar_lander/scripts/check_runs.py \
        --run-dir /path/to/visual-ppo-gym-default-128px/s42

    # All runs under a custom parent dir
    python lunar_lander/scripts/check_runs.py \
        --agents-dir /path/to/visual_rl_agents/gym-default

    # Show last N eval points for trend (default: 8)
    python lunar_lander/scripts/check_runs.py --trend 12
"""
import argparse
import json
import os
import sys
from pathlib import Path

# Lazy import — tensorboard is heavy
EventAccumulator = None


def _load_ea():
    global EventAccumulator
    if EventAccumulator is None:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator as EA,
        )
        EventAccumulator = EA


def _find_runs(agents_dir):
    """Find all seed directories (s42, s0, etc.) with config.json + events."""
    runs = []
    for run_name in sorted(os.listdir(agents_dir)):
        run_path = os.path.join(agents_dir, run_name)
        if not os.path.isdir(run_path) or "__BUGGED" in run_name:
            continue
        for seed_name in sorted(os.listdir(run_path)):
            seed_path = os.path.join(run_path, seed_name)
            config_path = os.path.join(seed_path, "config.json")
            if os.path.isdir(seed_path) and os.path.exists(config_path):
                runs.append(seed_path)
    return runs


def _load_events(seed_dir):
    """Load TensorBoard EventAccumulator for a seed directory."""
    _load_ea()
    tf_files = [f for f in os.listdir(seed_dir) if f.startswith("events.out")]
    if not tf_files:
        return None
    ea = EventAccumulator(os.path.join(seed_dir, tf_files[0]))
    ea.Reload()
    return ea


def _get_scalar_history(ea, tag, n=None):
    """Get scalar history. Returns list of (step, value) tuples."""
    if tag not in ea.Tags()["scalars"]:
        return []
    events = ea.Scalars(tag)
    pairs = [(e.step, e.value) for e in events]
    if n is not None:
        pairs = pairs[-n:]
    return pairs


def _format_trend(values, fmt=".0f"):
    """Format a list of values as a compact trend string."""
    if not values:
        return "—"
    return " → ".join(f"{v:{fmt}}" for v in values)


def check_single_run(seed_dir, n_trend=8):
    """Generate a status report for a single run."""
    config_path = os.path.join(seed_dir, "config.json")
    with open(config_path) as f:
        cfg = json.load(f)

    run_name = os.path.basename(os.path.dirname(seed_dir))
    seed_name = os.path.basename(seed_dir)

    report = []
    report.append(f"\n{'='*70}")
    report.append(f"  {run_name}/{seed_name}")
    report.append(f"{'='*70}")

    # Config summary
    backbone = cfg.get("cnn_backbone", "nature")
    lr = cfg.get("learning_rate", "?")
    lr_str = f"{lr:.1e}" if isinstance(lr, (int, float)) else str(lr)
    share = "shared" if cfg.get("share_features_extractor", True) else "separate"
    frame_size = cfg.get("frame_size", 84)
    total_steps = cfg.get("total_steps", 0)
    report.append(f"  Config: {backbone} | {frame_size}px | LR={lr_str} | {share} | {total_steps/1e6:.0f}M steps")

    ea = _load_events(seed_dir)
    if ea is None:
        report.append("  No TensorBoard events found.")
        return "\n".join(report)

    # Current progress
    rew_hist = _get_scalar_history(ea, "rollout/ep_rew_mean")
    current_step = rew_hist[-1][0] if rew_hist else 0
    pct = current_step / total_steps * 100 if total_steps > 0 else 0

    if pct >= 99:
        status = "DONE"
    elif pct > 0:
        status = "RUNNING"
    else:
        status = "STARTING"

    report.append(f"  Progress: {current_step/1e6:.1f}M / {total_steps/1e6:.0f}M ({pct:.0f}%) — {status}")

    # Reward trend
    rew_recent = _get_scalar_history(ea, "rollout/ep_rew_mean", n=n_trend)
    if rew_recent:
        vals = [v for _, v in rew_recent]
        report.append(f"  Reward:  {_format_trend(vals)}")
        report.append(f"           mean={sum(vals)/len(vals):.0f}  best={max(vals):.0f}  last={vals[-1]:.0f}")

    # Eval reward
    eval_rew = _get_scalar_history(ea, "eval/mean_reward", n=n_trend)
    if eval_rew:
        vals = [v for _, v in eval_rew]
        report.append(f"  Eval:    {_format_trend(vals)}")

    # Landed %
    landed = _get_scalar_history(ea, "eval/landed_pct", n=n_trend)
    if landed:
        vals = [v for _, v in landed]
        all_landed = _get_scalar_history(ea, "eval/landed_pct")
        max_landed = max(v for _, v in all_landed) if all_landed else 0
        report.append(f"  Landed%: {_format_trend(vals)}")
        report.append(f"           max_ever={max_landed:.0f}%  last={vals[-1]:.0f}%")

    # Episode length
    ep_len = _get_scalar_history(ea, "rollout/ep_len_mean", n=n_trend)
    if ep_len:
        vals = [v for _, v in ep_len]
        report.append(f"  Ep len:  {_format_trend(vals)}")

    # Training diagnostics
    report.append(f"  --- Training diagnostics (last {n_trend}) ---")

    clip = _get_scalar_history(ea, "train/clip_fraction", n=n_trend)
    if clip:
        vals = [v for _, v in clip]
        last = vals[-1]
        flag = " ⚠ HIGH" if last > 0.3 else " ✓" if last < 0.15 else ""
        report.append(f"  Clip:    {_format_trend(vals, '.2f')}{flag}")

    kl = _get_scalar_history(ea, "train/approx_kl", n=n_trend)
    if kl:
        vals = [v for _, v in kl]
        last = vals[-1]
        flag = " ⚠ HIGH" if last > 0.05 else ""
        report.append(f"  KL:      {_format_trend(vals, '.3f')}{flag}")

    entropy = _get_scalar_history(ea, "train/entropy_loss", n=n_trend)
    if entropy:
        vals = [v for _, v in entropy]
        report.append(f"  Entropy: {_format_trend(vals, '.2f')}  (more negative = more random)")

    vloss = _get_scalar_history(ea, "train/value_loss", n=n_trend)
    if vloss:
        vals = [v for _, v in vloss]
        report.append(f"  V-loss:  {_format_trend(vals, '.3f')}")

    std = _get_scalar_history(ea, "train/std", n=n_trend)
    if std:
        vals = [v for _, v in std]
        report.append(f"  Std:     {_format_trend(vals, '.3f')}")

    return "\n".join(report)


def summary_table(run_dirs):
    """Generate a compact summary table across all runs."""
    rows = []
    for seed_dir in run_dirs:
        config_path = os.path.join(seed_dir, "config.json")
        with open(config_path) as f:
            cfg = json.load(f)

        run_name = os.path.basename(os.path.dirname(seed_dir))
        short = run_name.replace("visual-ppo-gym-default", "vppo")

        backbone = cfg.get("cnn_backbone", "nature")
        lr = cfg.get("learning_rate")
        lr_str = f"{lr:.1e}" if isinstance(lr, (int, float)) and lr else "?"
        share = "yes" if cfg.get("share_features_extractor", True) else "no"
        frame_size = cfg.get("frame_size", 84)
        total_steps = cfg.get("total_steps", 0)

        ea = _load_events(seed_dir)
        current_step = 0
        last_reward = "—"
        landed_str = "—"
        clip_str = "—"

        if ea:
            rew = _get_scalar_history(ea, "rollout/ep_rew_mean")
            if rew:
                current_step = rew[-1][0]
                last_reward = f"{rew[-1][1]:.0f}"

            landed = _get_scalar_history(ea, "eval/landed_pct")
            if landed:
                max_l = max(v for _, v in landed)
                last_l = landed[-1][1]
                landed_str = f"{last_l:.0f}/{max_l:.0f}"

            clip = _get_scalar_history(ea, "train/clip_fraction")
            if clip:
                clip_str = f"{clip[-1][1]:.2f}"

        pct = current_step / total_steps * 100 if total_steps > 0 else 0
        status = "DONE" if pct >= 99 else "RUN" if pct > 0 else "?"
        progress = f"{current_step/1e6:.1f}M/{total_steps/1e6:.0f}M"

        rows.append((short, backbone, lr_str, share, frame_size, progress,
                      f"{pct:.0f}%", last_reward, landed_str, clip_str, status))

    # Print table
    header = f"{'Run':<40s} {'CNN':<8s} {'LR':<8s} {'Sh':<4s} {'Res':<4s} {'Progress':<12s} {'%':<5s} {'Rew':<7s} {'Land':<8s} {'Clip':<6s} {'St'}"
    sep = "-" * len(header)
    lines = ["\n" + header, sep]
    for r in rows:
        lines.append(f"{r[0]:<40s} {r[1]:<8s} {r[2]:<8s} {r[3]:<4s} {r[4]:<4d} {r[5]:<12s} {r[6]:<5s} {r[7]:<7s} {r[8]:<8s} {r[9]:<6s} {r[10]}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Check status of visual RL training runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--run-dir",
                       help="Single seed directory (e.g. .../128px/s42)")
    group.add_argument("--agents-dir",
                       help="Parent dir with run subdirs")

    parser.add_argument("--trend", type=int, default=8,
                        help="Number of recent points for trend display (default: 8)")
    parser.add_argument("--table-only", action="store_true",
                        help="Only show summary table, skip per-run details")

    args = parser.parse_args()

    # Default agents dir
    default_dir = "/media/hdd1/physics-priors-latent-space/lunar-lander-networks/visual_rl_agents/gym-default"

    if args.run_dir:
        run_dirs = [args.run_dir]
    elif args.agents_dir:
        run_dirs = _find_runs(args.agents_dir)
    else:
        run_dirs = _find_runs(default_dir)

    if not run_dirs:
        print("No runs found.")
        sys.exit(1)

    # Summary table
    print(summary_table(run_dirs))

    # Per-run details
    if not args.table_only:
        for seed_dir in run_dirs:
            print(check_single_run(seed_dir, n_trend=args.trend))

    print()


if __name__ == "__main__":
    main()
