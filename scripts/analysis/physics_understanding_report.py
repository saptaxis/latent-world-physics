#!/usr/bin/env python
"""Physics understanding evaluation for wm-ladder world models.

Extracts physical constants from model predictions on validation data,
measures consistency across state space, and reports compounding rates.

Usage:
    # Run on a wm-ladder checkpoint
    python lunar_lander/scripts/physics_understanding_report.py \
        --ladder-checkpoint /path/to/best.pt \
        --data-dir /path/to/episode/data

    # Custom rollout horizon and output dir
    python lunar_lander/scripts/physics_understanding_report.py \
        --ladder-checkpoint /path/to/best.pt \
        --data-dir /path/to/episode/data \
        --rollout-horizon 20 \
        --output-dir /path/to/output

    # Limit number of validation episodes
    python lunar_lander/scripts/physics_understanding_report.py \
        --ladder-checkpoint /path/to/best.pt \
        --data-dir /path/to/episode/data \
        --n-episodes 100
"""
import argparse
import os
import sys
from pathlib import Path


import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(
        description="Physics understanding evaluation for world models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--ladder-checkpoint", required=True,
        help="Path to wm-ladder .pt checkpoint file.",
    )
    parser.add_argument(
        "--data-dir", required=True,
        help="Path to episode data directory (contains .npz files).",
    )
    parser.add_argument(
        "--n-episodes", type=int, default=200,
        help="Max number of validation episodes to use (default: 200).",
    )
    parser.add_argument(
        "--rollout-horizon", type=int, default=10,
        help="Steps for rollout constant extraction (default: 10).",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Where to save JSON report (default: checkpoint_dir/physics_understanding/).",
    )
    parser.add_argument(
        "--device", default="cpu",
        help="Device for model inference (default: cpu).",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip if output JSON already exists.",
    )
    args = parser.parse_args()

    # --- Skip if already done ---
    if args.skip_existing:
        if args.output_dir:
            out_dir = Path(args.output_dir)
        else:
            out_dir = Path(args.ladder_checkpoint).parent / "physics_understanding"
        json_path = out_dir / "physics_understanding.json"
        if json_path.exists():
            print(f"SKIP — already exists: {json_path}")
            return

    # --- Load checkpoint ---
    from lwp.utils.checkpoint import load_checkpoint
    from lwp.models.factory import build_model

    print(f"Loading checkpoint: {args.ladder_checkpoint}")
    ckpt = load_checkpoint(args.ladder_checkpoint, device=args.device)
    config = ckpt["config"]
    model = build_model(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    model = model.to(args.device)

    # NormStats from checkpoint.
    from lwp.data.normalization import NormStats
    norm_stats = NormStats.from_dict(ckpt["norm_stats"])

    # Determine if recurrent.
    recurrent = config.arch in ("gru", "rssm")
    print(f"Model: {config.arch} | recurrent={recurrent} | state_dim={config.state_dim}")

    # --- Load episodes ---
    from lwp.wm.diagnostics import _load_episodes_from_dir

    data_dir = Path(args.data_dir)
    print(f"Loading episodes from: {data_dir}")
    episodes = _load_episodes_from_dir(data_dir, args.n_episodes)
    print(f"Loaded {len(episodes)} episodes")

    if not episodes:
        print("ERROR: No episodes found.")
        sys.exit(1)

    # --- Run evaluation ---
    from lwp.wm.physics_understanding import (
        generate_report, format_console_report, save_json_report,
    )

    print(f"\nRunning physics understanding evaluation...")
    print(f"  Rollout horizon: {args.rollout_horizon}")
    print(f"  Recurrent warmup: {'50 steps' if recurrent else 'N/A'}")

    results = generate_report(
        model, norm_stats, episodes,
        recurrent=recurrent,
        rollout_horizon=args.rollout_horizon,
    )

    # --- Output ---
    # Console report.
    run_name = Path(args.ladder_checkpoint).parent.name
    print("\n" + format_console_report(results, run_name=run_name))

    # JSON report.
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = Path(args.ladder_checkpoint).parent / "physics_understanding"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "physics_understanding.json"
    save_json_report(results, str(json_path))
    print(f"\nJSON report saved to: {json_path}")

    # Text report (same as console output).
    txt_path = out_dir / "physics_understanding.txt"
    report_text = format_console_report(results, run_name=run_name)
    with open(txt_path, "w") as f:
        f.write(report_text + "\n")
    print(f"Text report saved to: {txt_path}")


if __name__ == "__main__":
    main()
