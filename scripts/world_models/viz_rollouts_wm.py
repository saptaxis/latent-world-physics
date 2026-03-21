# lunar_lander/scripts/viz_rollouts_wm.py
"""Visualize autoregressive rollouts on validation episodes.

Renders side-by-side predicted vs ground-truth trajectories for
wm-ladder models. Shows what "val loss" actually looks like as
rollout behavior.

Usage:
    python lunar_lander/scripts/viz_rollouts_wm.py \
        --ladder-checkpoint /path/to/best.pt \
        --n-episodes 10 --horizon 30 --video --plot

Output: {checkpoint_dir}/val_rollouts/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


import numpy as np
import torch

from lwp.wm.rollout_viz import (
    render_trajectory_video,
    plot_state_overlay,
)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize autoregressive rollouts on val episodes.",
    )
    parser.add_argument(
        "--ladder-checkpoint", required=True,
        help="Path to wm-ladder checkpoint (.pt).",
    )
    parser.add_argument(
        "--n-episodes", type=int, default=10,
        help="Number of val episodes to visualize (default: 10).",
    )
    parser.add_argument(
        "--horizon", type=int, default=30,
        help="Rollout horizon in model steps (default: 30). "
             "At subsample=5 (10 FPS), 30 steps = 3 seconds.",
    )
    parser.add_argument(
        "--warmup", type=int, default=20,
        help="Number of warmup steps for recurrent models (default: 20).",
    )
    parser.add_argument(
        "--video", action="store_true",
        help="Generate trajectory videos (triangle renderer).",
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Generate per-dimension state overlay plots.",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: {checkpoint_dir}/val_rollouts/).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for episode selection.",
    )
    args = parser.parse_args()

    if not args.video and not args.plot:
        parser.error("At least one of --video or --plot is required.")

    # -- Load model --
    from lwp.utils.checkpoint import load_checkpoint as load_ladder_ckpt
    from lwp.models.factory import build_model as build_ladder_model
    from lwp.data.normalization import NormStats, normalize, denormalize
    from lwp.data.loader import EpisodeDataset

    print(f"\n  Loading model from {args.ladder_checkpoint}...")
    ckpt = load_ladder_ckpt(args.ladder_checkpoint)
    config = ckpt["config"]
    model = build_ladder_model(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    norm_stats = NormStats.from_dict(ckpt["norm_stats"])

    arch = config.arch
    subsample = getattr(config, "subsample", 1)
    effective_fps = 50 // subsample

    print(f"  Model: {arch}, state_dim={config.state_dim}")
    if subsample > 1:
        print(f"  Subsample: {subsample}x ({effective_fps} FPS)")

    # -- Load val episodes --
    print(f"  Loading validation episodes...")
    val_ds = EpisodeDataset(
        config.data_path, state_dim=config.state_dim,
        mode="single_step", split="val",
        val_fraction=config.val_fraction,
        subsample=subsample,
    )

    # Pick random (episode, start_point) pairs.
    # Start point is randomly chosen anywhere in the episode, as long as
    # there's enough room for warmup before it and horizon after it.
    rng = np.random.RandomState(args.seed)
    min_len = args.warmup + args.horizon
    candidates = []  # (episode_idx, branch_point)
    for i in range(val_ds.n_episodes):
        T = len(val_ds.actions[i])
        if T < min_len:
            continue
        # branch_point range: [warmup, T - horizon]
        lo = args.warmup
        hi = T - args.horizon
        candidates.append((i, lo, hi))
    if len(candidates) == 0:
        print(f"  ERROR: No episodes with >= {min_len} steps.")
        return
    n = min(args.n_episodes, len(candidates))
    chosen_eps = rng.choice(len(candidates), size=n, replace=False)
    chosen = []
    for idx in chosen_eps:
        ep_idx, lo, hi = candidates[idx]
        bp = rng.randint(lo, hi + 1)  # random branch point
        chosen.append((ep_idx, bp))
    print(f"  Episodes: {n} of {val_ds.n_episodes} val episodes "
          f"(random start points, need >= {min_len} steps)")

    # -- Output dir --
    output_dir = (
        Path(args.output_dir) if args.output_dir
        else Path(args.ladder_checkpoint).parent / "val_rollouts"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # -- Run rollouts and render --
    device = "cpu"
    ns = norm_stats.to(device)

    for k, (ep_idx, branch_point) in enumerate(chosen):
        states_full = torch.tensor(val_ds.states[ep_idx], dtype=torch.float32)
        actions_full = torch.tensor(val_ds.actions[ep_idx], dtype=torch.float32)

        rollout_end = branch_point + args.horizon

        # Ground truth states for the rollout portion.
        gt_states = states_full[branch_point:rollout_end + 1].numpy()  # +1 for initial

        # Autoregressive rollout in raw space.
        # First, warm up hidden state with teacher forcing up to branch_point.
        model_state = None
        with torch.no_grad():
            for t in range(branch_point):
                s = states_full[t].unsqueeze(0)  # [1, state_dim]
                a = actions_full[t].unsqueeze(0)  # [1, action_dim]
                s_n = normalize(s, ns.state_mean, ns.state_std)
                delta_n, model_state = model.step(s_n, a, model_state)

            # Now rollout autoregressively from branch_point.
            s = states_full[branch_point].unsqueeze(0)  # Start from true state
            pred_states = [s.squeeze(0).numpy()]

            for t in range(branch_point, rollout_end):
                a = actions_full[t].unsqueeze(0)
                s_n = normalize(s, ns.state_mean, ns.state_std)
                delta_n, model_state = model.step(s_n, a, model_state)
                delta = denormalize(delta_n, ns.delta_mean, ns.delta_std)
                s = s + delta
                pred_states.append(s.squeeze(0).numpy())

        pred_states = np.stack(pred_states)  # [H+1, state_dim]

        # Pad to at least 8 dims for the renderer (it reads x, y, angle).
        # Models with state_dim=6 are missing leg contact dims.
        def pad_to_8(arr):
            if arr.shape[1] < 8:
                pad = np.zeros((arr.shape[0], 8 - arr.shape[1]), dtype=arr.dtype)
                return np.concatenate([arr, pad], axis=1)
            return arr

        pred_padded = pad_to_8(pred_states)
        gt_padded = pad_to_8(gt_states)

        rollout = {
            "predicted_states": pred_padded,
            "actual_states": gt_padded,
        }

        tag = f"ep{ep_idx:03d}_t{branch_point:03d}"
        print(f"  [{k+1}/{n}] {tag}: {args.horizon} steps from t={branch_point} "
              f"({args.horizon / effective_fps:.1f}s)", flush=True)

        if args.plot:
            plot_state_overlay(
                rollout,
                output_path=output_dir / f"{tag}_overlay.png",
                title=f"{arch} ep{ep_idx} — {args.horizon}-step rollout",
                fps=effective_fps,
            )

        # Actions for the rollout portion (T actions for T+1 states).
        rollout_actions = actions_full[branch_point:rollout_end].numpy()

        if args.video:
            # Get dim names from config, falling back to defaults.
            vid_dim_names = getattr(config, "dim_names", None)

            render_trajectory_video(
                rollout,
                output_path=output_dir / f"{tag}_rollout.mp4",
                fps=effective_fps,
                title=f"{arch} ep{ep_idx} t={branch_point}",
                actions=rollout_actions,
                dim_names=vid_dim_names,
            )

    print(f"\n  Output: {output_dir}/")
    if args.video:
        print(f"  Videos: {n} x .mp4")
    if args.plot:
        print(f"  Plots:  {n} x .png")
    print(f"  Done.")


if __name__ == "__main__":
    main()
