#!/usr/bin/env python
"""Collect probe data (activations + targets) from trained Lunar Lander agents.

Loads a trained checkpoint, registers forward hooks on the policy network,
runs episodes, and saves per-timestep activations and probe targets
as a single .npz file. For probe training, pipe output to train_probes.py.

Works for both state-vector agents (hooks MLP hidden layers) and visual
agents (hooks CNN encoder output + MLP hidden layers). The variant is
auto-detected from config.json.

Usage:
    # Single agent
    python lunar_lander/scripts/collect_probe_data.py \
        --checkpoint-dir /path/to/agent/s42 --episodes 100

    # All agents under a parent directory
    python lunar_lander/scripts/collect_probe_data.py \
        --agents-dir /path/to/agents/ --episodes 100

    # Specific seeds only
    python lunar_lander/scripts/collect_probe_data.py \
        --agents-dir /path/to/agents/ --episodes 100 --seeds 42,123,456
"""
import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np


from lwp.agents.eval_utils import (
    resolve_model_path, resolve_vec_normalize_path,
    load_training_config, make_env_factory,
)
from lwp.probing.collection import collect_probe_data


def _find_agent_dirs(parent_dir, seeds=None):
    """Find agent checkpoint directories, optionally filtering by seed.

    Walks looking for dirs containing model.zip or config.json.
    If seeds is specified, only include dirs matching s{seed} pattern.
    """
    agent_dirs = []
    for root, dirs, files in os.walk(parent_dir, followlinks=True):
        if "model.zip" in files or "config.json" in files:
            if seeds is not None:
                dirname = os.path.basename(root)
                if dirname.startswith("s"):
                    try:
                        dir_seed = int(dirname[1:])
                        if dir_seed not in seeds:
                            dirs.clear()
                            continue
                    except ValueError:
                        pass
            agent_dirs.append(root)
            dirs.clear()
    return sorted(agent_dirs)


def main():
    parser = argparse.ArgumentParser(
        description="Collect probe data (activations + targets) from trained agents.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint-dir",
                       help="Single agent checkpoint directory")
    group.add_argument("--agents-dir",
                       help="Parent dir — collect from all agents underneath")

    parser.add_argument("--episodes", type=int, default=100,
                        help="Episodes per agent (default: 100)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Env random seed (default: 0)")
    parser.add_argument("--seeds", type=str, default=None,
                        help="Comma-separated agent seeds to include (e.g. '42,123,456')")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: {checkpoint-dir}/probe_data/)")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing probe data")

    args = parser.parse_args()

    # Parse seed filter
    seed_filter = None
    if args.seeds:
        seed_filter = [int(s.strip()) for s in args.seeds.split(",")]

    if args.checkpoint_dir:
        agent_dirs = [args.checkpoint_dir]
    else:
        agent_dirs = _find_agent_dirs(args.agents_dir, seeds=seed_filter)
        if not agent_dirs:
            print(f"ERROR: No agent checkpoints found under {args.agents_dir}")
            sys.exit(1)
        print(f"Found {len(agent_dirs)} agents under {args.agents_dir}")

    for agent_dir in agent_dirs:
        agent_name = os.path.basename(agent_dir)
        parent_name = os.path.basename(os.path.dirname(agent_dir))
        display_name = f"{parent_name}/{agent_name}"
        print(f"\n{'='*60}")
        print(f"Agent: {display_name}")
        print(f"{'='*60}")

        # Output directory
        output_dir = args.output_dir or os.path.join(agent_dir, "probe_data")
        npz_path = os.path.join(output_dir, "probe_data.npz")

        # Check existing
        if os.path.exists(npz_path) and not args.force:
            print(f"  SKIP: {npz_path} already exists (use --force to overwrite)")
            continue

        # Load config and model
        try:
            model_path = resolve_model_path(agent_dir)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue

        vec_norm_path = resolve_vec_normalize_path(agent_dir, model_path)

        try:
            train_config = load_training_config(agent_dir)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue

        variant = train_config["variant"]
        algo = train_config.get("algo", "ppo")
        n_rays = train_config.get("n_rays", 7)
        history_k = train_config.get("history_k", 8)

        is_visual = variant.startswith("visual")
        n_stack = train_config.get("n_stack", 4 if is_visual else 0)
        frame_size = train_config.get("frame_size", 84)

        # Auto-detect training profile
        profile = train_config.get("profile")

        print(f"  Config: {variant} / {algo.upper()}")
        if is_visual:
            print(f"  Visual: {frame_size}px, n_stack={n_stack}")
        if profile:
            print(f"  Profile: {profile}")

        # Load model
        from stable_baselines3 import PPO, SAC
        AlgoClass = PPO if algo == "ppo" else SAC
        model = AlgoClass.load(model_path, device="cpu")

        # Build env factory with the training profile
        env_fn = make_env_factory(
            variant=variant, n_rays=n_rays, history_k=history_k,
            profile=profile, frame_size=frame_size,
        )

        # Collect
        print(f"  Collecting {args.episodes} episodes...")
        result = collect_probe_data(
            model=model,
            env_fn=env_fn,
            n_episodes=args.episodes,
            seed=args.seed,
            vec_normalize_path=vec_norm_path,
            variant=variant,
            n_stack=n_stack,
        )

        # Save — write all activation arrays + targets
        os.makedirs(output_dir, exist_ok=True)
        save_dict = {}
        for key, val in result.items():
            if isinstance(val, np.ndarray):
                save_dict[key] = val
            elif isinstance(val, str):
                save_dict[key] = val
        np.savez(npz_path, **save_dict)

        # Print summary
        layer_names = json.loads(result["layer_names"])
        n_timesteps = len(result["episode_ids"])
        file_mb = os.path.getsize(npz_path) / (1024 * 1024)
        print(f"  Saved {n_timesteps} timesteps to {npz_path} ({file_mb:.1f} MB)")
        for name in layer_names:
            arr = result[f"activations_{name}"]
            print(f"  {name}: {arr.shape}")

    print("\nDone.")


if __name__ == "__main__":
    main()
