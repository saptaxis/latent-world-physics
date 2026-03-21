#!/usr/bin/env python
"""Collect trajectory data for world model training.

Unified collection script supporting all policy source types: trained RL
agents (blind, labeled), heuristic controller, random policy, and noisy-expert.
Configured via YAML — one config per collection run, or a batch config
to orchestrate multiple runs.

Single collection:
    CUDA_VISIBLE_DEVICES="" python lunar_lander/scripts/collect_world_model_data.py \
        --config configs/wm-data-collection/random.yaml \
        --output-dir /path/to/output

Batch collection:
    CUDA_VISIBLE_DEVICES="" python lunar_lander/scripts/collect_world_model_data.py \
        --batch configs/wm-data-collection/batch-full-100k.yaml

See specs/data-collection.md for design and rationale.
"""

import sys
import argparse
from pathlib import Path

import numpy as np


from lwp.collection.wm_collection_config import CollectionConfig, BatchConfig
from lwp.collection.wm_collection import collect_simple, collect_rl_agent, collect_primitive
from lwp.collection.wm_policies import make_random_policy
from parametric_lunar_lander.heuristic import heuristic_policy


def _run_single_collection(config: CollectionConfig, output_dir: str, n_workers: int = 1) -> None:
    """Run one collection according to config, save to output_dir."""
    print(f"\n{'='*60}")
    print(f"Collection: {config.source_type} -> {output_dir}")
    print(f"Episodes: {config.n_episodes}, seed: {config.seed}")
    if config.twr_range is not None:
        print(f"TWR constraint: [{config.twr_range[0]}, {config.twr_range[1]}]")
    print(f"{'='*60}")

    if config.source_type == "random":
        policy_fn = make_random_policy(seed=config.seed)
        results = collect_simple(
            policy_fn=policy_fn,
            output_dir=output_dir,
            n_episodes=config.n_episodes,
            physics_ranges=config.physics_ranges,
            source_type="random",
            seed=config.seed,
            save_frames=config.save_frames,
            n_workers=n_workers,
            twr_range=config.twr_range,
        )

    elif config.source_type == "heuristic":
        results = collect_simple(
            policy_fn=heuristic_policy,
            output_dir=output_dir,
            n_episodes=config.n_episodes,
            physics_ranges=config.physics_ranges,
            source_type="heuristic",
            seed=config.seed,
            save_frames=config.save_frames,
            n_workers=n_workers,
            twr_range=config.twr_range,
        )

    elif config.source_type == "primitive":
        results = collect_primitive(
            output_dir=output_dir,
            n_episodes=config.n_episodes,
            physics_ranges=config.physics_ranges,
            maneuver_config=config.maneuver_config,
            start_config=config.start_config,
            seed=config.seed,
            max_steps=config.max_steps,
            save_frames=config.save_frames,
            allow_post_landing=config.allow_post_landing,
            twr_range=config.twr_range,
        )

    elif config.source_type in ("blind_agent", "labeled_agent", "noisy_expert"):
        from stable_baselines3 import PPO, SAC
        from lwp.agents.eval_utils import (
            resolve_model_path,
            resolve_vec_normalize_path,
            load_training_config,
        )

        model_path = resolve_model_path(config.checkpoint_dir)
        vec_norm_path = resolve_vec_normalize_path(config.checkpoint_dir, model_path)
        train_config = load_training_config(config.checkpoint_dir)

        algo = train_config.get("algo", "ppo")
        variant = train_config["variant"]
        AlgoClass = PPO if algo == "ppo" else SAC
        model = AlgoClass.load(model_path)  # runs on CPU via CUDA_VISIBLE_DEVICES=""

        print(f"  Agent: {config.checkpoint_dir}")
        print(f"  Variant: {variant}, algo: {algo.upper()}")

        noise_sigma_range = config.noise_sigma if config.source_type == "noisy_expert" else None

        results = collect_rl_agent(
            model=model,
            output_dir=output_dir,
            n_episodes=config.n_episodes,
            variant=variant,
            physics_ranges=config.physics_ranges,
            source_type=config.source_type,
            seed=config.seed,
            vec_normalize_path=vec_norm_path,
            deterministic=config.deterministic,
            save_frames=config.save_frames,
            noise_sigma_range=noise_sigma_range,
            n_rays=train_config.get("n_rays", 7),
            history_k=train_config.get("history_k", 8),
            twr_range=config.twr_range,
        )

    else:
        raise ValueError(f"Unknown source_type: {config.source_type}")

    n_landed = sum(1 for r in results if r["outcome"] == "landed")
    mean_reward = np.mean([r["reward"] for r in results])
    print(f"\n  Done: {len(results)} episodes")
    print(f"  Landed: {n_landed}/{len(results)} ({100*n_landed/len(results):.0f}%)")
    print(f"  Mean reward: {mean_reward:.1f}")


def main():
    parser = argparse.ArgumentParser(
        description="Collect trajectory data for world model training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", help="Path to single collection YAML config")
    group.add_argument("--batch", help="Path to batch YAML config")

    parser.add_argument("--output-dir", help="Output directory (required for --config, optional override for --batch)")
    parser.add_argument("--sample", type=int, default=None,
                        help="Override n_episodes for quick validation (e.g., --sample 10)")
    parser.add_argument("--save-frames", action="store_true",
                        help="Override save_frames=true (store RGB frames in .npz)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel workers. Single collection: env workers for simple "
                             "sources. Batch mode: parallel collections. (default: 1)")

    args = parser.parse_args()

    if args.config:
        if not args.output_dir:
            parser.error("--output-dir is required with --config")
        config = CollectionConfig.load(args.config)
        if args.sample:
            config.n_episodes = args.sample
        if args.save_frames:
            config.save_frames = True
        _run_single_collection(config, args.output_dir, n_workers=args.workers)

    elif args.batch:
        batch = BatchConfig.load(args.batch)
        if args.output_dir:
            batch.output_base = args.output_dir
        for entry in batch.entries:
            if args.sample:
                entry.config.n_episodes = args.sample
            if args.save_frames:
                entry.config.save_frames = True
        print(f"Batch: {len(batch.entries)} collections -> {batch.output_base}")

        if args.workers > 1:
            from concurrent.futures import ProcessPoolExecutor, as_completed
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures = {}
                for entry in batch.entries:
                    output_dir = str(Path(batch.output_base) / entry.output_name)
                    future = pool.submit(_run_single_collection, entry.config, output_dir)
                    futures[future] = entry.output_name
                for future in as_completed(futures):
                    name = futures[future]
                    try:
                        future.result()
                        print(f"  Completed: {name}")
                    except Exception as e:
                        print(f"  FAILED: {name}: {e}")
        else:
            for entry in batch.entries:
                output_dir = str(Path(batch.output_base) / entry.output_name)
                _run_single_collection(entry.config, output_dir)

    print("\nAll done.")


if __name__ == "__main__":
    main()
