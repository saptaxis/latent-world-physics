#!/usr/bin/env python
"""Train linear probes from collected activation data.

Loads probe_data.npz (from collect_probe_data.py), trains ridge regression
probes for each layer x target combination, and saves R² results as JSON.

Usage:
    # Single agent
    python lunar_lander/scripts/train_probes.py \
        --probe-data /path/to/agent/s42/probe_data/probe_data.npz

    # All probe_data.npz files under a directory tree
    python lunar_lander/scripts/train_probes.py \
        --agents-dir /path/to/agents/

    # Specific layers/targets
    python lunar_lander/scripts/train_probes.py \
        --probe-data /path/to/probe_data.npz \
        --layers L1,L2 --targets twr,gravity
"""
import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np


from lwp.probing.training import train_all_probes
from lwp.probing.targets import ALL_TARGET_NAMES


def _find_probe_data_files(parent_dir):
    """Find all probe_data.npz files under a directory tree."""
    paths = []
    for root, dirs, files in os.walk(parent_dir, followlinks=True):
        if "probe_data.npz" in files:
            paths.append(os.path.join(root, "probe_data.npz"))
    return sorted(paths)


def _infer_agent_info(npz_path):
    """Infer agent name, seed, variant from the directory structure.

    Expected: .../config-name/s{seed}/probe_data/probe_data.npz
    """
    probe_dir = os.path.dirname(npz_path)
    seed_dir = os.path.dirname(probe_dir)
    config_dir = os.path.dirname(seed_dir)

    seed_name = os.path.basename(seed_dir)
    config_name = os.path.basename(config_dir)

    seed = None
    if seed_name.startswith("s"):
        try:
            seed = int(seed_name[1:])
        except ValueError:
            pass

    # Try to load config.json for variant info
    variant = None
    config_json = os.path.join(seed_dir, "config.json")
    if os.path.exists(config_json):
        with open(config_json) as f:
            cfg = json.load(f)
            variant = cfg.get("variant")

    return {
        "agent": config_name,
        "seed": seed,
        "variant": variant,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train linear probes on collected activation data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--probe-data",
                       help="Path to a single probe_data.npz file")
    group.add_argument("--agents-dir",
                       help="Parent dir — train probes for all probe_data.npz underneath")

    parser.add_argument("--layers", type=str, default=None,
                        help="Comma-separated layers to probe (default: L1,L2)")
    parser.add_argument("--targets", type=str, default=None,
                        help=f"Comma-separated targets (default: all {len(ALL_TARGET_NAMES)})")
    parser.add_argument("--kinematic-only", action="store_true",
                        help="Only probe kinematic state targets (x, y, vx, vy, angle, etc.). "
                             "Use for fixed-physics agents where physics probes are trivial.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: same dir as probe_data.npz)")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing probe_results.json")

    args = parser.parse_args()

    layers = args.layers.split(",") if args.layers else None
    if args.kinematic_only:
        from lwp.probing.targets import KINEMATIC_TARGET_NAMES
        targets = list(KINEMATIC_TARGET_NAMES)
    else:
        targets = args.targets.split(",") if args.targets else None

    if args.probe_data:
        npz_files = [args.probe_data]
    else:
        npz_files = _find_probe_data_files(args.agents_dir)
        if not npz_files:
            print(f"ERROR: No probe_data.npz found under {args.agents_dir}")
            sys.exit(1)
        print(f"Found {len(npz_files)} probe data files")

    for npz_path in npz_files:
        agent_info = _infer_agent_info(npz_path)
        display = f"{agent_info['agent']}/s{agent_info['seed']}"
        print(f"\n{'='*60}")
        print(f"Agent: {display}")
        print(f"{'='*60}")

        output_dir = args.output_dir or os.path.dirname(npz_path)
        results_path = os.path.join(output_dir, "probe_results.json")

        if os.path.exists(results_path) and not args.force:
            print(f"  SKIP: {results_path} exists (use --force)")
            continue

        # Load probe data
        data = dict(np.load(npz_path, allow_pickle=True))
        n_timesteps = len(data["episode_ids"])
        n_episodes = len(np.unique(data["episode_ids"]))
        print(f"  Data: {n_timesteps} timesteps, {n_episodes} episodes")

        # Detect available activation layers
        act_keys = sorted([k for k in data if k.startswith("activations_")])
        for ak in act_keys:
            layer_name = ak.replace("activations_", "")
            print(f"  {layer_name}: {data[ak].shape}")

        # Train probes
        print(f"  Training probes...")
        probes, coefficients = train_all_probes(data, layers=layers, targets=targets)

        # Build full results object
        result = {
            **agent_info,
            "n_timesteps": n_timesteps,
            "n_episodes": n_episodes,
            "probes": probes,
        }

        # Save JSON (R² scores, top units — no large arrays)
        os.makedirs(output_dir, exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(result, f, indent=2)

        # Save full coefficient matrices as .npz (for Phase C ablation/steering)
        coeff_path = os.path.join(output_dir, "probe_coefficients.npz")
        np.savez(coeff_path, **coefficients)
        coeff_mb = os.path.getsize(coeff_path) / (1024 * 1024)
        print(f"  Saved coefficients to {coeff_path} ({coeff_mb:.2f} MB)")

        # Print summary
        for layer_name, layer_results in probes.items():
            print(f"\n  {layer_name}:")
            for target_name, probe_result in layer_results.items():
                r2 = probe_result["r2_mean"]
                marker = "***" if r2 > 0.7 else "**" if r2 > 0.4 else "*" if r2 > 0.2 else ""
                print(f"    {target_name:<25s} R²={r2:.3f} ±{probe_result['r2_std']:.3f} {marker}")

        print(f"\n  Saved to {results_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
