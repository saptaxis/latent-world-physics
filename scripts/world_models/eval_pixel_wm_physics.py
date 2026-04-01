#!/usr/bin/env python3
"""Run pixel world model physics evaluation (all 4 layers).

Usage:
    python lunar_lander/scripts/eval_pixel_wm_physics.py \
        --vae-checkpoint path/to/vae/best.pt \
        --dynamics-checkpoint path/to/dynamics/best.pt \
        --data-path /path/to/episodes \
        --output-dir /path/to/output \
        --n-episodes 200
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

# Add project root to path so imports work when running as script.
# __file__ is lunar_lander/scripts/eval_pixel_wm_physics.py, so two levels
# up is the repo root (latent-world-geometry/).


def parse_args():
    p = argparse.ArgumentParser(description="Pixel WM Physics Evaluation")
    p.add_argument("--vae-checkpoint", type=str, required=True)
    p.add_argument("--dynamics-checkpoint", type=str, required=True)
    # nargs="+" allows multiple data paths (e.g. multiple episode directories)
    p.add_argument("--data-path", type=str, nargs="+", required=True)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--n-episodes", type=int, default=200)
    # horizons as comma-separated string for CLI ergonomics
    p.add_argument("--horizons", type=str, default="1,5,10,20,50")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--n-per-policy", type=int, default=25,
                   help="Episodes per policy subdir (stratified sampling)")
    p.add_argument("--per-policy", action="store_true",
                   help="Run eval separately per policy and produce per-policy reports")
    p.add_argument("--start-stride", type=int, default=0,
                   help="Dream start stride (0=single start from frame 0)")
    p.add_argument("--max-horizon", type=int, default=0,
                   help="Max dream horizon per start (0=dream to end)")
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed for episode sampling. Different seeds select "
                        "different eval episodes for independent variance estimates.")
    return p.parse_args()


def load_vae(checkpoint_path: str, device: str):
    """Load VAE from checkpoint, dispatching on model_type.

    Supports standard PixelVAE and FactoredPixelVAE. The checkpoint's
    config dict determines which class to instantiate. Must have
    state_dim > 0 (either via MLP state head or z_kin slice) for
    physics eval to extract per-dim kinematics.
    """
    from lwp.models.pixel_vae import PixelVAE
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model_type = cfg.get("model_type", "standard")

    if model_type == "factored":
        from lwp.models.factored_pixel_vae import FactoredPixelVAE
        vae = FactoredPixelVAE(
            in_channels=cfg["in_channels"],
            latent_dim=cfg["latent_dim"],
            frame_size=cfg["frame_size"],
            channels=cfg.get("channels", [32, 64, 128, 256]),
            kin_targets=cfg.get("kin_targets", [0, 1, 2, 3, 4, 5]),
            decoder_type=cfg.get("decoder_type", "concat"),
            coord_conv=cfg.get("coord_conv", False),
        )
    else:
        vae = PixelVAE(
            in_channels=cfg["in_channels"],
            latent_dim=cfg["latent_dim"],
            frame_size=cfg["frame_size"],
            channels=cfg.get("channels", [32, 64, 128, 256]),
            state_dim=cfg.get("state_dim", 0),
            coord_conv=cfg.get("coord_conv", False),
        )

    vae.load_state_dict(ckpt["model_state_dict"])
    vae.to(device)
    vae.eval()
    if vae.state_dim == 0:
        print("ERROR: VAE has state_dim=0 (no state head). "
              "Pixel physics eval requires state_dim > 0.")
        sys.exit(1)
    print(f"  VAE ({model_type}): latent_dim={cfg['latent_dim']}, "
          f"frame_size={cfg['frame_size']}, state_dim={vae.state_dim}, "
          f"{sum(p.numel() for p in vae.parameters()):,} params")
    return vae, cfg


def load_dynamics(checkpoint_path: str, latent_dim: int, device: str):
    """Load dynamics model, dispatching on model_type in config.

    Supports GRU (LatentDynamicsModel) and RSSM (LatentRSSM).
    The model_type key in config determines which class is instantiated.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model_type = cfg.get("model_type", "gru")
    if model_type == "rssm":
        from lwp.models.pixel_rssm import LatentRSSM
        dynamics = LatentRSSM(
            latent_dim=latent_dim,
            action_dim=cfg.get("action_dim", 2),
            deter_dim=cfg.get("deter_dim", 200),
            stoch_dim=cfg.get("stoch_dim", 30),
            hidden_dim=cfg.get("hidden_size", 200),
        )
    elif model_type == "film":
        from lwp.models.pixel_dynamics import FiLMDynamicsModel
        dynamics = FiLMDynamicsModel(
            latent_dim=latent_dim,
            action_dim=cfg.get("action_dim", 2),
            hidden_size=cfg.get("hidden_size", 256),
        )
    elif model_type == "factored-dyn":
        from lwp.models.factored_dynamics import FactoredDynamicsModel
        dynamics = FactoredDynamicsModel(
            latent_dim=latent_dim,
            action_dim=cfg.get("action_dim", 2),
            hidden_size=cfg.get("hidden_size", 256),
            kin_dims=cfg.get("kin_dims", 6),
        )
    else:
        from lwp.models.pixel_dynamics import LatentDynamicsModel
        dynamics = LatentDynamicsModel(
            latent_dim=latent_dim,
            action_dim=cfg.get("action_dim", 2),
            hidden_size=cfg.get("hidden_size", 256),
        )
    dynamics.load_state_dict(ckpt["model_state_dict"])
    dynamics.to(device)
    dynamics.eval()
    print(f"  Dynamics: {model_type}, {sum(p.numel() for p in dynamics.parameters()):,} params")
    return dynamics


def collect_episode_paths(
    data_paths: list[str], n_per_policy: int = 25, seed: int | None = None,
) -> list[str]:
    """Collect npz episode paths with stratified sampling across subdirs.

    Samples n_per_policy episodes from EACH subdir that contains .npz files.
    This ensures balanced representation across trajectory types
    (heuristic, random, free-fall, impulse-main, etc.) instead of
    biasing toward whichever subdir has the most episodes.

    When seed is provided, episodes are randomly sampled within each subdir
    (different seed = different eval episodes). When seed is None, uses
    deterministic evenly-spaced sampling (backward compatible).
    """
    import random as _random
    rng = _random.Random(seed) if seed is not None else None

    all_paths = []
    for dp in data_paths:
        # Find all subdirs containing .npz files.
        # Use os.walk with followlinks=True because the combined prims
        # data dir uses symlinks to the actual episode directories.
        # Path.rglob does NOT follow symlinks by default.
        subdirs: dict[str, list[str]] = {}
        for root, _dirs, files in os.walk(dp, followlinks=True):
            if "cache" in root or "prepared" in root:
                continue
            for fname in sorted(files):
                if fname.endswith(".npz") and fname.startswith("episode_"):
                    fpath = os.path.join(root, fname)
                    subdir = os.path.basename(root)
                    if subdir not in subdirs:
                        subdirs[subdir] = []
                    subdirs[subdir].append(fpath)

        # Sample n_per_policy from each subdir
        for subdir, files in sorted(subdirs.items()):
            if len(files) <= n_per_policy:
                all_paths.extend(files)
            elif rng is not None:
                # Random sampling — different seed = different episodes
                all_paths.extend(rng.sample(files, n_per_policy))
            else:
                # Evenly spaced sampling (deterministic, backward compatible)
                step = len(files) // n_per_policy
                all_paths.extend([files[i * step] for i in range(n_per_policy)])

    return all_paths


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy arrays and scalar types.

    json.dump fails on np.ndarray, np.float32, etc. by default —
    this encoder converts them to native Python types for clean JSON output.
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return super().default(obj)


def main():
    args = parse_args()
    horizons = [int(h) for h in args.horizons.split(",")]

    print(f"Loading VAE from {args.vae_checkpoint}...")
    vae, vae_cfg = load_vae(args.vae_checkpoint, args.device)
    frame_size = vae_cfg["frame_size"]

    print(f"Loading dynamics from {args.dynamics_checkpoint}...")
    dynamics = load_dynamics(args.dynamics_checkpoint, vae_cfg["latent_dim"], args.device)

    print(f"Collecting episodes from {args.data_path}...")
    # Use stratified sampling: n_per_policy episodes from each subdir
    paths = collect_episode_paths(args.data_path, args.n_per_policy, seed=args.seed)
    print(f"  Found {len(paths)} episodes")
    if not paths:
        print("ERROR: No episode files found. Check --data-path.")
        sys.exit(1)

    # Import here (after sys.path is set) so the script is self-contained
    from lwp.wm.pixel_physics_eval import load_eval_episodes, run_full_eval, format_report

    print(f"Loading episodes (frame_size={frame_size})...")
    episodes = load_eval_episodes(paths, frame_size=frame_size)
    print(f"  Loaded {len(episodes)} episodes")

    # Derive a model name from the checkpoint basename for the report header
    model_name = Path(args.vae_checkpoint).parent.name

    if args.per_policy:
        # Group episodes by policy label
        by_policy: dict[str, list[dict]] = {}
        for ep in episodes:
            policy = ep.get("policy", "unknown")
            by_policy.setdefault(policy, []).append(ep)

        all_results = {}
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        for policy, policy_eps in sorted(by_policy.items()):
            print(f"\n=== Policy: {policy} ({len(policy_eps)} episodes) ===")
            policy_results = run_full_eval(
                vae, dynamics, policy_eps, horizons=horizons, device=args.device,
                start_stride=args.start_stride, max_horizon=args.max_horizon)
            all_results[policy] = policy_results

            # Write per-policy outputs
            policy_dir = out_dir / policy
            policy_dir.mkdir(parents=True, exist_ok=True)
            report = format_report(policy_results, model_name=f"{model_name}/{policy}")
            (policy_dir / "report.md").write_text(report)
            # JSON per layer
            for key, val in policy_results.items():
                with open(policy_dir / f"{key}.json", "w") as f:
                    json.dump(val, f, indent=2, cls=_NumpyEncoder)

        # Also write aggregate report combining all policies
        print(f"\n=== Aggregate ({len(episodes)} episodes) ===")
        results = run_full_eval(vae, dynamics, episodes, horizons=horizons, device=args.device,
                                start_stride=args.start_stride, max_horizon=args.max_horizon)
    else:
        # Aggregate (existing behavior)
        print(f"Running evaluation on {args.device} (horizons={horizons})...")
        results = run_full_eval(vae, dynamics, episodes, horizons=horizons, device=args.device,
                                start_stride=args.start_stride, max_horizon=args.max_horizon)

    # Write outputs — one JSON file per layer key for easy downstream loading,
    # plus a combined markdown report for human review.
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON per layer
    for key, val in results.items():
        json_path = out_dir / f"{key}.json"
        with open(json_path, "w") as f:
            json.dump(val, f, indent=2, cls=_NumpyEncoder)

    # Markdown report
    report = format_report(results, model_name=model_name)
    report_path = out_dir / "report.md"
    with open(report_path, "w") as f:
        f.write(report)

    print(f"\nResults written to {out_dir}/")
    print(f"  Report: {report_path}")
    print(f"\n{report}")


if __name__ == "__main__":
    main()
