#!/usr/bin/env python3
"""Probe encoder features with linear and/or MLP probes.

Loads a pre-trained or fine-tuned encoder, extracts features from the
encoder pretraining dataset, and runs probes on kinematic targets
[x, y, vx, vy, angle, ang_vel]. Compares linear vs MLP R² to test
whether information is linearly accessible or nonlinearly encoded.

Usage:
    # Pre-trained encoder (.pt file)
    python scripts/perception/probe_encoder_features.py \
        --encoder /path/to/best_encoder.pt \
        --data /path/to/prepared-npy \
        --probe-types linear,mlp \
        --output-dir ~/vsr-tmp/encoder-probe-pretrained

    # Fine-tuned encoder (SB3 model.zip — extracts features_extractor)
    python scripts/perception/probe_encoder_features.py \
        --encoder /path/to/model.zip \
        --data /path/to/prepared-npy \
        --probe-types linear,mlp \
        --output-dir ~/vsr-tmp/encoder-probe-finetuned
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Scripts import via sys.path (see conftest.py pattern).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.perception.pretrain_encoder import StandaloneImpalaCNN
from lwp.agents.encoder_dataset import PreparedEncoderDataset
from lwp.probing.training import train_single_probe, train_single_mlp_probe

TARGET_NAMES = ["x", "y", "vx", "vy", "angle", "ang_vel"]


def load_encoder(encoder_path: str) -> StandaloneImpalaCNN:
    """Load encoder from .pt (standalone) or .zip (SB3 model)."""
    path = Path(encoder_path)
    encoder = StandaloneImpalaCNN(in_channels=4, features_dim=256)

    if path.suffix == ".pt":
        state_dict = torch.load(str(path), map_location="cpu", weights_only=True)
        encoder.load_state_dict(state_dict)
        print(f"Loaded pre-trained encoder from {path.name}")
    elif path.suffix == ".zip":
        # Extract features_extractor weights directly from the zip file.
        # Can't use PPO.load() because old models pickle-reference
        # lunar_lander.src.visual_backbones which no longer exists.
        import zipfile
        import io
        with zipfile.ZipFile(str(path), "r") as zf:
            with zf.open("policy.pth") as f:
                # policy.pth contains the full policy state_dict
                buf = io.BytesIO(f.read())
                policy_state = torch.load(buf, map_location="cpu", weights_only=True)
        # Filter to features_extractor keys and strip the prefix.
        prefix = "features_extractor."
        fe_state = {
            k[len(prefix):]: v for k, v in policy_state.items()
            if k.startswith(prefix)
        }
        if not fe_state:
            raise ValueError(f"No features_extractor keys found in {path}")
        encoder.load_state_dict(fe_state)
        print(f"Extracted fine-tuned encoder from {path.name} ({len(fe_state)} params)")
    else:
        raise ValueError(f"Unknown encoder format: {path.suffix} (expected .pt or .zip)")

    encoder.eval()
    return encoder


def extract_features(
    encoder: StandaloneImpalaCNN,
    dataset: PreparedEncoderDataset,
    data_path: str,
    batch_size: int = 256,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract encoder features and kinematic targets from dataset.

    Returns:
        features: (N, 256) float32 — encoder output features
        targets: (N, 6) float32 — kinematic targets
        episode_ids: (N,) int32 — episode index per sample (for CV splitting)
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_features = []
    all_targets = []

    with torch.no_grad():
        for frames, tgt in tqdm(loader, desc="Extracting features"):
            frames = frames.to(device)
            feats = encoder(frames)
            all_features.append(feats.cpu().numpy())
            all_targets.append(tgt.numpy())

    features = np.concatenate(all_features, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    # Build episode IDs from the dataset's prepared data structure.
    # PreparedEncoderDataset builds a flat _index of valid frame-stack
    # start positions. We reconstruct which episode each stack belongs to
    # by reloading episode_ends and using searchsorted.
    data_path = Path(data_path)
    if data_path.is_dir():
        episode_ends = np.load(str(data_path / "episode_ends.npy"))
    else:
        episode_ends = np.load(str(data_path))["episode_ends"]
    index_arr = np.array(dataset._index)
    episode_ids = np.searchsorted(episode_ends, index_arr, side="right").astype(np.int32)

    return features, targets, episode_ids


def run_probes(
    features: np.ndarray,
    targets: np.ndarray,
    episode_ids: np.ndarray,
    probe_types: list[str],
    n_folds: int = 5,
    mlp_hidden_sizes: tuple[int, ...] = (64,),
) -> dict:
    """Run probes on all kinematic targets.

    Returns:
        {target_name: {probe_type: {r2_mean, r2_std, r2_folds, ...}}}
    """
    results = {}

    for i, name in enumerate(TARGET_NAMES):
        target = targets[:, i]
        results[name] = {}

        for ptype in probe_types:
            print(f"  {ptype:6s} {name:10s}...", end="", flush=True)

            t_probe = time.time()
            if ptype == "linear":
                result = train_single_probe(features, target, episode_ids, n_folds=n_folds)
                # Remove numpy arrays for JSON serialization
                result.pop("coefficients", None)
                result.pop("intercept", None)
            elif ptype == "mlp":
                result = train_single_mlp_probe(
                    features, target, episode_ids,
                    n_folds=n_folds, hidden_sizes=mlp_hidden_sizes,
                )
            else:
                raise ValueError(f"Unknown probe type: {ptype}")
            elapsed_probe = time.time() - t_probe

            r2 = result["r2_mean"]
            marker = "***" if r2 > 0.9 else "**" if r2 > 0.7 else "*" if r2 > 0.4 else ""
            print(f" R²={r2:.4f} ±{result['r2_std']:.4f} {marker}  ({elapsed_probe:.0f}s)")
            results[name][ptype] = result

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Probe encoder features with linear and/or MLP probes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--encoder", required=True,
                        help="Path to encoder weights (.pt) or SB3 model (.zip)")
    parser.add_argument("--data", required=True,
                        help="Path to prepared-npy directory or prepared.npz")
    parser.add_argument("--output-dir", required=True,
                        help="Directory for output JSON")
    parser.add_argument("--probe-types", default="linear,mlp",
                        help="Comma-separated probe types (default: linear,mlp)")
    parser.add_argument("--split", default="val",
                        help="Dataset split: train, val, or all (default: val)")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size for feature extraction (default: 256)")
    parser.add_argument("--n-folds", type=int, default=5,
                        help="CV folds (default: 5)")
    parser.add_argument("--mlp-hidden", default="64",
                        help="MLP hidden layer sizes, comma-separated (default: 64)")
    parser.add_argument("--device", default="cpu",
                        help="Device for encoder (default: cpu)")

    args = parser.parse_args()

    probe_types = [p.strip() for p in args.probe_types.split(",")]
    mlp_hidden = tuple(int(x) for x in args.mlp_hidden.split(","))
    split = None if args.split == "all" else args.split

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load encoder
    print(f"\n=== Loading encoder: {args.encoder}")
    encoder = load_encoder(args.encoder)
    if args.device != "cpu":
        encoder = encoder.to(args.device)

    # Load dataset
    print(f"\n=== Loading dataset: {args.data} (split={split})")
    ds = PreparedEncoderDataset(args.data, n_stack=4, split=split, seed=42)
    print(f"  {len(ds)} samples")

    # Extract features
    print(f"\n=== Extracting features (batch_size={args.batch_size})")
    t0 = time.time()
    features, targets, episode_ids = extract_features(
        encoder, ds, data_path=args.data,
        batch_size=args.batch_size, device=args.device,
    )
    print(f"  Features: {features.shape}, targets: {targets.shape}")
    print(f"  Unique episodes: {len(np.unique(episode_ids))}")
    print(f"  Extraction took {time.time() - t0:.1f}s")

    # Run probes
    print(f"\n=== Running probes: {probe_types}")
    t0 = time.time()
    results = run_probes(
        features, targets, episode_ids,
        probe_types=probe_types,
        n_folds=args.n_folds,
        mlp_hidden_sizes=mlp_hidden,
    )
    print(f"  Probing took {time.time() - t0:.1f}s")

    # Summary table
    print(f"\n=== Summary")
    header = f"  {'target':12s}"
    for pt in probe_types:
        header += f"  {pt:>12s}"
    if len(probe_types) == 2:
        header += f"  {'Δ(mlp-lin)':>12s}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for name in TARGET_NAMES:
        row = f"  {name:12s}"
        r2s = []
        for pt in probe_types:
            r2 = results[name][pt]["r2_mean"]
            r2s.append(r2)
            row += f"  {r2:12.4f}"
        if len(probe_types) == 2:
            delta = r2s[1] - r2s[0]
            row += f"  {delta:+12.4f}"
        print(row)

    # Save results
    output_file = output_dir / "encoder_probe_results.json"
    output = {
        "encoder_path": str(args.encoder),
        "data_path": str(args.data),
        "split": args.split,
        "n_samples": int(features.shape[0]),
        "n_episodes": int(len(np.unique(episode_ids))),
        "probe_types": probe_types,
        "mlp_hidden_sizes": list(mlp_hidden),
        "n_folds": args.n_folds,
        "results": results,
    }
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved to {output_file}")


if __name__ == "__main__":
    main()
