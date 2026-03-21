"""Episode index builder for world model training splits.

Scans .npz episode metadata (without loading full state arrays), assigns
episodes to train/val/test/ood splits using quantile grid on physics params,
and persists the assignment as a JSON index.

The split is region-based (not episode-based): the physics space is binned
into a 3D grid, and entire regions are assigned to splits. This prevents
near-identical physics from leaking across splits.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from lwp.wm.mix_config import MixConfig


def build_episode_index(
    mix_config: MixConfig,
    save_path: str | Path | None = None,
    seed: int = 42,
) -> dict[str, str]:
    """Scan episodes, assign to splits, return {episode_path: split_label}.

    Steps:
      1. Scan all .npz files across selected profiles.
      2. Read metadata_json from each (fast — no state arrays loaded).
      3. Apply OOD holdout first (episodes matching ALL OOD conditions → "ood").
      4. Apply policy holdout (episodes from held-out source → "policy_holdout").
      5. Remaining: compute quantile bins per axis, assign 3D bin index,
         randomly assign regions to train/val/test.
      6. Persist as JSON if save_path is given.

    Args:
        mix_config: MixConfig instance defining profiles, data_base, split rules.
        save_path: Optional path to save the JSON index.
        seed: RNG seed for deterministic region assignment.

    Returns:
        Dict mapping episode file paths (str) to split labels.
    """
    rng = np.random.default_rng(seed)

    # -- Step 1-2: Scan episodes and extract physics metadata --
    # Only reads metadata_json from each .npz — no state/action arrays loaded.
    episodes = []  # list of (path_str, physics_dict, source_type)
    base = Path(mix_config.data_base)
    for profile in mix_config.profiles:
        profile_dir = base / profile["path"]
        # Sorted for determinism across filesystems.
        all_npz = sorted(profile_dir.rglob("*.npz"))
        for npz_path in tqdm(all_npz, desc=f"Scanning {profile['path']}", disable=len(all_npz) < 50):
            data = np.load(str(npz_path), allow_pickle=False)
            meta = json.loads(str(data["metadata_json"]))
            physics = meta["physics_config"]
            source_type = meta.get("source_type", "unknown")
            episodes.append((str(npz_path), physics, source_type))

    index = {}

    # -- Step 3: OOD holdout (applied first) --
    # Episodes where ALL physics params fall within OOD ranges → "ood" split.
    remaining = []
    if mix_config.ood_holdout:
        for path, physics, source_type in episodes:
            in_ood = all(
                lo <= physics.get(param, float("inf")) <= hi
                for param, (lo, hi) in mix_config.ood_holdout.items()
            )
            if in_ood:
                index[path] = "ood"
            else:
                remaining.append((path, physics, source_type))
    else:
        remaining = list(episodes)

    # -- Step 4: Policy holdout --
    # Episodes from the held-out source type → "policy_holdout" split.
    if mix_config.policy_holdout:
        after_policy = []
        for path, physics, source_type in remaining:
            if source_type == mix_config.policy_holdout:
                index[path] = "policy_holdout"
            else:
                after_policy.append((path, physics, source_type))
        remaining = after_policy

    # -- Step 5: Assign remaining episodes to train/val/test --
    if not remaining:
        if save_path:
            Path(save_path).write_text(json.dumps(index, indent=2))
        return index

    if mix_config.split_method == "random":
        # Simple episode-level random split. Appropriate for fixed-physics
        # datasets where region-based splitting is meaningless (all episodes
        # share identical physics, so there's only one "region").
        paths = [path for path, _, _ in remaining]
        rng.shuffle(paths)
        n = len(paths)
        n_train = max(1, int(n * mix_config.train_ratio))
        n_val = max(1, int(n * mix_config.val_ratio))
        for i, path in enumerate(paths):
            if i < n_train:
                index[path] = "train"
            elif i < n_train + n_val:
                index[path] = "val"
            else:
                index[path] = "test"
    else:
        # Quantile grid: bin episodes by physics params, assign whole regions.
        # Prevents near-identical physics from leaking across splits.
        axes = mix_config.split_axes
        n_bins = mix_config.bins
        values = {axis: np.array([p[axis] for _, p, _ in remaining]) for axis in axes}

        # Compute quantile bin edges per axis.
        # For bins=3, we get 2 inner edges (tercile boundaries).
        bin_edges = {}
        for axis in axes:
            quantiles = np.linspace(0, 1, n_bins + 1)[1:-1]  # inner edges only
            bin_edges[axis] = np.quantile(values[axis], quantiles)

        # Assign each episode a multi-dimensional bin index.
        episode_bins = []
        for path, physics, source_type in remaining:
            bin_idx = tuple(
                int(np.searchsorted(bin_edges[axis], physics[axis]))
                for axis in axes
            )
            episode_bins.append((path, bin_idx))

        # Collect unique regions (bin index tuples) and randomly assign to splits.
        regions = sorted(set(idx for _, idx in episode_bins))
        rng.shuffle(regions)

        n_train = max(1, int(len(regions) * mix_config.train_ratio))
        n_val = max(1, int(len(regions) * mix_config.val_ratio))
        region_map = {}
        for i, region in enumerate(regions):
            if i < n_train:
                region_map[region] = "train"
            elif i < n_train + n_val:
                region_map[region] = "val"
            else:
                region_map[region] = "test"

        # Map episodes to splits via their region assignment.
        for path, bin_idx in episode_bins:
            index[path] = region_map[bin_idx]

    # -- Step 6: Persist --
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        Path(save_path).write_text(json.dumps(index, indent=2))

    return index


def load_episode_index(path: str | Path) -> dict[str, str]:
    """Load a persisted split index from JSON."""
    return json.loads(Path(path).read_text())
