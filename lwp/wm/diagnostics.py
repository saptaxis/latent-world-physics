"""World model data loading utilities.

Helpers for loading episodes, configs, and applying supervision masks.
Used by physics test scripts, evaluation scripts, and physics understanding reports.

TODO: Move these to a better home (lwp/wm/dataset.py or lwp/data/) in post-migration cleanup.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from lwp.wm.train_config import TrainConfig


def _load_config(run_dir: Path) -> TrainConfig:
    """Load training config from run directory.

    Prefers train_config.yaml (original nested YAML, saved by Task 1).
    Falls back to config.json (flat dict) for older runs.
    """
    yaml_path = run_dir / "train_config.yaml"
    json_path = run_dir / "config.json"

    if yaml_path.exists():
        return TrainConfig.load(yaml_path)
    elif json_path.exists():
        with open(json_path) as f:
            flat = json.load(f)
        return TrainConfig.from_flat_dict(flat)
    else:
        raise FileNotFoundError(
            f"No config found in {run_dir}. Expected train_config.yaml or config.json. "
            f"For older runs, copy the original YAML config to {yaml_path}."
        )


def _apply_supervision_mask(states: np.ndarray, supervision: str) -> np.ndarray:
    """Slice to kinematic dims (0:8) for blind supervision, matching training."""
    if supervision == "blind":
        return states[:, :8].copy()
    return states


def _load_episodes_from_dir(data_dir: Path, n_episodes: int) -> list[dict]:
    """Load episode .npz files directly from a directory.

    Used by the wm-ladder code path where episodes come from a data
    directory rather than a split_index.json file. Recursively finds
    all .npz files. If multiple subdirectories exist, samples
    proportionally from each to ensure coverage across all data types
    (e.g., free-fall, thrust, side-thrust primitives).

    Args:
        data_dir: Directory containing episode .npz files (searched recursively).
        n_episodes: Maximum number of episodes to load.

    Returns:
        List of episode dicts with 'states', 'actions', and optionally
        'metadata_json' keys.
    """
    # Group .npz files by immediate subdirectory.
    from collections import defaultdict
    files_by_subdir = defaultdict(list)
    for f in sorted(data_dir.glob("**/*.npz")):
        # Key = first subdir relative to data_dir, or "" for files in root.
        rel = f.relative_to(data_dir)
        key = rel.parts[0] if len(rel.parts) > 1 else ""
        files_by_subdir[key].append(f)

    # If only one group (flat dir or single subdir), just take first n.
    if len(files_by_subdir) <= 1:
        all_files = sorted(data_dir.glob("**/*.npz"))[:n_episodes]
    else:
        # Proportional sampling across subdirs.
        all_npz = list(data_dir.glob("**/*.npz"))
        total = len(all_npz)
        rng = np.random.default_rng(42)
        all_files = []
        for key in sorted(files_by_subdir.keys()):
            subdir_files = files_by_subdir[key]
            # Proportion of total, but at least 1 per subdir.
            n_from_subdir = max(1, round(len(subdir_files) / total * n_episodes))
            if n_from_subdir >= len(subdir_files):
                all_files.extend(subdir_files)
            else:
                idx = rng.choice(len(subdir_files), n_from_subdir, replace=False)
                all_files.extend(subdir_files[int(i)] for i in sorted(idx))
        # Trim to n_episodes if rounding pushed us over.
        if len(all_files) > n_episodes:
            all_files = all_files[:n_episodes]

    episodes = []
    for f in all_files:
        data = np.load(f)
        ep = {"states": data["states"], "actions": data["actions"]}
        if "metadata_json" in data:
            ep["metadata_json"] = str(data["metadata_json"])
        episodes.append(ep)
    return episodes


def _load_episodes(split_index_path: Path, split: str, max_episodes: int | None = None):
    """Load episodes for a given split from the split index."""
    with open(split_index_path) as f:
        index = json.load(f)
    paths = [p for p, s in index.items() if s == split]
    if max_episodes:
        paths = paths[:max_episodes]
    episodes = []
    for p in paths:
        data = np.load(p, allow_pickle=True)
        ep = {
            "states": data["states"],
            "actions": data["actions"],
            "rewards": data["rewards"],
        }
        # Include metadata if available (contains seed, physics_config, etc.).
        if "metadata_json" in data:
            ep["metadata_json"] = str(data["metadata_json"])
        episodes.append(ep)
    return episodes
