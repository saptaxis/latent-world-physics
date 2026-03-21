#!/usr/bin/env python3
"""Prepare raw episode .npz files into a single training-ready dataset.

Loads all episodes from one or more directories, resizes RGB frames to
(frame_size × frame_size) grayscale, extracts kinematic targets, and
saves everything to a single .npz file for fast in-memory training.

Usage:
    python lunar_lander/scripts/prepare_encoder_dataset.py \
        --data-dir /path/to/random /path/to/heuristic \
        --output /path/to/prepared.npz \
        --frame-size 128

Output format:
    frames: (N_total, H, W) uint8 — all grayscale frames concatenated
    states: (N_total, 6) float32 — [x, y, vx, vy, angle, ang_vel] per frame
    episode_ends: (n_episodes,) int64 — cumulative frame count per episode
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm



def prepare_dataset(
    data_dirs: list[str | Path],
    output_path: str | Path,
    frame_size: int = 128,
) -> dict:
    """Convert raw episodes to a single prepared dataset file.

    Args:
        data_dirs: List of directories containing .npz episode files.
        output_path: Where to save the prepared .npz.
        frame_size: Target resolution (square) for frame resize.

    Returns:
        Summary dict with n_episodes, n_frames, file_size_mb.
    """
    # Collect all episode paths, sorted for reproducibility.
    all_paths = []
    for d in data_dirs:
        all_paths.extend(sorted(Path(d).rglob("*.npz")))

    if not all_paths:
        raise ValueError(f"No .npz files found in {data_dirs}")

    frames_list = []
    states_list = []
    episode_ends = []
    total_frames = 0

    for path in tqdm(all_paths, desc="Processing episodes", unit="ep"):

        data = np.load(str(path))
        rgb_frames = data["rgb_frames"]  # (T+1, H, W, 3) uint8
        ep_states = data["states"][:, :6].astype(np.float32)  # (T+1, 6)

        # Resize + grayscale each frame.
        processed = []
        for frame in rgb_frames:
            frame = cv2.resize(frame, (frame_size, frame_size))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  # BT.709 luminance
            processed.append(frame)

        ep_frames = np.stack(processed, axis=0)  # (T+1, H, W) uint8
        frames_list.append(ep_frames)
        states_list.append(ep_states)

        total_frames += ep_frames.shape[0]
        episode_ends.append(total_frames)

    # Concatenate all episodes into flat arrays.
    all_frames = np.concatenate(frames_list, axis=0)   # (N_total, H, W) uint8
    all_states = np.concatenate(states_list, axis=0)   # (N_total, 6) float32
    all_ends = np.array(episode_ends, dtype=np.int64)  # (n_episodes,)

    print(f"  Total: {len(all_paths)} episodes, {total_frames:,} frames")
    print(f"  Frames shape: {all_frames.shape}, dtype: {all_frames.dtype}")
    print(f"  States shape: {all_states.shape}")

    # Save.
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(output_path),
        frames=all_frames,
        states=all_states,
        episode_ends=all_ends,
    )

    file_size_mb = output_path.stat().st_size / 1e6
    print(f"  Saved to {output_path} ({file_size_mb:.1f} MB)")

    return {
        "n_episodes": len(all_paths),
        "n_frames": total_frames,
        "file_size_mb": file_size_mb,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Prepare raw episodes into a single training-ready dataset."
    )
    parser.add_argument(
        "--data-dir", type=str, required=True, nargs="+",
        help="Directory(ies) containing raw .npz episodes with rgb_frames.",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output path for prepared .npz file.",
    )
    parser.add_argument("--frame-size", type=int, default=128)
    args = parser.parse_args()

    print(f"Preparing dataset from {args.data_dir}...")
    prepare_dataset(
        [Path(d) for d in args.data_dir],
        args.output,
        frame_size=args.frame_size,
    )


if __name__ == "__main__":
    main()
