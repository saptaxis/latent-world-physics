#!/usr/bin/env python3
"""Prepare raw episodes into memory-mapped .npy files (low-RAM variant).

Same output format as prepare_encoder_dataset.py, but writes frames.npy,
states.npy, episode_ends.npy as memory-mapped arrays. Peak RAM is bounded
to one episode at a time — avoids the OOM problem of the in-memory variant
when the dataset exceeds available RAM.

Output format (same as PreparedEncoderDataset's directory mode):
    frames.npy:       (N_total, H, W) uint8
    states.npy:       (N_total, 6) float32
    episode_ends.npy: (n_episodes,) int64

Usage:
    python scripts/perception/prepare_encoder_dataset_mmap.py \
        --data-dir /path/to/random /path/to/heuristic \
        --output /path/to/prepared-npy \
        --frame-size 128
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def prepare_dataset_mmap(
    data_dirs: list[str | Path],
    output_dir: str | Path,
    frame_size: int = 128,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect episode paths, sorted for reproducibility.
    all_paths = []
    for d in data_dirs:
        all_paths.extend(sorted(Path(d).rglob("*.npz")))

    if not all_paths:
        raise ValueError(f"No .npz files found in {data_dirs}")

    # Pass 1: count frames per episode by loading only 'states' (small).
    print(f"Pass 1/2: counting frames across {len(all_paths)} episodes...")
    ep_lengths = []
    for path in tqdm(all_paths, desc="Counting", unit="ep"):
        data = np.load(str(path))
        ep_lengths.append(int(data["states"].shape[0]))

    episode_ends = np.cumsum(ep_lengths, dtype=np.int64)
    total_frames = int(episode_ends[-1])
    print(f"  Total frames: {total_frames:,} across {len(all_paths)} episodes")

    # Allocate memory-mapped output arrays.
    frames_path = output_dir / "frames.npy"
    states_path = output_dir / "states.npy"
    ends_path = output_dir / "episode_ends.npy"

    frames_mmap = np.lib.format.open_memmap(
        str(frames_path),
        mode="w+",
        dtype=np.uint8,
        shape=(total_frames, frame_size, frame_size),
    )
    states_mmap = np.lib.format.open_memmap(
        str(states_path),
        mode="w+",
        dtype=np.float32,
        shape=(total_frames, 6),
    )

    # Pass 2: process each episode, write directly to mmap slices.
    print("Pass 2/2: resizing/grayscaling and writing mmap...")
    cursor = 0
    for path in tqdm(all_paths, desc="Processing", unit="ep"):
        data = np.load(str(path))
        rgb_frames = data["rgb_frames"]              # (T+1, H, W, 3) uint8
        ep_states = data["states"][:, :6].astype(np.float32)  # (T+1, 6)
        n = rgb_frames.shape[0]

        # Resize + grayscale each frame, write directly into mmap slice.
        for i, frame in enumerate(rgb_frames):
            frame = cv2.resize(frame, (frame_size, frame_size))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            frames_mmap[cursor + i] = frame

        states_mmap[cursor:cursor + n] = ep_states
        cursor += n

    assert cursor == total_frames, f"cursor mismatch: {cursor} vs {total_frames}"

    # Flush mmaps to disk.
    frames_mmap.flush()
    states_mmap.flush()
    del frames_mmap, states_mmap

    np.save(str(ends_path), episode_ends)

    frames_mb = frames_path.stat().st_size / 1e6
    states_mb = states_path.stat().st_size / 1e6
    print(f"  Saved {output_dir}")
    print(f"    frames.npy:       {frames_mb:.1f} MB")
    print(f"    states.npy:       {states_mb:.1f} MB")
    print(f"    episode_ends.npy: {ends_path.stat().st_size} bytes")

    return {
        "n_episodes": len(all_paths),
        "n_frames": total_frames,
        "total_mb": frames_mb + states_mb,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Prepare raw episodes into memory-mapped .npy files (low-RAM)."
    )
    parser.add_argument("--data-dir", type=str, required=True, nargs="+",
                        help="Directory(ies) containing raw .npz episodes with rgb_frames.")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory for frames.npy, states.npy, episode_ends.npy.")
    parser.add_argument("--frame-size", type=int, default=128)
    args = parser.parse_args()

    print(f"Preparing dataset from {args.data_dir}...")
    prepare_dataset_mmap(
        [Path(d) for d in args.data_dir],
        args.output,
        frame_size=args.frame_size,
    )


if __name__ == "__main__":
    main()
