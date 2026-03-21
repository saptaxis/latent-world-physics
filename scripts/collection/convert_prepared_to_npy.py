#!/usr/bin/env python3
"""Create memory-mapped .npy dataset from raw episode .npz files.

Processes episodes one at a time — never holds more than one episode
in RAM. Writes directly to pre-allocated memory-mapped files.

Output: a directory with:
  - frames.npy:       (N_total, H, W) uint8
  - states.npy:       (N_total, 6) float32
  - episode_ends.npy: (n_episodes,) int64

Usage:
    python lunar_lander/scripts/convert_prepared_to_npy.py \
        --data-dir /path/to/random /path/to/heuristic \
        --output-dir /path/to/prepared-npy/ \
        --frame-size 128
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(
        description="Create mmap-ready .npy dataset from raw episodes (streaming, low RAM)."
    )
    parser.add_argument("--data-dir", type=str, required=True, nargs="+",
                        help="Directory(ies) containing raw .npz episodes.")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to write frames.npy, states.npy, episode_ends.npy")
    parser.add_argument("--frame-size", type=int, default=128)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fs = args.frame_size

    # Collect all episode paths.
    all_paths = []
    for d in args.data_dir:
        all_paths.extend(sorted(Path(d).rglob("*.npz")))
    if not all_paths:
        print(f"ERROR: No .npz files found in {args.data_dir}")
        sys.exit(1)
    print(f"Found {len(all_paths)} episodes")

    # Pass 1: count total frames to pre-allocate.
    print("Pass 1: counting frames...")
    episode_frame_counts = []
    for path in tqdm(all_paths, desc="Counting", unit="ep"):
        with np.load(str(path)) as data:
            episode_frame_counts.append(data["states"].shape[0])
    total_frames = sum(episode_frame_counts)
    print(f"  Total: {total_frames:,} frames across {len(all_paths)} episodes")

    # Pre-allocate memory-mapped output files.
    # np.lib.format.open_memmap creates the file with the right header
    # and returns a writable memmap. No RAM allocation — writes go to disk.
    frames_path = str(output_dir / "frames.npy")
    states_path = str(output_dir / "states.npy")

    print(f"Allocating frames.npy ({total_frames} × {fs} × {fs} uint8)...")
    frames_mmap = np.lib.format.open_memmap(
        frames_path, mode="w+", dtype=np.uint8, shape=(total_frames, fs, fs)
    )
    print(f"Allocating states.npy ({total_frames} × 6 float32)...")
    states_mmap = np.lib.format.open_memmap(
        states_path, mode="w+", dtype=np.float32, shape=(total_frames, 6)
    )

    # Pass 2: process episodes and write to memmap.
    print("Pass 2: processing episodes...")
    episode_ends = []
    write_idx = 0

    for i, path in enumerate(tqdm(all_paths, desc="Processing", unit="ep")):
        with np.load(str(path)) as data:
            rgb_frames = data["rgb_frames"]  # (T+1, H, W, 3) uint8
            ep_states = data["states"][:, :6].astype(np.float32)

        n_frames = rgb_frames.shape[0]

        # Resize + grayscale each frame, write directly to memmap.
        for j in range(n_frames):
            frame = cv2.resize(rgb_frames[j], (fs, fs))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            frames_mmap[write_idx + j] = frame

        states_mmap[write_idx:write_idx + n_frames] = ep_states
        write_idx += n_frames
        episode_ends.append(write_idx)

    # Flush memmap to disk.
    del frames_mmap
    del states_mmap

    # Save episode_ends (tiny, no memmap needed).
    np.save(str(output_dir / "episode_ends.npy"),
            np.array(episode_ends, dtype=np.int64))

    # Verify.
    print(f"\nDone. Output at {output_dir}/")
    for name in ["frames.npy", "states.npy", "episode_ends.npy"]:
        size_mb = (output_dir / name).stat().st_size / 1e6
        print(f"  {name}: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
