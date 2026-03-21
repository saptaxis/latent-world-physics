"""PyTorch Datasets for encoder pre-training.

Two modes for loading frame-stack training data:

1. EncoderPretrainDataset (raw mode):
   - Loads per-episode .npz files from disk on the fly
   - Resizes + grayscales each frame during __getitem__
   - Works at any dataset scale (50K+ episodes), but slower
   - LRU cache (64 episodes) reduces re-reads for sequential access

2. PreparedEncoderDataset (prepared mode):
   - Loads a single pre-processed .npz into RAM at init
   - All frames already resized to target size + grayscaled (uint8)
   - Zero I/O during training — just numpy slicing + normalization
   - Requires running prepare_encoder_dataset.py first
   - Only works when prepared file fits in RAM (~13 GB for 5K episodes)

Both return identical output: (frames, targets) where
  frames: (n_stack, H, W) float32 in [0, 1]
  targets: (6,) float32 = [x, y, vx, vy, angle, ang_vel]

The target for each frame stack is the kinematic state at the LAST frame
in the stack. This matches the observation the policy sees when it acts.

NOTE on resize interpolation: EncoderPretrainDataset uses cv2.resize
(INTER_LINEAR) while Gymnasium's ResizeObservation uses PIL.Image.resize.
The pixel-level differences are negligible for encoder pre-training.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class EncoderPretrainDataset(Dataset):
    """Raw mode: loads per-episode .npz files on the fly.

    Frame processing pipeline (matches SB3 visual RL):
    1. Resize RGB frame from (H, W, 3) to (frame_size, frame_size, 3)
    2. Convert to grayscale via cv2.COLOR_RGB2GRAY (BT.709 luminance,
       same formula as Gymnasium's GrayscaleObservation)
    3. Stack n_stack consecutive frames: (n_stack, frame_size, frame_size)
    4. Normalize to [0, 1] float32 (divide by 255.0)

    Args:
        data_dirs: Path or list of paths to directories containing .npz episode files.
        n_stack: Number of consecutive frames to stack (default 4).
        frame_size: Resize target in pixels (default 128).
        split: "train", "val", or None (all data). Split by episode index.
        val_fraction: Fraction of episodes held out for validation (default 0.1).
        seed: RNG seed for reproducible train/val split (default 42).
    """

    def __init__(
        self,
        data_dirs: str | Path | list[str | Path],
        n_stack: int = 4,
        frame_size: int = 128,
        split: str | None = None,
        val_fraction: float = 0.1,
        seed: int = 42,
    ):
        self.n_stack = n_stack
        self.frame_size = frame_size

        # Collect all .npz episode paths, sorted for reproducibility.
        if isinstance(data_dirs, (str, Path)):
            data_dirs = [data_dirs]
        all_paths = []
        for d in data_dirs:
            all_paths.extend(sorted(Path(d).rglob("*.npz")))

        if not all_paths:
            raise ValueError(f"No .npz files found in {data_dirs}")

        # Split by episode index (not by frame) to avoid data leakage.
        # Adjacent frames within an episode are highly correlated.
        rng = np.random.RandomState(seed)
        n_val = max(1, int(len(all_paths) * val_fraction))
        val_indices = set(rng.choice(len(all_paths), size=n_val, replace=False))

        if split == "train":
            self.episode_paths = [p for i, p in enumerate(all_paths) if i not in val_indices]
        elif split == "val":
            self.episode_paths = [p for i, p in enumerate(all_paths) if i in val_indices]
        else:
            self.episode_paths = all_paths

        # Build flat index: (episode_idx, frame_start_idx) per valid stack.
        self._index = []
        for ep_idx, path in enumerate(self.episode_paths):
            with np.load(str(path)) as data:
                n_frames = data["states"].shape[0]
            for t in range(n_frames - n_stack + 1):
                self._index.append((ep_idx, t))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        ep_idx, frame_start = self._index[idx]
        episode = self._load_episode(ep_idx)

        # Build frame stack: resize + grayscale each frame.
        stack = []
        for t in range(frame_start, frame_start + self.n_stack):
            frame = episode["rgb_frames"][t]
            frame = cv2.resize(frame, (self.frame_size, self.frame_size))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            stack.append(frame)

        frames = np.stack(stack, axis=0).astype(np.float32) / 255.0
        target_idx = frame_start + self.n_stack - 1
        targets = episode["states"][target_idx, :6].copy()

        return torch.from_numpy(frames), torch.from_numpy(targets)

    @lru_cache(maxsize=64)
    def _load_episode(self, ep_idx: int) -> dict:
        """Load episode from disk. LRU-cached (64 episodes max)."""
        data = np.load(str(self.episode_paths[ep_idx]))
        return {
            "rgb_frames": data["rgb_frames"],       # (T+1, H, W, 3) uint8
            "states": data["states"].astype(np.float32),  # (T+1, 15)
        }


class PreparedEncoderDataset(Dataset):
    """Prepared mode: loads pre-processed frames from .npz or mmap'd .npy files.

    Two loading modes:
    1. **npz** (prepared_path is a .npz file): Loads everything into RAM.
       Fast training but uses ~19 GB RAM for 5K episodes at 128×128.
    2. **npy/mmap** (prepared_path is a directory with .npy files): Memory-mapped.
       Near-zero RAM — the OS pages frames from disk on demand. Multiple
       training runs can share the same mmap'd files simultaneously.
       Create with convert_prepared_to_npy.py.

    The prepared data contains:
        frames: (N_total, H, W) uint8 — all frames concatenated, already
                resized and grayscaled by prepare_encoder_dataset.py
        states: (N_total, 6) float32 — kinematic targets per frame
        episode_ends: (n_episodes,) int64 — cumulative frame count, used
                      to prevent frame stacks from crossing episode boundaries

    Args:
        prepared_path: Path to prepared .npz file OR directory with .npy files.
        n_stack: Number of consecutive frames to stack (default 4).
        split: "train", "val", or None. Split by episode index.
        val_fraction: Fraction of episodes for validation (default 0.1).
        seed: RNG seed for reproducible split (default 42).
    """

    def __init__(
        self,
        prepared_path: str | Path,
        n_stack: int = 4,
        split: str | None = None,
        val_fraction: float = 0.1,
        seed: int = 42,
    ):
        self.n_stack = n_stack
        prepared_path = Path(prepared_path)

        # Load frames, states, episode_ends — either from .npz or mmap'd .npy.
        if prepared_path.is_dir():
            # Memory-mapped mode: near-zero RAM, OS pages from disk on demand.
            all_frames = np.load(str(prepared_path / "frames.npy"), mmap_mode="r")
            all_states = np.load(str(prepared_path / "states.npy"), mmap_mode="r")
            episode_ends = np.load(str(prepared_path / "episode_ends.npy"))
        else:
            # npz mode: loads everything into RAM.
            data = np.load(str(prepared_path))
            all_frames = data["frames"]       # (N_total, H, W) uint8
            all_states = data["states"]       # (N_total, 6) float32
            episode_ends = data["episode_ends"]  # (n_episodes,) int64

        # Derive episode start indices from cumulative ends.
        # episode_ends = [11, 27, 48] means ep0=[0:11], ep1=[11:27], ep2=[27:48]
        episode_starts = np.concatenate([[0], episode_ends[:-1]])
        n_episodes = len(episode_ends)

        # Split by episode index.
        rng = np.random.RandomState(seed)
        n_val = max(1, int(n_episodes * val_fraction))
        val_indices = set(rng.choice(n_episodes, size=n_val, replace=False))

        if split == "train":
            ep_mask = [i for i in range(n_episodes) if i not in val_indices]
        elif split == "val":
            ep_mask = [i for i in range(n_episodes) if i in val_indices]
        else:
            ep_mask = list(range(n_episodes))

        # Build flat index of valid frame-stack positions.
        # Store (global_frame_start,) — simpler than (ep_idx, local_offset)
        # since all data is in a flat array.
        self._index = []
        for ep_idx in ep_mask:
            start = int(episode_starts[ep_idx])
            end = int(episode_ends[ep_idx])
            n_frames = end - start
            for t in range(n_frames - n_stack + 1):
                self._index.append(start + t)

        # Keep references to the data arrays.
        self._frames = all_frames
        self._states = all_states

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        frame_start = self._index[idx]

        # Slice n_stack consecutive frames from the flat array.
        # np.array() copies from mmap into a writable array (mmap pages
        # are read-only). The copy is tiny: 4 × 128 × 128 = 64 KB.
        frames = np.array(self._frames[frame_start:frame_start + self.n_stack])
        frames = frames.astype(np.float32) / 255.0

        # Target: kinematic state at the last frame in the stack.
        target_idx = frame_start + self.n_stack - 1
        targets = np.array(self._states[target_idx])  # (6,) float32

        return torch.from_numpy(frames), torch.from_numpy(targets)
