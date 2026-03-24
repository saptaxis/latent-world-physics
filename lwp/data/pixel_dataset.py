# data/pixel_dataset.py
"""Pixel-space datasets for VAE and dynamics training.

PixelFrameDataset: individual frames for VAE reconstruction training.
PixelEpisodeDataset: sequential (frames, actions) for dynamics training.

Both load .npz files with 'rgb_frames' key. All frames are pre-loaded
into RAM at init (resized + grayscaled), so __getitem__ is a pure
array index with zero I/O. For 5K episodes at 84x84 grayscale,
this uses ~2-3 GB RAM.
"""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


def _preprocess_frame(frame: np.ndarray, frame_size: int,
                      grayscale: bool) -> np.ndarray:
    """Resize and optionally convert to grayscale. Returns (H, W) or (H, W, 3) uint8."""
    # Convert before resize — grayscale reduces channels from 3→1, which
    # means INTER_AREA downsampling operates on a single channel (faster
    # and avoids colour-bleeding artefacts at small resolutions).
    if grayscale and frame.ndim == 3 and frame.shape[2] == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # INTER_AREA is the correct interpolation for downscaling — it averages
    # pixel areas rather than point-sampling, avoiding aliasing artefacts
    # that INTER_LINEAR or INTER_NEAREST would produce at 4-6x downscale.
    resized = cv2.resize(frame, (frame_size, frame_size),
                         interpolation=cv2.INTER_AREA)
    return resized


def _frame_to_tensor(frame: np.ndarray) -> torch.Tensor:
    """Convert preprocessed frame to (C, H, W) float32 tensor in [0, 1]."""
    if frame.ndim == 2:
        # Grayscale (H, W) → (1, H, W): unsqueeze adds the channel dim
        # that Conv2d expects. Divide by 255 to normalise uint8 → [0, 1].
        t = torch.from_numpy(frame).float().unsqueeze(0) / 255.0
    else:
        # RGB (H, W, 3) → (3, H, W): permute from HWC (numpy/cv2 layout)
        # to CHW (PyTorch conv layout).
        t = torch.from_numpy(frame).float().permute(2, 0, 1) / 255.0
    return t


def _split_episodes(npz_files: list[Path], split: str | None,
                    val_fraction: float, seed: int) -> list[Path]:
    """Split episode files into train/val by episode."""
    # Fixed-seed RNG so train/val splits are identical across runs — critical
    # to avoid data leakage between VAE and dynamics training phases.
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(npz_files))
    # At least 1 val episode even for tiny datasets, otherwise val metrics
    # would be undefined.
    n_val = max(1, int(len(npz_files) * val_fraction))
    if split == "val":
        # First n_val of the shuffled indices become validation
        selected = sorted(indices[:n_val])
    elif split == "train":
        # Remaining indices become training
        selected = sorted(indices[n_val:])
    else:
        # No split requested — return everything (used for inference/eval)
        selected = list(range(len(npz_files)))
    return [npz_files[i] for i in selected]


def _load_one_episode(args: tuple) -> np.ndarray | None:
    """Load and preprocess a single episode's frames. For multiprocessing.Pool."""
    # Takes a tuple (not kwargs) because multiprocessing.Pool.imap requires
    # a single-argument callable — we pack/unpack manually.
    path, frame_size, grayscale = args
    try:
        # np.load with context manager ensures the npz file handle is closed
        # promptly, avoiding file-descriptor exhaustion with thousands of episodes.
        with np.load(str(path)) as data:
            raw_frames = data["rgb_frames"]
        # Stack into a single (T+1, H, W) array — contiguous memory for
        # efficient concatenation later in the parent process.
        processed = np.stack([
            _preprocess_frame(raw_frames[i], frame_size, grayscale)
            for i in range(len(raw_frames))
        ])
        return processed
    except Exception:
        # Silently skip corrupt/truncated episodes rather than crashing —
        # a few missing episodes won't affect training quality.
        return None


def _load_one_episode_states(args: tuple) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Load frames + states from a single episode. For multiprocessing.Pool.

    The 4th arg can be:
      - int (state_dim): legacy, slices states[:, :state_dim]
      - list[int] (state_targets): index-based, slices states[:, state_targets]
    """
    path, frame_size, grayscale, state_spec = args
    try:
        with np.load(str(path)) as data:
            raw_frames = data["rgb_frames"]
            if "states" in data:
                all_states = data["states"]
                # Dispatch on type: int = first-N slice, list = index selection.
                # Index selection is needed for kin_targets=[4,5] (angle-only)
                # where we want specific non-contiguous dims from the 8D state.
                if isinstance(state_spec, (list, tuple)):
                    states = all_states[:, list(state_spec)].astype(np.float32)
                else:
                    states = all_states[:, :state_spec].astype(np.float32)
            else:
                states = None
        processed = np.stack([
            _preprocess_frame(raw_frames[i], frame_size, grayscale)
            for i in range(len(raw_frames))
        ])
        return processed, states
    except Exception:
        return None, None


def _load_one_episode_with_actions(args: tuple) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Load and preprocess a single episode's frames + actions. For multiprocessing.Pool."""
    path, frame_size, grayscale = args
    try:
        with np.load(str(path)) as data:
            raw_frames = data["rgb_frames"]
            actions = data["actions"].astype(np.float32)
        # Episodes store T+1 frames and T actions (one action per transition).
        # Some episodes may have mismatched counts due to early termination
        # or recording bugs, so we clamp to the minimum usable length.
        n_frames = raw_frames.shape[0]
        n_actions = actions.shape[0]
        usable = min(n_frames, n_actions + 1)
        processed = np.stack([
            _preprocess_frame(raw_frames[i], frame_size, grayscale)
            for i in range(usable)
        ])
        # Return usable-1 actions to maintain the (T+1 frames, T actions) invariant.
        return processed, actions[:usable - 1]
    except Exception:
        return None, None


def _load_and_preprocess_all_frames(
    episode_files: list[Path], frame_size: int, grayscale: bool,
    n_workers: int = 8,
) -> list[np.ndarray]:
    """Load all episodes in parallel, preprocess frames.

    Uses multiprocessing.Pool to parallelize npz decompression + resize.
    Returns list of (T+1, H, W) or (T+1, H, W, 3) uint8 arrays.
    """
    from multiprocessing import Pool

    args = [(path, frame_size, grayscale) for path in episode_files]

    all_episodes = []
    # imap (not imap_unordered) preserves episode order for deterministic
    # dataset construction. The tqdm wrapper shows progress since loading
    # thousands of npz files can take several minutes.
    with Pool(n_workers) as pool:
        for result in tqdm(
            pool.imap(_load_one_episode, args),
            total=len(args), desc="Loading frames", unit="ep",
        ):
            if result is not None:
                all_episodes.append(result)
    return all_episodes


class PixelFrameDataset(Dataset):
    """Individual frames for VAE training.

    Pre-loads all frames into RAM at init. __getitem__ is a pure
    array index — zero I/O during training.

    If cache_path is provided, saves/loads preprocessed frames as a
    single .npy file for instant loading on subsequent runs (~2s vs ~10min).
    Cache is per-split: include the split name in cache_path.

    When state_dim > 0, also loads kinematic states and returns
    (frame, state) tuples instead of just frames.

    Args:
        data_path: directory with .npz episode files
        frame_size: target resolution (square)
        grayscale: convert to grayscale
        split: "train", "val", or None
        val_fraction: fraction for validation
        seed: RNG seed for split
        cache_path: optional path to save/load preprocessed frames (.npy)
        state_dim: if > 0, load and return states (first N dims from .npz states)
    """

    def __init__(self, data_path: str | Path | list[str | Path],
                 frame_size: int = 84,
                 grayscale: bool = True, split: str | None = None,
                 val_fraction: float = 0.1, seed: int = 0,
                 n_workers: int = 8, cache_path: str | Path | None = None,
                 state_dim: int = 0,
                 state_targets: list[int] | None = None):
        self.frame_size = frame_size
        self.grayscale = grayscale
        # state_targets overrides state_dim: if state_targets=[4,5] is given,
        # state_dim becomes 2 (the number of selected dims), not the raw 6.
        if state_targets is not None:
            self.state_dim = len(state_targets)
            # _state_spec is what we pass to the worker — list triggers index selection
            self._state_spec = list(state_targets)
        else:
            self.state_dim = state_dim
            # int triggers legacy first-N slice
            self._state_spec = state_dim
        self._states = None  # (N, state_dim) float32 or None; only populated when state_dim > 0

        # --- Cache fast path ---
        # Try loading from cache first. Cache is a directory with frames.npy
        # and optionally states.npy. Uses mmap_mode='r' — memory-mapped, no
        # RAM allocation. The OS pages data from disk on demand, so even 10GB
        # of frames uses near-zero resident memory. DataLoader workers each
        # get their own page faults, which the OS handles transparently.
        if cache_path is not None and Path(cache_path).is_dir():
            frames_file = Path(cache_path) / "frames.npy"
            if frames_file.exists():
                print(f"Loading from cache (mmap): {cache_path} ...")
                self._frames = np.load(str(frames_file), mmap_mode='r')
                states_file = Path(cache_path) / "states.npy"
                if self.state_dim > 0:
                    if not states_file.exists():
                        raise ValueError(
                            f"Cache at {cache_path} has no states.npy, but state_dim={self.state_dim}. "
                            f"Delete the cache dir and rerun."
                        )
                    self._states = np.load(str(states_file), mmap_mode='r')
                n = self._frames.shape[0]
                disk_mb = os.path.getsize(str(frames_file)) / 1024 / 1024
                state_str = f" + {self._states.shape[1]}D states" if self._states is not None else ""
                print(f"PixelFrameDataset: {n} frames{state_str} ({disk_mb:.0f} MB on disk, mmap)")
                return

        # --- Try streaming cache build (low-memory) ---
        # For large datasets at high resolution (e.g., 19K episodes at 128x128),
        # loading all frames into RAM first causes OOM (~70GB). Instead, process
        # episodes one at a time and write directly to a pre-allocated mmap file.
        # Only used when cache_path is given and cache doesn't exist yet.
        if cache_path is not None and not Path(cache_path).is_dir():
            self._build_cache_streaming(
                data_path, cache_path, frame_size, grayscale, split,
                val_fraction, seed, n_workers, self._state_spec)
            # Now load from the freshly-built cache via mmap (top of __init__)
            frames_file = Path(cache_path) / "frames.npy"
            if frames_file.exists():
                self._frames = np.load(str(frames_file), mmap_mode='r')
                if self.state_dim > 0:
                    states_file = Path(cache_path) / "states.npy"
                    if states_file.exists():
                        self._states = np.load(str(states_file), mmap_mode='r')
                n = self._frames.shape[0]
                disk_mb = os.path.getsize(str(frames_file)) / 1024 / 1024
                state_str = f" + {self.state_dim}D states" if self._states is not None else ""
                print(f"PixelFrameDataset: {n} frames{state_str} ({disk_mb:.0f} MB on disk, mmap)")
                return

        # --- Full load path (no cache or cache miss) ---
        # Normalize to list of paths so callers can pass a single string or
        # multiple data directories (e.g., heuristic + random episodes).
        if isinstance(data_path, (str, Path)):
            data_path = [data_path]
        npz_files = []
        # Walk with followlinks=True so symlinked data dirs work (common on
        # shared compute where data lives on a different filesystem).
        for dp in data_path:
            for root, _dirs, files in os.walk(str(Path(dp)), followlinks=True):
                for f in files:
                    if f.endswith(".npz"):
                        npz_files.append(Path(root) / f)
        # Sort for deterministic ordering across platforms/filesystems.
        npz_files.sort()
        # Filter out "prepared" cache files that PixelEpisodeDataset may have
        # saved alongside the raw episodes — they use a different format.
        npz_files = [f for f in npz_files if "prepared" not in f.name]
        episode_files = _split_episodes(npz_files, split, val_fraction, seed)

        if self.state_dim > 0:
            # Load frames + kinematic states for the auxiliary state prediction
            # head. States are used as supervised targets during VAE training
            # to encourage the latent space to encode physical quantities.
            load_args = [(path, frame_size, grayscale, self._state_spec) for path in episode_files]
            all_frames = []
            all_states = []
            if n_workers > 1:
                from multiprocessing import Pool
                with Pool(n_workers) as pool:
                    iterator = pool.imap(_load_one_episode_states, load_args)
                    for frames, states in tqdm(iterator, total=len(load_args),
                                               desc="Loading frames+states", unit="ep"):
                        if frames is not None:
                            all_frames.append(frames)
                            if states is not None:
                                all_states.append(states)
            else:
                for a in tqdm(load_args, desc="Loading frames+states", unit="ep"):
                    frames, states = _load_one_episode_states(a)
                    if frames is not None:
                        all_frames.append(frames)
                        if states is not None:
                            all_states.append(states)

            if all_frames:
                # Concatenate all episodes into a single flat array of frames.
                # After this, episode boundaries are lost — each frame is an
                # independent training sample for the VAE.
                self._frames = np.concatenate(all_frames, axis=0)
                if all_states:
                    self._states = np.concatenate(all_states, axis=0).astype(np.float32)
            else:
                # Empty placeholder so len() and __getitem__ work gracefully
                # even if all episodes failed to load.
                self._frames = np.zeros((0, frame_size, frame_size), dtype=np.uint8)
        else:
            # Frames only (original path)
            episodes = _load_and_preprocess_all_frames(
                episode_files, frame_size, grayscale, n_workers=n_workers)
            if episodes:
                self._frames = np.concatenate(episodes, axis=0)
            else:
                if grayscale:
                    self._frames = np.zeros((0, frame_size, frame_size), dtype=np.uint8)
                else:
                    self._frames = np.zeros((0, frame_size, frame_size, 3), dtype=np.uint8)

        n_total = self._frames.shape[0]
        mb = self._frames.nbytes / 1024 / 1024
        if self._states is not None:
            mb += self._states.nbytes / 1024 / 1024
        print(f"PixelFrameDataset: {n_total} frames"
              f"{f' + {self.state_dim}D states' if self._states is not None else ''}"
              f" from {len(episode_files)} episodes ({mb:.0f} MB in RAM)")

        # Save cache, then free RAM and reload as mmap.
        # np.save writes directly without copying. After saving, delete
        # the in-memory arrays and reload as memory-mapped — OS pages
        # from disk on demand, near-zero RAM.
        if cache_path is not None and n_total > 0:
            cache_dir = Path(cache_path)
            cache_dir.mkdir(parents=True, exist_ok=True)
            frames_file = cache_dir / "frames.npy"
            np.save(str(frames_file), self._frames)
            states_file = cache_dir / "states.npy"
            if self._states is not None:
                np.save(str(states_file), self._states)
            print(f"Saved cache to {cache_dir}")

            # Free RAM and reload as mmap
            del self._frames
            self._frames = np.load(str(frames_file), mmap_mode='r')
            if self._states is not None:
                del self._states
                self._states = np.load(str(states_file), mmap_mode='r')
            print(f"Reloaded as mmap (RAM freed)")

    def _build_cache_streaming(
        self, data_path, cache_path, frame_size, grayscale,
        split, val_fraction, seed, n_workers, state_spec,
    ):
        """Build the frame cache without loading all episodes into RAM.

        Processes episodes one at a time: load npz → preprocess → write
        directly to a pre-allocated numpy mmap file. Peak RAM usage is
        one episode (~a few MB), not the entire dataset (~70GB at 128x128).

        Two passes:
        1. Count total frames (fast — just read npz metadata)
        2. Process episodes and write to mmap
        """
        # Resolve episode files (same logic as full load path)
        if isinstance(data_path, (str, Path)):
            data_path = [data_path]
        npz_files = []
        for dp in data_path:
            for root, _dirs, files in os.walk(str(Path(dp)), followlinks=True):
                for f in files:
                    if f.endswith(".npz"):
                        npz_files.append(Path(root) / f)
        npz_files.sort()
        npz_files = [f for f in npz_files if "prepared" not in f.name]
        episode_files = _split_episodes(npz_files, split, val_fraction, seed)

        if not episode_files:
            return

        # Pass 1: count total frames across all episodes
        # Read only rgb_frames shape from each npz — fast, no decompression
        print(f"Streaming cache build: counting frames in {len(episode_files)} episodes...")
        total_frames = 0
        frame_counts = []
        for path in tqdm(episode_files, desc="Counting", unit="ep"):
            try:
                with np.load(str(path)) as data:
                    n = data["rgb_frames"].shape[0]
                    frame_counts.append(n)
                    total_frames += n
            except Exception:
                frame_counts.append(0)

        if total_frames == 0:
            return

        # Pass 2: allocate mmap file and fill it episode by episode
        cache_dir = Path(cache_path)
        cache_dir.mkdir(parents=True, exist_ok=True)
        frames_file = cache_dir / "frames.npy"
        states_file = cache_dir / "states.npy"

        # Pre-allocate the output arrays as mmap files.
        # np.lib.format.open_memmap creates a .npy file with the right
        # header and returns a writable mmap view.
        if grayscale:
            frame_shape = (total_frames, frame_size, frame_size)
        else:
            frame_shape = (total_frames, frame_size, frame_size, 3)

        print(f"Allocating {frames_file} ({total_frames} frames, "
              f"{np.prod(frame_shape) / 1e9:.1f} GB)...")
        frames_mmap = np.lib.format.open_memmap(
            str(frames_file), mode='w+', dtype=np.uint8, shape=frame_shape)

        # Derive the number of state dims from the spec for mmap allocation.
        # state_spec is either int (first-N slice) or list[int] (index selection).
        if isinstance(state_spec, (list, tuple)):
            n_state_dims = len(state_spec)
        else:
            n_state_dims = state_spec

        states_mmap = None
        if n_state_dims > 0:
            states_mmap = np.lib.format.open_memmap(
                str(states_file), mode='w+', dtype=np.float32,
                shape=(total_frames, n_state_dims))

        # Fill the mmap using parallel workers. Each worker loads one episode,
        # preprocesses its frames (decompress npz + resize + grayscale — CPU heavy),
        # and returns the preprocessed arrays. The main thread writes to the mmap
        # sequentially. Peak RAM = n_workers episodes in flight (~few MB each).
        offset = 0
        load_workers = max(1, n_workers)

        if load_workers > 1:
            from multiprocessing import Pool

            # Build args: each worker gets (path, frame_size, grayscale, state_spec)
            # and returns preprocessed (frames, states_or_None)
            worker_args = []
            for i, path in enumerate(episode_files):
                if frame_counts[i] > 0:
                    if n_state_dims > 0:
                        worker_args.append((path, frame_size, grayscale, state_spec))
                    else:
                        worker_args.append((path, frame_size, grayscale))

            # Use the existing worker functions — they return preprocessed arrays
            worker_fn = _load_one_episode_states if n_state_dims > 0 else _load_one_episode

            with Pool(load_workers) as pool:
                for result in tqdm(
                    pool.imap(worker_fn, worker_args),
                    total=len(worker_args), desc="Building cache", unit="ep",
                ):
                    if n_state_dims > 0:
                        frames, states = result
                    else:
                        frames = result
                        states = None

                    if frames is None:
                        continue
                    n = len(frames)
                    # Write this episode's preprocessed frames to the mmap
                    frames_mmap[offset:offset + n] = frames
                    if states_mmap is not None and states is not None:
                        # Worker already selects the right dims, just write directly
                        states_mmap[offset:offset + n] = states[:n]
                    offset += n
                    # frames/states go out of scope here — GC reclaims per iteration
        else:
            # Single-threaded fallback
            for i, path in enumerate(tqdm(episode_files, desc="Building cache", unit="ep")):
                n = frame_counts[i]
                if n == 0:
                    continue
                try:
                    with np.load(str(path), allow_pickle=True) as data:
                        raw_frames = data["rgb_frames"]
                        for j in range(n):
                            frame = _preprocess_frame(raw_frames[j], frame_size, grayscale)
                            frames_mmap[offset + j] = frame
                        if states_mmap is not None and "states" in data:
                            all_states = data["states"]
                            # Same dispatch as the worker: list = index, int = slice
                            if isinstance(state_spec, (list, tuple)):
                                states = all_states[:n, list(state_spec)].astype(np.float32)
                            else:
                                states = all_states[:n, :state_spec].astype(np.float32)
                            states_mmap[offset:offset + n] = states
                    offset += n
                except Exception:
                    offset += n  # skip but keep alignment

        # Flush to disk
        del frames_mmap
        if states_mmap is not None:
            del states_mmap
        print(f"Streaming cache complete: {total_frames} frames → {cache_dir}")

    def __len__(self) -> int:
        return self._frames.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # np.array() copies the mmap slice into a writable array — without
        # this, PyTorch warns about non-writable tensors (mmap pages are
        # read-only). The copy is tiny (one frame: 84*84 = 7KB).
        frame = _frame_to_tensor(np.array(self._frames[idx]))
        if self._states is not None:
            state = torch.from_numpy(np.array(self._states[idx])).float()
            return frame, state
        return frame


class PixelEpisodeDataset(Dataset):
    """Sequential frame+action chunks for dynamics training.

    Pre-loads all frames and actions into RAM. __getitem__ builds
    stacked-frame windows from the pre-loaded arrays — zero I/O.

    Args:
        data_path: directory with .npz episode files
        frame_size: target resolution
        grayscale: convert to grayscale
        seq_len: number of timesteps per window
        frame_stack: number of frames to stack (1=no stacking)
        split, val_fraction, seed: train/val splitting
    """

    def __init__(self, data_path: str | Path | list[str | Path],
                 frame_size: int = 84,
                 grayscale: bool = True, seq_len: int = 20,
                 frame_stack: int = 1, split: str | None = None,
                 val_fraction: float = 0.1, seed: int = 0,
                 n_workers: int = 8, cache_path: str | Path | None = None):
        self.frame_size = frame_size
        self.grayscale = grayscale
        self.seq_len = seq_len
        self.frame_stack = frame_stack

        # Minimum episode length to yield at least one training window.
        # We need seq_len timesteps, plus (frame_stack - 1) extra past frames
        # for the initial stacked observation, plus 1 for the frame→frame
        # transition at the last timestep.
        min_frames = seq_len + frame_stack - 1 + 1

        # Per-episode storage — we keep episodes separate (not flattened)
        # because windows must not cross episode boundaries.
        self._episode_frames = []  # list of (T+1, H, W) uint8 arrays
        self._episode_actions = []  # list of (T, action_dim) float32 arrays
        # Flat index mapping: window_index[i] = (episode_idx, start_frame)
        # so __getitem__ can jump directly to any valid window in O(1).
        self._window_index = []

        # Try cache first
        if cache_path is not None and Path(cache_path).exists():
            print(f"Loading cached episodes from {cache_path} ...")
            cached = np.load(str(cache_path), allow_pickle=True)
            all_frames = cached["all_frames"]  # object array of per-episode frames
            all_actions = cached["all_actions"]  # object array of per-episode actions
            for i in range(len(all_frames)):
                frames = all_frames[i]
                if frames.shape[0] < min_frames:
                    continue
                ep_idx = len(self._episode_frames)
                self._episode_frames.append(frames)
                self._episode_actions.append(all_actions[i])
                max_start = frames.shape[0] - seq_len - frame_stack + 1
                for s in range(frame_stack - 1, frame_stack - 1 + max_start):
                    self._window_index.append((ep_idx, s))
        else:
            if isinstance(data_path, (str, Path)):
                data_path = [data_path]
            npz_files = []
            for dp in data_path:
                for root, dirs, files in os.walk(str(Path(dp)), followlinks=True):
                    for f in files:
                        if f.endswith(".npz"):
                            npz_files.append(Path(root) / f)
            npz_files.sort()
            npz_files = [f for f in npz_files if "prepared" not in f.name]
            episode_files = _split_episodes(npz_files, split, val_fraction, seed)

            from multiprocessing import Pool

            args = [(path, frame_size, grayscale) for path in episode_files]
            with Pool(n_workers) as pool:
                for frames, actions in tqdm(
                    pool.imap(_load_one_episode_with_actions, args),
                    total=len(args), desc="Loading episodes", unit="ep",
                ):
                    if frames is None:
                        continue
                    if frames.shape[0] < min_frames:
                        continue

                    ep_idx = len(self._episode_frames)
                    self._episode_frames.append(frames)
                    self._episode_actions.append(actions)

                    # Enumerate all valid window start positions within this
                    # episode. Start at (frame_stack - 1) so there are enough
                    # past frames to build the initial stacked observation.
                    max_start = frames.shape[0] - seq_len - frame_stack + 1
                    for s in range(frame_stack - 1, frame_stack - 1 + max_start):
                        self._window_index.append((ep_idx, s))

            # Save cache so subsequent runs skip the expensive npz loading
            if cache_path is not None and self._episode_frames:
                Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
                # Use object arrays because episodes have different lengths —
                # a regular ndarray would require padding. Object arrays let
                # np.savez pickle each variable-length array independently.
                np.savez(
                    str(cache_path),
                    all_frames=np.array(self._episode_frames, dtype=object),
                    all_actions=np.array(self._episode_actions, dtype=object),
                )
                print(f"Saved episode cache to {cache_path}")

        total_frames = sum(f.shape[0] for f in self._episode_frames)
        mb = sum(f.nbytes for f in self._episode_frames) / 1024 / 1024
        mb += sum(a.nbytes for a in self._episode_actions) / 1024 / 1024
        print(f"PixelEpisodeDataset: {len(self._window_index)} windows from "
              f"{len(self._episode_frames)} episodes, {total_frames} frames "
              f"({mb:.0f} MB in RAM)")

    def __len__(self) -> int:
        return len(self._window_index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        ep_idx, start = self._window_index[idx]
        frames = self._episode_frames[ep_idx]
        actions = self._episode_actions[ep_idx]

        # Reach back (frame_stack - 1) frames before `start` to have enough
        # history for the initial stacked observation at t=0.
        frame_start = start - (self.frame_stack - 1)
        frame_end = start + self.seq_len
        raw_chunk = frames[frame_start:frame_end]  # already preprocessed uint8
        act_chunk = actions[start:start + self.seq_len]

        # Convert each raw frame to a (C, H, W) float tensor in [0, 1]
        processed = [_frame_to_tensor(raw_chunk[i]) for i in range(len(raw_chunk))]

        # Build stacked-frame observations by concatenating `frame_stack`
        # consecutive frames along the channel dimension. This gives the
        # dynamics model access to short-term motion information (velocity
        # cues) without requiring an explicit velocity input.
        stacked_frames = []
        for t in range(self.seq_len):
            # processed[t:t+frame_stack] are frame_stack consecutive frames;
            # cat along dim=0 produces (C*frame_stack, H, W).
            stack = torch.cat(processed[t:t + self.frame_stack], dim=0)
            stacked_frames.append(stack)

        # Final shapes: frames (seq_len, C*frame_stack, H, W), actions (seq_len, action_dim)
        frames_tensor = torch.stack(stacked_frames, dim=0)
        # .copy() because the numpy slice may share memory with the episode
        # array, and PyTorch prefers owning the underlying storage.
        actions_tensor = torch.from_numpy(act_chunk.copy()).float()

        return frames_tensor, actions_tensor
