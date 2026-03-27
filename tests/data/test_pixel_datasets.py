# tests/test_pixel_datasets.py
"""Tests for pixel datasets."""
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from lwp.data.pixel_dataset import PixelFrameDataset, PixelEpisodeDataset


def _make_fake_episode(path: Path, n_steps: int = 20, h: int = 100, w: int = 150):
    """Create a minimal .npz episode with rgb_frames and actions."""
    np.savez(
        path,
        rgb_frames=np.random.randint(0, 255, (n_steps + 1, h, w, 3), dtype=np.uint8),
        actions=np.random.randn(n_steps, 2).astype(np.float32),
        states=np.random.randn(n_steps + 1, 8).astype(np.float32),
    )


@pytest.fixture
def fake_data_dir(tmp_path):
    """Create a temp dir with 5 fake episodes."""
    for i in range(5):
        _make_fake_episode(tmp_path / f"episode_{i:04d}.npz", n_steps=30)
    return tmp_path


class TestPixelFrameDataset:
    """Tests for single-frame VAE training dataset."""

    def test_loads_frames(self, fake_data_dir):
        ds = PixelFrameDataset(fake_data_dir, frame_size=64, grayscale=True)
        assert len(ds) > 0
        frame = ds[0]
        assert frame.shape == (1, 64, 64)
        assert frame.dtype == torch.float32
        assert 0.0 <= frame.min() and frame.max() <= 1.0

    def test_rgb_mode(self, fake_data_dir):
        ds = PixelFrameDataset(fake_data_dir, frame_size=64, grayscale=False)
        frame = ds[0]
        assert frame.shape == (3, 64, 64)

    def test_val_split(self, fake_data_dir):
        train_ds = PixelFrameDataset(fake_data_dir, frame_size=64,
                                     split="train", val_fraction=0.2)
        val_ds = PixelFrameDataset(fake_data_dir, frame_size=64,
                                   split="val", val_fraction=0.2)
        assert len(train_ds) + len(val_ds) > 0


class TestPixelEpisodeDataset:
    """Tests for sequential dynamics training dataset."""

    def test_returns_sequences(self, fake_data_dir):
        ds = PixelEpisodeDataset(fake_data_dir, frame_size=64, grayscale=True,
                                 seq_len=10, frame_stack=1)
        frames, actions = ds[0]
        assert frames.shape == (10, 1, 64, 64)
        assert actions.shape == (10, 2)

    def test_frame_stacking(self, fake_data_dir):
        ds = PixelEpisodeDataset(fake_data_dir, frame_size=64, grayscale=True,
                                 seq_len=10, frame_stack=4)
        frames, actions = ds[0]
        assert frames.shape == (10, 4, 64, 64)

    def test_skips_short_episodes(self, tmp_path):
        _make_fake_episode(tmp_path / "short.npz", n_steps=3)
        _make_fake_episode(tmp_path / "long.npz", n_steps=30)
        ds = PixelEpisodeDataset(tmp_path, frame_size=64, grayscale=True,
                                 seq_len=20, frame_stack=1)
        assert len(ds) > 0  # long episode provides windows


class TestPixelFrameDatasetStateTargets:
    """Tests for state_targets index-based dim selection."""

    def test_state_targets_selects_specific_dims(self, tmp_path):
        """state_targets=[4,5] should load only angle + ang_vel dims."""
        frames = np.random.randint(0, 255, (5, 50, 50, 3), dtype=np.uint8)
        states = np.random.randn(5, 8).astype(np.float32)
        actions = np.random.randn(4, 2).astype(np.float32)
        np.savez(str(tmp_path / "episode_0000.npz"), rgb_frames=frames,
                 states=states, actions=actions)
        ds = PixelFrameDataset(
            str(tmp_path), frame_size=64, grayscale=True,
            state_targets=[4, 5],
        )
        frame, state = ds[0]
        assert state.shape == (2,)
        expected = states[0, [4, 5]]
        assert np.allclose(state.numpy(), expected, atol=1e-6)

    def test_state_targets_all_six(self, tmp_path):
        """state_targets=[0,1,2,3,4,5] equivalent to state_dim=6."""
        frames = np.random.randint(0, 255, (5, 50, 50, 3), dtype=np.uint8)
        states = np.random.randn(5, 8).astype(np.float32)
        actions = np.random.randn(4, 2).astype(np.float32)
        np.savez(str(tmp_path / "episode_0000.npz"), rgb_frames=frames,
                 states=states, actions=actions)
        ds = PixelFrameDataset(
            str(tmp_path), frame_size=64, grayscale=True,
            state_targets=[0, 1, 2, 3, 4, 5],
        )
        frame, state = ds[0]
        assert state.shape == (6,)
        expected = states[0, :6]
        assert np.allclose(state.numpy(), expected, atol=1e-6)

    def test_state_targets_non_contiguous(self, tmp_path):
        """state_targets=[0,2,4] picks non-adjacent dims."""
        frames = np.random.randint(0, 255, (5, 50, 50, 3), dtype=np.uint8)
        states = np.random.randn(5, 8).astype(np.float32)
        np.savez(str(tmp_path / "episode_0000.npz"), rgb_frames=frames,
                 states=states)
        ds = PixelFrameDataset(
            str(tmp_path), frame_size=64, grayscale=True,
            state_targets=[0, 2, 4],
        )
        frame, state = ds[0]
        assert state.shape == (3,)
        expected = states[0, [0, 2, 4]]
        assert np.allclose(state.numpy(), expected, atol=1e-6)

    def test_state_dim_backward_compat(self, fake_data_dir):
        """Existing state_dim=6 still works unchanged."""
        ds = PixelFrameDataset(
            fake_data_dir, frame_size=64, grayscale=True,
            state_dim=6,
        )
        frame, state = ds[0]
        assert state.shape == (6,)

    def test_state_targets_overrides_state_dim(self, tmp_path):
        """If both state_dim and state_targets given, state_targets wins."""
        frames = np.random.randint(0, 255, (5, 50, 50, 3), dtype=np.uint8)
        states = np.random.randn(5, 8).astype(np.float32)
        np.savez(str(tmp_path / "episode_0000.npz"), rgb_frames=frames,
                 states=states)
        ds = PixelFrameDataset(
            str(tmp_path), frame_size=64, grayscale=True,
            state_dim=6,  # would give 6 dims
            state_targets=[4, 5],  # but this should override to 2 dims
        )
        frame, state = ds[0]
        assert state.shape == (2,)
