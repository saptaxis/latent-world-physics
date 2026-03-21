"""Tests for encoder pre-training datasets (raw and prepared modes)."""
import json
import numpy as np
import pytest
import torch
from pathlib import Path

from lwp.agents.encoder_dataset import (
    EncoderPretrainDataset,
    PreparedEncoderDataset,
)


def _make_fake_episodes(tmp_path, n_episodes=3, step_counts=None):
    """Create fake episodes with rgb_frames and states.

    Returns (episode_paths, data_dir).
    """
    step_counts = step_counts or [20 + i * 5 for i in range(n_episodes)]
    episode_paths = []
    for i, n_steps in enumerate(step_counts):
        states = np.random.randn(n_steps + 1, 15).astype(np.float32)
        actions = np.random.randn(n_steps, 2).astype(np.float32)
        rewards = np.random.randn(n_steps).astype(np.float32)
        dones = np.zeros(n_steps, dtype=bool)
        dones[-1] = True
        # RGB frames at 400x600 (raw env render size)
        rgb_frames = np.random.randint(
            0, 255, (n_steps + 1, 400, 600, 3), dtype=np.uint8
        )
        metadata = {
            "physics_config": {"gravity": -10.0},
            "outcome": "crashed",
            "seed": i,
        }

        path = tmp_path / f"episode_{i:04d}.npz"
        np.savez_compressed(
            str(path),
            states=states,
            actions=actions,
            rewards=rewards,
            dones=dones,
            rgb_frames=rgb_frames,
            metadata_json=json.dumps(metadata),
        )
        episode_paths.append(path)
    return episode_paths, tmp_path


class TestEncoderPretrainDataset:
    """Test the raw (on-the-fly) dataset."""

    @pytest.fixture
    def fake_episodes(self, tmp_path):
        return _make_fake_episodes(tmp_path)

    def test_dataset_length(self, fake_episodes):
        """Dataset length = sum of valid frame stack positions across episodes."""
        paths, data_dir = fake_episodes
        ds = EncoderPretrainDataset(data_dir, n_stack=4, frame_size=128)
        # Episode 0: 21 frames, valid stacks = 21 - 4 + 1 = 18
        # Episode 1: 26 frames, valid stacks = 26 - 4 + 1 = 23
        # Episode 2: 31 frames, valid stacks = 31 - 4 + 1 = 28
        assert len(ds) == 18 + 23 + 28

    def test_item_shapes(self, fake_episodes):
        """Each item is (frames, targets) with correct shapes."""
        paths, data_dir = fake_episodes
        ds = EncoderPretrainDataset(data_dir, n_stack=4, frame_size=128)
        frames, targets = ds[0]
        assert frames.shape == (4, 128, 128)
        assert frames.dtype == torch.float32
        assert targets.shape == (6,)
        assert targets.dtype == torch.float32

    def test_pixel_normalization(self, fake_episodes):
        """Frames are normalized to [0, 1] float32."""
        paths, data_dir = fake_episodes
        ds = EncoderPretrainDataset(data_dir, n_stack=4, frame_size=128)
        frames, _ = ds[0]
        assert frames.min() >= 0.0
        assert frames.max() <= 1.0

    def test_targets_are_kinematic_state(self, fake_episodes):
        """Targets are states[:6] at the LAST frame in the stack."""
        paths, data_dir = fake_episodes
        ds = EncoderPretrainDataset(data_dir, n_stack=4, frame_size=128)
        _, targets = ds[0]
        ep = np.load(str(paths[0]))
        expected = ep["states"][3, :6]  # Index 3 = last frame of stack [0,1,2,3]
        np.testing.assert_allclose(targets.numpy(), expected, atol=1e-6)

    def test_frame_stack_temporal_order(self, fake_episodes):
        """Frame stack has oldest frame first, newest last."""
        paths, data_dir = fake_episodes
        ds = EncoderPretrainDataset(data_dir, n_stack=4, frame_size=128)
        frames_0, _ = ds[0]  # Stack of frames [0,1,2,3]
        frames_1, _ = ds[1]  # Stack of frames [1,2,3,4]
        # frames_0[1:] should equal frames_1[:3] (overlapping region)
        torch.testing.assert_close(frames_0[1:], frames_1[:3])

    def test_cross_episode_boundary(self, fake_episodes):
        """Stacks don't cross episode boundaries."""
        paths, data_dir = fake_episodes
        ds = EncoderPretrainDataset(data_dir, n_stack=4, frame_size=128)
        _, targets_ep0_last = ds[17]  # Last stack in episode 0
        _, targets_ep1_first = ds[18]  # First stack in episode 1
        ep0 = np.load(str(paths[0]))
        ep1 = np.load(str(paths[1]))
        np.testing.assert_allclose(targets_ep0_last.numpy(), ep0["states"][20, :6], atol=1e-6)
        np.testing.assert_allclose(targets_ep1_first.numpy(), ep1["states"][3, :6], atol=1e-6)

    def test_validation_split(self, fake_episodes):
        """Dataset supports train/val split with non-overlapping episode indices."""
        paths, data_dir = fake_episodes
        ds_train = EncoderPretrainDataset(data_dir, n_stack=4, frame_size=128, split="train", val_fraction=0.34)
        ds_val = EncoderPretrainDataset(data_dir, n_stack=4, frame_size=128, split="val", val_fraction=0.34)
        assert len(ds_train) + len(ds_val) == 18 + 23 + 28
        assert len(ds_train) > 0
        assert len(ds_val) > 0

    def test_multi_directory(self, tmp_path):
        """Dataset loads from multiple directories (random + heuristic)."""
        dir_a = tmp_path / "random"
        dir_b = tmp_path / "heuristic"
        dir_a.mkdir()
        dir_b.mkdir()
        for d in [dir_a, dir_b]:
            for i in range(2):
                n_steps = 10
                np.savez_compressed(
                    str(d / f"ep_{i:04d}.npz"),
                    states=np.random.randn(n_steps + 1, 15).astype(np.float32),
                    actions=np.random.randn(n_steps, 2).astype(np.float32),
                    rewards=np.random.randn(n_steps).astype(np.float32),
                    dones=np.zeros(n_steps, dtype=bool),
                    rgb_frames=np.random.randint(0, 255, (n_steps + 1, 64, 96, 3), dtype=np.uint8),
                    metadata_json=json.dumps({"physics_config": {}, "outcome": "crashed", "seed": i}),
                )
        ds = EncoderPretrainDataset([dir_a, dir_b], n_stack=4, frame_size=64)
        assert len(ds) == 4 * 8  # 4 episodes × (11 - 4 + 1) = 32


class TestPreparedEncoderDataset:
    """Test the prepared (in-memory) dataset."""

    @pytest.fixture
    def prepared_file(self, tmp_path):
        """Create a small prepared dataset file.

        Prepared format:
            frames: (N_total, H, W) uint8 — all processed grayscale frames concatenated
            states: (N_total, 6) float32 — kinematic targets per frame
            episode_ends: (n_episodes,) int64 — cumulative frame count per episode
        """
        # 3 episodes with 11, 16, 21 frames (= 10, 15, 20 steps)
        frames_list = []
        states_list = []
        episode_ends = []
        for n_frames in [11, 16, 21]:
            frames_list.append(np.random.randint(0, 255, (n_frames, 64, 64), dtype=np.uint8))
            states_list.append(np.random.randn(n_frames, 6).astype(np.float32))
            episode_ends.append(sum(f.shape[0] for f in frames_list))

        path = tmp_path / "prepared.npz"
        np.savez_compressed(
            str(path),
            frames=np.concatenate(frames_list, axis=0),
            states=np.concatenate(states_list, axis=0),
            episode_ends=np.array(episode_ends, dtype=np.int64),
        )
        return path

    def test_dataset_length(self, prepared_file):
        """Length matches valid stacks across all episodes."""
        ds = PreparedEncoderDataset(prepared_file, n_stack=4)
        # Episode 0: 11 frames → 8 stacks
        # Episode 1: 16 frames → 13 stacks
        # Episode 2: 21 frames → 18 stacks
        assert len(ds) == 8 + 13 + 18

    def test_item_shapes(self, prepared_file):
        """Returns (n_stack, H, W) float32 and (6,) float32."""
        ds = PreparedEncoderDataset(prepared_file, n_stack=4)
        frames, targets = ds[0]
        assert frames.shape == (4, 64, 64)
        assert frames.dtype == torch.float32
        assert targets.shape == (6,)
        assert targets.dtype == torch.float32

    def test_pixel_normalization(self, prepared_file):
        """Frames are normalized to [0, 1]."""
        ds = PreparedEncoderDataset(prepared_file, n_stack=4)
        frames, _ = ds[0]
        assert frames.min() >= 0.0
        assert frames.max() <= 1.0

    def test_cross_episode_boundary(self, prepared_file):
        """Stacks don't cross episode boundaries."""
        ds = PreparedEncoderDataset(prepared_file, n_stack=4)
        # Episode 0 ends at index 8 (11 frames → 8 valid stacks)
        # Check that stack index 7 (last in ep0) and 8 (first in ep1) use different frames
        f0, _ = ds[7]   # Last stack in episode 0
        f1, _ = ds[8]   # First stack in episode 1
        # These should NOT share frames (different episodes)
        assert not torch.equal(f0[-1], f1[0])

    def test_validation_split(self, prepared_file):
        """Train/val split works on prepared datasets."""
        ds_train = PreparedEncoderDataset(prepared_file, n_stack=4, split="train", val_fraction=0.34)
        ds_val = PreparedEncoderDataset(prepared_file, n_stack=4, split="val", val_fraction=0.34)
        assert len(ds_train) + len(ds_val) == 8 + 13 + 18
        assert len(ds_train) > 0
        assert len(ds_val) > 0
