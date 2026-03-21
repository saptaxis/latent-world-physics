"""Tests for dataset preparation script."""
import json
import numpy as np
import pytest
from pathlib import Path

from scripts.perception.prepare_encoder_dataset import prepare_dataset


class TestPrepareDataset:

    @pytest.fixture
    def raw_episodes(self, tmp_path):
        """Create 3 small raw episodes with rgb_frames."""
        data_dir = tmp_path / "raw"
        data_dir.mkdir()
        expected_total_frames = 0
        for i, n_steps in enumerate([10, 15, 20]):
            np.savez_compressed(
                str(data_dir / f"ep_{i:04d}.npz"),
                states=np.random.randn(n_steps + 1, 15).astype(np.float32),
                actions=np.random.randn(n_steps, 2).astype(np.float32),
                rewards=np.random.randn(n_steps).astype(np.float32),
                dones=np.zeros(n_steps, dtype=bool),
                rgb_frames=np.random.randint(0, 255, (n_steps + 1, 400, 600, 3), dtype=np.uint8),
                metadata_json=json.dumps({"physics_config": {}, "outcome": "crashed", "seed": i}),
            )
            expected_total_frames += n_steps + 1
        return data_dir, expected_total_frames  # 11 + 16 + 21 = 48

    def test_prepare_creates_valid_file(self, raw_episodes, tmp_path):
        """Prepared file has correct keys, shapes, and dtypes."""
        data_dir, expected_frames = raw_episodes
        out_path = tmp_path / "prepared.npz"
        prepare_dataset([data_dir], out_path, frame_size=64)

        data = np.load(str(out_path))
        assert "frames" in data
        assert "states" in data
        assert "episode_ends" in data

        assert data["frames"].shape == (expected_frames, 64, 64)
        assert data["frames"].dtype == np.uint8
        assert data["states"].shape == (expected_frames, 6)
        assert data["states"].dtype == np.float32
        assert data["episode_ends"].shape == (3,)
        assert data["episode_ends"][-1] == expected_frames

    def test_episode_ends_are_cumulative(self, raw_episodes, tmp_path):
        """episode_ends marks cumulative frame boundaries."""
        data_dir, _ = raw_episodes
        out_path = tmp_path / "prepared.npz"
        prepare_dataset([data_dir], out_path, frame_size=64)

        ends = np.load(str(out_path))["episode_ends"]
        # Episodes have 11, 16, 21 frames → ends = [11, 27, 48]
        assert ends[0] == 11
        assert ends[1] == 27
        assert ends[2] == 48

    def test_multi_directory(self, tmp_path):
        """Preparation works with multiple input directories."""
        for name in ["random", "heuristic"]:
            d = tmp_path / name
            d.mkdir()
            np.savez_compressed(
                str(d / "ep_0000.npz"),
                states=np.random.randn(11, 15).astype(np.float32),
                actions=np.random.randn(10, 2).astype(np.float32),
                rewards=np.random.randn(10).astype(np.float32),
                dones=np.zeros(10, dtype=bool),
                rgb_frames=np.random.randint(0, 255, (11, 64, 96, 3), dtype=np.uint8),
                metadata_json=json.dumps({"physics_config": {}, "outcome": "crashed", "seed": 0}),
            )
        out_path = tmp_path / "prepared.npz"
        prepare_dataset([tmp_path / "random", tmp_path / "heuristic"], out_path, frame_size=64)

        data = np.load(str(out_path))
        assert data["frames"].shape[0] == 22  # 11 + 11
        assert len(data["episode_ends"]) == 2
