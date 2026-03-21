"""Tests for collect_trajectories.py — trajectory collection from trained agents.

Uses a mock model with constant actions so we don't need a real trained
agent. Tests the core _collect_episodes() function, not the CLI.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest


class _ConstantPolicy:
    """Mock SB3 model returning gentle main thrust for testing.

    Mimics the model.predict() interface. Returns (action, None) where
    action is shaped (1, 2) to match the VecEnv convention.
    """

    def predict(self, obs, deterministic=False):
        action = np.array([[0.3, 0.0]], dtype=np.float32)
        return action, None


def _make_test_env(seed=42):
    """Create a basic wrapped lunar lander env for testing."""
    from parametric_lunar_lander.wrappers import make_lunar_lander_env
    return make_lunar_lander_env(variant="labeled", seed=seed)


class TestCollectEpisodes:
    """Test _collect_episodes saves well-formed .npz files."""

    def test_creates_expected_number_of_npz_files(self):
        from scripts.collection.collect_trajectories import _collect_episodes

        outdir = tempfile.mkdtemp()
        results = _collect_episodes(
            model=_ConstantPolicy(),
            env_fn=_make_test_env,
            output_dir=outdir,
            n_episodes=3,
            seed=42,
        )

        npz_files = sorted(Path(outdir).glob("*.npz"))
        assert len(npz_files) == 3
        assert len(results) == 3

    def test_npz_arrays_have_correct_shapes(self):
        from scripts.collection.collect_trajectories import _collect_episodes

        outdir = tempfile.mkdtemp()
        results = _collect_episodes(
            model=_ConstantPolicy(),
            env_fn=_make_test_env,
            output_dir=outdir,
            n_episodes=1,
            seed=42,
        )

        data = np.load(results[0]["npz_path"], allow_pickle=False)
        states = data["states"]
        actions = data["actions"]
        rewards = data["rewards"]
        dones = data["dones"]

        T = len(actions)
        assert states.shape == (T + 1, 15)  # T+1 states for T actions
        assert actions.shape == (T, 2)       # (main_thrust, side_thrust)
        assert rewards.shape == (T,)
        assert dones.shape == (T,)
        assert dones[-1] == True             # last step is terminal

    def test_metadata_has_physics_config_and_outcome(self):
        from scripts.collection.collect_trajectories import _collect_episodes

        outdir = tempfile.mkdtemp()
        results = _collect_episodes(
            model=_ConstantPolicy(),
            env_fn=_make_test_env,
            output_dir=outdir,
            n_episodes=1,
            seed=42,
            profile="test-profile",
        )

        data = np.load(results[0]["npz_path"], allow_pickle=False)
        metadata = json.loads(str(data["metadata_json"]))

        assert "physics_config" in metadata
        assert "outcome" in metadata
        assert metadata["outcome"] in ("landed", "crashed", "timeout", "out_of_bounds")
        assert metadata["profile"] == "test-profile"
        assert "episode_length" in metadata
        assert "total_reward" in metadata

    def test_results_dicts_have_expected_keys(self):
        from scripts.collection.collect_trajectories import _collect_episodes

        outdir = tempfile.mkdtemp()
        results = _collect_episodes(
            model=_ConstantPolicy(),
            env_fn=_make_test_env,
            output_dir=outdir,
            n_episodes=2,
            seed=42,
        )

        for r in results:
            assert "npz_path" in r
            assert "outcome" in r
            assert "reward" in r
            assert "steps" in r
            assert Path(r["npz_path"]).exists()

    def test_no_rgb_frames_saved(self):
        """Collection saves data without RGB frames to keep files small."""
        from scripts.collection.collect_trajectories import _collect_episodes

        outdir = tempfile.mkdtemp()
        results = _collect_episodes(
            model=_ConstantPolicy(),
            env_fn=_make_test_env,
            output_dir=outdir,
            n_episodes=1,
            seed=42,
        )

        data = np.load(results[0]["npz_path"], allow_pickle=False)
        assert "rgb_frames" not in data
