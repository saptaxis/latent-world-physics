"""Tests for world model episode dataset."""
import json
import pytest
import numpy as np
import torch
from pathlib import Path

from lwp.wm.dataset import EpisodeDataset


def _make_episode(path: Path, physics: dict, T: int = 20):
    """Create a synthetic episode .npz with realistic shapes."""
    rng = np.random.default_rng(42)
    states = rng.standard_normal((T + 1, 15)).astype(np.float32)
    # Stamp physics params into state dims 8-14 (constant within episode).
    for i, key in enumerate(["gravity", "main_engine_power", "side_engine_power",
                             "lander_density", "angular_damping", "wind_power",
                             "turbulence_power"]):
        states[:, 8 + i] = physics[key]
    metadata = {"physics_config": physics, "outcome": "landed", "source_type": "random"}
    np.savez_compressed(str(path),
        states=states,
        actions=rng.standard_normal((T, 2)).astype(np.float32),
        rewards=rng.standard_normal(T).astype(np.float32),
        dones=np.zeros(T, dtype=bool),
        metadata_json=json.dumps(metadata),
    )


def _make_test_data(tmp_path, n_episodes=30, T=20):
    """Create episodes + split index."""
    source_dir = tmp_path / "full-variation" / "random"
    source_dir.mkdir(parents=True)
    physics = {
        "gravity": -10.0, "main_engine_power": 13.0,
        "side_engine_power": 0.6, "lander_density": 5.0,
        "angular_damping": 0.0, "wind_power": 15.0, "turbulence_power": 1.5,
    }
    index = {}
    for i in range(n_episodes):
        p = source_dir / f"ep_{i:04d}.npz"
        _make_episode(p, physics, T=T)
        # Simple split: first 20 train, next 5 val, last 5 test.
        if i < 20:
            index[str(p)] = "train"
        elif i < 25:
            index[str(p)] = "val"
        else:
            index[str(p)] = "test"

    index_path = tmp_path / "split_index.json"
    import json as _json
    index_path.write_text(_json.dumps(index))
    return index_path


class TestEpisodeDataset:

    def test_load_split(self, tmp_path):
        index_path = _make_test_data(tmp_path)
        ds = EpisodeDataset(index_path, split="train")
        assert len(ds) == 20

    def test_sample_transitions(self, tmp_path):
        index_path = _make_test_data(tmp_path)
        ds = EpisodeDataset(index_path, split="train")
        batch = ds.sample_transitions(batch_size=8)
        s, a, s_next, r = batch
        assert s.shape == (8, 15)
        assert a.shape == (8, 2)
        assert s_next.shape == (8, 15)
        assert r.shape == (8,)
        assert s.dtype == torch.float32

    def test_sample_with_context(self, tmp_path):
        index_path = _make_test_data(tmp_path)
        ds = EpisodeDataset(index_path, split="train")
        batch = ds.sample_with_context(batch_size=4, K=5)
        s, a, s_next, r, context = batch
        assert s.shape == (4, 15)
        assert context.shape == (4, 5, 32)  # K=5, each context = [s, a, s'] = 15+2+15

    def test_sample_sequences(self, tmp_path):
        index_path = _make_test_data(tmp_path, T=50)
        ds = EpisodeDataset(index_path, split="train")
        states, actions, rewards = ds.sample_sequences(batch_size=4, seq_len=20)
        assert states.shape == (4, 21, 15)  # seq_len+1 states
        assert actions.shape == (4, 20, 2)
        assert rewards.shape == (4, 20)

    def test_blind_slices_to_kinematic_dims(self, tmp_path):
        """In blind mode, states should be sliced to 8 kinematic dims."""
        index_path = _make_test_data(tmp_path)
        ds = EpisodeDataset(index_path, split="train", supervision="blind")
        s, a, s_next, r = ds.sample_transitions(batch_size=4)
        # Blind mode produces 8-dim states (kinematic only).
        assert s.shape[1] == 8
        assert s_next.shape[1] == 8

    def test_labeled_preserves_physics(self, tmp_path):
        index_path = _make_test_data(tmp_path)
        ds = EpisodeDataset(index_path, split="train", supervision="labeled")
        s, a, s_next, r = ds.sample_transitions(batch_size=4)
        # Physics dims should be non-zero (they were set to known values).
        assert not (s[:, 8:15] == 0).all()

    def test_corrupt_file_skipped(self, tmp_path):
        """A corrupt .npz should be skipped with a warning, not crash loading."""
        # Create one valid episode and one corrupt file.
        physics = {
            "gravity": -10.0, "main_engine_power": 13.0,
            "side_engine_power": 0.6, "lander_density": 5.0,
            "angular_damping": 0.0, "wind_power": 15.0, "turbulence_power": 1.5,
        }
        valid_path = str(tmp_path / "valid.npz")
        _make_episode(Path(valid_path), physics, T=20)

        corrupt_path = str(tmp_path / "corrupt.npz")
        Path(corrupt_path).write_bytes(b"not a valid npz file")

        index = {valid_path: "train", corrupt_path: "train"}
        index_path = tmp_path / "index.json"
        index_path.write_text(json.dumps(index))

        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ds = EpisodeDataset(index_path, split="train")
            assert len(ds) == 1
            assert len(w) >= 1
            assert any("skip" in str(warning.message).lower() or "corrupt" in str(warning.message).lower()
                        for warning in w)

    def test_shape_validation(self, tmp_path):
        """Episodes with wrong state shape should be skipped."""
        # Valid episode
        physics = {
            "gravity": -10.0, "main_engine_power": 13.0,
            "side_engine_power": 0.6, "lander_density": 5.0,
            "angular_damping": 0.0, "wind_power": 15.0, "turbulence_power": 1.5,
        }
        valid_path = str(tmp_path / "valid.npz")
        _make_episode(Path(valid_path), physics, T=20)

        # Invalid: wrong state dim (12 instead of 15)
        bad_path = str(tmp_path / "bad_shape.npz")
        rng = np.random.default_rng(0)
        np.savez(bad_path,
            states=rng.standard_normal((21, 12)).astype(np.float32),
            actions=rng.standard_normal((20, 2)).astype(np.float32),
            rewards=rng.standard_normal(20).astype(np.float32),
            metadata_json=json.dumps({"physics_config": physics, "source_type": "random"}),
        )

        index = {valid_path: "train", bad_path: "train"}
        index_path = tmp_path / "index.json"
        index_path.write_text(json.dumps(index))

        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ds = EpisodeDataset(index_path, split="train")
            assert len(ds) == 1

    def test_seeded_sampling_is_reproducible(self, tmp_path):
        """Two calls with same rng seed should produce identical batches."""
        index_path = _make_test_data(tmp_path)
        ds = EpisodeDataset(index_path, split="train")
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        s1, a1, sn1, r1 = ds.sample_transitions(16, rng=rng1)
        s2, a2, sn2, r2 = ds.sample_transitions(16, rng=rng2)
        assert torch.equal(s1, s2)
        assert torch.equal(sn1, sn2)

    def test_delta_target(self, tmp_path):
        """When prediction_target='delta', s_next should be the delta."""
        index_path = _make_test_data(tmp_path)
        ds_abs = EpisodeDataset(index_path, split="train", prediction_target="absolute")
        ds_delta = EpisodeDataset(index_path, split="train", prediction_target="delta")

        # Use identical numpy RNGs so both datasets sample the same episodes/timesteps.
        rng1 = np.random.default_rng(0)
        rng2 = np.random.default_rng(0)
        s_abs, a_abs, snext_abs, _ = ds_abs.sample_transitions(8, rng=rng1)
        s_del, a_del, delta, _ = ds_delta.sample_transitions(8, rng=rng2)

        # delta should equal s_next - s
        torch.testing.assert_close(delta, snext_abs - s_abs, atol=1e-6, rtol=1e-6)
