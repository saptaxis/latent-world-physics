"""Tests for world model data splitting."""
import json
import pytest
import numpy as np
from pathlib import Path

from lwp.wm.mix_config import MixConfig
from lwp.wm.split import build_episode_index, load_episode_index


def _make_episode(path: Path, physics: dict, source_type: str = "random"):
    """Create a minimal .npz episode for testing."""
    T = 5
    metadata = {"physics_config": physics, "outcome": "landed", "source_type": source_type}
    np.savez_compressed(
        str(path),
        states=np.zeros((T + 1, 15), dtype=np.float32),
        actions=np.zeros((T, 2), dtype=np.float32),
        rewards=np.zeros(T, dtype=np.float32),
        dones=np.zeros(T, dtype=bool),
        metadata_json=json.dumps(metadata),
    )


class TestBuildEpisodeIndex:

    def _make_dataset(self, tmp_path, n_episodes=50):
        """Create a fake dataset with known physics spread."""
        rng = np.random.default_rng(42)
        profile_dir = tmp_path / "full-variation"
        source_dir = profile_dir / "random"
        source_dir.mkdir(parents=True)

        for i in range(n_episodes):
            physics = {
                "gravity": float(rng.uniform(-12.0, -2.0)),
                "main_engine_power": float(rng.uniform(5.0, 25.0)),
                "side_engine_power": float(rng.uniform(0.2, 1.5)),
                "lander_density": float(rng.uniform(2.5, 10.0)),
                "angular_damping": float(rng.uniform(0.0, 5.0)),
                "wind_power": float(rng.uniform(0.0, 30.0)),
                "turbulence_power": float(rng.uniform(0.0, 5.0)),
            }
            _make_episode(source_dir / f"ep_{i:04d}.npz", physics)

        return tmp_path

    def _make_mix_config(self, data_base: str) -> MixConfig:
        return MixConfig.from_dict({
            "profiles": [{"name": "full-variation", "path": "full-variation"}],
            "data_base": data_base,
            "split": {
                "method": "quantile_grid",
                "axes": ["gravity", "main_engine_power", "lander_density"],
                "bins": 3,
                "train_ratio": 0.8,
                "val_ratio": 0.1,
                "ood_holdout": None,
                "policy_holdout": None,
            },
        })

    def test_all_episodes_assigned(self, tmp_path):
        base = self._make_dataset(tmp_path)
        cfg = self._make_mix_config(str(base))
        index = build_episode_index(cfg)
        assert len(index) == 50
        assert all(v in ("train", "val", "test", "ood", "policy_holdout") for v in index.values())

    def test_splits_non_empty(self, tmp_path):
        base = self._make_dataset(tmp_path, n_episodes=100)
        cfg = self._make_mix_config(str(base))
        index = build_episode_index(cfg)
        splits = set(index.values())
        assert "train" in splits
        assert "val" in splits
        assert "test" in splits

    def test_train_ratio_approximate(self, tmp_path):
        base = self._make_dataset(tmp_path, n_episodes=200)
        cfg = self._make_mix_config(str(base))
        index = build_episode_index(cfg)
        train_frac = sum(1 for v in index.values() if v == "train") / len(index)
        # Region-based split won't be exact, but should be roughly right.
        assert 0.6 < train_frac < 0.95

    def test_ood_holdout(self, tmp_path):
        """Episodes in OOD corner get assigned to 'ood' split."""
        profile_dir = tmp_path / "full-variation" / "random"
        profile_dir.mkdir(parents=True)

        # Create episodes: some in OOD corner, some not.
        ood_physics = {
            "gravity": -11.0, "main_engine_power": 6.0,
            "side_engine_power": 0.5, "lander_density": 9.0,
            "angular_damping": 1.0, "wind_power": 5.0, "turbulence_power": 1.0,
        }
        normal_physics = {
            "gravity": -5.0, "main_engine_power": 15.0,
            "side_engine_power": 0.5, "lander_density": 5.0,
            "angular_damping": 1.0, "wind_power": 5.0, "turbulence_power": 1.0,
        }
        for i in range(10):
            _make_episode(profile_dir / f"ood_{i:04d}.npz", ood_physics)
        for i in range(40):
            _make_episode(profile_dir / f"normal_{i:04d}.npz", normal_physics)

        cfg = MixConfig.from_dict({
            "profiles": [{"name": "full-variation", "path": "full-variation"}],
            "data_base": str(tmp_path),
            "split": {
                "method": "quantile_grid",
                "axes": ["gravity", "main_engine_power", "lander_density"],
                "bins": 3, "train_ratio": 0.8, "val_ratio": 0.1,
                "ood_holdout": {
                    "gravity": [-12.0, -9.0],
                    "main_engine_power": [5.0, 8.0],
                    "lander_density": [7.5, 10.0],
                },
                "policy_holdout": None,
            },
        })
        index = build_episode_index(cfg)
        ood_episodes = [k for k, v in index.items() if v == "ood"]
        assert len(ood_episodes) == 10

    def test_policy_holdout(self, tmp_path):
        """Episodes from held-out source go to 'policy_holdout' split."""
        profile_dir = tmp_path / "full-variation"
        physics = {
            "gravity": -5.0, "main_engine_power": 15.0,
            "side_engine_power": 0.5, "lander_density": 5.0,
            "angular_damping": 1.0, "wind_power": 5.0, "turbulence_power": 1.0,
        }
        for source in ["random", "heuristic"]:
            d = profile_dir / source
            d.mkdir(parents=True)
            for i in range(20):
                _make_episode(d / f"ep_{i:04d}.npz", physics, source_type=source)

        cfg = MixConfig.from_dict({
            "profiles": [{"name": "full-variation", "path": "full-variation"}],
            "data_base": str(tmp_path),
            "split": {
                "method": "quantile_grid",
                "axes": ["gravity", "main_engine_power", "lander_density"],
                "bins": 3, "train_ratio": 0.8, "val_ratio": 0.1,
                "ood_holdout": None,
                "policy_holdout": "random",
            },
        })
        index = build_episode_index(cfg)
        holdout = [k for k, v in index.items() if v == "policy_holdout"]
        assert len(holdout) == 20
        assert all("random" in k for k in holdout)

    def test_persist_and_reload(self, tmp_path):
        base = self._make_dataset(tmp_path)
        cfg = self._make_mix_config(str(base))
        index = build_episode_index(cfg)

        index_path = tmp_path / "split_index.json"
        build_episode_index(cfg, save_path=index_path)
        loaded = load_episode_index(index_path)
        assert loaded == index

    def test_deterministic(self, tmp_path):
        base = self._make_dataset(tmp_path)
        cfg = self._make_mix_config(str(base))
        index1 = build_episode_index(cfg, seed=42)
        index2 = build_episode_index(cfg, seed=42)
        assert index1 == index2
