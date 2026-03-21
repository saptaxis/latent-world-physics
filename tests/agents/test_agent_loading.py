"""Tests for load_trained_agent() in eval_utils.py.

Uses mock checkpoint directories with fake config.json files — no actual
trained model needed. Tests the loading logic: config resolution,
path construction, env factory, VecNormalize path.
"""

import json
import os
import tempfile
from pathlib import Path

import pytest


def _make_mock_checkpoint(
    tmpdir,
    variant="labeled",
    algo="ppo",
    history_k=8,
    n_rays=7,
    total_steps=100000,
    has_vec_normalize=True,
    has_config=True,
):
    """Create a mock checkpoint directory with config.json.

    Does NOT create a real model.zip — tests that need model loading
    are integration tests that use real checkpoints.
    """
    if has_config:
        config = {
            "variant": variant,
            "algo": algo,
            "history_k": history_k,
            "n_rays": n_rays,
            "total_steps": total_steps,
        }
        with open(os.path.join(tmpdir, "config.json"), "w") as f:
            json.dump(config, f)

    if has_vec_normalize:
        # Create dummy vec_normalize.pkl (just needs to exist for path resolution)
        with open(os.path.join(tmpdir, "vec_normalize.pkl"), "w") as f:
            f.write("dummy")

    return tmpdir


class TestResolveModelPath:
    """Test model path resolution logic."""

    def test_default_model_zip(self, tmp_path):
        from lwp.agents.eval_utils import resolve_model_path
        # Create model.zip
        (tmp_path / "model.zip").touch()
        path = resolve_model_path(str(tmp_path), model_name=None)
        assert path == str(tmp_path / "model.zip")

    def test_specific_model_in_checkpoints_subdir(self, tmp_path):
        from lwp.agents.eval_utils import resolve_model_path
        (tmp_path / "checkpoints").mkdir()
        (tmp_path / "checkpoints" / "rl_model_1000000_steps.zip").touch()
        path = resolve_model_path(str(tmp_path), model_name="rl_model_1000000_steps.zip")
        assert "checkpoints" in path
        assert path.endswith("rl_model_1000000_steps.zip")

    def test_specific_model_toplevel(self, tmp_path):
        from lwp.agents.eval_utils import resolve_model_path
        (tmp_path / "rl_model_500000_steps.zip").touch()
        path = resolve_model_path(str(tmp_path), model_name="rl_model_500000_steps.zip")
        assert path.endswith("rl_model_500000_steps.zip")

    def test_best_subdir_model(self, tmp_path):
        from lwp.agents.eval_utils import resolve_model_path
        (tmp_path / "best").mkdir()
        (tmp_path / "best" / "model.zip").touch()
        path = resolve_model_path(str(tmp_path), model_name=None)
        assert path == str(tmp_path / "best" / "model.zip")

    def test_missing_model_raises(self, tmp_path):
        from lwp.agents.eval_utils import resolve_model_path
        with pytest.raises(FileNotFoundError):
            resolve_model_path(str(tmp_path), model_name=None)


class TestResolveVecNormalizePath:
    """Test VecNormalize stats path resolution."""

    def test_default_vec_normalize(self, tmp_path):
        from lwp.agents.eval_utils import resolve_vec_normalize_path
        (tmp_path / "vec_normalize.pkl").touch()
        model_path = str(tmp_path / "model.zip")
        path = resolve_vec_normalize_path(str(tmp_path), model_path)
        assert path.endswith("vec_normalize.pkl")

    def test_periodic_checkpoint_vec_normalize(self, tmp_path):
        from lwp.agents.eval_utils import resolve_vec_normalize_path
        (tmp_path / "checkpoints").mkdir()
        (tmp_path / "checkpoints" / "rl_model_vecnormalize_1000000_steps.pkl").touch()
        model_path = str(tmp_path / "checkpoints" / "rl_model_1000000_steps.zip")
        path = resolve_vec_normalize_path(str(tmp_path), model_path)
        assert "vecnormalize" in path
        assert "1000000" in path

    def test_missing_vec_normalize_returns_none(self, tmp_path):
        from lwp.agents.eval_utils import resolve_vec_normalize_path
        model_path = str(tmp_path / "model.zip")
        path = resolve_vec_normalize_path(str(tmp_path), model_path)
        assert path is None


class TestLoadTrainingConfig:
    """Test config.json loading."""

    def test_loads_config(self, tmp_path):
        from lwp.agents.eval_utils import load_training_config
        _make_mock_checkpoint(str(tmp_path), variant="blind", algo="sac")
        config = load_training_config(str(tmp_path))
        assert config["variant"] == "blind"
        assert config["algo"] == "sac"

    def test_missing_config_raises(self, tmp_path):
        from lwp.agents.eval_utils import load_training_config
        with pytest.raises(FileNotFoundError):
            load_training_config(str(tmp_path))

    def test_defaults_for_optional_fields(self, tmp_path):
        from lwp.agents.eval_utils import load_training_config
        # Config with only variant
        with open(tmp_path / "config.json", "w") as f:
            json.dump({"variant": "labeled"}, f)
        config = load_training_config(str(tmp_path))
        assert config["variant"] == "labeled"
        assert config.get("algo", "ppo") == "ppo"


class TestMakeEnvFactory:
    """Test env factory construction."""

    def test_returns_callable(self):
        from lwp.agents.eval_utils import make_env_factory
        fn = make_env_factory(variant="labeled", n_rays=7, history_k=8)
        assert callable(fn)

    def test_factory_creates_env(self):
        from lwp.agents.eval_utils import make_env_factory
        fn = make_env_factory(variant="labeled", n_rays=7, history_k=8)
        env = fn(seed=42)
        obs, _ = env.reset()
        assert obs.shape[0] == 22  # labeled = 15 base + 7 rays
        env.close()

    def test_blind_variant_obs_shape(self):
        from lwp.agents.eval_utils import make_env_factory
        fn = make_env_factory(variant="blind", n_rays=7, history_k=8)
        env = fn(seed=42)
        obs, _ = env.reset()
        assert obs.shape[0] == 15  # blind = 8 kinematic + 7 rays
        env.close()

    def test_factory_with_profile(self):
        from lwp.agents.eval_utils import make_env_factory
        fn = make_env_factory(
            variant="labeled", n_rays=7, history_k=8, profile="easy"
        )
        env = fn(seed=42)
        obs, _ = env.reset()
        assert obs.shape[0] == 22
        env.close()


class TestBuildEvalBatches:
    """Test eval batch construction from profile strings."""

    def test_single_default_batch(self):
        from lwp.agents.eval_utils import build_eval_batches, make_env_factory
        env_fn = make_env_factory(variant="labeled", n_rays=7, history_k=8)
        batches = build_eval_batches(
            variant="labeled", n_rays=7, history_k=8,
            profiles_str=None, default_env_fn=env_fn,
        )
        assert len(batches) == 1
        assert batches[0][0] == "default"

    def test_multi_profile_batches(self):
        from lwp.agents.eval_utils import build_eval_batches, make_env_factory
        env_fn = make_env_factory(variant="labeled", n_rays=7, history_k=8)
        batches = build_eval_batches(
            variant="labeled", n_rays=7, history_k=8,
            profiles_str="easy,medium,hard", default_env_fn=env_fn,
        )
        assert len(batches) == 3
        assert [b[0] for b in batches] == ["easy", "medium", "hard"]
        # Each batch has a callable env_fn
        for name, fn in batches:
            assert callable(fn)
