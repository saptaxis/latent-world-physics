"""Tests for training config YAML loading."""

import os
import tempfile

import pytest
import yaml

from lwp.agents.training_config import (
    TRAINING_DEFAULTS,
    load_training_config,
    load_batch_config,
)


class TestTrainingDefaults:
    """TRAINING_DEFAULTS has sane values for all expected keys."""

    def test_has_required_keys(self):
        required = [
            "variant", "algo", "total_steps", "n_envs", "seed",
            "eval_freq", "checkpoint_freq", "video_freq",
            "history_k", "n_rays", "net_arch", "run_dir",
        ]
        for key in required:
            assert key in TRAINING_DEFAULTS, f"Missing default for '{key}'"

    def test_required_fields_are_none(self):
        """Operational params must be explicitly set — no silent defaults."""
        for key in ("variant", "algo", "total_steps", "run_dir"):
            assert TRAINING_DEFAULTS[key] is None, f"'{key}' should be None (required)"

    def test_structural_defaults_have_values(self):
        """Structural/hyperparameter defaults are fine — rarely change."""
        assert TRAINING_DEFAULTS["n_rays"] == 7
        assert TRAINING_DEFAULTS["history_k"] == 8
        assert TRAINING_DEFAULTS["net_arch"] is None  # None = default 3x256
        assert TRAINING_DEFAULTS["n_envs"] == 8

    def test_ent_coef_default_is_none(self):
        """ent_coef defaults to None (use rl_common default)."""
        assert "ent_coef" in TRAINING_DEFAULTS
        assert TRAINING_DEFAULTS["ent_coef"] is None


class TestLoadTrainingConfig:
    """load_training_config() merges YAML with TRAINING_DEFAULTS."""

    def test_empty_yaml_returns_defaults(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({}, f)
            f.flush()
            config = load_training_config(f.name)
        os.unlink(f.name)
        # Required fields stay None — train_rl.py validates these
        assert config["variant"] is None
        assert config["algo"] is None
        assert config["total_steps"] is None
        assert config["run_dir"] is None
        # Structural defaults populated
        assert config["n_envs"] == 8
        assert config["n_rays"] == 7

    def test_yaml_overrides_defaults(self):
        data = {"variant": "blind", "algo": "sac", "total_steps": 500_000, "seed": 123}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(data, f)
            f.flush()
            config = load_training_config(f.name)
        os.unlink(f.name)
        assert config["variant"] == "blind"
        assert config["algo"] == "sac"
        assert config["total_steps"] == 500_000
        assert config["seed"] == 123
        # Unspecified structural defaults kept
        assert config["n_envs"] == 8

    def test_net_arch_list_preserved(self):
        data = {"variant": "labeled", "net_arch": [64, 64]}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(data, f)
            f.flush()
            config = load_training_config(f.name)
        os.unlink(f.name)
        assert config["net_arch"] == [64, 64]

    def test_net_arch_dict_preserved(self):
        """PPO-style separate pi/vf architecture."""
        arch = {"pi": [256, 256, 256], "vf": [256, 256, 256]}
        data = {"variant": "labeled", "net_arch": arch}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(data, f)
            f.flush()
            config = load_training_config(f.name)
        os.unlink(f.name)
        assert config["net_arch"] == arch

    def test_run_dir_preserved(self):
        data = {"variant": "labeled", "run_dir": "runs/lunar_lander/labeled-ppo"}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(data, f)
            f.flush()
            config = load_training_config(f.name)
        os.unlink(f.name)
        assert config["run_dir"] == "runs/lunar_lander/labeled-ppo"

    def test_unknown_keys_silently_ignored(self):
        data = {"variant": "blind", "unknown_future_key": 42}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(data, f)
            f.flush()
            config = load_training_config(f.name)
        os.unlink(f.name)
        assert config["variant"] == "blind"
        assert "unknown_future_key" not in config

    def test_builtin_name_not_found_raises(self):
        with pytest.raises(FileNotFoundError, match="No builtin config"):
            load_training_config("nonexistent_config_name_xyz")

    def test_file_path_not_found_raises(self):
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            load_training_config("/tmp/definitely_not_a_real_config.yaml")

    def test_file_path_loads(self):
        data = {"variant": "history", "algo": "sac"}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(data, f)
            f.flush()
            config = load_training_config(f.name)
        os.unlink(f.name)
        assert config["variant"] == "history"
        assert config["algo"] == "sac"

    def test_builtin_labeled_ppo_loads(self):
        """Verify a real builtin config loads correctly."""
        config = load_training_config("full-variation/labeled-ppo-easy")
        assert config["variant"] == "labeled"
        assert config["algo"] == "ppo"
        assert config["profile"] == "easy"
        assert config["total_steps"] == 3_000_000
        assert config["net_arch"] == {"pi": [256, 256, 256], "vf": [256, 256, 256]}

    def test_builtin_small_has_net_arch(self):
        """Small-network configs specify net_arch."""
        config = load_training_config("full-variation/labeled-ppo-easy-small")
        assert config["net_arch"] == [64, 64]
        assert config["variant"] == "labeled"

    def test_ent_coef_zero_preserved(self):
        """ent_coef=0.0 in YAML is loaded (not confused with None/falsy)."""
        data = {"variant": "labeled", "algo": "ppo", "ent_coef": 0.0}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(data, f)
            f.flush()
            config = load_training_config(f.name)
        os.unlink(f.name)
        assert config["ent_coef"] == 0.0

    def test_ent_coef_absent_is_none(self):
        """Config without ent_coef gets None (don't override rl_common default)."""
        data = {"variant": "labeled", "algo": "ppo"}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(data, f)
            f.flush()
            config = load_training_config(f.name)
        os.unlink(f.name)
        assert config["ent_coef"] is None

    def test_builtin_noent_has_ent_coef_zero(self):
        """noent configs set ent_coef=0.0 explicitly."""
        config = load_training_config("full-variation/labeled-ppo-easy-noent")
        assert config["ent_coef"] == 0.0
        assert config["variant"] == "labeled"
        assert config["net_arch"] == {"pi": [256, 256, 256], "vf": [256, 256, 256]}


class TestLoadBatchConfig:
    """load_batch_config() returns a flat list of config names."""

    def test_flat_list(self):
        data = {"runs": ["labeled-ppo", "blind-ppo", "history-ppo"]}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(data, f)
            f.flush()
            runs = load_batch_config(f.name)
        os.unlink(f.name)
        assert runs == ["labeled-ppo", "blind-ppo", "history-ppo"]

    def test_empty_runs_raises(self):
        data = {"runs": []}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(data, f)
            f.flush()
            with pytest.raises(ValueError, match="no runs"):
                load_batch_config(f.name)
        os.unlink(f.name)

    def test_missing_runs_key_raises(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"something_else": True}, f)
            f.flush()
            with pytest.raises(ValueError, match="no runs"):
                load_batch_config(f.name)
        os.unlink(f.name)

    def test_builtin_name_not_found_raises(self):
        with pytest.raises(FileNotFoundError, match="No builtin batch"):
            load_batch_config("nonexistent_batch_xyz")

    def test_builtin_all_ppo_loads(self):
        """Verify a real builtin batch config loads."""
        runs = load_batch_config("all-ppo")
        assert runs == ["labeled-ppo-easy", "blind-ppo", "history-ppo"]

    def test_builtin_full_matrix_loads(self):
        runs = load_batch_config("full-matrix")
        assert len(runs) == 10
        assert "labeled-ppo-easy" in runs
        assert "labeled-ppo-gym-default-small" in runs


class TestSubdirectoryResolution:
    """Config names with subdirectories should resolve as builtins."""

    def test_subdir_config_loads(self):
        """A config in a subdirectory should load via 'subdir/name'."""
        config = load_training_config("full-variation/labeled-ppo-easy")
        assert config["variant"] == "labeled"
        assert config["profile"] == "easy"

    def test_subdir_config_matches_flat(self):
        """Subdir config should produce identical result to flat config (both use subdir path now)."""
        config = load_training_config("full-variation/labeled-ppo-easy")
        assert config["variant"] == "labeled"
        assert config["profile"] == "easy"

    def test_absolute_path_still_works(self):
        """Absolute paths should still bypass builtin resolution."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"variant": "blind", "algo": "ppo"}, f)
            f.flush()
            config = load_training_config(f.name)
        os.unlink(f.name)
        assert config["variant"] == "blind"

    def test_nonexistent_subdir_raises(self):
        """A subdir/name that doesn't exist should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="No builtin config"):
            load_training_config("nonexistent-subdir/fake-config")
