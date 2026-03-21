"""Tests for world model mix config parsing."""
import pytest
import yaml
from pathlib import Path

from lwp.wm.mix_config import MixConfig


class TestMixConfig:

    def _sample_mix_yaml(self) -> dict:
        return {
            "profiles": [
                {"name": "full-variation", "path": "full-variation"},
            ],
            "data_base": "/fake/base",
            "split": {
                "method": "quantile_grid",
                "axes": ["gravity", "main_engine_power", "lander_density"],
                "bins": 3,
                "train_ratio": 0.8,
                "val_ratio": 0.1,
                "ood_holdout": {
                    "gravity": [-12.0, -9.0],
                    "main_engine_power": [5.0, 8.0],
                    "lander_density": [7.5, 10.0],
                },
                "policy_holdout": None,
            },
        }

    def test_parse_basic(self):
        cfg = MixConfig.from_dict(self._sample_mix_yaml())
        assert len(cfg.profiles) == 1
        assert cfg.profiles[0]["name"] == "full-variation"
        assert cfg.split_method == "quantile_grid"
        assert cfg.split_axes == ["gravity", "main_engine_power", "lander_density"]
        assert cfg.train_ratio == 0.8
        assert cfg.val_ratio == 0.1

    def test_ood_holdout_parsed(self):
        cfg = MixConfig.from_dict(self._sample_mix_yaml())
        assert cfg.ood_holdout is not None
        assert cfg.ood_holdout["gravity"] == (-12.0, -9.0)

    def test_multiple_profiles(self):
        d = self._sample_mix_yaml()
        d["profiles"].append({"name": "easy", "path": "easy"})
        cfg = MixConfig.from_dict(d)
        assert len(cfg.profiles) == 2

    def test_policy_holdout_default_none(self):
        cfg = MixConfig.from_dict(self._sample_mix_yaml())
        assert cfg.policy_holdout is None

    def test_load_from_yaml(self, tmp_path):
        d = self._sample_mix_yaml()
        p = tmp_path / "mix.yaml"
        p.write_text(yaml.dump(d))
        cfg = MixConfig.load(p)
        assert cfg.split_method == "quantile_grid"
