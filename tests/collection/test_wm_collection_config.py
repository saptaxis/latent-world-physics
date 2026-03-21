"""Tests for world model collection YAML config parsing."""

import pytest
import yaml

from lwp.collection.wm_collection_config import (
    CollectionConfig,
    BatchConfig,
    ManeuverConfig,
    StartConfig,
)


class TestCollectionConfig:
    """Single-run collection config from YAML."""

    def _sample_yaml(self) -> dict:
        return {
            "source_type": "blind_agent",
            "checkpoint_dir": "/path/to/agent",
            "n_episodes": 100,
            "physics_sampling": {
                "method": "uniform",
                "ranges": {
                    "gravity": [-12.0, -2.0],
                    "main_engine_power": [5.0, 25.0],
                    "side_engine_power": [0.2, 1.5],
                    "lander_density": [2.5, 10.0],
                    "angular_damping": [0.0, 5.0],
                    "wind_power": [0.0, 30.0],
                    "turbulence_power": [0.0, 5.0],
                },
            },
            "seed": 0,
            "deterministic": True,
            "save_frames": False,
        }

    def test_parse_rl_agent_config(self):
        cfg = CollectionConfig.from_dict(self._sample_yaml())
        assert cfg.source_type == "blind_agent"
        assert cfg.n_episodes == 100
        assert cfg.checkpoint_dir == "/path/to/agent"
        assert cfg.deterministic is True
        assert cfg.physics_ranges["gravity"] == (-12.0, -2.0)

    def test_parse_random_config(self):
        d = self._sample_yaml()
        d["source_type"] = "random"
        del d["checkpoint_dir"]
        del d["deterministic"]
        cfg = CollectionConfig.from_dict(d)
        assert cfg.source_type == "random"
        assert cfg.checkpoint_dir is None

    def test_parse_noisy_expert_fixed_sigma(self):
        d = self._sample_yaml()
        d["source_type"] = "noisy_expert"
        d["noise_sigma"] = 0.1
        cfg = CollectionConfig.from_dict(d)
        assert cfg.noise_sigma == (0.1, 0.1)  # scalar normalized to range

    def test_parse_noisy_expert_sigma_range(self):
        d = self._sample_yaml()
        d["source_type"] = "noisy_expert"
        d["noise_sigma"] = [0.01, 0.5]
        cfg = CollectionConfig.from_dict(d)
        assert cfg.noise_sigma == (0.01, 0.5)

    def test_invalid_source_type_raises(self):
        d = self._sample_yaml()
        d["source_type"] = "invalid"
        with pytest.raises(ValueError, match="source_type"):
            CollectionConfig.from_dict(d)

    def test_rl_agent_requires_checkpoint_dir(self):
        d = self._sample_yaml()
        del d["checkpoint_dir"]
        with pytest.raises(ValueError, match="checkpoint_dir"):
            CollectionConfig.from_dict(d)

    def test_load_from_yaml_file(self, tmp_path):
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text(yaml.dump(self._sample_yaml()))
        cfg = CollectionConfig.load(str(yaml_path))
        assert cfg.source_type == "blind_agent"

    def test_physics_ranges_all_7_params(self):
        cfg = CollectionConfig.from_dict(self._sample_yaml())
        assert len(cfg.physics_ranges) == 7

    def test_default_noise_sigma(self):
        """noise_sigma defaults to (0.01, 0.5) range for noisy_expert."""
        d = self._sample_yaml()
        d["source_type"] = "noisy_expert"
        cfg = CollectionConfig.from_dict(d)
        assert cfg.noise_sigma == (0.01, 0.5)


VALID_SOURCE_TYPES = ["blind_agent", "labeled_agent", "heuristic", "random", "noisy_expert"]
RL_SOURCE_TYPES = ["blind_agent", "labeled_agent", "noisy_expert"]
SIMPLE_SOURCE_TYPES = ["random", "heuristic"]


class TestManeuverConfig:
    """Maneuver config parsing from YAML dicts."""

    def test_constant_thrust(self):
        d = {"type": "constant_thrust", "main": 0.75, "side": 0.0}
        cfg = ManeuverConfig.from_dict(d)
        assert cfg.type == "constant_thrust"
        assert cfg.main == 0.75
        assert cfg.side == 0.0

    def test_free_fall_defaults(self):
        d = {"type": "free_fall"}
        cfg = ManeuverConfig.from_dict(d)
        assert cfg.main == 0.0
        assert cfg.side == 0.0

    def test_impulse_with_timing(self):
        d = {
            "type": "impulse",
            "channel": "main",
            "thrust_level": 1.0,
            "pulse_duration": [5, 50],
            "gap_duration": [5, 50],
            "n_cycles": [2, 5],
        }
        cfg = ManeuverConfig.from_dict(d)
        assert cfg.type == "impulse"
        assert cfg.channel == "main"
        assert cfg.pulse_duration == (5, 50)

    def test_direction_reversal(self):
        d = {
            "type": "direction_reversal",
            "channel": "side",
            "thrust_level": 1.0,
            "first_duration": [10, 50],
            "gap_duration": [0, 20],
            "second_duration": [10, 50],
        }
        cfg = ManeuverConfig.from_dict(d)
        assert cfg.type == "direction_reversal"
        assert cfg.gap_duration == (0, 20)

    def test_range_valued_main(self):
        """ground_thrust_sweep uses main: [lo, hi] for per-episode sampling."""
        d = {"type": "ground_thrust_sweep", "main": [0.0, 1.0]}
        cfg = ManeuverConfig.from_dict(d)
        assert cfg.main == (0.0, 1.0)

    def test_range_valued_side(self):
        d = {"type": "ground_side_thrust", "side": [-1.0, 1.0]}
        cfg = ManeuverConfig.from_dict(d)
        assert cfg.side == (-1.0, 1.0)

    def test_controlled_descent_range(self):
        d = {"type": "controlled_descent", "main": [0.2, 0.4]}
        cfg = ManeuverConfig.from_dict(d)
        assert cfg.main == (0.2, 0.4)

    def test_hover(self):
        d = {"type": "hover"}
        cfg = ManeuverConfig.from_dict(d)
        assert cfg.type == "hover"

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError, match="Unknown maneuver type"):
            ManeuverConfig.from_dict({"type": "nosuch"})


class TestStartConfig:
    """Start config parsing from YAML dicts."""

    def test_fresh_reset(self):
        d = {
            "mode": "fresh_reset",
            "initial_state": {
                "x": [-0.6, 0.6],
                "y": [0.2, 1.2],
                "vx": [-1.5, 1.5],
                "vy": [-1.5, 0.5],
                "angle": [-1.0, 1.0],
                "angular_vel": [-1.0, 1.0],
            },
        }
        cfg = StartConfig.from_dict(d)
        assert cfg.mode == "fresh_reset"
        assert cfg.initial_state["x"] == (-0.6, 0.6)

    def test_replay(self):
        d = {
            "mode": "replay",
            "source_dir": "/path/to/episodes",
            "min_step": 20,
            "max_step_fraction": 0.8,
        }
        cfg = StartConfig.from_dict(d)
        assert cfg.mode == "replay"
        assert cfg.source_dir == "/path/to/episodes"

    def test_replay_to_landing(self):
        d = {
            "mode": "replay_to_landing",
            "source_dir": "/path/to/episodes",
        }
        cfg = StartConfig.from_dict(d)
        assert cfg.mode == "replay_to_landing"

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown start mode"):
            StartConfig.from_dict({"mode": "nosuch"})


class TestPrimitiveCollectionConfig:
    """Primitive source type in CollectionConfig."""

    def test_parse_primitive_config(self):
        d = {
            "source_type": "primitive",
            "n_episodes": 200,
            "max_steps": 1000,
            "physics_sampling": {
                "profile": "gym-default",
            },
            "maneuver": {
                "type": "free_fall",
            },
            "start": {
                "mode": "fresh_reset",
                "initial_state": {
                    "x": [-0.6, 0.6],
                    "y": [0.2, 1.2],
                    "vx": [-1.5, 1.5],
                    "vy": [-1.5, 0.5],
                    "angle": [-1.0, 1.0],
                    "angular_vel": [-1.0, 1.0],
                },
            },
        }
        cfg = CollectionConfig.from_dict(d)
        assert cfg.source_type == "primitive"
        assert cfg.maneuver_config is not None
        assert cfg.maneuver_config.type == "free_fall"
        assert cfg.start_config is not None
        assert cfg.start_config.mode == "fresh_reset"
        assert cfg.max_steps == 1000

    def test_primitive_with_post_landing(self):
        d = {
            "source_type": "primitive",
            "n_episodes": 200,
            "max_steps": 1000,
            "allow_post_landing": True,
            "physics_sampling": {"profile": "gym-default"},
            "maneuver": {"type": "ground_stationary"},
            "start": {
                "mode": "replay_to_landing",
                "source_dir": "/path/to/episodes",
            },
        }
        cfg = CollectionConfig.from_dict(d)
        assert cfg.allow_post_landing is True
        assert cfg.start_config.mode == "replay_to_landing"

    def test_primitive_missing_maneuver_raises(self):
        d = {
            "source_type": "primitive",
            "n_episodes": 200,
            "physics_sampling": {"profile": "gym-default"},
            "start": {"mode": "fresh_reset", "initial_state": {"x": [0, 1]}},
        }
        with pytest.raises(ValueError, match="maneuver"):
            CollectionConfig.from_dict(d)

    def test_primitive_missing_start_raises(self):
        d = {
            "source_type": "primitive",
            "n_episodes": 200,
            "physics_sampling": {"profile": "gym-default"},
            "maneuver": {"type": "free_fall"},
        }
        with pytest.raises(ValueError, match="start"):
            CollectionConfig.from_dict(d)


class TestBatchConfig:
    """Batch config that lists multiple collection configs."""

    def test_parse_batch(self, tmp_path):
        # Create two collection configs
        for name in ["a.yaml", "b.yaml"]:
            cfg = {
                "source_type": "random",
                "n_episodes": 10,
                "physics_sampling": {
                    "method": "uniform",
                    "ranges": {
                        "gravity": [-12.0, -2.0],
                        "main_engine_power": [5.0, 25.0],
                        "side_engine_power": [0.2, 1.5],
                        "lander_density": [2.5, 10.0],
                        "angular_damping": [0.0, 5.0],
                        "wind_power": [0.0, 30.0],
                        "turbulence_power": [0.0, 5.0],
                    },
                },
                "seed": 0,
            }
            (tmp_path / name).write_text(yaml.dump(cfg))

        batch = {
            "output_base": "/media/hdd1/world_model_data",
            "collections": [
                {"config": str(tmp_path / "a.yaml"), "output_name": "run-a"},
                {"config": str(tmp_path / "b.yaml"), "output_name": "run-b"},
            ],
        }
        batch_path = tmp_path / "batch.yaml"
        batch_path.write_text(yaml.dump(batch))

        bc = BatchConfig.load(str(batch_path))
        assert bc.output_base == "/media/hdd1/world_model_data"
        assert len(bc.entries) == 2
        assert bc.entries[0].output_name == "run-a"
        assert bc.entries[0].config.source_type == "random"
