"""Tests for extended episode format — primitive metadata round-trip.

Verifies that metadata with primitive collection fields (maneuver_type,
start_mode, termination_event, etc.) round-trips correctly through
save_episode/load_episode. Since metadata is JSON-serialized, this
should work out of the box — this is a contract test.
"""

import numpy as np
import pytest

from parametric_lunar_lander.episode_io import save_episode, load_episode


class TestExtendedEpisodeFormat:
    """Extended episode format with primitive metadata."""

    def test_save_load_with_primitive_metadata(self, tmp_path):
        """Primitive episodes have maneuver_type, start_mode, etc. in metadata."""
        T = 50
        metadata = {
            "physics_config": {"gravity": -10.0, "main_engine_power": 13.0,
                               "side_engine_power": 0.6, "lander_density": 5.0,
                               "angular_damping": 2.5, "wind_power": 0.0,
                               "turbulence_power": 0.0},
            "outcome": "crashed",
            "seed": 42,
            "n_steps": T,
            "total_reward": -100.0,
            "source_type": "primitive",
            "maneuver_type": "free_fall",
            "maneuver_params": {"main": 0.0, "side": 0.0},
            "start_mode": "fresh_reset",
            "start_state": {"x": 0.1, "y": 0.8, "vx": 0.0, "vy": 0.0,
                           "angle": 0.0, "angular_vel": 0.0},
        }
        path = tmp_path / "ep.npz"
        save_episode(
            path=path,
            states=np.zeros((T + 1, 15), dtype=np.float32),
            actions=np.zeros((T, 2), dtype=np.float32),
            rewards=np.zeros(T, dtype=np.float32),
            dones=np.zeros(T, dtype=bool),
            metadata=metadata,
        )
        ep = load_episode(path)
        assert ep["metadata"]["source_type"] == "primitive"
        assert ep["metadata"]["maneuver_type"] == "free_fall"
        assert ep["metadata"]["start_mode"] == "fresh_reset"

    def test_save_load_with_termination_event(self, tmp_path):
        """Post-landing episodes record termination_event in metadata."""
        T = 100
        metadata = {
            "physics_config": {"gravity": -10.0, "main_engine_power": 13.0,
                               "side_engine_power": 0.6, "lander_density": 5.0,
                               "angular_damping": 2.5, "wind_power": 0.0,
                               "turbulence_power": 0.0},
            "outcome": "timeout",
            "seed": 42,
            "n_steps": T,
            "total_reward": -50.0,
            "source_type": "primitive",
            "maneuver_type": "ground_stationary",
            "allow_post_landing": True,
            "termination_event": {"type": "landed", "step": 60},
        }
        path = tmp_path / "ep.npz"
        save_episode(
            path=path,
            states=np.zeros((T + 1, 15), dtype=np.float32),
            actions=np.zeros((T, 2), dtype=np.float32),
            rewards=np.zeros(T, dtype=np.float32),
            dones=np.zeros(T, dtype=bool),
            metadata=metadata,
        )
        ep = load_episode(path)
        assert ep["metadata"]["termination_event"]["type"] == "landed"
        assert ep["metadata"]["termination_event"]["step"] == 60
