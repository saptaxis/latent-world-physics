"""Tests for primitive maneuver action generators and replay-to-branch logic."""

import numpy as np
import pytest

from lwp.collection.wm_primitives import generate_actions, replay_to_branch_point
from lwp.collection.wm_collection_config import ManeuverConfig
from parametric_lunar_lander.episode_io import load_episode


class TestConstantActions:
    """Constant thrust action generation."""

    def test_free_fall_all_zeros(self):
        cfg = ManeuverConfig(type="free_fall")
        actions, params = generate_actions(cfg, n_steps=100, rng=np.random.default_rng(0))
        assert actions.shape == (100, 2)
        np.testing.assert_array_equal(actions, 0.0)
        assert params == {"main": 0.0, "side": 0.0}

    def test_constant_main_thrust(self):
        cfg = ManeuverConfig(type="constant_thrust", main=0.75, side=0.0)
        actions, params = generate_actions(cfg, n_steps=50, rng=np.random.default_rng(0))
        assert actions.shape == (50, 2)
        np.testing.assert_allclose(actions[:, 0], 0.75)
        np.testing.assert_allclose(actions[:, 1], 0.0)
        assert params == {"main": 0.75, "side": 0.0}

    def test_constant_side_thrust(self):
        cfg = ManeuverConfig(type="constant_thrust", main=0.0, side=-0.5)
        actions, _ = generate_actions(cfg, n_steps=50, rng=np.random.default_rng(0))
        np.testing.assert_allclose(actions[:, 0], 0.0)
        np.testing.assert_allclose(actions[:, 1], -0.5)

    def test_combined_thrust(self):
        cfg = ManeuverConfig(type="constant_thrust", main=0.5, side=0.5)
        actions, _ = generate_actions(cfg, n_steps=30, rng=np.random.default_rng(0))
        np.testing.assert_allclose(actions[:, 0], 0.5)
        np.testing.assert_allclose(actions[:, 1], 0.5)

    def test_ground_stationary_zero_actions(self):
        cfg = ManeuverConfig(type="ground_stationary")
        actions, _ = generate_actions(cfg, n_steps=100, rng=np.random.default_rng(0))
        np.testing.assert_array_equal(actions, 0.0)

    def test_ground_thrust_sweep_scalar(self):
        cfg = ManeuverConfig(type="ground_thrust_sweep", main=0.4, side=0.0)
        actions, params = generate_actions(cfg, n_steps=50, rng=np.random.default_rng(0))
        np.testing.assert_allclose(actions[:, 0], 0.4)
        assert params["main"] == 0.4

    def test_ground_thrust_sweep_range(self):
        cfg = ManeuverConfig(type="ground_thrust_sweep", main=(0.0, 1.0), side=0.0)
        a1, p1 = generate_actions(cfg, n_steps=50, rng=np.random.default_rng(1))
        a2, p2 = generate_actions(cfg, n_steps=50, rng=np.random.default_rng(2))
        assert np.all(a1[:, 0] == a1[0, 0]), "Should be constant within episode"
        assert a1[0, 0] != a2[0, 0], "Different seeds should sample different values"
        assert 0.0 <= a1[0, 0] <= 1.0
        assert p1["main"] == a1[0, 0]
        assert p2["main"] == a2[0, 0]

    def test_ground_liftoff(self):
        cfg = ManeuverConfig(type="ground_liftoff", main=0.9, side=0.0)
        actions, _ = generate_actions(cfg, n_steps=50, rng=np.random.default_rng(0))
        np.testing.assert_allclose(actions[:, 0], 0.9)


class TestImpulseActions:
    """Impulse on/off pattern generation."""

    def test_main_impulse_shape(self):
        cfg = ManeuverConfig(
            type="impulse", channel="main", thrust_level=1.0,
            pulse_duration=(10, 10), gap_duration=(10, 10), n_cycles=(2, 2),
        )
        actions, _ = generate_actions(cfg, n_steps=100, rng=np.random.default_rng(0))
        assert actions.shape == (100, 2)

    def test_main_impulse_pattern_correct(self):
        cfg = ManeuverConfig(
            type="impulse", channel="main", thrust_level=0.8,
            pulse_duration=(10, 10), gap_duration=(10, 10), n_cycles=(2, 2),
        )
        actions, params = generate_actions(cfg, n_steps=50, rng=np.random.default_rng(0))
        np.testing.assert_allclose(actions[:10, 0], 0.8)
        np.testing.assert_allclose(actions[10:20, 0], 0.0)
        np.testing.assert_allclose(actions[20:30, 0], 0.8)
        np.testing.assert_allclose(actions[30:40, 0], 0.0)
        np.testing.assert_allclose(actions[:, 1], 0.0)
        assert params == {
            "channel": "main", "thrust_level": 0.8,
            "pulse_duration": 10, "gap_duration": 10, "n_cycles": 2,
        }

    def test_side_impulse_uses_side_channel(self):
        cfg = ManeuverConfig(
            type="impulse", channel="side", thrust_level=1.0,
            pulse_duration=(5, 5), gap_duration=(5, 5), n_cycles=(1, 1),
        )
        actions, _ = generate_actions(cfg, n_steps=20, rng=np.random.default_rng(0))
        np.testing.assert_allclose(actions[:5, 1], 1.0)
        np.testing.assert_allclose(actions[5:10, 1], 0.0)
        np.testing.assert_allclose(actions[:, 0], 0.0)

    def test_random_timing_varies(self):
        cfg = ManeuverConfig(
            type="impulse", channel="main", thrust_level=1.0,
            pulse_duration=(5, 50), gap_duration=(5, 50), n_cycles=(2, 5),
        )
        a1, p1 = generate_actions(cfg, n_steps=200, rng=np.random.default_rng(1))
        a2, p2 = generate_actions(cfg, n_steps=200, rng=np.random.default_rng(2))
        assert not np.array_equal(a1, a2)
        assert p1 != p2


class TestDirectionReversalActions:
    """Direction reversal pattern generation."""

    def test_side_reversal_shape(self):
        cfg = ManeuverConfig(
            type="direction_reversal", channel="side", thrust_level=1.0,
            first_duration=(20, 20), gap_duration=(0, 0), second_duration=(20, 20),
        )
        actions, _ = generate_actions(cfg, n_steps=100, rng=np.random.default_rng(0))
        assert actions.shape == (100, 2)

    def test_side_reversal_pattern(self):
        cfg = ManeuverConfig(
            type="direction_reversal", channel="side", thrust_level=1.0,
            first_duration=(20, 20), gap_duration=(0, 0), second_duration=(20, 20),
        )
        actions, params = generate_actions(cfg, n_steps=50, rng=np.random.default_rng(0))
        np.testing.assert_allclose(actions[:20, 1], 1.0)
        np.testing.assert_allclose(actions[20:40, 1], -1.0)
        np.testing.assert_allclose(actions[40:, 1], 0.0)
        np.testing.assert_allclose(actions[:, 0], 0.0)
        assert params == {
            "channel": "side", "thrust_level": 1.0,
            "first_duration": 20, "gap_duration": 0, "second_duration": 20,
        }

    def test_with_gap(self):
        cfg = ManeuverConfig(
            type="direction_reversal", channel="side", thrust_level=0.5,
            first_duration=(10, 10), gap_duration=(5, 5), second_duration=(10, 10),
        )
        actions, _ = generate_actions(cfg, n_steps=30, rng=np.random.default_rng(0))
        np.testing.assert_allclose(actions[:10, 1], 0.5)
        np.testing.assert_allclose(actions[10:15, 1], 0.0)
        np.testing.assert_allclose(actions[15:25, 1], -0.5)


class TestHoverActions:
    """Hover equilibrium action generation."""

    def test_hover_requires_physics_config(self):
        cfg = ManeuverConfig(type="hover")
        with pytest.raises(ValueError, match="physics_config"):
            generate_actions(cfg, n_steps=50, rng=np.random.default_rng(0))

    def test_hover_produces_constant_thrust(self):
        from parametric_lunar_lander.physics_config import LunarLanderPhysicsConfig
        physics = LunarLanderPhysicsConfig()  # gym defaults
        cfg = ManeuverConfig(type="hover")
        actions, params = generate_actions(
            cfg, n_steps=50, rng=np.random.default_rng(0),
            physics_config=physics,
        )
        assert np.all(actions[:, 0] == actions[0, 0])
        np.testing.assert_allclose(actions[:, 1], 0.0)
        assert actions[0, 0] > 0.0
        assert params["main"] == actions[0, 0]
        assert params["side"] == 0.0
        assert "hover_m_power" in params


class TestReplayToBranchPoint:
    """Replaying source episodes to a branch point."""

    def _collect_source_episode(self, tmp_path):
        """Collect a short episode to use as source data."""
        from parametric_lunar_lander.episode_io import save_episode, run_episode
        from parametric_lunar_lander.env import ParameterizedLunarLander

        env = ParameterizedLunarLander()
        ep = run_episode(env, lambda obs: np.array([0.0, 0.0], dtype=np.float32),
                         seed=42, max_steps=100)
        env.close()
        path = tmp_path / "source_ep.npz"
        save_episode(path, ep["states"], ep["actions"], ep["rewards"],
                     ep["dones"], ep["metadata"])
        return str(path)

    def test_returns_env_at_branch_state(self, tmp_path):
        """After replay, env state matches source episode at branch point."""
        ep_path = self._collect_source_episode(tmp_path)
        source_ep = load_episode(ep_path)
        branch_step = 30

        env = replay_to_branch_point(source_ep, branch_step)

        # The env's current state should match the source episode at step 30.
        # states[30] = the observation after 30 steps of action replay.
        # _last_obs is set by the env's step() method on each call.
        env_obs = env.unwrapped._last_obs
        source_state = source_ep["states"][branch_step]
        np.testing.assert_allclose(env_obs, source_state, atol=1e-4)
        env.close()

    def test_branch_point_at_episode_start_raises(self, tmp_path):
        """branch_point=0 is invalid — use fresh start instead of replay."""
        ep_path = self._collect_source_episode(tmp_path)
        source_ep = load_episode(ep_path)

        with pytest.raises(ValueError, match="branch_point must be >= 1"):
            replay_to_branch_point(source_ep, branch_point=0)
