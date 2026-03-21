"""Tests for collection-specific env wrappers."""

import numpy as np
import pytest

from parametric_lunar_lander.env import ParameterizedLunarLander
from lwp.collection.env_wrappers_collection import (
    PostLandingWrapper,
    override_initial_state,
)


class TestPostLandingWrapper:
    """PostLandingWrapper suppresses landing/crash termination."""

    def _make_env(self):
        env = ParameterizedLunarLander()
        return PostLandingWrapper(env)

    def test_wraps_env(self):
        env = self._make_env()
        obs, info = env.reset(seed=42)
        assert obs.shape == (15,)
        env.close()

    def test_does_not_terminate_on_crash(self):
        """Run with zero thrust — lander crashes but env continues."""
        env = self._make_env()
        env.reset(seed=42)

        terminated_ever = False
        for step in range(500):
            obs, reward, terminated, truncated, info = env.step(
                np.array([0.0, 0.0], dtype=np.float32)
            )
            if terminated:
                terminated_ever = True
                break

        assert not terminated_ever
        assert env.termination_event is not None
        assert env.termination_event["type"] in ("crashed", "landed", "out_of_bounds")
        assert isinstance(env.termination_event["step"], int)
        env.close()

    def test_oob_still_terminates(self):
        """Out-of-bounds should still terminate.

        Full main engine + full side thrust keeps the lander airborne
        while pushing it sideways past the viewport edge.
        """
        env = self._make_env()
        env.reset(seed=42)

        terminated = False
        for step in range(1000):
            # Main engine (1.0) keeps lander airborne; side thrust (1.0)
            # pushes it right until it exits the viewport.
            obs, reward, term, truncated, info = env.step(
                np.array([1.0, 1.0], dtype=np.float32)
            )
            if term:
                terminated = True
                break

        assert terminated, "OOB should still terminate"
        assert info["outcome"] == "out_of_bounds"
        env.close()

    def test_termination_event_none_before_event(self):
        env = self._make_env()
        env.reset(seed=42)
        assert env.termination_event is None
        env.step(np.array([0.0, 0.0], dtype=np.float32))
        env.close()


class TestOverrideInitialState:
    """override_initial_state() sets Box2D body state after reset."""

    def test_sets_position(self):
        """Position override produces correct Box2D body coordinates."""
        env = ParameterizedLunarLander()
        env.reset(seed=42)

        override_initial_state(env, x=0.3, y=0.8)

        # Step once to let Box2D propagate the overridden state.
        obs, _, _, _, _ = env.step(np.array([0.0, 0.0], dtype=np.float32))

        # Check the Box2D body directly — position should be close to the
        # requested values (one physics step causes small drift from gravity).
        lander = env.unwrapped.lander
        W = 600 / 30.0
        H = 400 / 30.0
        expected_x = 0.3 * (W / 2) + W / 2
        expected_y = 0.8 * (H / 2) + env.unwrapped.helipad_y + 18 / 30.0

        # Allow tolerance for one physics step of drift.
        assert abs(lander.position[0] - expected_x) < 0.5
        assert abs(lander.position[1] - expected_y) < 0.5
        env.close()

    def test_sets_velocity(self):
        """Velocity override sets Box2D linearVelocity correctly."""
        env = ParameterizedLunarLander()
        env.reset(seed=42)
        override_initial_state(env, vx=0.5, vy=-0.3)

        # Check Box2D body velocity directly (no _get_obs method available).
        lander = env.unwrapped.lander
        W = 600 / 30.0
        H = 400 / 30.0
        FPS = 50
        expected_vx = 0.5 * FPS / (W / 2)
        expected_vy = -0.3 * FPS / (H / 2)

        assert abs(lander.linearVelocity[0] - expected_vx) < 0.01
        assert abs(lander.linearVelocity[1] - expected_vy) < 0.01
        env.close()

    def test_sets_angle(self):
        """Angle override sets Box2D body angle directly."""
        env = ParameterizedLunarLander()
        env.reset(seed=42)
        override_initial_state(env, angle=0.5)

        lander = env.unwrapped.lander
        assert abs(lander.angle - 0.5) < 0.01
        env.close()

    def test_underground_check(self):
        """Requesting y below terrain height should raise ValueError."""
        env = ParameterizedLunarLander()
        env.reset(seed=42)

        with pytest.raises(ValueError, match="below terrain"):
            override_initial_state(env, x=0.0, y=-0.5)
        env.close()

    def test_partial_override(self):
        """Can override just some dimensions, leave others at reset default."""
        env = ParameterizedLunarLander()
        env.reset(seed=42)

        # Record pre-override position.
        pre_x = env.unwrapped.lander.position[0]

        override_initial_state(env, angle=0.3)

        # Angle should match; x position should be unchanged.
        assert abs(env.unwrapped.lander.angle - 0.3) < 0.01
        assert abs(env.unwrapped.lander.position[0] - pre_x) < 0.01
        env.close()

    def test_sets_angular_velocity(self):
        """Angular velocity override sets Box2D angularVelocity correctly."""
        env = ParameterizedLunarLander()
        env.reset(seed=42)
        override_initial_state(env, angular_vel=2.0)

        lander = env.unwrapped.lander
        # Invert: obs_angvel = world_angvel * 20 / FPS
        # So: world_angvel = obs_angvel * FPS / 20
        expected = 2.0 * 50 / 20.0  # = 5.0 rad/s
        assert abs(lander.angularVelocity - expected) < 0.01
        env.close()

    def test_wakes_sleeping_body(self):
        """Override should wake a sleeping body (awake=True)."""
        env = ParameterizedLunarLander()
        env.reset(seed=42)

        # Force the body to sleep, then override.
        env.unwrapped.lander.awake = False
        override_initial_state(env, angle=0.1)

        assert env.unwrapped.lander.awake is True
        env.close()
