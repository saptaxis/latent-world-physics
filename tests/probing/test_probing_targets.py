"""Tests for behavioral target computation."""
import numpy as np
import pytest

from parametric_lunar_lander.physics_config import LunarLanderPhysicsConfig
from lwp.probing.targets import (
    compute_behavioral_targets,
    BEHAVIORAL_TARGET_NAMES,
    PARAMETRIC_TARGET_NAMES,
    ALL_TARGET_NAMES,
)


class TestBehavioralTargetNames:
    def test_parametric_names_match_physics_config(self):
        assert PARAMETRIC_TARGET_NAMES == LunarLanderPhysicsConfig.PARAM_NAMES

    def test_behavioral_names_are_five(self):
        assert len(BEHAVIORAL_TARGET_NAMES) == 5

    def test_all_targets_is_parametric_plus_behavioral(self):
        assert ALL_TARGET_NAMES == PARAMETRIC_TARGET_NAMES + BEHAVIORAL_TARGET_NAMES


class TestComputeBehavioralTargets:
    def test_returns_array_of_five(self):
        config = LunarLanderPhysicsConfig()  # defaults
        result = compute_behavioral_targets(config)
        assert isinstance(result, np.ndarray)
        assert result.shape == (5,)
        assert result.dtype == np.float32

    def test_twr_matches_physics_config(self):
        config = LunarLanderPhysicsConfig(
            gravity=-10.0, main_engine_power=13.0, lander_density=5.0,
        )
        result = compute_behavioral_targets(config)
        expected_twr = config.twr()
        np.testing.assert_allclose(result[0], expected_twr, rtol=1e-5)

    def test_descent_rate_increases_with_gravity(self):
        low_g = LunarLanderPhysicsConfig(gravity=-3.0)
        high_g = LunarLanderPhysicsConfig(gravity=-12.0)
        low_result = compute_behavioral_targets(low_g)
        high_result = compute_behavioral_targets(high_g)
        # Higher gravity -> faster descent
        assert high_result[1] > low_result[1]

    def test_angular_responsiveness_increases_with_side_power(self):
        low = LunarLanderPhysicsConfig(side_engine_power=0.2)
        high = LunarLanderPhysicsConfig(side_engine_power=1.5)
        low_result = compute_behavioral_targets(low)
        high_result = compute_behavioral_targets(high)
        assert high_result[2] > low_result[2]

    def test_hover_cost_is_inverse_twr(self):
        config = LunarLanderPhysicsConfig()
        result = compute_behavioral_targets(config)
        expected = 1.0 / config.twr()
        np.testing.assert_allclose(result[3], expected, rtol=1e-5)

    def test_effective_weight_increases_with_density(self):
        low = LunarLanderPhysicsConfig(lander_density=2.5)
        high = LunarLanderPhysicsConfig(lander_density=10.0)
        low_result = compute_behavioral_targets(low)
        high_result = compute_behavioral_targets(high)
        assert high_result[4] > low_result[4]

    def test_all_values_finite(self):
        """Verify no NaN/Inf for extreme but valid configs."""
        config = LunarLanderPhysicsConfig(
            gravity=-2.0, main_engine_power=25.0, side_engine_power=0.2,
            lander_density=2.5, angular_damping=5.0, wind_power=30.0,
            turbulence_power=5.0,
        )
        result = compute_behavioral_targets(config)
        assert np.all(np.isfinite(result))
