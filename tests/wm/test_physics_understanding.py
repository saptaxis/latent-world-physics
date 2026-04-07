"""Tests for physics understanding evaluation."""
import pytest
import numpy as np
import torch
from lwp.wm.physics_understanding import X, Y, VX, VY, ANGLE, ANGULAR_VEL


class TestGeneralFilter:
    """Test general exclusion filter applied to all measurements."""

    def test_excludes_near_ground(self):
        """y <= 0.3 should be excluded."""
        from lwp.wm.physics_understanding import passes_general_filter
        state = np.array([0.0, 0.2, 0.0, 0.0, 0.0, 0.0])
        assert not passes_general_filter(state, timestep=10)

    def test_excludes_near_oob(self):
        """|x| >= 0.8 should be excluded."""
        from lwp.wm.physics_understanding import passes_general_filter
        state = np.array([0.85, 0.5, 0.0, 0.0, 0.0, 0.0])
        assert not passes_general_filter(state, timestep=10)

    def test_excludes_early_timesteps(self):
        """First 3 steps of episode should be excluded."""
        from lwp.wm.physics_understanding import passes_general_filter
        state = np.array([0.0, 0.5, 0.0, 0.0, 0.0, 0.0])
        assert not passes_general_filter(state, timestep=2)
        assert passes_general_filter(state, timestep=3)

    def test_passes_clean_state(self):
        """State in clean region should pass."""
        from lwp.wm.physics_understanding import passes_general_filter
        state = np.array([0.0, 0.5, 0.0, -0.1, 0.0, 0.0])
        assert passes_general_filter(state, timestep=10)


class TestGravityFilter:
    """Test gravity-specific filter: no engines, upright, not spinning."""

    def test_rejects_main_thrust(self):
        from lwp.wm.physics_understanding import passes_gravity_filter
        state = np.array([0.0, 0.5, 0.0, 0.0, 0.0, 0.0])
        action = np.array([0.5, 0.0])
        assert not passes_gravity_filter(state, action)

    def test_rejects_side_thrust(self):
        from lwp.wm.physics_understanding import passes_gravity_filter
        state = np.array([0.0, 0.5, 0.0, 0.0, 0.0, 0.0])
        action = np.array([0.0, 0.6])
        assert not passes_gravity_filter(state, action)

    def test_rejects_tilted(self):
        from lwp.wm.physics_understanding import passes_gravity_filter
        state = np.array([0.0, 0.5, 0.0, 0.0, 0.2, 0.0])
        action = np.array([0.0, 0.0])
        assert not passes_gravity_filter(state, action)

    def test_rejects_spinning(self):
        from lwp.wm.physics_understanding import passes_gravity_filter
        state = np.array([0.0, 0.5, 0.0, 0.0, 0.0, 0.5])
        action = np.array([0.0, 0.0])
        assert not passes_gravity_filter(state, action)

    def test_passes_clean_transition(self):
        from lwp.wm.physics_understanding import passes_gravity_filter
        state = np.array([0.0, 0.5, 0.0, -0.1, 0.05, 0.1])
        action = np.array([0.0, 0.0])
        assert passes_gravity_filter(state, action)


class TestThrustFilter:
    """Test main thrust filter: strong main, no side, upright."""

    def test_rejects_weak_thrust(self):
        from lwp.wm.physics_understanding import passes_main_thrust_filter
        state = np.array([0.0, 0.5, 0.0, 0.0, 0.0, 0.0])
        action = np.array([0.3, 0.0])
        assert not passes_main_thrust_filter(state, action)

    def test_rejects_side_engine_active(self):
        from lwp.wm.physics_understanding import passes_main_thrust_filter
        state = np.array([0.0, 0.5, 0.0, 0.0, 0.0, 0.0])
        action = np.array([0.8, 0.6])
        assert not passes_main_thrust_filter(state, action)

    def test_passes_clean_thrust(self):
        from lwp.wm.physics_understanding import passes_main_thrust_filter
        state = np.array([0.0, 0.5, 0.0, 0.0, 0.05, 0.1])
        action = np.array([0.8, 0.0])
        assert passes_main_thrust_filter(state, action)


class TestSideThrustFilter:
    """Test side thrust filter: strong side, no main, upright."""

    def test_rejects_main_engine(self):
        from lwp.wm.physics_understanding import passes_side_thrust_filter
        state = np.array([0.0, 0.5, 0.0, 0.0, 0.0, 0.0])
        action = np.array([0.1, 0.8])
        assert not passes_side_thrust_filter(state, action)

    def test_passes_clean_side_thrust(self):
        from lwp.wm.physics_understanding import passes_side_thrust_filter
        state = np.array([0.0, 0.5, 0.0, 0.0, 0.0, 0.0])
        action = np.array([0.0, 0.8])
        assert passes_side_thrust_filter(state, action)


class TestKinematicFilter:
    """Test kinematic consistency filter: velocities not near zero."""

    def test_rejects_near_zero_vx(self):
        from lwp.wm.physics_understanding import passes_kinematic_filter
        state = np.array([0.0, 0.5, 0.05, -0.2, 0.0, 0.0])
        assert not passes_kinematic_filter(state)

    def test_passes_with_velocity(self):
        from lwp.wm.physics_understanding import passes_kinematic_filter
        state = np.array([0.0, 0.5, 0.2, -0.2, 0.0, 0.0])
        assert passes_kinematic_filter(state)


class TestAngleThrustFilter:
    """Test angle-thrust coupling filter: tilted, thrusting, no side."""

    def test_rejects_near_upright(self):
        from lwp.wm.physics_understanding import passes_angle_thrust_filter
        state = np.array([0.0, 0.5, 0.0, 0.0, 0.1, 0.0])
        action = np.array([0.8, 0.0])
        assert not passes_angle_thrust_filter(state, action, gravity_model=-0.13)

    def test_passes_tilted_thrusting(self):
        from lwp.wm.physics_understanding import passes_angle_thrust_filter
        state = np.array([0.0, 0.5, 0.0, 0.0, 0.3, 0.0])
        action = np.array([0.8, 0.0])
        assert passes_angle_thrust_filter(state, action, gravity_model=-0.13)


class TestAngularDampingFilter:
    """Test angular damping filter: no engines, enough rotation."""

    def test_rejects_slow_rotation(self):
        from lwp.wm.physics_understanding import passes_angular_damping_filter
        state = np.array([0.0, 0.5, 0.0, 0.0, 0.0, 0.2])
        action = np.array([0.0, 0.0])
        assert not passes_angular_damping_filter(state, action)

    def test_passes_rotating(self):
        from lwp.wm.physics_understanding import passes_angular_damping_filter
        state = np.array([0.0, 0.5, 0.0, 0.0, 0.0, 0.5])
        action = np.array([0.0, 0.0])
        assert passes_angular_damping_filter(state, action)


class FakeLinearModel:
    """Minimal model stub for testing: predicts a fixed delta.

    Mimics wm-ladder model.step() interface:
        model.step(obs_norm, action, model_state) -> (delta_norm, new_state)

    This fake model predicts a constant gravity-like delta on vy (dim 3)
    and identity on other dims, in normalized space.
    """

    def __init__(self, gravity_delta_norm: float = -0.5):
        self.gravity_delta_norm = gravity_delta_norm

    def eval(self):
        pass

    def step(self, obs, action, model_state=None):
        batch = obs.shape[0]
        delta = torch.zeros(batch, 6)
        # Predict a constant gravity-like effect on vy.
        delta[:, 3] = self.gravity_delta_norm
        return delta, None


class FakeNormStats:
    """Minimal NormStats stub for testing.

    Identity normalization: mean=0, std=1 for both state and delta.
    This way normalized == raw, simplifying test assertions.
    """

    def __init__(self, state_dim: int = 6):
        self.state_mean = torch.zeros(state_dim)
        self.state_std = torch.ones(state_dim)
        self.delta_mean = torch.zeros(state_dim)
        self.delta_std = torch.ones(state_dim)


def _make_clean_episode(n_steps: int = 100, gravity: float = -0.13) -> dict:
    """Create a synthetic episode with known gravity for testing.

    All states are in the clean measurement region:
    - y=0.5 (above ground), x=0.0 (centered)
    - angle=0, angular_vel=0
    - Actions: all zero (no thrust)
    - vy decreases by `gravity` each step

    Returns episode dict with 'states' (n_steps+1, 6) and 'actions' (n_steps, 2).
    """
    states = np.zeros((n_steps + 1, 6), dtype=np.float32)
    actions = np.zeros((n_steps, 2), dtype=np.float32)
    for t in range(n_steps + 1):
        states[t, Y] = 0.5
        states[t, VY] = gravity * t  # vy accumulates gravity
    return {"states": states, "actions": actions}


class TestOracleGravityExtraction:
    """Test 1-step oracle extraction of gravity constant."""

    def test_extracts_gravity_from_clean_episode(self):
        """Model with constant gravity delta should extract that value."""
        from lwp.wm.physics_understanding import extract_gravity_oracle

        model = FakeLinearModel(gravity_delta_norm=-0.13)
        norm_stats = FakeNormStats()
        episode = _make_clean_episode(n_steps=50, gravity=-0.13)

        result = extract_gravity_oracle(model, norm_stats, [episode])
        assert result["n_samples"] > 10
        np.testing.assert_allclose(result["model_mean"], -0.13, atol=0.01)

    def test_returns_gt_gravity(self):
        """GT gravity should come from episode deltas, not model."""
        from lwp.wm.physics_understanding import extract_gravity_oracle

        model = FakeLinearModel(gravity_delta_norm=-0.10)  # model is wrong
        norm_stats = FakeNormStats()
        episode = _make_clean_episode(n_steps=50, gravity=-0.13)

        result = extract_gravity_oracle(model, norm_stats, [episode])
        np.testing.assert_allclose(result["gt_mean"], -0.13, atol=0.01)
        assert abs(result["model_mean"] - result["gt_mean"]) > 0.01

    def test_filters_near_ground(self):
        """Transitions near ground should be excluded."""
        from lwp.wm.physics_understanding import extract_gravity_oracle

        model = FakeLinearModel(gravity_delta_norm=-0.13)
        norm_stats = FakeNormStats()
        episode = _make_clean_episode(n_steps=50, gravity=-0.13)
        episode["states"][:, 1] = 0.1  # Y index

        result = extract_gravity_oracle(model, norm_stats, [episode])
        assert result["n_samples"] == 0

    def test_multiple_episodes(self):
        """Should aggregate across multiple episodes."""
        from lwp.wm.physics_understanding import extract_gravity_oracle

        model = FakeLinearModel(gravity_delta_norm=-0.13)
        norm_stats = FakeNormStats()
        episodes = [_make_clean_episode(n_steps=30, gravity=-0.13) for _ in range(5)]

        result = extract_gravity_oracle(model, norm_stats, episodes)
        assert result["n_samples"] > 50


def _make_thrust_episode(n_steps: int = 50, gravity: float = -0.13,
                         thrust: float = 0.09) -> dict:
    """Synthetic episode with constant main thrust + gravity.

    All states upright, y=0.5, full main thrust action.
    vy changes by (gravity + thrust) each step.
    """
    states = np.zeros((n_steps + 1, 6), dtype=np.float32)
    actions = np.zeros((n_steps, 2), dtype=np.float32)
    actions[:, 0] = 1.0  # full main thrust
    for t in range(n_steps + 1):
        states[t, Y] = 0.5
        states[t, VY] = (gravity + thrust) * t
    return {"states": states, "actions": actions}


def _make_side_thrust_episode(n_steps: int = 50,
                              side_accel: float = 0.025) -> dict:
    """Synthetic episode with constant side thrust, no main.

    All states upright, y=0.5, full side thrust action.
    vx changes by side_accel each step.
    """
    states = np.zeros((n_steps + 1, 6), dtype=np.float32)
    actions = np.zeros((n_steps, 2), dtype=np.float32)
    actions[:, 1] = 1.0  # full side thrust
    for t in range(n_steps + 1):
        states[t, Y] = 0.5
        states[t, VX] = side_accel * t
    return {"states": states, "actions": actions}


def _make_kinematic_episode(n_steps: int = 50) -> dict:
    """Synthetic episode with constant velocity for kinematic check.

    vx=0.2, vy=-0.2 constant. dx = vx * dt, dy = vy * dt per step.
    Effective dt = 1.0 in normalized coords.
    """
    states = np.zeros((n_steps + 1, 6), dtype=np.float32)
    actions = np.zeros((n_steps, 2), dtype=np.float32)
    vx, vy = 0.2, -0.2
    for t in range(n_steps + 1):
        states[t, X] = vx * t
        states[t, Y] = 0.5 + vy * t
        states[t, VX] = vx
        states[t, VY] = vy
    return {"states": states, "actions": actions}


def _make_damping_episode(n_steps: int = 50, damping: float = 0.98) -> dict:
    """Synthetic episode with angular velocity decaying by damping factor.

    No thrust. angular_vel starts at 1.0 and decays by damping^t.
    """
    states = np.zeros((n_steps + 1, 6), dtype=np.float32)
    actions = np.zeros((n_steps, 2), dtype=np.float32)
    for t in range(n_steps + 1):
        states[t, Y] = 0.5
        states[t, ANGULAR_VEL] = 1.0 * (damping ** t)
    return {"states": states, "actions": actions}


class FakeGravityAndThrustModel:
    """Model that predicts gravity + thrust proportional to action_main."""

    def __init__(self, gravity: float = -0.13, thrust_scale: float = 0.09,
                 side_scale: float = 0.025, damping: float = 0.98):
        self.gravity = gravity
        self.thrust_scale = thrust_scale
        self.side_scale = side_scale
        self.damping = damping

    def eval(self):
        pass

    def step(self, obs, action, model_state=None):
        batch = obs.shape[0]
        delta = torch.zeros(batch, 6)
        # Gravity always present on vy.
        delta[:, VY] = self.gravity
        # Main thrust adds to vy proportional to action[0].
        delta[:, VY] += self.thrust_scale * action[:, 0]
        # Side thrust adds to vx proportional to action[1].
        delta[:, VX] = self.side_scale * action[:, 1]
        # Kinematics: dx = vx, dy = vy (effective dt=1 in norm coords).
        delta[:, X] = obs[:, VX]
        delta[:, Y] = obs[:, VY]
        # Angular damping: next_avel = damping * current_avel.
        # delta = next - current = (damping - 1) * current.
        delta[:, ANGULAR_VEL] = (self.damping - 1.0) * obs[:, ANGULAR_VEL]
        return delta, None


class FakeAngleThrustModel:
    """Model that correctly vectors thrust with sin/cos of angle."""

    def __init__(self, gravity: float = -0.13, thrust_magnitude: float = 0.20):
        self.gravity = gravity
        self.thrust_magnitude = thrust_magnitude

    def eval(self):
        pass

    def step(self, obs, action, model_state=None):
        batch = obs.shape[0]
        delta = torch.zeros(batch, 6)
        angle = obs[:, ANGLE]
        # Gravity on vy.
        delta[:, VY] = self.gravity
        # Thrust vectored by angle: vy += thrust * cos(angle), vx += -thrust * sin(angle).
        thrust = self.thrust_magnitude * action[:, 0]
        delta[:, VY] += thrust * torch.cos(angle)
        delta[:, VX] = -thrust * torch.sin(angle)
        return delta, None


def _make_angle_thrust_episode(n_steps: int = 50, gravity: float = -0.13,
                               thrust_magnitude: float = 0.20) -> dict:
    """Synthetic episode with tilted thrust for angle-coupling test.

    Angles vary across steps (0.2 to 0.8 rad). Full main thrust, no side.
    """
    states = np.zeros((n_steps + 1, 6), dtype=np.float32)
    actions = np.zeros((n_steps, 2), dtype=np.float32)
    actions[:, 0] = 1.0  # full main thrust
    angles = np.linspace(0.2, 0.8, n_steps + 1)
    for t in range(n_steps + 1):
        states[t, Y] = 0.5
        states[t, ANGLE] = angles[t]
    # Compute GT deltas from thrust vectoring.
    for t in range(n_steps):
        a = states[t, ANGLE]
        states[t + 1, VY] = states[t, VY] + gravity + thrust_magnitude * np.cos(a)
        states[t + 1, VX] = states[t, VX] - thrust_magnitude * np.sin(a)
    return {"states": states, "actions": actions}


class TestOracleAllConstants:
    """Test oracle extraction for thrust, side, kinematics, damping."""

    def test_main_thrust_extraction(self):
        from lwp.wm.physics_understanding import (
            extract_gravity_oracle, extract_main_thrust_oracle,
        )

        model = FakeGravityAndThrustModel()
        ns = FakeNormStats()
        grav_ep = _make_clean_episode(n_steps=50, gravity=-0.13)
        grav_result = extract_gravity_oracle(model, ns, [grav_ep])
        gravity_model = grav_result["model_mean"]
        gravity_gt = grav_result["gt_mean"]

        thrust_ep = _make_thrust_episode()
        result = extract_main_thrust_oracle(
            model, ns, [thrust_ep], gravity_model, gravity_gt,
        )
        assert result["n_samples"] > 10
        np.testing.assert_allclose(result["model_mean"], 0.09, atol=0.02)

    def test_side_thrust_extraction(self):
        from lwp.wm.physics_understanding import extract_side_thrust_oracle

        model = FakeGravityAndThrustModel()
        ns = FakeNormStats()
        ep = _make_side_thrust_episode()
        result = extract_side_thrust_oracle(model, ns, [ep])
        assert result["n_samples"] > 10
        np.testing.assert_allclose(result["model_mean"], 0.025, atol=0.005)

    def test_kinematic_consistency(self):
        from lwp.wm.physics_understanding import extract_kinematics_oracle

        model = FakeGravityAndThrustModel()
        ns = FakeNormStats()
        ep = _make_kinematic_episode()
        result = extract_kinematics_oracle(model, ns, [ep])
        assert result["n_samples"] > 10
        np.testing.assert_allclose(result["model_mean"], 1.0, atol=0.05)

    def test_angular_damping(self):
        from lwp.wm.physics_understanding import extract_damping_oracle

        model = FakeGravityAndThrustModel(damping=0.98)
        ns = FakeNormStats()
        ep = _make_damping_episode(damping=0.98)
        result = extract_damping_oracle(model, ns, [ep])
        assert result["n_samples"] > 5
        # Stored as 1 - ratio (decay amount), so damping=0.98 -> 0.02.
        np.testing.assert_allclose(result["model_mean"], 0.02, atol=0.005)

    def test_angle_thrust_coupling(self):
        """Model that correctly vectors thrust should match -tan(angle)."""
        from lwp.wm.physics_understanding import extract_angle_thrust_oracle

        model = FakeAngleThrustModel(gravity=-0.13, thrust_magnitude=0.20)
        ns = FakeNormStats()
        ep = _make_angle_thrust_episode(n_steps=50, gravity=-0.13,
                                        thrust_magnitude=0.20)
        result = extract_angle_thrust_oracle(model, ns, [ep], gravity_model=-0.13)
        assert result["n_samples"] > 5
        assert result["n_samples"] > 0


class TestConsistencyChecks:
    """Test R² consistency checks for spurious dependencies."""

    def test_constant_values_have_zero_r2(self):
        """If extracted constant doesn't vary, R² should be ~0."""
        from lwp.wm.physics_understanding import compute_consistency_r2

        values = np.full(100, -0.13)
        states = np.random.RandomState(42).randn(100, 6)
        r2_dict = compute_consistency_r2(values, states)
        for dim_name, r2 in r2_dict.items():
            assert r2 < 0.05, f"R² for {dim_name} should be ~0 for constant values"

    def test_correlated_values_have_high_r2(self):
        """If extracted constant correlates with y, R² for y should be high."""
        from lwp.wm.physics_understanding import compute_consistency_r2

        rng = np.random.RandomState(42)
        states = rng.randn(200, 6)
        values = 3.0 * states[:, 1] + 0.1 * rng.randn(200)  # Y index
        r2_dict = compute_consistency_r2(values, states)
        assert r2_dict["y"] > 0.8, "R² for y should be high when value correlates"
        assert r2_dict["vx"] < 0.1

    def test_too_few_samples_returns_nan(self):
        """With < 3 samples, R² should be NaN (can't fit regression)."""
        from lwp.wm.physics_understanding import compute_consistency_r2

        values = np.array([-0.13, -0.13])
        states = np.array([[0.0, 0.5, 0.0, 0.0, 0.0, 0.0],
                           [0.1, 0.6, 0.0, 0.0, 0.0, 0.0]])
        r2_dict = compute_consistency_r2(values, states)
        for r2 in r2_dict.values():
            assert np.isnan(r2)


class TestRolloutExtraction:
    """Test short-rollout constant extraction."""

    def test_rollout_gravity_matches_oracle_for_perfect_model(self):
        """A model with no compounding error should match oracle exactly."""
        from lwp.wm.physics_understanding import (
            extract_gravity_oracle, extract_constants_rollout,
        )

        model = FakeLinearModel(gravity_delta_norm=-0.13)
        ns = FakeNormStats()
        episode = _make_clean_episode(n_steps=80, gravity=-0.13)

        oracle = extract_gravity_oracle(model, ns, [episode])
        rollout = extract_constants_rollout(
            model, ns, [episode], horizon=10, recurrent=False,
        )
        assert rollout["gravity"]["n_samples"] > 0
        np.testing.assert_allclose(
            rollout["gravity"]["model_mean"], oracle["model_mean"], atol=0.02,
        )

    def test_rollout_all_constants(self):
        """Rollout should extract all constants from model's own trajectory."""
        from lwp.wm.physics_understanding import extract_constants_rollout

        model = FakeGravityAndThrustModel()
        ns = FakeNormStats()
        episodes = [
            _make_clean_episode(n_steps=80, gravity=-0.13),
            _make_thrust_episode(n_steps=80),
            _make_side_thrust_episode(n_steps=80),
            _make_damping_episode(n_steps=80),
            _make_kinematic_episode(n_steps=80),
        ]

        rollout = extract_constants_rollout(
            model, ns, episodes, horizon=10, recurrent=False,
        )
        for key in ["gravity", "main_thrust", "side_thrust",
                    "kinematics", "angular_damping"]:
            assert key in rollout


class TestCompoundingCurve:
    """Test MSE-vs-horizon compounding diagnostic."""

    def test_perfect_model_has_zero_mse(self):
        """A model that predicts exact GT should have ~0 MSE at all horizons."""
        from lwp.wm.physics_understanding import compute_compounding_curve

        model = FakeLinearModel(gravity_delta_norm=-0.13)
        ns = FakeNormStats()
        episode = _make_clean_episode(n_steps=100, gravity=-0.13)

        result = compute_compounding_curve(
            model, ns, [episode], horizons=[1, 5, 10],
            recurrent=False,
        )
        assert "mse_by_horizon" in result
        assert len(result["mse_by_horizon"]) == 3
        for h, mse in result["mse_by_horizon"].items():
            assert mse < 0.01, f"MSE at h={h} should be ~0 for perfect model"

    def test_wrong_model_has_growing_mse(self):
        """A model with wrong gravity should show growing MSE with horizon."""
        from lwp.wm.physics_understanding import compute_compounding_curve

        model = FakeLinearModel(gravity_delta_norm=-0.2)
        ns = FakeNormStats()
        episode = _make_clean_episode(n_steps=100, gravity=-0.13)

        result = compute_compounding_curve(
            model, ns, [episode], horizons=[1, 5, 10],
            recurrent=False,
        )
        mse_1 = result["mse_by_horizon"][1]
        mse_10 = result["mse_by_horizon"][10]
        assert mse_10 > mse_1, "MSE should grow with horizon for wrong model"

    def test_per_dim_mse(self):
        """Should report per-dimension MSE breakdown."""
        from lwp.wm.physics_understanding import compute_compounding_curve

        model = FakeLinearModel(gravity_delta_norm=-0.2)
        ns = FakeNormStats()
        episode = _make_clean_episode(n_steps=100, gravity=-0.13)

        result = compute_compounding_curve(
            model, ns, [episode], horizons=[1, 5],
            recurrent=False,
        )
        assert "per_dim_mse" in result
        for h in [1, 5]:
            assert result["per_dim_mse"][h]["vy"] > result["per_dim_mse"][h]["x"]

    def test_power_law_fit(self):
        """Should fit MSE(h) = a * h^b and report exponent."""
        from lwp.wm.physics_understanding import compute_compounding_curve

        model = FakeLinearModel(gravity_delta_norm=-0.2)
        ns = FakeNormStats()
        episode = _make_clean_episode(n_steps=100, gravity=-0.13)

        result = compute_compounding_curve(
            model, ns, [episode], horizons=[1, 2, 5, 10],
            recurrent=False,
        )
        assert "fit_params" in result
        assert not np.isnan(result["fit_params"]["b"])
        assert result["fit_params"]["b"] > 1.0

    def test_useful_horizon_and_diverge_order(self):
        """Should report useful horizon per dim and first-to-diverge ordering."""
        from lwp.wm.physics_understanding import compute_compounding_curve

        model = FakeLinearModel(gravity_delta_norm=-0.2)
        ns = FakeNormStats()
        episode = _make_clean_episode(n_steps=100, gravity=-0.13)

        result = compute_compounding_curve(
            model, ns, [episode], horizons=[1, 5, 10, 20],
            recurrent=False,
        )
        assert "useful_horizon" in result
        assert "first_to_diverge" in result
        assert isinstance(result["first_to_diverge"], list)
        if result["first_to_diverge"]:
            first_dim = result["first_to_diverge"][0][0]
            assert first_dim == "vy", f"Expected vy to diverge first, got {first_dim}"


class TestReportGeneration:
    """Test the full report orchestration and formatting."""

    def test_generate_full_report(self):
        """generate_report() should produce structured results dict."""
        from lwp.wm.physics_understanding import generate_report

        model = FakeGravityAndThrustModel()
        ns = FakeNormStats()
        episodes = [
            _make_clean_episode(n_steps=80, gravity=-0.13),
            _make_thrust_episode(n_steps=80),
            _make_side_thrust_episode(n_steps=80),
            _make_damping_episode(n_steps=80),
            _make_kinematic_episode(n_steps=80),
        ]

        results = generate_report(model, ns, episodes, recurrent=False)

        assert "gravity" in results["constants"]
        assert "main_thrust" in results["constants"]
        assert "side_thrust" in results["constants"]
        assert "kinematics" in results["constants"]
        assert "angular_damping" in results["constants"]
        assert "angle_thrust" in results["constants"]
        for name, const in results["constants"].items():
            assert "oracle" in const, f"{name} missing oracle"
            assert "rollout" in const, f"{name} missing rollout"
            assert "consistency" in const, f"{name} missing consistency"

        assert "compounding" in results

    def test_format_console_report(self):
        """format_console_report() should produce readable string."""
        from lwp.wm.physics_understanding import (
            generate_report, format_console_report,
        )

        model = FakeGravityAndThrustModel()
        ns = FakeNormStats()
        episodes = [_make_clean_episode(n_steps=80, gravity=-0.13)]

        results = generate_report(model, ns, episodes, recurrent=False)
        text = format_console_report(results, run_name="test-model")

        assert "test-model" in text
        assert "Gravity" in text
        assert "Oracle" in text or "oracle" in text.lower()


class FakeGRUModel:
    """Fake recurrent model that returns better predictions with more context.

    Simulates a model that needs ~10 steps of warmup before predictions
    stabilize. Hidden state is just a counter of steps seen.
    """

    def __init__(self, gravity: float = -0.13):
        self.gravity = gravity

    def eval(self):
        pass

    def step(self, obs, action, model_state=None):
        batch = obs.shape[0]
        steps_seen = 0 if model_state is None else model_state
        steps_seen += 1
        delta = torch.zeros(batch, 6)
        # Add noise that decreases with warmup (simulates context helping).
        noise_scale = 0.1 / (1 + steps_seen)
        delta[:, 3] = self.gravity + noise_scale  # VY index
        return delta, steps_seen


class TestWarmupDiagnostic:
    """Test warmup length diagnostic for recurrent models."""

    def test_warmup_curve_computed(self):
        from lwp.wm.physics_understanding import compute_warmup_curve

        model = FakeGRUModel(gravity=-0.13)
        ns = FakeNormStats()
        episode = _make_clean_episode(n_steps=100, gravity=-0.13)

        result = compute_warmup_curve(
            model, ns, [episode],
            warmup_lengths=[0, 1, 5, 10, 20, 50],
        )
        assert "oracle_mse_by_warmup" in result
        assert len(result["oracle_mse_by_warmup"]) == 6

    def test_more_warmup_reduces_error(self):
        from lwp.wm.physics_understanding import compute_warmup_curve

        model = FakeGRUModel(gravity=-0.13)
        ns = FakeNormStats()
        episode = _make_clean_episode(n_steps=100, gravity=-0.13)

        result = compute_warmup_curve(
            model, ns, [episode],
            warmup_lengths=[0, 5, 20, 50],
        )
        mse = result["oracle_mse_by_warmup"]
        # Error should decrease (or stay flat) with more warmup.
        assert mse[0] >= mse[50]


class FakeGRUModelColdWarm:
    """GRU-like model whose prediction depends on hidden state.

    Returns gravity_cold on first step (no hidden state),
    gravity_warm on subsequent steps (has hidden state).
    """

    def __init__(self, gravity_cold=-0.5, gravity_warm=-0.13):
        self.gravity_cold = gravity_cold
        self.gravity_warm = gravity_warm

    def eval(self):
        pass

    def parameters(self):
        return iter([torch.zeros(1)])

    def step(self, obs, action, model_state=None):
        batch = obs.shape[0]
        delta = torch.zeros(batch, 6)
        if model_state is None:
            delta[:, 3] = self.gravity_cold
            new_state = torch.ones(1)
        else:
            delta[:, 3] = self.gravity_warm
            new_state = model_state
        return delta, new_state


class TestRecurrentOracleExtraction:
    """Test that oracle extraction threads hidden state for recurrent models."""

    def test_oracle_uses_hidden_state_for_recurrent(self):
        from lwp.wm.physics_understanding import extract_gravity_oracle

        model = FakeGRUModelColdWarm(gravity_cold=-0.5, gravity_warm=-0.13)
        norm_stats = FakeNormStats()
        episode = _make_clean_episode(n_steps=50, gravity=-0.13)
        result = extract_gravity_oracle(model, norm_stats, [episode], recurrent=True)
        assert result["n_samples"] > 5
        # With recurrent=True, most steps use warm prediction (-0.13)
        np.testing.assert_allclose(result["model_mean"], -0.13, atol=0.01)

    def test_oracle_cold_start_without_recurrent_flag(self):
        from lwp.wm.physics_understanding import extract_gravity_oracle

        model = FakeGRUModelColdWarm(gravity_cold=-0.5, gravity_warm=-0.13)
        norm_stats = FakeNormStats()
        episode = _make_clean_episode(n_steps=50, gravity=-0.13)
        result = extract_gravity_oracle(model, norm_stats, [episode], recurrent=False)
        assert result["n_samples"] > 5
        # Without recurrent, every step is cold-start (-0.5)
        np.testing.assert_allclose(result["model_mean"], -0.5, atol=0.01)


class TestRolloutOffByOne:
    """Verify recurrent rollout extraction doesn't double-apply the branch action."""

    def test_recurrent_rollout_branch_alignment(self):
        """Hidden state at branch time should reflect context UP TO (not including) branch step."""
        from lwp.wm.physics_understanding import extract_constants_rollout

        class StepCountingModel:
            def eval(self): pass
            def parameters(self): return iter([torch.zeros(1)])
            def step(self, obs, action, model_state=None):
                batch = obs.shape[0]
                count = (model_state or 0)
                delta = torch.zeros(batch, 6)
                delta[:, 3] = -0.13
                return delta, count + 1

        model = StepCountingModel()
        norm_stats = FakeNormStats()
        episode = _make_clean_episode(n_steps=100, gravity=-0.13)

        import lwp.wm.physics_understanding as pu
        original_rollout = pu._rollout_trajectory
        branch_states = []
        branch_times = []

        def recording_rollout(model, norm_stats, start_state, actions, model_state=None):
            branch_states.append(model_state)
            ep_states = episode["states"][:, :6]
            for t in range(len(ep_states)):
                if np.allclose(start_state, ep_states[t], atol=1e-6):
                    branch_times.append(t)
                    break
            return original_rollout(model, norm_stats, start_state, actions, model_state=model_state)

        pu._rollout_trajectory = recording_rollout
        try:
            extract_constants_rollout(
                model, norm_stats, [episode],
                horizon=10, recurrent=True, warmup_steps=5,
            )
        finally:
            pu._rollout_trajectory = original_rollout

        assert len(branch_states) > 0, "No branches were taken"
        for state_count, t in zip(branch_states, branch_times):
            assert state_count == t, (
                f"Branch at t={t}: hidden state count={state_count}, expected {t}. "
                f"Hidden state was advanced past branch point (off-by-one)."
            )

    def test_recurrent_rollout_gravity_correct(self):
        """Perfect-gravity recurrent model should extract correct gravity from rollouts."""
        from lwp.wm.physics_understanding import extract_constants_rollout

        class PerfectGravityGRU:
            def eval(self): pass
            def parameters(self): return iter([torch.zeros(1)])
            def step(self, obs, action, model_state=None):
                batch = obs.shape[0]
                delta = torch.zeros(batch, 6)
                delta[:, 3] = -0.13
                return delta, (model_state or 0) + 1

        model = PerfectGravityGRU()
        norm_stats = FakeNormStats()
        episode = _make_clean_episode(n_steps=100, gravity=-0.13)

        results = extract_constants_rollout(
            model, norm_stats, [episode],
            horizon=10, recurrent=True, warmup_steps=5,
            gravity_model=-0.13, gravity_gt=-0.13,
        )

        assert results["gravity"]["n_samples"] > 0
        np.testing.assert_allclose(
            results["gravity"]["model_mean"], -0.13, atol=0.02,
        )
