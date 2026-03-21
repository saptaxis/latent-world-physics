"""Tests for trajectory_metrics.py — per-episode metric computation.

Uses synthetic .npz files created via episode_io.save_episode() so
we don't need real trained agents. Tests metric shapes, value ranges,
and landed-vs-crashed differences.
"""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest


def _make_fake_episode(
    outdir,
    episode_idx=0,
    n_steps=150,
    outcome="landed",
    seed=42,
    profile="default",
    landing_x_error=0.05,
):
    """Create a synthetic .npz file for metric testing.

    Generates plausible (but not realistic) state/action trajectories
    that exercise the metric computation code paths. The trajectory
    simulates a simple descent with configurable outcome.

    Args:
        outdir: Directory for the .npz file.
        episode_idx: Episode number for filename.
        n_steps: Number of timesteps.
        outcome: "landed", "crashed", or "timeout".
        seed: Random seed for reproducibility.
        profile: Profile name for metadata tagging.
        landing_x_error: Final x position (only matters for landed).

    Returns:
        Path to the saved .npz file.
    """
    from parametric_lunar_lander.episode_io import save_episode

    rng = np.random.RandomState(seed)

    # State vector: 15D = 8 kinematic + 7 physics params
    # [0] x_pos, [1] y_pos, [2] vx, [3] vy, [4] angle,
    # [5] angular_vel, [6] left_leg, [7] right_leg,
    # [8] gravity, [9] main_power, [10] side_power,
    # [11] density, [12] ang_damping, [13] wind, [14] turbulence
    states = np.zeros((n_steps + 1, 15), dtype=np.float32)

    # Physics params (constant across episode)
    states[:, 8] = -10.0   # gravity
    states[:, 9] = 13.0    # main_engine_power
    states[:, 10] = 0.6    # side_engine_power
    states[:, 11] = 5.0    # lander_density
    states[:, 12] = 2.5    # angular_damping
    states[:, 13] = 10.0   # wind_power
    states[:, 14] = 1.5    # turbulence_power

    # Kinematic trajectory: start high, descend
    for t in range(n_steps + 1):
        frac = t / n_steps
        states[t, 0] = landing_x_error * frac           # x: drift toward landing
        states[t, 1] = 1.0 - 0.9 * frac                 # y: descend from 1.0
        states[t, 2] = rng.uniform(-0.3, 0.3)           # vx: small lateral vel
        states[t, 3] = -0.5 + 0.3 * frac                # vy: descending
        states[t, 4] = rng.uniform(-0.2, 0.2)           # angle: small tilt
        states[t, 5] = rng.uniform(-0.3, 0.3)           # angular vel

    if outcome == "landed":
        # Last state: on ground, legs in contact, low velocity
        states[-1, 0] = landing_x_error
        states[-1, 1] = 0.02
        states[-1, 2] = 0.0
        states[-1, 3] = -0.05
        states[-1, 4] = 0.01
        states[-1, 5] = 0.0
        states[-1, 6] = 1.0  # left leg contact
        states[-1, 7] = 1.0  # right leg contact
    elif outcome == "crashed":
        # Last state: on ground, high downward velocity
        states[-1, 1] = 0.0
        states[-1, 3] = -2.0

    # Actions: (T, 2) — main thrust [0,1] and side thrust [-1,1]
    actions = np.zeros((n_steps, 2), dtype=np.float32)
    actions[:, 0] = rng.uniform(0.0, 0.8, n_steps)      # main thrust
    actions[:, 1] = rng.uniform(-0.5, 0.5, n_steps)     # side thrust

    # Rewards
    rewards = np.full(n_steps, -0.5, dtype=np.float32)   # shaping penalty
    if outcome == "landed":
        rewards[-1] = 100.0
    elif outcome == "crashed":
        rewards[-1] = -100.0

    # Dones
    dones = np.zeros(n_steps, dtype=bool)
    dones[-1] = True

    # Metadata
    metadata = {
        "physics_config": {
            "gravity": -10.0,
            "main_engine_power": 13.0,
            "side_engine_power": 0.6,
            "lander_density": 5.0,
            "angular_damping": 2.5,
            "wind_power": 10.0,
            "turbulence_power": 1.5,
        },
        "outcome": outcome,
        "seed": seed,
        "episode_length": n_steps,
        "total_reward": float(rewards.sum()),
        "policy": "test",
        "profile": profile,
    }

    npz_path = os.path.join(outdir, f"episode_{episode_idx:04d}.npz")
    save_episode(
        path=npz_path,
        states=states,
        actions=actions,
        rewards=rewards,
        dones=dones,
        metadata=metadata,
    )
    return npz_path


class TestComputeMetrics:
    """Test compute_metrics() on individual episodes."""

    def test_returns_dict_with_expected_keys(self, tmp_path):
        from lwp.analysis.trajectory_metrics import compute_metrics

        npz_path = _make_fake_episode(str(tmp_path), outcome="landed")
        m = compute_metrics(npz_path)

        # Identity keys
        assert "outcome" in m
        assert "episode_steps" in m
        assert "total_reward" in m
        assert "profile" in m
        assert "npz_path" in m

        # Physics keys
        assert "gravity" in m
        assert "main_engine_power" in m
        assert "twr" in m

        # Spatial keys
        assert "max_altitude" in m
        assert "max_lateral_drift" in m

        # Action keys
        assert "mean_main_thrust" in m
        assert "thrust_duty_cycle" in m
        assert "total_fuel" in m

        # Control keys
        assert "mean_abs_angular_vel" in m

    def test_landed_has_landing_metrics(self, tmp_path):
        from lwp.analysis.trajectory_metrics import compute_metrics

        npz_path = _make_fake_episode(str(tmp_path), outcome="landed")
        m = compute_metrics(npz_path)

        assert m["outcome"] == "landed"
        assert m["landing_x_error"] is not None
        assert m["landing_vy"] is not None
        assert m["fuel_efficiency"] is not None

    def test_crashed_has_null_landing_metrics(self, tmp_path):
        from lwp.analysis.trajectory_metrics import compute_metrics

        npz_path = _make_fake_episode(str(tmp_path), outcome="crashed")
        m = compute_metrics(npz_path)

        assert m["outcome"] == "crashed"
        assert m["landing_x_error"] is None
        assert m["landing_vy"] is None
        assert m["fuel_efficiency"] is None

    def test_spatial_metrics_reasonable(self, tmp_path):
        from lwp.analysis.trajectory_metrics import compute_metrics

        npz_path = _make_fake_episode(str(tmp_path), outcome="landed")
        m = compute_metrics(npz_path)

        assert m["max_altitude"] >= 0
        assert m["max_lateral_drift"] >= 0

    def test_action_metrics_reasonable(self, tmp_path):
        from lwp.analysis.trajectory_metrics import compute_metrics

        npz_path = _make_fake_episode(str(tmp_path), outcome="landed")
        m = compute_metrics(npz_path)

        assert 0.0 <= m["thrust_duty_cycle"] <= 1.0
        assert m["total_fuel"] > 0
        assert 0.0 <= m["mean_main_thrust"] <= 1.0

    def test_physics_params_from_metadata(self, tmp_path):
        from lwp.analysis.trajectory_metrics import compute_metrics

        npz_path = _make_fake_episode(str(tmp_path), outcome="landed")
        m = compute_metrics(npz_path)

        assert m["gravity"] == -10.0
        assert m["main_engine_power"] == 13.0
        assert m["lander_density"] == 5.0

    def test_episode_steps_matches_actions(self, tmp_path):
        from lwp.analysis.trajectory_metrics import compute_metrics

        npz_path = _make_fake_episode(str(tmp_path), n_steps=80, outcome="crashed")
        m = compute_metrics(npz_path)

        assert m["episode_steps"] == 80

    # --- New scalars from Plan 17 ---

    def test_action_distribution_shape_scalars(self, tmp_path):
        """Action distribution shape: std and fraction scalars."""
        from lwp.analysis.trajectory_metrics import compute_metrics

        npz_path = _make_fake_episode(str(tmp_path), outcome="landed")
        m = compute_metrics(npz_path)

        # Standard deviations should be non-negative
        assert m["std_main_thrust"] >= 0.0
        assert m["std_side_thrust"] >= 0.0

        # Fractions should be in [0, 1]
        assert 0.0 <= m["main_thrust_frac_full"] <= 1.0
        assert 0.0 <= m["main_thrust_frac_zero"] <= 1.0

    def test_phase_fraction_scalars(self, tmp_path):
        """Phase fractions should be in [0, 1] and describe flight behavior."""
        from lwp.analysis.trajectory_metrics import compute_metrics

        npz_path = _make_fake_episode(str(tmp_path), outcome="landed")
        m = compute_metrics(npz_path)

        for key in ["frac_descending", "frac_hovering", "frac_approaching", "frac_correcting"]:
            assert 0.0 <= m[key] <= 1.0, f"{key} = {m[key]} not in [0, 1]"

    def test_control_smoothness_scalars(self, tmp_path):
        """Autocorrelation at lag 1 should be in [-1, 1]."""
        from lwp.analysis.trajectory_metrics import compute_metrics

        npz_path = _make_fake_episode(str(tmp_path), outcome="landed", n_steps=100)
        m = compute_metrics(npz_path)

        assert -1.0 <= m["thrust_autocorr_lag1"] <= 1.0
        assert -1.0 <= m["side_thrust_autocorr_lag1"] <= 1.0

    def test_autocorr_constant_signal(self, tmp_path):
        """Constant thrust should give autocorrelation 0.0 (zero variance convention)."""
        from parametric_lunar_lander.episode_io import save_episode
        from lwp.analysis.trajectory_metrics import compute_metrics

        n_steps = 100
        states = np.zeros((n_steps + 1, 15), dtype=np.float32)
        states[:, 8] = -10.0  # gravity
        states[:, 9] = 13.0   # main_engine_power
        states[:, 11] = 5.0   # lander_density
        states[:, 1] = np.linspace(1.0, 0.1, n_steps + 1)  # descending y

        # Constant main thrust = 0.5, side thrust = 0.0
        actions = np.full((n_steps, 2), 0.5, dtype=np.float32)
        actions[:, 1] = 0.0

        rewards = np.full(n_steps, -0.5, dtype=np.float32)
        dones = np.zeros(n_steps, dtype=bool)
        dones[-1] = True

        metadata = {
            "physics_config": {"gravity": -10.0, "main_engine_power": 13.0,
                               "side_engine_power": 0.6, "lander_density": 5.0,
                               "angular_damping": 0.0, "wind_power": 0.0,
                               "turbulence_power": 0.0},
            "outcome": "landed",
            "total_reward": -50.0,
            "profile": "default",
        }

        npz_path = str(tmp_path / "constant.npz")
        save_episode(npz_path, states, actions, rewards, dones, metadata)
        m = compute_metrics(npz_path)

        # Constant signal has zero variance -> autocorr = 0.0 by convention
        assert m["thrust_autocorr_lag1"] == 0.0

    def test_short_episode_smoothness(self, tmp_path):
        """Very short episodes (< 2 steps) should handle autocorr gracefully."""
        from lwp.analysis.trajectory_metrics import compute_metrics

        npz_path = _make_fake_episode(str(tmp_path), n_steps=1, outcome="crashed")
        m = compute_metrics(npz_path)

        # Should not crash — return 0.0 for undefined autocorrelation
        assert m["thrust_autocorr_lag1"] == 0.0
        assert m["side_thrust_autocorr_lag1"] == 0.0


class TestComputeCollectionMetrics:
    """Test compute_collection_metrics() on a directory of episodes."""

    def test_collection_returns_dataframe(self, tmp_path):
        from lwp.analysis.trajectory_metrics import compute_collection_metrics

        for i in range(5):
            _make_fake_episode(str(tmp_path), episode_idx=i,
                               outcome="landed" if i % 2 == 0 else "crashed",
                               seed=i)

        df = compute_collection_metrics(str(tmp_path), workers=1)

        assert len(df) == 5
        assert "outcome" in df.columns
        assert "total_reward" in df.columns
        assert "twr" in df.columns

    def test_parallel_matches_sequential(self, tmp_path):
        from lwp.analysis.trajectory_metrics import compute_collection_metrics

        for i in range(6):
            _make_fake_episode(str(tmp_path), episode_idx=i,
                               outcome="landed" if i < 3 else "crashed",
                               seed=i * 10)

        df_seq = compute_collection_metrics(str(tmp_path), workers=1)
        df_par = compute_collection_metrics(str(tmp_path), workers=4)

        # Sort by npz_path to align rows
        df_seq = df_seq.sort_values("npz_path").reset_index(drop=True)
        df_par = df_par.sort_values("npz_path").reset_index(drop=True)

        # All columns should match
        for col in df_seq.columns:
            if df_seq[col].dtype == float:
                np.testing.assert_allclose(
                    df_seq[col].values, df_par[col].values,
                    rtol=1e-5, err_msg=f"Mismatch in column {col}"
                )
            else:
                assert list(df_seq[col]) == list(df_par[col]), f"Mismatch in column {col}"
