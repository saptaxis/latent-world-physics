"""Tests for probe training (ridge regression with episode-level CV)."""
import json
import numpy as np
import pytest

from lwp.probing.training import (
    train_single_probe,
    train_all_probes,
)


def _make_synthetic_probe_data(n_episodes=20, steps_per_episode=50, hidden_dim=128):
    """Create synthetic probe data with known linear relationship.

    Target = weighted sum of first 3 hidden units + noise.
    R² should be high (>0.8) when the relationship is clean.
    """
    rng = np.random.default_rng(42)
    n_total = n_episodes * steps_per_episode

    # Activations: random but structured
    activations = rng.standard_normal((n_total, hidden_dim)).astype(np.float32)

    # True weights for a linear target
    true_weights = np.zeros(hidden_dim, dtype=np.float32)
    true_weights[0] = 1.0
    true_weights[1] = -0.5
    true_weights[2] = 0.3

    # Target is a linear function of activations + small noise
    target = activations @ true_weights + rng.normal(0, 0.1, n_total).astype(np.float32)

    # Episode IDs
    episode_ids = np.repeat(np.arange(n_episodes), steps_per_episode).astype(np.int32)

    return activations, target, episode_ids


class TestTrainSingleProbe:
    def test_returns_expected_keys(self):
        acts, target, ep_ids = _make_synthetic_probe_data()
        result = train_single_probe(acts, target, ep_ids)
        assert "r2_mean" in result
        assert "r2_std" in result
        assert "r2_folds" in result
        assert "alpha" in result
        assert "top_units" in result
        assert "top_weights" in result
        assert "coefficients" in result
        assert "intercept" in result

    def test_coefficients_shape_matches_hidden_dim(self):
        acts, target, ep_ids = _make_synthetic_probe_data(hidden_dim=128)
        result = train_single_probe(acts, target, ep_ids)
        assert isinstance(result["coefficients"], np.ndarray)
        assert result["coefficients"].shape == (128,)
        assert isinstance(result["intercept"], float)

    def test_high_r2_for_linear_target(self):
        acts, target, ep_ids = _make_synthetic_probe_data()
        result = train_single_probe(acts, target, ep_ids)
        # Linear relationship with low noise -> R² should be high
        assert result["r2_mean"] > 0.8

    def test_low_r2_for_random_target(self):
        rng = np.random.default_rng(99)
        acts = rng.standard_normal((1000, 128)).astype(np.float32)
        target = rng.standard_normal(1000).astype(np.float32)
        ep_ids = np.repeat(np.arange(20), 50).astype(np.int32)
        result = train_single_probe(acts, target, ep_ids)
        # Random target -> R² should be near 0
        assert result["r2_mean"] < 0.1

    def test_five_folds_by_default(self):
        acts, target, ep_ids = _make_synthetic_probe_data()
        result = train_single_probe(acts, target, ep_ids)
        assert len(result["r2_folds"]) == 5

    def test_top_units_are_sorted_by_importance(self):
        acts, target, ep_ids = _make_synthetic_probe_data()
        result = train_single_probe(acts, target, ep_ids)
        weights = [abs(w) for w in result["top_weights"]]
        assert weights == sorted(weights, reverse=True)

    def test_top_units_identifies_true_features(self):
        """The known-important units (0, 1, 2) should appear in top_units."""
        acts, target, ep_ids = _make_synthetic_probe_data()
        result = train_single_probe(acts, target, ep_ids)
        # At least 2 of the 3 true features should be in top 5
        true_features = {0, 1, 2}
        found = true_features.intersection(result["top_units"])
        assert len(found) >= 2


class TestTrainAllProbes:
    def test_trains_probes_for_all_layers_and_targets(self):
        rng = np.random.default_rng(42)
        n = 1000
        data = {
            "activations_L1": rng.standard_normal((n, 16)).astype(np.float32),
            "activations_L2": rng.standard_normal((n, 16)).astype(np.float32),
            "physics_params": rng.standard_normal((n, 7)).astype(np.float32),
            "behavioral": rng.standard_normal((n, 5)).astype(np.float32),
            "episode_ids": np.repeat(np.arange(20), 50).astype(np.int32),
        }
        result, coefficients = train_all_probes(data)
        assert "L1" in result
        assert "L2" in result
        # Should have entries for all 12 targets
        assert len(result["L1"]) == 12
        assert len(result["L2"]) == 12

    def test_coefficients_dict_has_all_keys(self):
        rng = np.random.default_rng(42)
        n = 1000
        data = {
            "activations_L1": rng.standard_normal((n, 16)).astype(np.float32),
            "activations_L2": rng.standard_normal((n, 16)).astype(np.float32),
            "physics_params": rng.standard_normal((n, 7)).astype(np.float32),
            "behavioral": rng.standard_normal((n, 5)).astype(np.float32),
            "episode_ids": np.repeat(np.arange(20), 50).astype(np.int32),
        }
        _, coefficients = train_all_probes(data)
        # 2 layers × 12 targets × 2 (coef + intercept) = 48 keys
        assert len(coefficients) == 48
        assert "L1/gravity" in coefficients
        assert "L1/gravity_intercept" in coefficients
        assert coefficients["L1/gravity"].shape == (16,)

    def test_filter_layers(self):
        rng = np.random.default_rng(42)
        n = 1000
        data = {
            "activations_L1": rng.standard_normal((n, 16)).astype(np.float32),
            "activations_L2": rng.standard_normal((n, 16)).astype(np.float32),
            "physics_params": rng.standard_normal((n, 7)).astype(np.float32),
            "behavioral": rng.standard_normal((n, 5)).astype(np.float32),
            "episode_ids": np.repeat(np.arange(20), 50).astype(np.int32),
        }
        result, _ = train_all_probes(data, layers=["L1"])
        assert "L1" in result
        assert "L2" not in result

    def test_filter_targets(self):
        rng = np.random.default_rng(42)
        n = 1000
        data = {
            "activations_L1": rng.standard_normal((n, 16)).astype(np.float32),
            "activations_L2": rng.standard_normal((n, 16)).astype(np.float32),
            "physics_params": rng.standard_normal((n, 7)).astype(np.float32),
            "behavioral": rng.standard_normal((n, 5)).astype(np.float32),
            "episode_ids": np.repeat(np.arange(20), 50).astype(np.int32),
        }
        result, _ = train_all_probes(data, targets=["gravity", "twr"])
        assert len(result["L1"]) == 2
        assert "gravity" in result["L1"]
        assert "twr" in result["L1"]
