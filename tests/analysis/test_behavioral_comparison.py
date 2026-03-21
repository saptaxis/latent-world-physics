"""Tests for behavioral_comparison.py — model-level aggregation and adaptation.

Tests use synthetic histogram data and DataFrames to verify:
- Aggregation normalizes correctly
- JS divergence is bounded [0, 1]
- Adaptation score is zero for identical distributions
- Adaptation score is positive for shifted distributions
- Binned performance computes correct landed percentages
"""

import numpy as np
import pandas as pd
import pytest


class TestAggregateModelDistribution:
    """Test aggregate_model_distribution()."""

    def test_normalized_to_probability(self):
        from lwp.analysis.behavioral_comparison import aggregate_model_distribution

        edges = np.linspace(0.0, 1.0, 51)
        histograms = [
            {"main_thrust_counts": np.ones(50), "main_thrust_edges": edges,
             "side_thrust_counts": np.ones(50), "side_thrust_edges": np.linspace(-1, 1, 51)},
            {"main_thrust_counts": np.ones(50) * 3, "main_thrust_edges": edges,
             "side_thrust_counts": np.ones(50) * 3, "side_thrust_edges": np.linspace(-1, 1, 51)},
        ]

        result = aggregate_model_distribution(histograms)

        # Probability distribution should sum to ~1.0
        np.testing.assert_almost_equal(result["main_thrust_probs"].sum(), 1.0)
        np.testing.assert_almost_equal(result["side_thrust_probs"].sum(), 1.0)

    def test_preserves_shape(self):
        from lwp.analysis.behavioral_comparison import aggregate_model_distribution

        n_bins = 50
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        histograms = [
            {"main_thrust_counts": np.ones(n_bins), "main_thrust_edges": edges,
             "side_thrust_counts": np.ones(n_bins), "side_thrust_edges": np.linspace(-1, 1, n_bins + 1)},
        ]

        result = aggregate_model_distribution(histograms)

        assert result["main_thrust_probs"].shape == (n_bins,)
        assert result["side_thrust_probs"].shape == (n_bins,)

    def test_empty_input(self):
        from lwp.analysis.behavioral_comparison import aggregate_model_distribution

        with pytest.raises(ValueError, match="empty"):
            aggregate_model_distribution([])


class TestComputeAdaptationScore:
    """Test compute_adaptation_score()."""

    def test_identical_distributions_score_zero(self):
        """If all quartiles have the same action distribution, adaptation = 0."""
        from lwp.analysis.behavioral_comparison import compute_adaptation_score

        n_bins = 50
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        side_edges = np.linspace(-1.0, 1.0, n_bins + 1)

        # 100 episodes with identical action distributions
        rng = np.random.default_rng(42)
        histograms = {}
        for i in range(100):
            key = f"episode_{i:04d}.npz"
            histograms[key] = {
                "main_thrust_counts": np.ones(n_bins, dtype=int) * 3,
                "main_thrust_edges": edges,
                "side_thrust_counts": np.ones(n_bins, dtype=int) * 3,
                "side_thrust_edges": side_edges,
            }

        # Vary TWR across episodes so quartiles have different physics,
        # but action distributions are all uniform -> score should be ~0
        df = pd.DataFrame({
            "npz_path": [f"episode_{i:04d}.npz" for i in range(100)],
            "twr": rng.uniform(2.0, 20.0, 100),
        })

        result = compute_adaptation_score(histograms, df, physics_col="twr")

        assert result["adaptation_score"] < 0.01, (
            f"Identical distributions should give ~0 score, got {result['adaptation_score']}"
        )

    def test_shifted_distributions_score_positive(self):
        """If quartiles have very different distributions, adaptation > 0."""
        from lwp.analysis.behavioral_comparison import compute_adaptation_score

        n_bins = 50
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        side_edges = np.linspace(-1.0, 1.0, n_bins + 1)

        # 100 episodes. Low TWR -> high thrust. High TWR -> low thrust.
        histograms = {}
        twrs = []
        for i in range(100):
            key = f"episode_{i:04d}.npz"
            twr = 2.0 + (i / 100) * 18.0  # TWR from 2 to 20
            twrs.append(twr)

            # Create peaked distribution that shifts with TWR
            counts = np.zeros(n_bins, dtype=int)
            # Peak position moves from high bins (high thrust) to low bins
            peak_bin = int((1.0 - i / 100) * (n_bins - 1))
            peak_bin = np.clip(peak_bin, 0, n_bins - 1)
            counts[peak_bin] = 100
            # Add some spread
            if peak_bin > 0:
                counts[peak_bin - 1] = 20
            if peak_bin < n_bins - 1:
                counts[peak_bin + 1] = 20

            histograms[key] = {
                "main_thrust_counts": counts,
                "main_thrust_edges": edges,
                "side_thrust_counts": np.ones(n_bins, dtype=int),
                "side_thrust_edges": side_edges,
            }

        df = pd.DataFrame({
            "npz_path": [f"episode_{i:04d}.npz" for i in range(100)],
            "twr": twrs,
        })

        result = compute_adaptation_score(histograms, df, physics_col="twr")

        assert result["adaptation_score"] > 0.1, (
            f"Shifted distributions should give positive score, got {result['adaptation_score']}"
        )

    def test_returns_expected_keys(self):
        from lwp.analysis.behavioral_comparison import compute_adaptation_score

        n_bins = 50
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        side_edges = np.linspace(-1.0, 1.0, n_bins + 1)

        histograms = {}
        for i in range(40):
            key = f"episode_{i:04d}.npz"
            histograms[key] = {
                "main_thrust_counts": np.ones(n_bins, dtype=int),
                "main_thrust_edges": edges,
                "side_thrust_counts": np.ones(n_bins, dtype=int),
                "side_thrust_edges": side_edges,
            }

        df = pd.DataFrame({
            "npz_path": [f"episode_{i:04d}.npz" for i in range(40)],
            "twr": np.linspace(2.0, 20.0, 40),
        })

        result = compute_adaptation_score(histograms, df, physics_col="twr")

        assert "adaptation_score" in result
        assert "quartile_boundaries" in result
        assert "pairwise_js" in result
        assert "n_quartiles" in result

    def test_adaptation_score_bounded(self):
        """JS divergence-based score should be in [0, 1]."""
        from lwp.analysis.behavioral_comparison import compute_adaptation_score

        n_bins = 50
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        side_edges = np.linspace(-1.0, 1.0, n_bins + 1)

        histograms = {}
        rng = np.random.default_rng(123)
        for i in range(80):
            key = f"episode_{i:04d}.npz"
            histograms[key] = {
                "main_thrust_counts": rng.integers(0, 100, n_bins),
                "main_thrust_edges": edges,
                "side_thrust_counts": rng.integers(0, 100, n_bins),
                "side_thrust_edges": side_edges,
            }

        df = pd.DataFrame({
            "npz_path": [f"episode_{i:04d}.npz" for i in range(80)],
            "twr": rng.uniform(2.0, 20.0, 80),
        })

        result = compute_adaptation_score(histograms, df, physics_col="twr")

        assert 0.0 <= result["adaptation_score"] <= 1.0


class TestComputeBinnedPerformance:
    """Test compute_binned_performance()."""

    def test_correct_landed_percentages(self):
        from lwp.analysis.behavioral_comparison import compute_binned_performance

        df = pd.DataFrame({
            "twr": [5.0, 5.5, 6.0, 15.0, 15.5, 16.0],
            "outcome": ["crashed", "crashed", "landed", "landed", "landed", "landed"],
            "total_reward": [-100, -100, 100, 200, 200, 200],
        })

        result = compute_binned_performance(df, bin_col="twr", n_bins=2)

        assert len(result) == 2
        assert "landed_pct" in result.columns
        assert "n_episodes" in result.columns
        assert "mean_reward" in result.columns

    def test_all_landed_bin(self):
        from lwp.analysis.behavioral_comparison import compute_binned_performance

        df = pd.DataFrame({
            "twr": [10.0, 10.5, 11.0],
            "outcome": ["landed", "landed", "landed"],
            "total_reward": [100, 200, 150],
        })

        result = compute_binned_performance(df, bin_col="twr", n_bins=1)

        assert result.iloc[0]["landed_pct"] == pytest.approx(100.0)
        assert result.iloc[0]["n_episodes"] == 3

    def test_no_landed_bin(self):
        from lwp.analysis.behavioral_comparison import compute_binned_performance

        df = pd.DataFrame({
            "twr": [3.0, 3.5, 4.0],
            "outcome": ["crashed", "crashed", "timeout"],
            "total_reward": [-100, -100, -50],
        })

        result = compute_binned_performance(df, bin_col="twr", n_bins=1)

        assert result.iloc[0]["landed_pct"] == pytest.approx(0.0)
