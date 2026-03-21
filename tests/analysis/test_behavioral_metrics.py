"""Tests for behavioral_metrics.py — per-episode action histograms.

Uses the same _make_fake_episode() helper from test_trajectory_metrics.py
to create synthetic .npz files.
"""

import numpy as np
import pytest

from tests.analysis.test_trajectory_metrics import _make_fake_episode


class TestComputeActionHistograms:
    """Test compute_action_histograms() on individual episodes."""

    def test_returns_expected_keys(self, tmp_path):
        from lwp.analysis.behavioral_metrics import compute_action_histograms

        npz_path = _make_fake_episode(str(tmp_path), outcome="landed")
        result = compute_action_histograms(npz_path)

        assert "main_thrust_counts" in result
        assert "main_thrust_edges" in result
        assert "side_thrust_counts" in result
        assert "side_thrust_edges" in result

    def test_histogram_shapes(self, tmp_path):
        from lwp.analysis.behavioral_metrics import compute_action_histograms

        n_bins = 50
        npz_path = _make_fake_episode(str(tmp_path), outcome="landed")
        result = compute_action_histograms(npz_path, n_bins=n_bins)

        assert result["main_thrust_counts"].shape == (n_bins,)
        assert result["main_thrust_edges"].shape == (n_bins + 1,)
        assert result["side_thrust_counts"].shape == (n_bins,)
        assert result["side_thrust_edges"].shape == (n_bins + 1,)

    def test_main_thrust_domain(self, tmp_path):
        """Main thrust bins should span [0, 1]."""
        from lwp.analysis.behavioral_metrics import compute_action_histograms

        npz_path = _make_fake_episode(str(tmp_path), outcome="landed")
        result = compute_action_histograms(npz_path)

        np.testing.assert_almost_equal(result["main_thrust_edges"][0], 0.0)
        np.testing.assert_almost_equal(result["main_thrust_edges"][-1], 1.0)

    def test_side_thrust_domain(self, tmp_path):
        """Side thrust bins should span [-1, 1]."""
        from lwp.analysis.behavioral_metrics import compute_action_histograms

        npz_path = _make_fake_episode(str(tmp_path), outcome="landed")
        result = compute_action_histograms(npz_path)

        np.testing.assert_almost_equal(result["side_thrust_edges"][0], -1.0)
        np.testing.assert_almost_equal(result["side_thrust_edges"][-1], 1.0)

    def test_counts_sum_to_episode_length(self, tmp_path):
        """Total histogram counts should equal the number of timesteps."""
        from lwp.analysis.behavioral_metrics import compute_action_histograms

        n_steps = 120
        npz_path = _make_fake_episode(str(tmp_path), n_steps=n_steps, outcome="crashed")
        result = compute_action_histograms(npz_path)

        assert result["main_thrust_counts"].sum() == n_steps
        assert result["side_thrust_counts"].sum() == n_steps

    def test_custom_bin_count(self, tmp_path):
        """Should respect the n_bins parameter."""
        from lwp.analysis.behavioral_metrics import compute_action_histograms

        npz_path = _make_fake_episode(str(tmp_path), outcome="landed")
        result = compute_action_histograms(npz_path, n_bins=25)

        assert result["main_thrust_counts"].shape == (25,)

    def test_very_short_episode(self, tmp_path):
        """Should handle episodes with very few steps (edge case)."""
        from lwp.analysis.behavioral_metrics import compute_action_histograms

        npz_path = _make_fake_episode(str(tmp_path), n_steps=3, outcome="crashed")
        result = compute_action_histograms(npz_path)

        assert result["main_thrust_counts"].sum() == 3


class TestComputeCollectionHistograms:
    """Test compute_collection_histograms() batch processing."""

    def test_returns_dict_with_correct_count(self, tmp_path):
        from lwp.analysis.behavioral_metrics import compute_collection_histograms

        for i in range(4):
            _make_fake_episode(str(tmp_path), episode_idx=i, seed=i)

        result = compute_collection_histograms(str(tmp_path), workers=1)

        assert len(result) == 4

    def test_parallel_matches_sequential(self, tmp_path):
        from lwp.analysis.behavioral_metrics import compute_collection_histograms

        for i in range(6):
            _make_fake_episode(str(tmp_path), episode_idx=i, seed=i)

        seq = compute_collection_histograms(str(tmp_path), workers=1)
        par = compute_collection_histograms(str(tmp_path), workers=4)

        # Same keys
        assert set(seq.keys()) == set(par.keys())

        # Same histogram counts for each episode
        for key in seq:
            np.testing.assert_array_equal(
                seq[key]["main_thrust_counts"],
                par[key]["main_thrust_counts"],
            )
