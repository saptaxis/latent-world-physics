# tests/wm/test_gt_model.py
"""Tests for GroundTruthModel — a 'model' that returns GT deltas."""
import numpy as np
import torch
import pytest
from lwp.wm.gt_model import GroundTruthModel, prepare_episodes_for_norm
from lwp.data.normalization import NormStats, compute_norm_stats


class TestGroundTruthModel:
    @pytest.fixture
    def simple_episodes(self):
        """Tiny episodes with non-degenerate variance on all 6 dims.

        Must have variance on every dim to avoid division-by-zero in
        normalization. Mimics a lander drifting with slight rotation.
        """
        states1 = np.array([
            [0.1, 1.0, 0.2, 0.0, 0.05, 0.1],
            [0.15, 0.9, 0.18, -0.5, 0.1, 0.08],
            [0.22, 0.7, 0.15, -1.0, 0.18, 0.05],
            [0.30, 0.4, 0.10, -1.5, 0.25, 0.02],
            [0.38, 0.0, 0.05, -2.0, 0.30, -0.01],
        ], dtype=np.float32)
        actions1 = np.array([
            [0.0, 0.0],  # no thrust
            [0.0, 0.3],  # slight side
            [0.8, 0.0],  # main thrust
            [0.0, -0.6], # side thrust other direction
        ], dtype=np.float32)
        # Raw episode format (as loaded from npz)
        return [
            {"states": states1, "actions": actions1},
        ]

    @pytest.fixture
    def norm_stats(self, simple_episodes):
        # Preprocess: compute deltas, convert to tensors for compute_norm_stats
        prepared = prepare_episodes_for_norm(simple_episodes)
        return compute_norm_stats(prepared)

    def test_step_returns_correct_delta(self, simple_episodes, norm_stats):
        """Model should return the actual delta from the episode."""
        model = GroundTruthModel(simple_episodes, norm_stats)
        # Query with the exact state + action from transition t=0
        state_raw = simple_episodes[0]["states"][0]
        action = simple_episodes[0]["actions"][0]
        state_norm = (torch.tensor(state_raw).unsqueeze(0) - norm_stats.state_mean) / norm_stats.state_std
        action_t = torch.tensor(action).unsqueeze(0)

        delta_norm, _ = model.step(state_norm, action_t, None)

        # Denormalize and check against actual delta
        delta_raw = (delta_norm * norm_stats.delta_std + norm_stats.delta_mean).squeeze(0).numpy()
        expected_delta = simple_episodes[0]["states"][1] - simple_episodes[0]["states"][0]
        np.testing.assert_allclose(delta_raw, expected_delta, atol=1e-4)

    def test_step_interface(self, simple_episodes, norm_stats):
        """Should implement WorldModel-compatible interface."""
        model = GroundTruthModel(simple_episodes, norm_stats)
        state = torch.randn(1, 6)
        action = torch.zeros(1, 2)
        delta, model_state = model.step(state, action, None)
        assert delta.shape == (1, 6)
        assert model_state is None

    def test_parameters_for_device_detection(self, simple_episodes, norm_stats):
        """Should have a dummy buffer so _get_model_device() works."""
        model = GroundTruthModel(simple_episodes, norm_stats)
        # _get_model_device falls back to CPU for no parameters, but
        # register_buffer ensures the model can report its device
        assert hasattr(model, '_device_dummy')

    def test_eval_mode(self, simple_episodes, norm_stats):
        """Should support .eval() call."""
        model = GroundTruthModel(simple_episodes, norm_stats)
        model.eval()  # should not raise

    def test_different_actions_return_different_deltas(self, simple_episodes, norm_stats):
        """NN matching includes actions — same state + different action = different delta."""
        # This test constructs two transitions from similar states but different actions
        states = np.array([
            [0.1, 0.5, 0.0, 0.0, 0.0, 0.0],  # same start
            [0.1, 0.5, 0.0, -0.5, 0.0, 0.0],  # after no thrust
            [0.1, 0.5, 0.0, 0.0, 0.0, 0.0],   # same start again
            [0.1, 0.5, 0.0, 0.5, 0.0, 0.0],   # after thrust (opposite delta_vy)
        ], dtype=np.float32)
        actions = np.array([
            [0.0, 0.0],  # no thrust
            [0.0, 0.0],
            [1.0, 0.0],  # main thrust
        ], dtype=np.float32)
        eps = [{"states": states, "actions": actions}]
        prepared = prepare_episodes_for_norm(eps)
        ns = compute_norm_stats(prepared)
        model = GroundTruthModel(eps, ns)

        # Query with no-thrust action
        s_norm = (torch.tensor(states[0]).unsqueeze(0) - ns.state_mean) / ns.state_std
        a_no = torch.tensor([[0.0, 0.0]])
        a_thrust = torch.tensor([[1.0, 0.0]])

        d_no, _ = model.step(s_norm, a_no, None)
        d_thrust, _ = model.step(s_norm, a_thrust, None)

        # Should return different deltas for different actions
        assert not torch.allclose(d_no, d_thrust, atol=1e-3)
