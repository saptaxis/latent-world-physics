# tests/training/test_integration.py
import torch
import numpy as np
import pytest


class TestHybridStateUpdate:
    """Test the shared hybrid integration helper."""

    def test_zero_force_delta_integrates_positions(self):
        """With zero force deltas, positions still change from existing velocities."""
        from lwp.training.integration import hybrid_state_update

        # state: [x, y, vx, vy, angle, ang_vel]
        state = torch.tensor([[0.0, 0.0, 1.0, 2.0, 0.0, 0.5]])
        force_delta = torch.zeros(1, 3)  # [Δvx, Δvy, Δang_vel] = 0

        next_state = hybrid_state_update(state, force_delta, subsample=1)

        assert next_state.shape == (1, 6)
        # Velocities unchanged
        assert next_state[0, 2] == 1.0   # vx
        assert next_state[0, 3] == 2.0   # vy
        assert next_state[0, 5] == 0.5   # ang_vel
        # Positions integrated: x += vx_next * IC_X * subsample
        torch.testing.assert_close(next_state[0, 0], torch.tensor(1.0 * 0.01))
        torch.testing.assert_close(next_state[0, 1], torch.tensor(2.0 * 0.0225))
        torch.testing.assert_close(next_state[0, 4], torch.tensor(0.5 * 0.05))

    def test_nonzero_force_delta_updates_velocities_first(self):
        """Post-update velocity used for position integration (semi-implicit Euler)."""
        from lwp.training.integration import hybrid_state_update

        state = torch.tensor([[0.0, 0.0, 1.0, 2.0, 0.0, 0.5]])
        force_delta = torch.tensor([[0.1, -0.5, 0.2]])  # [Δvx, Δvy, Δang_vel]

        next_state = hybrid_state_update(state, force_delta, subsample=1)

        # Velocities updated first
        assert next_state[0, 2] == pytest.approx(1.1)     # vx + Δvx
        assert next_state[0, 3] == pytest.approx(1.5)     # vy + Δvy
        assert next_state[0, 5] == pytest.approx(0.7)     # ang_vel + Δang_vel
        # Positions use POST-update velocities
        assert next_state[0, 0] == pytest.approx(1.1 * 0.01)
        assert next_state[0, 1] == pytest.approx(1.5 * 0.0225)
        assert next_state[0, 4] == pytest.approx(0.7 * 0.05)

    def test_subsample_scales_integration(self):
        """With subsample=5, position integration constants are 5x larger."""
        from lwp.training.integration import hybrid_state_update

        state = torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
        force_delta = torch.zeros(1, 3)

        next_1 = hybrid_state_update(state, force_delta, subsample=1)
        next_5 = hybrid_state_update(state, force_delta, subsample=5)

        # Position change should be 5x larger with subsample=5
        torch.testing.assert_close(next_5[0, 0], next_1[0, 0] * 5)

    def test_batch_dimension(self):
        """Works with batched inputs."""
        from lwp.training.integration import hybrid_state_update

        state = torch.randn(8, 6)
        force_delta = torch.randn(8, 3)
        next_state = hybrid_state_update(state, force_delta)
        assert next_state.shape == (8, 6)

    def test_gradient_flows(self):
        """Gradients flow through the integration for backprop."""
        from lwp.training.integration import hybrid_state_update

        state = torch.randn(4, 6)
        force_delta = torch.randn(4, 3, requires_grad=True)
        next_state = hybrid_state_update(state, force_delta)
        next_state.sum().backward()
        assert force_delta.grad is not None
        assert force_delta.grad.shape == (4, 3)

    def test_full_delta_passthrough(self):
        """When delta_dim == state_dim (6D), acts as plain addition."""
        from lwp.training.integration import hybrid_state_update

        state = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
        delta = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]])
        next_state = hybrid_state_update(state, delta)
        expected = state + delta
        torch.testing.assert_close(next_state, expected)


class TestIntegrationConstants:
    """Verify integration constants match env derivation."""

    def test_constants_values(self):
        from lwp.training.integration import INTEGRATION_CONSTANTS
        assert INTEGRATION_CONSTANTS == {
            "IC_X": 0.01,
            "IC_Y": 0.0225,
            "IC_ANGLE": 0.05,
        }

    def test_force_target_indices(self):
        from lwp.training.integration import FORCE_TARGET_INDICES
        assert FORCE_TARGET_INDICES == [2, 3, 5]  # vx, vy, ang_vel
