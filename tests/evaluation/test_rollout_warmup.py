"""Tests for recurrent model warmup in rollout evaluation."""
import torch
from lwp.evaluation.metrics.core import _rollout_raw_space
from lwp.data.normalization import NormStats


def test_rollout_raw_space_with_warmup():
    """Recurrent rollout should use warmup steps when provided."""

    class WarmupSensitiveModel:
        def step(self, obs, action, model_state=None):
            batch = obs.shape[0]
            dim = obs.shape[1]
            delta = torch.zeros(batch, dim)
            if model_state is not None:
                delta[:, 3] = -0.13  # correct gravity when warmed up
            else:
                delta[:, 3] = -0.5  # wrong gravity cold
            return delta, (model_state or 0) + 1

    model = WarmupSensitiveModel()
    ns = NormStats(
        state_mean=torch.zeros(6),
        state_std=torch.ones(6),
        delta_mean=torch.zeros(6),
        delta_std=torch.ones(6),
    )

    s0 = torch.zeros(1, 6)
    actions = torch.zeros(1, 10, 2)

    # Without warmup — cold start
    cold_states = _rollout_raw_space(model, s0, actions, ns)

    # With warmup
    warmup_states = torch.zeros(5, 6)
    warmup_actions = torch.zeros(5, 2)
    warm_states = _rollout_raw_space(
        model,
        s0,
        actions,
        ns,
        warmup_states=warmup_states,
        warmup_actions=warmup_actions,
    )

    # Cold uses -0.5 gravity, warm uses -0.13
    assert not torch.allclose(cold_states[:, :, 3], warm_states[:, :, 3])
    # Verify warm model uses correct gravity (after 1 step: vy should be -0.13)
    assert abs(warm_states[0, 1, 3].item() - (-0.13)) < 0.01
    # Cold should use -0.5
    assert abs(cold_states[0, 1, 3].item() - (-0.5)) < 0.01


def test_rollout_raw_space_no_warmup_unchanged():
    """Without warmup args, behavior is identical to before."""

    class SimpleModel:
        def step(self, obs, action, model_state=None):
            batch = obs.shape[0]
            dim = obs.shape[1]
            return torch.ones(batch, dim) * 0.1, None

    model = SimpleModel()
    ns = NormStats(
        state_mean=torch.zeros(4),
        state_std=torch.ones(4),
        delta_mean=torch.zeros(4),
        delta_std=torch.ones(4),
    )

    s0 = torch.zeros(1, 4)
    actions = torch.zeros(1, 5, 2)
    states = _rollout_raw_space(model, s0, actions, ns)
    # After 5 steps of +0.1 delta, state should be 0.5
    assert abs(states[0, 5, 0].item() - 0.5) < 0.01


def test_rollout_raw_space_3d_delta():
    """_rollout_raw_space should work with a model that outputs 3D deltas."""
    import pytest

    class Force3Model:
        def step(self, obs, action, model_state=None):
            batch = obs.shape[0]
            delta = torch.zeros(batch, 3)
            delta[:, 1] = -0.13  # gravity on Δvy
            return delta, model_state
        def initial_state(self, batch_size, device='cpu'):
            return None

    model = Force3Model()
    ns = NormStats(
        state_mean=torch.zeros(6), state_std=torch.ones(6),
        delta_mean=torch.zeros(3), delta_std=torch.ones(3),
    )
    s0 = torch.zeros(1, 6)
    actions = torch.zeros(1, 5, 2)
    result = _rollout_raw_space(model, s0, actions, ns)
    assert result.shape == (1, 6, 6)  # [batch, T+1, state_dim=6]
    # vy should decrease by 0.13 each step
    assert result[0, 1, 3].item() == pytest.approx(-0.13, abs=0.01)
