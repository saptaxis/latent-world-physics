import torch
from lwp.models.mlp import MLPModel
from lwp.training.rollout import (
    rollout_open_loop,
    rollout_teacher_forced,
    rollout_warmup_then_branch,
    rollout_scheduled_sampling,
)

STATE_DIM = 8
ACTION_DIM = 2
BATCH = 4
T = 10


def _make_model():
    return MLPModel(state_dim=STATE_DIM, action_dim=ACTION_DIM, hidden_dims=[32])


def test_open_loop_shapes():
    model = _make_model()
    s0 = torch.randn(BATCH, STATE_DIM)
    actions = torch.randn(BATCH, T, ACTION_DIM)
    states, deltas, ms = rollout_open_loop(model, s0, actions)
    assert states.shape == (BATCH, T + 1, STATE_DIM)
    assert deltas.shape == (BATCH, T, STATE_DIM)
    assert ms is None  # stateless model
    # First state should be s0
    assert torch.allclose(states[:, 0], s0)


def test_teacher_forced_shapes():
    model = _make_model()
    true_states = torch.randn(BATCH, T, STATE_DIM)
    actions = torch.randn(BATCH, T, ACTION_DIM)
    deltas, ms = rollout_teacher_forced(model, true_states, actions)
    assert deltas.shape == (BATCH, T, STATE_DIM)
    assert ms is None


def test_warmup_then_branch_shapes():
    model = _make_model()
    prefix_states = torch.randn(BATCH, 5, STATE_DIM)
    prefix_actions = torch.randn(BATCH, 5, ACTION_DIM)
    branch_actions = torch.randn(BATCH, T, ACTION_DIM)
    states, deltas, ms = rollout_warmup_then_branch(
        model, prefix_states, prefix_actions, branch_actions)
    assert states.shape == (BATCH, T + 1, STATE_DIM)
    assert deltas.shape == (BATCH, T, STATE_DIM)


def test_scheduled_sampling_shapes():
    model = _make_model()
    true_states = torch.randn(BATCH, T, STATE_DIM)
    actions = torch.randn(BATCH, T, ACTION_DIM)
    deltas, ms = rollout_scheduled_sampling(model, true_states, actions, sampling_prob=0.5)
    assert deltas.shape == (BATCH, T, STATE_DIM)


def test_open_loop_1step_matches_step():
    """Rollout of 1 step should match a single model.step() call."""
    model = _make_model()
    model.eval()
    s0 = torch.randn(BATCH, STATE_DIM)
    a = torch.randn(BATCH, 1, ACTION_DIM)
    with torch.no_grad():
        states, deltas, _ = rollout_open_loop(model, s0, a)
        delta_direct, _ = model.step(s0, a[:, 0])
    assert torch.allclose(deltas[:, 0], delta_direct, atol=1e-6)


def test_teacher_forced_sampling_prob_0_matches():
    """Scheduled sampling with prob=0 should match teacher forcing."""
    model = _make_model()
    model.eval()
    true_states = torch.randn(BATCH, T, STATE_DIM)
    actions = torch.randn(BATCH, T, ACTION_DIM)
    with torch.no_grad():
        deltas_tf, _ = rollout_teacher_forced(model, true_states, actions)
        deltas_ss, _ = rollout_scheduled_sampling(
            model, true_states, actions, sampling_prob=0.0)
    assert torch.allclose(deltas_tf, deltas_ss, atol=1e-6)


from lwp.models.gru import GRUModel


def _make_gru():
    return GRUModel(state_dim=STATE_DIM, action_dim=ACTION_DIM, hidden_dim=16)


def test_gru_open_loop_shapes():
    model = _make_gru()
    s0 = torch.randn(BATCH, STATE_DIM)
    actions = torch.randn(BATCH, T, ACTION_DIM)
    states, deltas, ms = rollout_open_loop(model, s0, actions)
    assert states.shape == (BATCH, T + 1, STATE_DIM)
    assert deltas.shape == (BATCH, T, STATE_DIM)
    assert ms is not None  # GRU returns hidden state
    assert ms.shape == (1, BATCH, 16)


def test_gru_teacher_forced_shapes():
    model = _make_gru()
    true_states = torch.randn(BATCH, T, STATE_DIM)
    actions = torch.randn(BATCH, T, ACTION_DIM)
    deltas, ms = rollout_teacher_forced(model, true_states, actions)
    assert deltas.shape == (BATCH, T, STATE_DIM)
    assert ms is not None


def test_gru_warmup_then_branch():
    model = _make_gru()
    prefix_states = torch.randn(BATCH, 5, STATE_DIM)
    prefix_actions = torch.randn(BATCH, 5, ACTION_DIM)
    branch_actions = torch.randn(BATCH, T, ACTION_DIM)
    states, deltas, ms = rollout_warmup_then_branch(
        model, prefix_states, prefix_actions, branch_actions)
    assert states.shape == (BATCH, T + 1, STATE_DIM)
    assert ms is not None


def test_gru_scheduled_sampling():
    model = _make_gru()
    true_states = torch.randn(BATCH, T, STATE_DIM)
    actions = torch.randn(BATCH, T, ACTION_DIM)
    deltas, ms = rollout_scheduled_sampling(model, true_states, actions, sampling_prob=0.5)
    assert deltas.shape == (BATCH, T, STATE_DIM)
    assert ms is not None


from lwp.models.rssm import RSSMModel
from lwp.models.rssm_state import RSSMState


def _make_rssm():
    return RSSMModel(state_dim=STATE_DIM, action_dim=ACTION_DIM,
                     deter_dim=32, stoch_dim=8, hidden_dim=16)


def test_rssm_open_loop_shapes():
    model = _make_rssm()
    s0 = torch.randn(BATCH, STATE_DIM)
    actions = torch.randn(BATCH, T, ACTION_DIM)
    states, deltas, ms = rollout_open_loop(model, s0, actions)
    assert states.shape == (BATCH, T + 1, STATE_DIM)
    assert deltas.shape == (BATCH, T, STATE_DIM)
    assert isinstance(ms, RSSMState)


def test_rssm_teacher_forced_shapes():
    model = _make_rssm()
    true_states = torch.randn(BATCH, T, STATE_DIM)
    actions = torch.randn(BATCH, T, ACTION_DIM)
    deltas, ms = rollout_teacher_forced(model, true_states, actions)
    assert deltas.shape == (BATCH, T, STATE_DIM)
    assert isinstance(ms, RSSMState)


def test_rssm_warmup_then_branch():
    model = _make_rssm()
    prefix_states = torch.randn(BATCH, 5, STATE_DIM)
    prefix_actions = torch.randn(BATCH, 5, ACTION_DIM)
    branch_actions = torch.randn(BATCH, T, ACTION_DIM)
    states, deltas, ms = rollout_warmup_then_branch(
        model, prefix_states, prefix_actions, branch_actions)
    assert states.shape == (BATCH, T + 1, STATE_DIM)
    assert isinstance(ms, RSSMState)
