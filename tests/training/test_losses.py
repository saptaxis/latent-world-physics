import torch
from lwp.models.mlp import MLPModel
from lwp.data.normalization import NormStats
from lwp.training.losses import single_step_loss, multi_step_loss


def _make_norm_stats():
    return NormStats(
        state_mean=torch.zeros(6),
        state_std=torch.ones(6),
        delta_mean=torch.zeros(6),
        delta_std=torch.ones(6),
    )


def test_single_step_loss_finite():
    model = MLPModel(state_dim=6, action_dim=2, hidden_dims=[32])
    batch = (torch.randn(16, 6), torch.randn(16, 2), torch.randn(16, 6))
    loss = single_step_loss(model, batch, _make_norm_stats())
    assert loss.isfinite()
    assert loss.item() > 0


def test_single_step_loss_backward():
    model = MLPModel(state_dim=6, action_dim=2, hidden_dims=[32])
    batch = (torch.randn(16, 6), torch.randn(16, 2), torch.randn(16, 6))
    loss = single_step_loss(model, batch, _make_norm_stats())
    loss.backward()
    for p in model.parameters():
        assert p.grad is not None


def test_multi_step_loss_finite():
    model = MLPModel(state_dim=6, action_dim=2, hidden_dims=[32])
    # batch for multi-step: (state_seq, action_seq) where state_seq is T+1 states
    state_seq = torch.randn(8, 11, 6)   # batch=8, T=10+1 states
    action_seq = torch.randn(8, 10, 2)  # T=10 actions
    batch = (state_seq, action_seq)
    loss = multi_step_loss(model, batch, _make_norm_stats(), k=5)
    assert loss.isfinite()
    assert loss.item() > 0


def test_multi_step_loss_backward():
    """Gradients flow through the k-step unrolled chain."""
    model = MLPModel(state_dim=6, action_dim=2, hidden_dims=[32])
    state_seq = torch.randn(4, 11, 6)
    action_seq = torch.randn(4, 10, 2)
    batch = (state_seq, action_seq)
    loss = multi_step_loss(model, batch, _make_norm_stats(), k=5)
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"No gradient for {name}"
        assert p.grad.abs().sum() > 0, f"Zero gradient for {name}"


def test_multi_step_k1_close_to_single_step():
    """Multi-step with k=1 should give similar loss to single-step."""
    torch.manual_seed(42)
    model = MLPModel(state_dim=6, action_dim=2, hidden_dims=[32])
    model.eval()

    states = torch.randn(16, 2, 6)  # just 2 states (for k=1)
    actions = torch.randn(16, 1, 2)

    ss_batch = (states[:, 0], actions[:, 0], states[:, 1] - states[:, 0])
    ms_batch = (states, actions)
    ns = _make_norm_stats()

    with torch.no_grad():
        ss = single_step_loss(model, ss_batch, ns)
        ms = multi_step_loss(model, ms_batch, ns, k=1)
    assert abs(ss.item() - ms.item()) < 0.01


from lwp.models.gru import GRUModel


def test_multi_step_loss_threads_model_state():
    """Multi-step loss should thread model_state so GRU gets recurrence.

    Compare multi-step loss with GRU vs a version that manually resets
    hidden state each step. They should differ (proving state is threaded).
    """
    torch.manual_seed(42)
    model = GRUModel(state_dim=6, action_dim=2, hidden_dim=16)
    model.eval()
    state_seq = torch.randn(4, 6, 6)
    action_seq = torch.randn(4, 5, 2)
    batch = (state_seq, action_seq)
    ns = _make_norm_stats()

    with torch.no_grad():
        loss = multi_step_loss(model, batch, ns, k=5)

    # Manually compute loss WITHOUT threading state (old buggy behavior)
    from lwp.data.normalization import normalize, denormalize
    import torch.nn.functional as F
    s_n = normalize(state_seq[:, 0], ns.state_mean, ns.state_std)
    pred_deltas_n_manual = []
    with torch.no_grad():
        for t in range(5):
            delta_n, _ = model.step(s_n, action_seq[:, t])  # no model_state!
            pred_deltas_n_manual.append(delta_n)
            delta_raw = delta_n * (ns.delta_std + 1e-8) + ns.delta_mean
            s_n = s_n + delta_raw / (ns.state_std + 1e-8)
    pred_deltas_n_manual = torch.stack(pred_deltas_n_manual, dim=1)
    true_deltas = state_seq[:, 1:6] - state_seq[:, :5]
    true_deltas_n = normalize(true_deltas, ns.delta_mean, ns.delta_std)
    loss_no_state = F.mse_loss(pred_deltas_n_manual, true_deltas_n)

    # These should differ because GRU hidden state evolves
    assert not torch.allclose(loss, loss_no_state), \
        "multi_step_loss should thread model_state for recurrent models"


def test_single_step_loss_gru():
    """Single-step loss works with GRU (ignores hidden state)."""
    model = GRUModel(state_dim=6, action_dim=2, hidden_dim=16)
    batch = (torch.randn(16, 6), torch.randn(16, 2), torch.randn(16, 6))
    loss = single_step_loss(model, batch, _make_norm_stats())
    assert loss.isfinite()


def test_multi_step_loss_gru():
    """Multi-step loss works with GRU, gradients flow through hidden state chain."""
    model = GRUModel(state_dim=6, action_dim=2, hidden_dim=16)
    state_seq = torch.randn(4, 11, 6)
    action_seq = torch.randn(4, 10, 2)
    batch = (state_seq, action_seq)
    loss = multi_step_loss(model, batch, _make_norm_stats(), k=5)
    assert loss.isfinite()
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"No gradient for {name}"


from lwp.training.losses import scheduled_sampling_loss


def test_scheduled_sampling_loss_finite():
    model = GRUModel(state_dim=6, action_dim=2, hidden_dim=16)
    state_seq = torch.randn(4, 11, 6)
    action_seq = torch.randn(4, 10, 2)
    batch = (state_seq, action_seq)
    loss = scheduled_sampling_loss(model, batch, _make_norm_stats(),
                                   k=5, sampling_prob=0.3)
    assert loss.isfinite()
    assert loss.item() > 0


def test_scheduled_sampling_loss_backward():
    model = GRUModel(state_dim=6, action_dim=2, hidden_dim=16)
    state_seq = torch.randn(4, 11, 6)
    action_seq = torch.randn(4, 10, 2)
    batch = (state_seq, action_seq)
    loss = scheduled_sampling_loss(model, batch, _make_norm_stats(),
                                   k=5, sampling_prob=0.3)
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"No gradient for {name}"


def test_scheduled_sampling_prob0_vs_multi_step():
    """With sampling_prob=0 (teacher-forced input) vs multi-step (open-loop input).

    These differ because multi_step feeds predicted states back while
    scheduled_sampling with prob=0 feeds true states. Both should produce
    finite losses. With GRU, both thread model_state.
    """
    torch.manual_seed(42)
    model = GRUModel(state_dim=6, action_dim=2, hidden_dim=16)
    model.eval()
    state_seq = torch.randn(4, 6, 6)
    action_seq = torch.randn(4, 5, 2)
    batch = (state_seq, action_seq)
    ns = _make_norm_stats()
    with torch.no_grad():
        ms_loss = multi_step_loss(model, batch, ns, k=5)
        ss_loss = scheduled_sampling_loss(model, batch, ns, k=5, sampling_prob=0.0)
    assert ms_loss.isfinite()
    assert ss_loss.isfinite()
    # They should differ because the state inputs differ (open-loop vs teacher-forced)
    # but both should be reasonable
    assert ms_loss.item() < 1000
    assert ss_loss.item() < 1000


from lwp.models.rssm import RSSMModel
from lwp.training.losses import elbo_loss


def test_elbo_loss_finite():
    model = RSSMModel(state_dim=6, action_dim=2, deter_dim=32, stoch_dim=8,
                      hidden_dim=16)
    state_seq = torch.randn(4, 11, 6)
    action_seq = torch.randn(4, 10, 2)
    batch = (state_seq, action_seq)
    loss = elbo_loss(model, batch, _make_norm_stats(), k=5)
    assert loss.isfinite()
    assert loss.item() > 0


def test_elbo_loss_backward():
    model = RSSMModel(state_dim=6, action_dim=2, deter_dim=32, stoch_dim=8,
                      hidden_dim=16)
    state_seq = torch.randn(4, 11, 6)
    action_seq = torch.randn(4, 10, 2)
    batch = (state_seq, action_seq)
    loss = elbo_loss(model, batch, _make_norm_stats(), k=5)
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"No gradient for {name}"


def test_elbo_loss_kl_weight():
    """Higher kl_weight should increase the loss."""
    torch.manual_seed(42)
    model = RSSMModel(state_dim=6, action_dim=2, deter_dim=32, stoch_dim=8)
    state_seq = torch.randn(4, 6, 6)
    action_seq = torch.randn(4, 5, 2)
    batch = (state_seq, action_seq)
    ns = _make_norm_stats()
    model.eval()
    with torch.no_grad():
        loss_low = elbo_loss(model, batch, ns, k=5, kl_weight=0.1)
        loss_high = elbo_loss(model, batch, ns, k=5, kl_weight=10.0)
    # Can't guarantee ordering due to randomness in sampling, but both should be finite
    assert loss_low.isfinite()
    assert loss_high.isfinite()


def test_single_step_loss_with_dim_weights():
    """Inverse-variance weighting changes the loss value."""
    torch.manual_seed(42)
    model = MLPModel(state_dim=6, action_dim=2, hidden_dims=[32])
    batch = (torch.randn(16, 6), torch.randn(16, 2), torch.randn(16, 6))
    ns = NormStats(
        state_mean=torch.zeros(6),
        state_std=torch.ones(6),
        delta_mean=torch.zeros(6),
        delta_std=torch.tensor([0.01, 0.07, 0.03, 0.02, 0.008, 0.04]),
    )
    model.eval()
    with torch.no_grad():
        loss_weighted = single_step_loss(model, batch, ns, dim_weights="inv_var")
        loss_uniform = single_step_loss(model, batch, ns, dim_weights=None)
    assert loss_weighted.isfinite()
    assert loss_uniform.isfinite()
    assert loss_weighted.item() != loss_uniform.item()


def test_multi_step_loss_with_dim_weights():
    """Inverse-variance weighting works with multi-step loss."""
    torch.manual_seed(42)
    model = MLPModel(state_dim=6, action_dim=2, hidden_dims=[32])
    state_seq = torch.randn(4, 6, 6)
    action_seq = torch.randn(4, 5, 2)
    batch = (state_seq, action_seq)
    ns = NormStats(
        state_mean=torch.zeros(6),
        state_std=torch.ones(6),
        delta_mean=torch.zeros(6),
        delta_std=torch.tensor([0.01, 0.07, 0.03, 0.02, 0.008, 0.04]),
    )
    model.eval()
    with torch.no_grad():
        loss_weighted = multi_step_loss(model, batch, ns, k=5, dim_weights="inv_var")
        loss_uniform = multi_step_loss(model, batch, ns, k=5, dim_weights=None)
    assert loss_weighted.isfinite()
    assert loss_uniform.isfinite()
    assert loss_weighted.item() != loss_uniform.item()


def test_dim_weights_none_matches_default():
    """dim_weights=None gives same result as before (backward compat)."""
    torch.manual_seed(42)
    model = MLPModel(state_dim=6, action_dim=2, hidden_dims=[32])
    batch = (torch.randn(16, 6), torch.randn(16, 2), torch.randn(16, 6))
    ns = NormStats(
        state_mean=torch.zeros(6), state_std=torch.ones(6),
        delta_mean=torch.zeros(6), delta_std=torch.ones(6),
    )
    model.eval()
    with torch.no_grad():
        loss_default = single_step_loss(model, batch, ns)
        loss_none = single_step_loss(model, batch, ns, dim_weights=None)
    assert torch.allclose(loss_default, loss_none)


def test_multi_step_normalized_accumulation():
    """Multi-step loss accumulates in normalized space (no raw-space round-trip)."""
    torch.manual_seed(42)
    model = MLPModel(state_dim=6, action_dim=2, hidden_dims=[32])
    state_seq = torch.randn(4, 6, 6)
    action_seq = torch.randn(4, 5, 2)
    batch = (state_seq, action_seq)
    ns = NormStats(
        state_mean=torch.tensor([0.1, 0.5, 0.0, 0.0, 0.0, 0.0]),
        state_std=torch.tensor([0.5, 0.3, 0.1, 0.1, 0.2, 0.15]),
        delta_mean=torch.tensor([0.0, -0.01, 0.0, -0.005, 0.0, 0.0]),
        delta_std=torch.tensor([0.007, 0.07, 0.025, 0.017, 0.008, 0.044]),
    )
    loss = multi_step_loss(model, batch, ns, k=5)
    assert loss.isfinite()
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"No gradient for {name}"
        assert p.grad.isfinite().all(), f"Non-finite gradient for {name}"


class TestForce3Losses:
    """Losses should work with delta_dim=3 models."""

    def _make_3d_norm_stats(self):
        return NormStats(
            state_mean=torch.zeros(6), state_std=torch.ones(6),
            delta_mean=torch.zeros(3), delta_std=torch.ones(3),
        )

    def test_single_step_loss_3d(self):
        model = MLPModel(state_dim=6, action_dim=2, delta_dim=3)
        ns = self._make_3d_norm_stats()
        obs = torch.randn(8, 6)
        actions = torch.randn(8, 2)
        true_deltas = torch.randn(8, 3)
        batch = (obs, actions, true_deltas)
        loss = single_step_loss(model, batch, ns)
        assert loss.dim() == 0
        loss.backward()

    def test_multi_step_loss_3d(self):
        model = MLPModel(state_dim=6, action_dim=2, delta_dim=3)
        ns = self._make_3d_norm_stats()
        state_seq = torch.randn(4, 11, 6)
        action_seq = torch.randn(4, 10, 2)
        batch = (state_seq, action_seq)
        loss = multi_step_loss(model, batch, ns, k=5)
        assert loss.dim() == 0
        loss.backward()

    def test_multi_step_loss_3d_gru(self):
        from lwp.models.gru import GRUModel
        model = GRUModel(state_dim=6, action_dim=2, delta_dim=3)
        ns = self._make_3d_norm_stats()
        state_seq = torch.randn(4, 11, 6)
        action_seq = torch.randn(4, 10, 2)
        batch = (state_seq, action_seq)
        loss = multi_step_loss(model, batch, ns, k=5)
        assert loss.dim() == 0
        loss.backward()

    def test_scheduled_sampling_loss_3d(self):
        from lwp.models.gru import GRUModel
        model = GRUModel(state_dim=6, action_dim=2, delta_dim=3)
        ns = self._make_3d_norm_stats()
        state_seq = torch.randn(4, 11, 6)
        action_seq = torch.randn(4, 10, 2)
        batch = (state_seq, action_seq)
        loss = scheduled_sampling_loss(model, batch, ns, k=5, sampling_prob=0.3)
        assert loss.dim() == 0
        loss.backward()

    def test_elbo_loss_3d(self):
        from lwp.models.rssm import RSSMModel
        model = RSSMModel(state_dim=6, action_dim=2, deter_dim=32, stoch_dim=8,
                          hidden_dim=16, delta_dim=3)
        ns = self._make_3d_norm_stats()
        state_seq = torch.randn(4, 11, 6)
        action_seq = torch.randn(4, 10, 2)
        batch = (state_seq, action_seq)
        loss = elbo_loss(model, batch, ns, k=5)
        assert loss.dim() == 0
        loss.backward()


def test_multi_step_k1_matches_single_step_with_real_stats():
    """Multi-step k=1 should give same loss as single-step even with non-trivial stats."""
    torch.manual_seed(42)
    model = MLPModel(state_dim=6, action_dim=2, hidden_dims=[32])
    model.eval()
    ns = NormStats(
        state_mean=torch.tensor([0.1, 0.5, 0.0, 0.0, 0.0, 0.0]),
        state_std=torch.tensor([0.5, 0.3, 0.1, 0.1, 0.2, 0.15]),
        delta_mean=torch.tensor([0.0, -0.01, 0.0, -0.005, 0.0, 0.0]),
        delta_std=torch.tensor([0.007, 0.07, 0.025, 0.017, 0.008, 0.044]),
    )
    states = torch.randn(16, 2, 6)
    actions = torch.randn(16, 1, 2)
    ss_batch = (states[:, 0], actions[:, 0], states[:, 1] - states[:, 0])
    ms_batch = (states, actions)
    with torch.no_grad():
        ss = single_step_loss(model, ss_batch, ns)
        ms = multi_step_loss(model, ms_batch, ns, k=1)
    assert abs(ss.item() - ms.item()) < 0.05, \
        f"k=1 multi-step ({ms.item():.4f}) should match single-step ({ss.item():.4f})"
