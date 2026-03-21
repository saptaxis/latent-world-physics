import torch
from lwp.models.rssm_state import RSSMState


def test_rssm_state_creation():
    state = RSSMState(
        deter=torch.randn(4, 200),
        stoch=torch.randn(4, 30),
    )
    assert state.deter.shape == (4, 200)
    assert state.stoch.shape == (4, 30)
    assert state.prior_logits is None
    assert state.posterior_logits is None


def test_rssm_state_with_distribution_params():
    state = RSSMState(
        deter=torch.randn(4, 200),
        stoch=torch.randn(4, 30),
        prior_logits=torch.randn(4, 30),
        posterior_logits=torch.randn(4, 30),
    )
    assert state.prior_logits is not None
    assert state.posterior_logits is not None


def test_rssm_state_detach():
    state = RSSMState(
        deter=torch.randn(4, 200, requires_grad=True),
        stoch=torch.randn(4, 30, requires_grad=True),
        prior_logits=torch.randn(4, 30, requires_grad=True),
        posterior_logits=torch.randn(4, 30, requires_grad=True),
    )
    detached = state.detach()
    assert not detached.deter.requires_grad
    assert not detached.stoch.requires_grad
    assert not detached.prior_logits.requires_grad
    assert not detached.posterior_logits.requires_grad


def test_rssm_state_to_device():
    state = RSSMState(
        deter=torch.randn(4, 200),
        stoch=torch.randn(4, 30),
    )
    moved = state.to("cpu")
    assert moved.deter.device == torch.device("cpu")
    assert moved.stoch.device == torch.device("cpu")


def test_rssm_state_concat():
    """Feature vector is [deter, stoch]."""
    state = RSSMState(
        deter=torch.randn(4, 200),
        stoch=torch.randn(4, 30),
    )
    feat = state.features()
    assert feat.shape == (4, 230)


from lwp.models.rssm import RSSMModel
from lwp.models.base import WorldModel


def test_rssm_output_shape():
    model = RSSMModel(state_dim=8, action_dim=2, deter_dim=64, stoch_dim=16,
                      hidden_dim=32, encoder_dims=[32])
    obs = torch.randn(4, 8)
    action = torch.randn(4, 2)
    delta, ms = model.step(obs, action)
    assert delta.shape == (4, 8)
    assert isinstance(ms, RSSMState)
    assert ms.deter.shape == (4, 64)
    assert ms.stoch.shape == (4, 16)


def test_rssm_is_world_model():
    model = RSSMModel(state_dim=8, action_dim=2, deter_dim=64, stoch_dim=16)
    assert isinstance(model, WorldModel)


def test_rssm_initial_state():
    model = RSSMModel(state_dim=8, action_dim=2, deter_dim=64, stoch_dim=16)
    ms = model.initial_state(batch_size=4)
    assert isinstance(ms, RSSMState)
    assert ms.deter.shape == (4, 64)
    assert ms.stoch.shape == (4, 16)
    assert (ms.deter == 0).all()
    assert (ms.stoch == 0).all()


def test_rssm_step_produces_posterior():
    """step() is the observation path — should have posterior_logits."""
    model = RSSMModel(state_dim=8, action_dim=2, deter_dim=64, stoch_dim=16)
    obs = torch.randn(4, 8)
    action = torch.randn(4, 2)
    _, ms = model.step(obs, action)
    assert ms.posterior_logits is not None
    assert ms.prior_logits is not None


def test_rssm_imagine_step():
    """imagine_step uses prior only — no observation needed."""
    model = RSSMModel(state_dim=8, action_dim=2, deter_dim=64, stoch_dim=16)
    ms = model.initial_state(batch_size=4)
    action = torch.randn(4, 2)
    delta, new_ms = model.imagine_step(action, ms)
    assert delta.shape == (4, 8)
    assert isinstance(new_ms, RSSMState)
    assert new_ms.prior_logits is not None
    assert new_ms.posterior_logits is None  # no observation


def test_rssm_hidden_state_evolves():
    model = RSSMModel(state_dim=8, action_dim=2, deter_dim=64, stoch_dim=16)
    obs = torch.randn(4, 8)
    action = torch.randn(4, 2)
    _, ms1 = model.step(obs, action)
    _, ms2 = model.step(obs, action, ms1)
    assert not torch.allclose(ms1.deter, ms2.deter)


def test_rssm_gradient_flows():
    model = RSSMModel(state_dim=8, action_dim=2, deter_dim=32, stoch_dim=8,
                      hidden_dim=16, encoder_dims=[16])
    obs = torch.randn(4, 8)
    action = torch.randn(4, 2)
    delta, ms = model.step(obs, action)
    delta2, ms2 = model.step(obs, action, ms)
    loss = delta2.pow(2).mean() + model.kl_loss(ms2)
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"


def test_rssm_kl_loss():
    """KL divergence between prior and posterior should be non-negative."""
    model = RSSMModel(state_dim=8, action_dim=2, deter_dim=64, stoch_dim=16)
    obs = torch.randn(4, 8)
    action = torch.randn(4, 2)
    _, ms = model.step(obs, action)
    kl = model.kl_loss(ms)
    assert kl.isfinite()
    assert kl.item() >= 0


def test_rssm_kl_loss_backward():
    model = RSSMModel(state_dim=8, action_dim=2, deter_dim=32, stoch_dim=8)
    obs = torch.randn(4, 8)
    action = torch.randn(4, 2)
    _, ms = model.step(obs, action)
    kl = model.kl_loss(ms)
    kl.backward()
    # Prior and posterior networks should get gradients
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in model.parameters())
    assert has_grad
