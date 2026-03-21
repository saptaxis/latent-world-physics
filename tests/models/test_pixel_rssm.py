"""Tests for LatentRSSM — latent-space RSSM dynamics model."""
import torch
import pytest
from lwp.models.pixel_rssm import LatentRSSM
from lwp.models.rssm_state import RSSMState


class TestLatentRSSM:
    @pytest.fixture
    def model(self):
        return LatentRSSM(latent_dim=16, action_dim=2, deter_dim=32, stoch_dim=8, hidden_dim=32)

    def test_initial_state_shapes(self, model):
        """initial_state returns RSSMState with correct shapes."""
        state = model.initial_state(4, torch.device("cpu"))
        assert isinstance(state, RSSMState)
        assert state.deter.shape == (4, 32)
        assert state.stoch.shape == (4, 8)

    def test_step_returns_z_next_and_state(self, model):
        """step() (posterior) returns absolute z_next, not a delta."""
        z_obs = torch.randn(4, 16)  # observation: current encoded GT z
        action = torch.randn(4, 2)
        z_next, state = model.step(z_obs, action)
        assert z_next.shape == (4, 16)  # latent_dim, absolute z
        assert isinstance(state, RSSMState)
        # prior_logits and posterior_logits are (B, stoch_dim * 2) — mean + log_std concatenated
        assert state.prior_logits is not None
        assert state.prior_logits.shape == (4, 8 * 2)
        assert state.posterior_logits is not None
        assert state.posterior_logits.shape == (4, 8 * 2)

    def test_imagine_step_uses_prior_only(self, model):
        """imagine_step() (prior) has no posterior logits."""
        action = torch.randn(4, 2)
        state = model.initial_state(4, torch.device("cpu"))
        z_next, new_state = model.imagine_step(action, state)
        assert z_next.shape == (4, 16)
        assert new_state.posterior_logits is None
        assert new_state.prior_logits is not None
        assert new_state.prior_logits.shape == (4, 8 * 2)

    def test_forward_delegates_to_imagine_step(self, model):
        """forward() uses prior (imagine_step) for dreaming."""
        z = torch.randn(4, 16)
        action = torch.randn(4, 2)
        state = model.initial_state(4, torch.device("cpu"))
        z_next_fwd, state_fwd = model.forward(z, action, state)
        # forward should produce the same as imagine_step
        assert z_next_fwd.shape == (4, 16)
        assert state_fwd.posterior_logits is None

    def test_forward_with_none_state(self, model):
        """forward() auto-initializes state when None (needed for dream C>1 path)."""
        z = torch.randn(4, 16)
        action = torch.randn(4, 2)
        z_next, state = model.forward(z, action, None)
        assert z_next.shape == (4, 16)
        assert isinstance(state, RSSMState)

    def test_rollout_shape_and_seed(self, model):
        """rollout produces (B, T+1, latent_dim) with seed as first element."""
        z_start = torch.randn(4, 16)
        actions = torch.randn(4, 10, 2)
        z_seq, final_state = model.rollout(z_start, actions)
        assert z_seq.shape == (4, 11, 16)
        assert isinstance(final_state, RSSMState)
        # First element must be the seed (multi_step_latent_loss depends on this)
        assert torch.allclose(z_seq[:, 0], z_start)

    def test_kl_loss_zero_when_prior_equals_posterior(self, model):
        """KL is ~0 when prior and posterior distributions match."""
        # Construct a state where prior_logits == posterior_logits
        fake_logits = torch.zeros(4, 8 * 2)  # mean=0, log_std=0 for both
        state = RSSMState(
            deter=torch.zeros(4, 32),
            stoch=torch.zeros(4, 8),
            prior_logits=fake_logits,
            posterior_logits=fake_logits.clone(),
        )
        kl = model.kl_loss(state)
        assert kl.item() < 1e-5

    def test_kl_loss_is_scalar(self, model):
        """kl_loss returns a positive scalar after a posterior step."""
        z_obs = torch.randn(4, 16)
        action = torch.randn(4, 2)
        _, state = model.step(z_obs, action)
        kl = model.kl_loss(state)
        assert kl.dim() == 0
        assert kl.item() >= 0

    def test_kl_loss_requires_posterior(self, model):
        """kl_loss raises error if posterior_logits is None (prior-only state)."""
        action = torch.randn(4, 2)
        state = model.initial_state(4, torch.device("cpu"))
        _, prior_state = model.imagine_step(action, state)
        with pytest.raises(ValueError):
            model.kl_loss(prior_state)

    def test_step_and_imagine_differ(self, model):
        """Posterior step (with observation) differs from prior step."""
        torch.manual_seed(42)
        z_obs = torch.randn(4, 16)
        action = torch.randn(4, 2)
        state = model.initial_state(4, torch.device("cpu"))

        # Posterior path
        z_post, _ = model.step(z_obs, action, state)
        # Prior path (same action, same initial state, no observation)
        torch.manual_seed(42)
        z_prior, _ = model.imagine_step(action, state)
        # They should differ because posterior uses the observation
        assert not torch.allclose(z_post, z_prior, atol=1e-4)
