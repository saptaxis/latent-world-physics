# tests/test_pixel_losses.py
"""Tests for pixel-space loss functions."""
import torch
import pytest
from lwp.training.pixel_losses import vae_loss, latent_dynamics_loss, multi_step_latent_loss, latent_elbo_loss
from lwp.models.pixel_dynamics import LatentDynamicsModel
from lwp.models.pixel_rssm import LatentRSSM


class TestVAELoss:
    def test_zero_reconstruction_error(self):
        """Perfect reconstruction -> loss is just KL."""
        x = torch.rand(4, 1, 84, 84)
        recon = x.clone()
        mu = torch.zeros(4, 64)
        logvar = torch.zeros(4, 64)
        loss, recon_loss, kl_loss, state_loss = vae_loss(recon, x, mu, logvar, beta=1.0)
        assert recon_loss.item() < 1e-6
        assert kl_loss.item() < 1e-6
        assert state_loss.item() < 1e-6  # no state head → 0

    def test_beta_scales_kl(self):
        x = torch.rand(4, 1, 84, 84)
        recon = x + 0.1 * torch.randn_like(x)
        mu = torch.randn(4, 64)
        logvar = torch.randn(4, 64)
        loss_b1, _, kl1, _ = vae_loss(recon, x, mu, logvar, beta=1.0)
        loss_b01, _, kl01, _ = vae_loss(recon, x, mu, logvar, beta=0.1)
        assert torch.allclose(kl1, kl01)
        assert loss_b1 > loss_b01


class TestLatentDynamicsLoss:
    def test_mse_in_latent_space(self):
        z_pred = torch.randn(4, 10, 64)
        z_target = z_pred.clone()
        loss = latent_dynamics_loss(z_pred, z_target)
        assert loss.item() < 1e-6

    def test_nonzero_for_different(self):
        z_pred = torch.randn(4, 10, 64)
        z_target = torch.randn(4, 10, 64)
        loss = latent_dynamics_loss(z_pred, z_target)
        assert loss.item() > 0


class TestMultiStepLatentLoss:
    def test_output_is_scalar(self):
        """Multi-step loss returns a scalar tensor."""
        model = LatentDynamicsModel(latent_dim=8, action_dim=2, hidden_size=16)
        z_seq = torch.randn(2, 6, 8)  # B=2, T=6, D=8
        actions = torch.randn(2, 5, 2)  # T-1 actions
        loss = multi_step_latent_loss(model, z_seq, actions, k=4)
        assert loss.dim() == 0
        assert loss.item() > 0

    def test_gradient_flows_through_chain(self):
        """Gradients propagate from loss back through rollout to z_seq."""
        model = LatentDynamicsModel(latent_dim=8, action_dim=2, hidden_size=16)
        z_seq = torch.randn(2, 6, 8, requires_grad=True)
        actions = torch.randn(2, 5, 2)
        loss = multi_step_latent_loss(model, z_seq, actions, k=4)
        loss.backward()
        # z_seq[:, 0] is the seed — gradient must flow back to it
        assert z_seq.grad is not None
        assert z_seq.grad[:, 0].abs().sum() > 0

    def test_k_clamped_to_sequence_length(self):
        """k larger than sequence is silently clamped."""
        model = LatentDynamicsModel(latent_dim=8, action_dim=2, hidden_size=16)
        z_seq = torch.randn(2, 4, 8)  # T=4
        actions = torch.randn(2, 3, 2)  # T-1=3 actions
        # k=10 > T-1=3, should clamp to 3
        loss = multi_step_latent_loss(model, z_seq, actions, k=10)
        assert loss.dim() == 0


class TestMultiStepLatentLossKinWeight:
    """Tests for per-dim kinematic weighting in multi_step_latent_loss."""

    def test_kin_weight_1_matches_default(self):
        """kin_weight=1.0 produces identical loss to no weighting."""
        model = LatentDynamicsModel(latent_dim=8, action_dim=2, hidden_size=16)
        z_seq = torch.randn(2, 6, 8)
        actions = torch.randn(2, 5, 2)
        loss_default = multi_step_latent_loss(model, z_seq, actions, k=4)
        loss_w1 = multi_step_latent_loss(model, z_seq, actions, k=4,
                                          kin_weight=1.0, kin_dims=3)
        assert torch.allclose(loss_default, loss_w1, atol=1e-6)

    def test_kin_weight_changes_loss(self):
        """kin_weight != 1 produces different loss value."""
        model = LatentDynamicsModel(latent_dim=8, action_dim=2, hidden_size=16)
        z_seq = torch.randn(2, 6, 8)
        actions = torch.randn(2, 5, 2)
        loss_w1 = multi_step_latent_loss(model, z_seq, actions, k=4,
                                          kin_weight=1.0, kin_dims=3)
        loss_w10 = multi_step_latent_loss(model, z_seq, actions, k=4,
                                           kin_weight=10.0, kin_dims=3)
        assert not torch.allclose(loss_w1, loss_w10)

    def test_kin_weight_is_scalar(self):
        """Result is still a scalar tensor."""
        model = LatentDynamicsModel(latent_dim=8, action_dim=2, hidden_size=16)
        z_seq = torch.randn(2, 6, 8)
        actions = torch.randn(2, 5, 2)
        loss = multi_step_latent_loss(model, z_seq, actions, k=4,
                                       kin_weight=5.0, kin_dims=3)
        assert loss.dim() == 0
        assert loss.item() > 0

    def test_gradient_flows_with_kin_weight(self):
        """Gradients still propagate when kin_weight is active."""
        model = LatentDynamicsModel(latent_dim=8, action_dim=2, hidden_size=16)
        z_seq = torch.randn(2, 6, 8, requires_grad=True)
        actions = torch.randn(2, 5, 2)
        loss = multi_step_latent_loss(model, z_seq, actions, k=4,
                                       kin_weight=10.0, kin_dims=3)
        loss.backward()
        assert z_seq.grad is not None
        assert z_seq.grad[:, 0].abs().sum() > 0

    def test_kin_weight_with_teacher_forcing(self):
        """kin_weight works with scheduled sampling path too."""
        model = LatentDynamicsModel(latent_dim=8, action_dim=2, hidden_size=16)
        z_seq = torch.randn(2, 6, 8)
        actions = torch.randn(2, 5, 2)
        loss = multi_step_latent_loss(model, z_seq, actions, k=4,
                                       teacher_forcing=0.5,
                                       kin_weight=5.0, kin_dims=3)
        assert loss.dim() == 0
        assert loss.item() > 0


class TestLatentELBOLoss:
    def test_output_is_scalar(self):
        """ELBO loss returns a scalar tensor."""
        model = LatentRSSM(latent_dim=8, action_dim=2, deter_dim=16, stoch_dim=4, hidden_dim=16)
        z_seq = torch.randn(2, 6, 8)
        actions = torch.randn(2, 5, 2)
        loss = latent_elbo_loss(model, z_seq, actions, k=4, kl_weight=1.0)
        assert loss.dim() == 0
        assert loss.item() > 0

    def test_kl_weight_affects_loss(self):
        """Higher kl_weight increases total loss."""
        model = LatentRSSM(latent_dim=8, action_dim=2, deter_dim=16, stoch_dim=4, hidden_dim=16)
        z_seq = torch.randn(2, 6, 8)
        actions = torch.randn(2, 5, 2)
        loss_kl1 = latent_elbo_loss(model, z_seq, actions, k=4, kl_weight=1.0)
        loss_kl10 = latent_elbo_loss(model, z_seq, actions, k=4, kl_weight=10.0)
        # Higher KL weight should change the loss (unless KL is exactly 0)
        assert not torch.allclose(loss_kl1, loss_kl10)

    def test_returns_breakdown_when_requested(self):
        """Can return (total, recon, kl) breakdown."""
        model = LatentRSSM(latent_dim=8, action_dim=2, deter_dim=16, stoch_dim=4, hidden_dim=16)
        z_seq = torch.randn(2, 6, 8)
        actions = torch.randn(2, 5, 2)
        total, recon, kl = latent_elbo_loss(
            model, z_seq, actions, k=4, kl_weight=1.0, return_breakdown=True)
        assert total.dim() == 0
        assert recon.item() >= 0
        assert kl.item() >= 0
