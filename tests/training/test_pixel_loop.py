# tests/test_pixel_loop.py
"""Smoke tests for pixel training loops."""
import torch
import pytest
from lwp.models.pixel_vae import PixelVAE
from lwp.models.pixel_dynamics import LatentDynamicsModel
from lwp.training.callbacks import CallbackContext
from lwp.training.pixel_loop import pixel_vae_train_epoch, pixel_dynamics_train_epoch


class TestPixelVAETrainEpoch:
    def test_returns_loss_dict(self):
        """pixel_vae_train_epoch returns expected keys."""
        vae = PixelVAE(in_channels=1, latent_dim=16, frame_size=84,
                       channels=[8, 16, 32, 64])
        optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
        loader = [torch.rand(4, 1, 84, 84) for _ in range(3)]
        result = pixel_vae_train_epoch(vae, loader, optimizer, beta=0.0001)
        assert "train_loss" in result
        assert "recon_loss" in result
        assert "kl_loss" in result
        assert result["train_loss"] > 0

    def test_callbacks_dispatched(self):
        """Callbacks receive on_step calls."""
        from lwp.training.callbacks import TrainCallback
        calls = []
        class CountCallback(TrainCallback):
            def on_step(self, ctx):
                calls.append(ctx.global_step)
                return True

        vae = PixelVAE(in_channels=1, latent_dim=16, frame_size=84,
                       channels=[8, 16, 32, 64])
        optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
        loader = [torch.rand(4, 1, 84, 84) for _ in range(3)]
        ctx = CallbackContext(model=vae, optimizer=optimizer, writer=None,
                              global_step=0, epoch=0, run_dir="/tmp",
                              device="cpu", extras={})
        pixel_vae_train_epoch(vae, loader, optimizer, ctx=ctx,
                              callbacks=[CountCallback()])
        assert len(calls) == 3


class TestPixelDynamicsTrainEpoch:
    def test_returns_loss_dict(self):
        """pixel_dynamics_train_epoch returns expected keys."""
        vae = PixelVAE(in_channels=1, latent_dim=16, frame_size=84,
                       channels=[8, 16, 32, 64])
        vae.eval()
        for p in vae.parameters():
            p.requires_grad_(False)
        dynamics = LatentDynamicsModel(latent_dim=16, action_dim=2, hidden_size=32)
        optimizer = torch.optim.Adam(dynamics.parameters(), lr=1e-3)
        loader = [(torch.rand(2, 10, 1, 84, 84), torch.randn(2, 10, 2))
                   for _ in range(2)]
        result = pixel_dynamics_train_epoch(dynamics, vae, loader, optimizer)
        assert "train_loss" in result
        assert result["train_loss"] > 0

    def test_multi_step_latent_mode(self):
        """Training epoch with multi_step_latent mode returns loss dict."""
        vae = PixelVAE(in_channels=1, latent_dim=8, frame_size=64,
                       channels=[8, 16, 32, 64])
        vae.eval()
        for p in vae.parameters():
            p.requires_grad_(False)
        dynamics = LatentDynamicsModel(latent_dim=8, action_dim=2, hidden_size=16)
        optimizer = torch.optim.Adam(dynamics.parameters(), lr=1e-3)
        # T=10 frames, T-1=9 actions — standard convention for rollout modes
        frames = torch.rand(2, 10, 1, 64, 64)
        actions = torch.randn(2, 9, 2)
        loader = [(frames, actions)]
        result = pixel_dynamics_train_epoch(
            dynamics, vae, loader, optimizer,
            training_mode="multi_step_latent", rollout_k=4,
            device="cpu")
        assert "train_loss" in result
        assert result["train_loss"] > 0

    def test_multi_step_latent_with_kin_weight(self):
        """Training epoch with kin_weight > 1 runs without error."""
        vae = PixelVAE(in_channels=1, latent_dim=8, frame_size=64,
                       channels=[8, 16, 32, 64])
        vae.eval()
        for p in vae.parameters():
            p.requires_grad_(False)
        dynamics = LatentDynamicsModel(latent_dim=8, action_dim=2, hidden_size=16)
        optimizer = torch.optim.Adam(dynamics.parameters(), lr=1e-3)
        frames = torch.rand(2, 10, 1, 64, 64)
        actions = torch.randn(2, 9, 2)
        loader = [(frames, actions)]
        result = pixel_dynamics_train_epoch(
            dynamics, vae, loader, optimizer,
            training_mode="multi_step_latent", rollout_k=4,
            kin_weight=10.0, kin_dims=3,
            device="cpu")
        assert "train_loss" in result
        assert result["train_loss"] > 0

    def test_latent_elbo_mode(self):
        """Training epoch with latent_elbo mode returns loss dict with KL."""
        from lwp.models.pixel_rssm import LatentRSSM
        vae = PixelVAE(in_channels=1, latent_dim=8, frame_size=64,
                       channels=[8, 16, 32, 64])
        vae.eval()
        for p in vae.parameters():
            p.requires_grad_(False)
        dynamics = LatentRSSM(latent_dim=8, action_dim=2, deter_dim=16,
                              stoch_dim=4, hidden_dim=16)
        optimizer = torch.optim.Adam(dynamics.parameters(), lr=1e-3)
        # T=10 frames, T-1=9 actions
        frames = torch.rand(2, 10, 1, 64, 64)
        actions = torch.randn(2, 9, 2)
        loader = [(frames, actions)]
        result = pixel_dynamics_train_epoch(
            dynamics, vae, loader, optimizer,
            training_mode="latent_elbo", rollout_k=4, kl_weight=1.0,
            device="cpu")
        assert "train_loss" in result
        assert result["train_loss"] > 0
