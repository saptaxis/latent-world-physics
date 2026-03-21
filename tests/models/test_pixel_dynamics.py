# tests/test_pixel_dynamics.py
"""Tests for LatentDynamicsModel."""
import torch
import pytest
from lwp.models.pixel_dynamics import LatentDynamicsModel


class TestLatentDynamicsModel:
    """Test GRU-based latent dynamics."""

    def test_single_step(self):
        """Single step produces next latent and hidden state."""
        model = LatentDynamicsModel(latent_dim=64, action_dim=2, hidden_size=256)
        z = torch.randn(4, 64)
        action = torch.randn(4, 2)
        z_next, hidden = model(z, action)
        assert z_next.shape == (4, 64)
        assert hidden.shape == (1, 4, 256)

    def test_single_step_with_hidden(self):
        """Passing hidden state from previous step."""
        model = LatentDynamicsModel(latent_dim=64, action_dim=2, hidden_size=256)
        z = torch.randn(4, 64)
        action = torch.randn(4, 2)
        _, hidden = model(z, action)
        z_next2, hidden2 = model(z, action, hidden)
        assert z_next2.shape == (4, 64)
        assert not torch.allclose(hidden, hidden2)

    def test_rollout(self):
        """Multi-step rollout produces sequence of latents."""
        model = LatentDynamicsModel(latent_dim=64, action_dim=2, hidden_size=256)
        z_start = torch.randn(4, 64)
        actions = torch.randn(4, 10, 2)
        z_seq, hidden = model.rollout(z_start, actions)
        assert z_seq.shape == (4, 11, 64)

    def test_predict_sequence_teacher_forced(self):
        """predict_sequence with full teacher forcing."""
        model = LatentDynamicsModel(latent_dim=64, action_dim=2, hidden_size=256)
        z_seq = torch.randn(4, 10, 64)
        actions = torch.randn(4, 10, 2)
        z_pred, hidden = model.predict_sequence(z_seq, actions, teacher_forcing=1.0)
        assert z_pred.shape == (4, 10, 64)

    def test_predict_sequence_no_teacher_forcing(self):
        """predict_sequence with no teacher forcing (autoregressive)."""
        model = LatentDynamicsModel(latent_dim=64, action_dim=2, hidden_size=256)
        z_seq = torch.randn(4, 10, 64)
        actions = torch.randn(4, 10, 2)
        z_pred, hidden = model.predict_sequence(z_seq, actions, teacher_forcing=0.0)
        assert z_pred.shape == (4, 10, 64)

    def test_initial_state(self):
        """initial_state creates zero hidden state."""
        model = LatentDynamicsModel(latent_dim=64, action_dim=2, hidden_size=256)
        hidden = model.initial_state(4, torch.device("cpu"))
        assert hidden.shape == (1, 4, 256)
        assert (hidden == 0).all()

    def test_init_hidden_alias(self):
        """init_hidden is preserved as alias for backward compat."""
        model = LatentDynamicsModel(latent_dim=64, action_dim=2, hidden_size=256)
        assert model.init_hidden(2, torch.device("cpu")).shape == (1, 2, 256)
