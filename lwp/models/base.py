"""WorldModel base class — the step() contract."""
from __future__ import annotations

import torch
import torch.nn as nn


class WorldModel(nn.Module):
    """Base class for all world models in the ladder.

    Core contract: a world model is a transition function.

        (obs_t, action_t, model_state_t) -> (delta_t, model_state_{t+1})

    For stateless models (linear, MLP), model_state is None.
    For recurrent models (GRU), model_state is a tensor.
    """

    def initial_state(self, batch_size: int, device=None):
        """Return model state for a fresh rollout. None for stateless models."""
        return None

    def step(self, obs: torch.Tensor, action: torch.Tensor,
             model_state=None) -> tuple[torch.Tensor, any]:
        """One transition step.

        Args:
            obs: [batch, state_dim]
            action: [batch, action_dim]
            model_state: recurrent/latent state (type depends on architecture)

        Returns:
            delta_pred: [batch, state_dim]
            next_model_state: same type as input model_state
        """
        raise NotImplementedError
