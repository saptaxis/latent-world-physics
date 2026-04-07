"""Levels 1–2: Stateless MLP delta predictor."""
from __future__ import annotations

import torch
import torch.nn as nn

from lwp.models.base import WorldModel


class MLPModel(WorldModel):
    """MLP transition model: delta = MLP([obs, action]).

    Args:
        state_dim: observation dimension
        action_dim: action dimension
        hidden_dims: list of hidden layer sizes
        activation: "relu" or "tanh"
        dropout: dropout probability (0.0 = no dropout)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list[int] = None,
        activation: str = "relu",
        dropout: float = 0.0,
        delta_dim: int = None,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256]
        if delta_dim is None:
            delta_dim = state_dim
        act_fn = {"relu": nn.ReLU, "tanh": nn.Tanh}[activation]
        layers = []
        in_dim = state_dim + action_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(act_fn())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, delta_dim))
        self.net = nn.Sequential(*layers)

    def step(self, obs, action, model_state=None):
        x = torch.cat([obs, action], dim=-1)
        delta = self.net(x)
        return delta, None
