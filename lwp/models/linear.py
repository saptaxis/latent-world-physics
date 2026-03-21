"""Level 0: Linear delta predictor."""
from __future__ import annotations

import torch
import torch.nn as nn

from lwp.models.base import WorldModel


class LinearModel(WorldModel):
    """Linear transition model: delta = W @ [obs, action] + b."""

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.linear = nn.Linear(state_dim + action_dim, state_dim)

    def step(self, obs, action, model_state=None):
        x = torch.cat([obs, action], dim=-1)
        delta = self.linear(x)
        return delta, None
