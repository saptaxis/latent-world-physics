"""Level 0 baseline: predict delta = 0 (copy previous state)."""
from __future__ import annotations

import torch

from lwp.models.base import WorldModel


class CopyStateModel(WorldModel):
    """Trivial baseline that always predicts zero delta.

    s_{t+1} = s_t + 0 = s_t

    Has no trainable parameters. Useful as a floor baseline for evaluation.
    """

    def __init__(self, state_dim: int, **kwargs):
        super().__init__()
        self.state_dim = state_dim

    def step(self, obs, action, model_state=None):
        delta = torch.zeros(obs.shape[0], self.state_dim,
                            device=obs.device, dtype=obs.dtype)
        return delta, None
