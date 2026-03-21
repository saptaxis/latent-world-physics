"""RSSMState — structured model state for RSSM world models."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class RSSMState:
    """Structured state for RSSM models.

    Contains deterministic recurrent state, stochastic latent sample,
    and optional distribution parameters for KL computation.

    Attributes:
        deter: deterministic hidden state [batch, deter_dim]
        stoch: stochastic latent sample [batch, stoch_dim]
        prior_logits: prior distribution params [batch, stoch_dim] (for KL)
        posterior_logits: posterior distribution params [batch, stoch_dim] (for KL)
    """
    deter: torch.Tensor
    stoch: torch.Tensor
    prior_logits: torch.Tensor | None = None
    posterior_logits: torch.Tensor | None = None

    def features(self) -> torch.Tensor:
        """Concatenated feature vector [deter, stoch] for downstream use."""
        return torch.cat([self.deter, self.stoch], dim=-1)

    def detach(self) -> RSSMState:
        """Detach all tensors from computation graph."""
        return RSSMState(
            deter=self.deter.detach(),
            stoch=self.stoch.detach(),
            prior_logits=self.prior_logits.detach() if self.prior_logits is not None else None,
            posterior_logits=self.posterior_logits.detach() if self.posterior_logits is not None else None,
        )

    def to(self, device) -> RSSMState:
        """Move all tensors to device."""
        return RSSMState(
            deter=self.deter.to(device),
            stoch=self.stoch.to(device),
            prior_logits=self.prior_logits.to(device) if self.prior_logits is not None else None,
            posterior_logits=self.posterior_logits.to(device) if self.posterior_logits is not None else None,
        )
