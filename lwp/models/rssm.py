"""Level 5: RSSM — Recurrent State-Space Model with stochastic latent state."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from lwp.models.base import WorldModel
from lwp.models.rssm_state import RSSMState


def _build_mlp(in_dim: int, hidden_dims: list[int], out_dim: int) -> nn.Sequential:
    """Build a simple MLP with ReLU activations."""
    layers = []
    d = in_dim
    for h in hidden_dims:
        layers.append(nn.Linear(d, h))
        layers.append(nn.ReLU())
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class RSSMModel(WorldModel):
    """RSSM world model with deterministic + stochastic latent state.

    Architecture:
        Prior path (imagination):
            h_t = GRU(concat(prev_stoch, action), prev_deter)
            prior_logits = MLP(h_t)
            stoch_t ~ N(prior_mean, prior_std)

        Posterior path (observation):
            h_t = GRU(concat(prev_stoch, action), prev_deter)
            posterior_logits = MLP(concat(h_t, obs_encoded))
            stoch_t ~ N(post_mean, post_std)

        Decoder:
            delta = MLP(concat(h_t, stoch_t))

    step() uses posterior (has observation).
    imagine_step() uses prior (no observation).

    Args:
        state_dim: observation dimension
        action_dim: action dimension
        deter_dim: deterministic recurrent state dimension
        stoch_dim: stochastic latent dimension
        hidden_dim: MLP hidden layer size for prior/posterior/decoder networks
        encoder_dims: hidden layer sizes for obs encoder (empty = linear projection)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        deter_dim: int = 200,
        stoch_dim: int = 30,
        hidden_dim: int = 200,
        encoder_dims: list[int] = None,
        delta_dim: int = None,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim
        if delta_dim is None:
            delta_dim = state_dim

        # Input projection: [prev_stoch, action] -> deter_dim for GRU input
        self.input_proj = nn.Linear(stoch_dim + action_dim, deter_dim)

        # Recurrent core
        self.gru_cell = nn.GRUCell(deter_dim, deter_dim)

        # Prior: h_t -> (mean, std) for stochastic latent
        self.prior_net = _build_mlp(deter_dim, [hidden_dim], stoch_dim * 2)

        # Observation encoder
        if encoder_dims is None:
            encoder_dims = []
        if encoder_dims:
            self.obs_encoder = _build_mlp(state_dim, encoder_dims[:-1] if len(encoder_dims) > 1 else [], encoder_dims[-1])
            enc_out_dim = encoder_dims[-1]
        else:
            self.obs_encoder = nn.Identity()
            enc_out_dim = state_dim

        # Posterior: [h_t, obs_encoded] -> (mean, std) for stochastic latent
        self.posterior_net = _build_mlp(deter_dim + enc_out_dim, [hidden_dim], stoch_dim * 2)

        # Decoder: [h_t, stoch_t] -> delta
        self.decoder = _build_mlp(deter_dim + stoch_dim, [hidden_dim], delta_dim)

    def initial_state(self, batch_size: int, device=None):
        """Return zero RSSMState."""
        return RSSMState(
            deter=torch.zeros(batch_size, self.deter_dim, device=device),
            stoch=torch.zeros(batch_size, self.stoch_dim, device=device),
        )

    def _prior(self, prev_state: RSSMState, action: torch.Tensor):
        """Prior transition: predict next state from action without observation.

        Returns (new_deter, prior_logits, prior_sample).
        """
        x = torch.cat([prev_state.stoch, action], dim=-1)
        x = F.relu(self.input_proj(x))
        deter = self.gru_cell(x, prev_state.deter)

        prior_params = self.prior_net(deter)
        prior_mean, prior_log_std = prior_params.chunk(2, dim=-1)
        prior_std = F.softplus(prior_log_std) + 0.1  # min std for stability
        prior_sample = prior_mean + prior_std * torch.randn_like(prior_std)

        return deter, prior_params, prior_sample

    def _posterior(self, deter: torch.Tensor, obs: torch.Tensor):
        """Posterior: refine stochastic latent using observation.

        Returns (posterior_logits, posterior_sample).
        """
        obs_encoded = self.obs_encoder(obs)
        post_input = torch.cat([deter, obs_encoded], dim=-1)
        post_params = self.posterior_net(post_input)
        post_mean, post_log_std = post_params.chunk(2, dim=-1)
        post_std = F.softplus(post_log_std) + 0.1
        post_sample = post_mean + post_std * torch.randn_like(post_std)

        return post_params, post_sample

    def step(self, obs, action, model_state=None):
        """Observation step: uses posterior (obs-informed) path."""
        if model_state is None:
            model_state = self.initial_state(obs.shape[0], device=obs.device)

        deter, prior_logits, _ = self._prior(model_state, action)
        posterior_logits, post_sample = self._posterior(deter, obs)

        # Decode from posterior sample
        feat = torch.cat([deter, post_sample], dim=-1)
        delta = self.decoder(feat)

        new_state = RSSMState(
            deter=deter,
            stoch=post_sample,
            prior_logits=prior_logits,
            posterior_logits=posterior_logits,
        )
        return delta, new_state

    def imagine_step(self, action, model_state):
        """Imagination step: uses prior only (no observation).

        Not part of the base WorldModel interface. Used for
        RSSM-specific imagination rollouts.
        """
        deter, prior_logits, prior_sample = self._prior(model_state, action)

        feat = torch.cat([deter, prior_sample], dim=-1)
        delta = self.decoder(feat)

        new_state = RSSMState(
            deter=deter,
            stoch=prior_sample,
            prior_logits=prior_logits,
            posterior_logits=None,
        )
        return delta, new_state

    def kl_loss(self, model_state: RSSMState) -> torch.Tensor:
        """KL divergence between posterior and prior distributions.

        Both parameterized as diagonal Gaussians with softplus(log_std) + 0.1.
        """
        prior_mean, prior_log_std = model_state.prior_logits.chunk(2, dim=-1)
        post_mean, post_log_std = model_state.posterior_logits.chunk(2, dim=-1)

        prior_std = F.softplus(prior_log_std) + 0.1
        post_std = F.softplus(post_log_std) + 0.1

        # Analytic KL for diagonal Gaussians
        kl = (torch.log(prior_std / post_std)
              + (post_std.pow(2) + (post_mean - prior_mean).pow(2)) / (2 * prior_std.pow(2))
              - 0.5)

        return kl.sum(dim=-1).mean()  # sum over latent dims, mean over batch
