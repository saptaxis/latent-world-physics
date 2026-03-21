"""LatentRSSM — RSSM dynamics model operating in VAE latent space.

This is the dynamics backbone for the pixel-based world model.  It mirrors
the state-space RSSMModel architecture (prior / posterior / GRU core) but
differs in three key ways:

1. **Inputs are latent codes**, not raw states.  The obs_encoder is a simple
   linear projection from latent_dim to hidden_dim (the VAE encoder has
   already done the heavy lifting).

2. **Outputs are absolute z_next**, not deltas.  The decoder predicts the
   next latent code directly — there is no meaningful "current state + delta"
   identity in latent space because the latent topology is learned.

3. **forward() delegates to imagine_step()** (prior only) so the model
   can be used as a drop-in dreaming module by PixelWorldModel.  The
   posterior path is only used during training via step().
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from lwp.models.rssm_state import RSSMState


def _build_mlp(in_dim: int, hidden_dims: list[int], out_dim: int) -> nn.Sequential:
    """Build a simple MLP with ReLU activations between hidden layers."""
    layers: list[nn.Module] = []
    d = in_dim
    for h in hidden_dims:
        layers.append(nn.Linear(d, h))
        layers.append(nn.ReLU())
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class LatentRSSM(nn.Module):
    """RSSM dynamics model that operates on VAE latent codes.

    Architecture follows the standard RSSM pattern:

        Prior path (imagination / dreaming):
            h_t = GRU(proj(prev_stoch, action), prev_deter)
            prior_params = MLP(h_t)           -> (mean, log_std)
            stoch_t ~ N(mean, softplus(log_std) + 0.1)

        Posterior path (training with ground-truth observations):
            h_t = GRU(proj(prev_stoch, action), prev_deter)
            post_params = MLP(h_t, encode(z_obs))  -> (mean, log_std)
            stoch_t ~ N(mean, softplus(log_std) + 0.1)

        Decoder:
            z_next = MLP(h_t, stoch_t)        -> absolute latent code

    Args:
        latent_dim: dimension of the VAE latent code (input and output)
        action_dim: dimension of the action vector
        deter_dim: dimension of the deterministic GRU hidden state
        stoch_dim: dimension of the stochastic latent variable
        hidden_dim: width of hidden layers in prior/posterior/decoder MLPs
    """

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        deter_dim: int = 200,
        stoch_dim: int = 30,
        hidden_dim: int = 200,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim

        # --- Recurrent core ---
        # Input projection: [prev_stoch, action] -> deter_dim sized vector for GRU
        self.input_proj = nn.Linear(stoch_dim + action_dim, deter_dim)
        # GRU cell maintains the deterministic recurrent state h_t
        self.gru_cell = nn.GRUCell(deter_dim, deter_dim)

        # --- Prior network (imagination) ---
        # Maps deterministic state to stochastic distribution params (mean, log_std)
        self.prior_net = _build_mlp(deter_dim, [hidden_dim], stoch_dim * 2)

        # --- Observation encoder ---
        # Simple linear projection from VAE latent to hidden_dim.
        # The VAE encoder has already compressed pixels into a meaningful
        # latent code, so a single linear layer suffices here.
        self.obs_encoder = nn.Linear(latent_dim, hidden_dim)

        # --- Posterior network (training) ---
        # Combines deterministic state with encoded observation to produce
        # a more informed stochastic distribution than the prior alone.
        self.posterior_net = _build_mlp(deter_dim + hidden_dim, [hidden_dim], stoch_dim * 2)

        # --- Decoder ---
        # Maps combined [deter, stoch] features to an absolute next latent code.
        # Unlike the state-space RSSM which predicts deltas, we predict absolute
        # z_next because there is no meaningful additive structure in learned
        # latent spaces.
        self.decoder = _build_mlp(deter_dim + stoch_dim, [hidden_dim], latent_dim)

    # ------------------------------------------------------------------
    # State initialisation
    # ------------------------------------------------------------------

    def initial_state(self, batch_size: int, device=None) -> RSSMState:
        """Return a zero-initialized RSSMState for the start of a sequence."""
        return RSSMState(
            deter=torch.zeros(batch_size, self.deter_dim, device=device),
            stoch=torch.zeros(batch_size, self.stoch_dim, device=device),
        )

    # ------------------------------------------------------------------
    # Internal transition helpers
    # ------------------------------------------------------------------

    def _prior(self, prev_state: RSSMState, action: torch.Tensor):
        """Prior transition: advance the recurrent state and sample from
        the prior (no observation).

        Returns:
            deter: new deterministic state [B, deter_dim]
            prior_params: concatenated (mean, log_std) [B, stoch_dim * 2]
            prior_sample: sampled stochastic latent [B, stoch_dim]
        """
        # Concatenate previous stochastic sample with action and project
        x = torch.cat([prev_state.stoch, action], dim=-1)
        x = F.relu(self.input_proj(x))

        # GRU update: deterministic transition
        deter = self.gru_cell(x, prev_state.deter)

        # Prior distribution: predict mean and log_std from deterministic state
        prior_params = self.prior_net(deter)
        prior_mean, prior_log_std = prior_params.chunk(2, dim=-1)
        # softplus + min_std=0.1 prevents posterior collapse and ensures
        # non-degenerate distributions for stable KL computation
        prior_std = F.softplus(prior_log_std) + 0.1
        prior_sample = prior_mean + prior_std * torch.randn_like(prior_std)

        return deter, prior_params, prior_sample

    def _posterior(self, deter: torch.Tensor, z_obs: torch.Tensor):
        """Posterior: refine the stochastic latent using the observed VAE code.

        Returns:
            post_params: concatenated (mean, log_std) [B, stoch_dim * 2]
            post_sample: sampled stochastic latent [B, stoch_dim]
        """
        # Encode the VAE latent code into hidden_dim features
        obs_encoded = self.obs_encoder(z_obs)
        # Combine deterministic state with encoded observation
        post_input = torch.cat([deter, obs_encoded], dim=-1)

        # Posterior distribution: more informed than prior because it sees z_obs
        post_params = self.posterior_net(post_input)
        post_mean, post_log_std = post_params.chunk(2, dim=-1)
        post_std = F.softplus(post_log_std) + 0.1
        post_sample = post_mean + post_std * torch.randn_like(post_std)

        return post_params, post_sample

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def step(
        self,
        z_obs: torch.Tensor,
        action: torch.Tensor,
        model_state: RSSMState | None = None,
    ) -> tuple[torch.Tensor, RSSMState]:
        """Posterior step: advance state using both action and ground-truth
        observation (VAE latent code).  Used during training.

        Args:
            z_obs: encoded ground-truth observation [B, latent_dim]
            action: action taken [B, action_dim]
            model_state: previous RSSMState (auto-initialised if None)

        Returns:
            z_next: predicted next latent code [B, latent_dim] (absolute, not delta)
            new_state: updated RSSMState with both prior and posterior logits
        """
        if model_state is None:
            model_state = self.initial_state(z_obs.shape[0], device=z_obs.device)

        # Run the prior transition to get the new deterministic state
        deter, prior_logits, _ = self._prior(model_state, action)
        # Run the posterior to get an observation-informed stochastic sample
        posterior_logits, post_sample = self._posterior(deter, z_obs)

        # Decode absolute z_next from combined deterministic + stochastic features
        feat = torch.cat([deter, post_sample], dim=-1)
        z_next = self.decoder(feat)

        new_state = RSSMState(
            deter=deter,
            stoch=post_sample,
            prior_logits=prior_logits,
            posterior_logits=posterior_logits,
        )
        return z_next, new_state

    def imagine_step(
        self,
        action: torch.Tensor,
        model_state: RSSMState,
    ) -> tuple[torch.Tensor, RSSMState]:
        """Prior-only step: advance state using action alone (no observation).
        Used during imagination / dreaming.

        Args:
            action: action taken [B, action_dim]
            model_state: previous RSSMState

        Returns:
            z_next: predicted next latent code [B, latent_dim]
            new_state: updated RSSMState (posterior_logits is None)
        """
        deter, prior_logits, prior_sample = self._prior(model_state, action)

        # Decode absolute z_next from prior features (no observation available)
        feat = torch.cat([deter, prior_sample], dim=-1)
        z_next = self.decoder(feat)

        new_state = RSSMState(
            deter=deter,
            stoch=prior_sample,
            prior_logits=prior_logits,
            posterior_logits=None,  # no posterior without observation
        )
        return z_next, new_state

    def forward(
        self,
        z: torch.Tensor,
        action: torch.Tensor,
        model_state: RSSMState | None = None,
    ) -> tuple[torch.Tensor, RSSMState]:
        """Dreaming interface: delegates to imagine_step() (prior only).

        This matches the signature expected by PixelWorldModel.dream(),
        which calls forward(z, action, hidden) in a loop.  The z argument
        is not used as an observation — imagine_step relies solely on the
        recurrent state and action.

        Args:
            z: current latent code [B, latent_dim] (unused by imagine_step,
               but present for API compatibility with the dream loop)
            action: action to take [B, action_dim]
            model_state: previous RSSMState, or None to auto-initialise

        Returns:
            z_next: predicted next latent code [B, latent_dim]
            new_state: updated RSSMState
        """
        # Auto-initialise state if None — needed for the first call in
        # PixelWorldModel.dream() when C>1 (hidden starts as None)
        if model_state is None:
            model_state = self.initial_state(z.shape[0], device=z.device)

        return self.imagine_step(action, model_state)

    def rollout(
        self,
        z_start: torch.Tensor,
        actions: torch.Tensor,
        model_state: RSSMState | None = None,
    ) -> tuple[torch.Tensor, RSSMState]:
        """Unroll the prior dynamics for T steps, prepending the seed.

        Same contract as the GRU world model's rollout(): output has shape
        (B, T+1, latent_dim) with z_start as the first element.  This is
        important because multi_step_latent_loss expects the seed at index 0.

        Args:
            z_start: seed latent code [B, latent_dim]
            actions: action sequence [B, T, action_dim]
            model_state: initial RSSMState (auto-initialised if None)

        Returns:
            z_seq: predicted latent sequence [B, T+1, latent_dim]
            final_state: RSSMState after the last step
        """
        B, T, _ = actions.shape

        # Collect predictions, starting with the seed
        z_preds = [z_start]
        state = model_state
        z = z_start

        for t in range(T):
            # forward() handles None state on the first call
            z, state = self.forward(z, actions[:, t], state)
            z_preds.append(z)

        # Stack into (B, T+1, latent_dim)
        z_seq = torch.stack(z_preds, dim=1)
        return z_seq, state

    # ------------------------------------------------------------------
    # KL divergence
    # ------------------------------------------------------------------

    def kl_loss(self, model_state: RSSMState,
                free_bits: float = 0.0) -> torch.Tensor:
        """Analytic KL divergence: KL(posterior || prior).

        Both distributions are diagonal Gaussians parameterised as
        (mean, log_std) with std = softplus(log_std) + 0.1.

        Args:
            model_state: RSSMState with both prior and posterior logits.
            free_bits: minimum KL per stochastic dimension (in nats).
                When > 0, clamps per-dim KL to this floor, preventing
                posterior collapse where the stochastic branch goes unused
                because the deterministic GRU can predict everything alone.
                Dreamer uses free_bits=1.0. Set to 0 to disable.

        Raises ValueError if posterior_logits is None (prior-only state).
        """
        if model_state.posterior_logits is None:
            raise ValueError(
                "Cannot compute KL loss: posterior_logits is None. "
                "Use step() (posterior path) before calling kl_loss()."
            )

        # Unpack distribution parameters
        prior_mean, prior_log_std = model_state.prior_logits.chunk(2, dim=-1)
        post_mean, post_log_std = model_state.posterior_logits.chunk(2, dim=-1)

        # Reconstruct std with the same softplus + min_std transform used
        # during sampling — must match exactly for correct KL
        prior_std = F.softplus(prior_log_std) + 0.1
        post_std = F.softplus(post_log_std) + 0.1

        # Analytic KL for two diagonal Gaussians:
        # KL(q || p) = log(p_std/q_std) + (q_std^2 + (q_mean - p_mean)^2) / (2 * p_std^2) - 0.5
        kl_per_dim = (
            torch.log(prior_std / post_std)
            + (post_std.pow(2) + (post_mean - prior_mean).pow(2)) / (2 * prior_std.pow(2))
            - 0.5
        )

        # Free bits: clamp per-dim KL to a minimum floor. This forces the
        # stochastic branch to carry at least free_bits nats of information
        # per dimension — without it, the deterministic GRU can handle
        # everything and the posterior collapses onto the prior (KL → 0).
        if free_bits > 0:
            kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)

        # Sum over stochastic dims, mean over batch — standard RSSM convention
        return kl_per_dim.sum(dim=-1).mean()
