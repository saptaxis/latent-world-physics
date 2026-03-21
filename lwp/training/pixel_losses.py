# training/pixel_losses.py
"""Loss functions for pixel world model training.

Separate from state-space losses (training/losses.py) because pixel
models have no delta normalization, no NormStats, and use different
loss structures (reconstruction + KL for VAE, latent MSE for dynamics).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _foreground_weight_mask(target: torch.Tensor, fg_weight: float,
                            lo: float = 0.04, hi: float = 0.78) -> torch.Tensor:
    """Compute per-pixel weight mask from target frame.

    Pixels in the foreground intensity band (between black sky and white
    terrain) get fg_weight multiplier. Background pixels get weight 1.

    For grayscale [0,1]: black sky < lo (~10/255), white terrain > hi (~200/255).
    Everything in between is lander, legs, flames, flags.

    Args:
        target: (B, C, H, W) in [0, 1]
        fg_weight: multiplier for foreground pixels
        lo: lower intensity threshold (below = sky)
        hi: upper intensity threshold (above = terrain)

    Returns:
        (B, C, H, W) weight tensor, same shape as target
    """
    # Foreground: pixels with intensity in (lo, hi) — the lander, legs,
    # flames, and flags sit in this band. Sky is near-black (< lo) and
    # terrain is near-white (> hi), both easy for the VAE to reconstruct.
    fg_mask = (target > lo) & (target < hi)
    # Start with uniform weight 1; only upweight foreground pixels
    weights = torch.ones_like(target)
    weights[fg_mask] = fg_weight
    return weights


def vae_loss(recon: torch.Tensor, target: torch.Tensor,
             mu: torch.Tensor, logvar: torch.Tensor,
             beta: float = 0.0001, fg_weight: float = 1.0,
             state_pred: torch.Tensor | None = None,
             state_target: torch.Tensor | None = None,
             state_weight: float = 0.0,
             ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """VAE loss: foreground-weighted reconstruction + KL + auxiliary state prediction.

    When fg_weight > 1, pixels in the foreground band (lander, flames,
    flags — between black sky and white terrain) are upweighted in the
    reconstruction loss.

    When state_pred and state_target are provided with state_weight > 0,
    adds MSE between predicted and ground-truth kinematic state. This
    forces the latent space to encode physical state (position, velocity,
    angle), giving z spatial meaning and preventing discontinuous jumps.

    Args:
        recon: reconstructed frames (B, C, H, W)
        target: original frames (B, C, H, W)
        mu: encoder mean (B, latent_dim)
        logvar: encoder log-variance (B, latent_dim)
        beta: KL weight
        fg_weight: foreground pixel weight (1.0 = uniform MSE)
        state_pred: predicted kinematic state (B, state_dim) or None
        state_target: ground truth kinematic state (B, state_dim) or None
        state_weight: weight for state prediction loss (0.0 = disabled)

    Returns:
        total_loss, recon_loss, kl_loss, state_loss (all scalar tensors)
    """
    if fg_weight > 1.0:
        # Foreground-weighted MSE — upweight lander/flames/flags pixels so the
        # VAE allocates capacity to small, visually important objects instead
        # of spending it all on the easy-to-reconstruct sky and terrain
        weights = _foreground_weight_mask(target, fg_weight)
        recon_loss = (weights * (recon - target).pow(2)).mean()
    else:
        # Uniform MSE — default when no foreground weighting is requested
        recon_loss = F.mse_loss(recon, target)

    # KL divergence D_KL(q(z|x) || N(0,I)) — closed-form for diagonal Gaussian.
    # Sum over latent dims per sample, then average over the batch.
    # This regularises z to stay near the prior, preventing posterior collapse
    # into a point estimate (which would make the latent non-smooth).
    kl_per_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
    kl_loss = kl_per_sample.mean()

    # Auxiliary state prediction loss — zero tensor when disabled so callers
    # can always unpack the same 4-tuple without branching
    state_loss = torch.tensor(0.0, device=recon.device)
    if state_pred is not None and state_target is not None and state_weight > 0:
        # MSE between VAE's state-head output and ground-truth kinematics.
        # Forces the latent z to encode physical state (pos, vel, angle),
        # giving z spatial meaning and preventing discontinuous jumps.
        state_loss = F.mse_loss(state_pred, state_target)

    # Weighted sum — beta << 1 keeps KL from overwhelming reconstruction,
    # state_weight is tuned per-experiment to balance grounding vs fidelity
    total = recon_loss + beta * kl_loss + state_weight * state_loss
    return total, recon_loss, kl_loss, state_loss


def latent_dynamics_loss(z_pred: torch.Tensor,
                         z_target: torch.Tensor) -> torch.Tensor:
    """MSE loss in latent space for dynamics prediction."""
    return F.mse_loss(z_pred, z_target)


def multi_step_latent_loss(dynamics: torch.nn.Module,
                           z_seq: torch.Tensor,
                           actions: torch.Tensor,
                           k: int,
                           teacher_forcing: float = 0.0,
                           kin_weight: float = 1.0,
                           kin_dims: int = 6,
                           ) -> torch.Tensor:
    """k-step loss with full gradient flow and optional scheduled sampling.

    Rolls out k steps from z_seq[:, 0], computing MSE at every step
    against encoded-GT z. Gradients flow through the full autoregressive
    chain (NO detach), teaching the model to produce z's that work as
    inputs for future steps.

    When teacher_forcing > 0, some steps randomly receive GT z instead
    of the model's own prediction. This acts as a gradient chain
    curriculum: at teacher_forcing=0.5, expected autoregressive chain
    length is ~2 steps (geometric distribution); at 0.0, it's the full
    k steps. Annealing from high to low teacher_forcing provides stable
    early training followed by full BPTT.

    Unlike predict_sequence() which detaches own predictions, here
    gradients ALWAYS flow through model predictions when they're used
    as input — this is standard scheduled sampling (Bengio et al. 2015)
    and matches Dreamer's BPTT training.

    Args:
        dynamics: model with forward(z, action, state) -> (z_next, state).
            Both LatentDynamicsModel (GRU) and LatentRSSM implement this.
        z_seq: (B, T, latent_dim) encoded GT latent sequence.
        actions: (B, T-1, action_dim) actions between frames.
        k: rollout horizon. Clamped to min(k, T-1, len(actions)).
        teacher_forcing: probability of using GT z instead of model's
            own prediction at each step. 0.0 = pure autoregressive
            (original behavior), 1.0 = fully teacher-forced (each step
            independent). Values in between provide scheduled sampling
            with gradient flow.
        kin_weight: upweight factor for z[0:kin_dims] in MSE. 1.0 = uniform
            (default). >1 prioritizes kinematic dims in the loss. Only valid
            with a factored VAE where z[0:kin_dims] are kinematics by construction.
        kin_dims: number of leading z dims to upweight. Typically 6 (all
            kinematics) or len(kin_targets).

    Returns:
        Scalar MSE loss averaged over all k steps and batch.
    """
    T = z_seq.size(1)
    n_actions = actions.size(1)
    # Clamp k to available sequence length — avoids index-out-of-bounds
    # when caller requests a horizon longer than the episode
    k = min(k, T - 1, n_actions)

    if teacher_forcing == 0.0:
        # Pure autoregressive — use rollout() for efficiency (single call,
        # no per-step branching). This is the common fast path.
        z_pred, _ = dynamics.rollout(z_seq[:, 0], actions[:, :k])
        z_pred_k = z_pred[:, 1:]
        z_target = z_seq[:, 1:k + 1]
    else:
        # Scheduled sampling with gradient flow — manual loop so we can
        # randomly substitute GT z at each step while keeping gradients
        # flowing through the autoregressive segments.
        z = z_seq[:, 0]
        state = None
        z_preds = []

        for t in range(k):
            z_next, state = dynamics.forward(z, actions[:, t], state)
            z_preds.append(z_next)

            if t < k - 1:
                # Randomly choose GT or own prediction as next input.
                # When using own prediction: NO detach — gradients flow
                # through the chain. When using GT: chain breaks here,
                # but the prediction at step t still gets gradient from
                # its own loss term.
                if torch.rand(1).item() < teacher_forcing:
                    z = z_seq[:, t + 1]  # GT — breaks gradient chain
                else:
                    z = z_next           # Own prediction — gradient flows

        z_pred_k = torch.stack(z_preds, dim=1)  # (B, k, latent_dim)
        z_target = z_seq[:, 1:k + 1]

    # --- Weighted MSE ---
    # When kin_weight > 1, upweight the first kin_dims latent dimensions
    # so the dynamics model prioritizes getting kinematics right. Only
    # meaningful with a factored VAE where z[0:kin_dims] are kinematics
    # by construction (position, velocity, angle).
    if kin_weight != 1.0 and kin_dims > 0:
        latent_dim = z_seq.size(-1)
        # Build per-dim weight vector: kin dims get kin_weight, rest get 1.0
        dim_weights = torch.ones(latent_dim, device=z_seq.device)
        dim_weights[:kin_dims] = kin_weight
        # Normalize so mean weight = 1 — keeps loss magnitude comparable
        # across different kin_weight values, preventing LR re-tuning
        dim_weights = dim_weights / dim_weights.mean()
        sq_err = (z_pred_k - z_target).pow(2)
        return (sq_err * dim_weights).mean()
    else:
        return F.mse_loss(z_pred_k, z_target)


def latent_elbo_loss(model: torch.nn.Module,
                     z_seq: torch.Tensor,
                     actions: torch.Tensor,
                     k: int,
                     kl_weight: float = 1.0,
                     free_bits: float = 0.0,
                     return_breakdown: bool = False,
                     ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """ELBO loss for latent RSSM: reconstruction MSE + KL(posterior || prior).

    Loops k steps calling model.step() (posterior path). At each step t:
    - Posterior observes z_t (current encoded GT) to refine latent state
    - Decoder predicts z_{t+1} from [deter, posterior_sample]
    - Reconstruction loss: MSE(z_{t+1}_pred, z_{t+1}_gt)
    - KL: pushes posterior toward prior so prior is good for dreaming

    The observation (z_t) and target (z_{t+1}) are different — the posterior
    sees the current frame, the loss evaluates the next-frame prediction.
    This prevents trivial solutions where the posterior just passes through.

    Args:
        model: LatentRSSM with step(z_obs, action, state) and kl_loss(state).
        z_seq: (B, T, latent_dim) encoded GT latent sequence.
        actions: (B, T-1, action_dim) actions between frames.
        k: number of steps. Clamped to min(k, T-1, len(actions)).
        kl_weight: scalar weight for KL divergence term.
        free_bits: minimum KL per stochastic dimension (nats). Prevents
            posterior collapse by ensuring the stochastic branch carries at
            least this much information. Dreamer uses 1.0. Set 0 to disable.
        return_breakdown: if True, return (total, recon_loss, kl_loss).

    Returns:
        Scalar loss, or (total, recon, kl) tuple if return_breakdown=True.
    """
    B = z_seq.size(0)
    T = z_seq.size(1)
    n_actions = actions.size(1)
    # Clamp k to available data — avoids index-out-of-bounds
    k = min(k, T - 1, n_actions)
    device = z_seq.device

    recon_terms = []
    kl_terms = []
    # Start from a zero-initialized recurrent state
    model_state = model.initial_state(B, device)

    for t in range(k):
        # Posterior observes z_t (current GT frame), predicts z_{t+1}
        z_next_pred, model_state = model.step(z_seq[:, t], actions[:, t], model_state)
        # Reconstruction: how well z_{t+1}_pred matches actual z_{t+1}
        recon_terms.append(F.mse_loss(z_next_pred, z_seq[:, t + 1]))
        # KL(posterior || prior): ensures prior can dream without observations.
        # free_bits prevents collapse by clamping per-dim KL to a floor.
        kl_terms.append(model.kl_loss(model_state, free_bits=free_bits))

    # Average over time steps — treats each step equally regardless of k
    recon_loss = torch.stack(recon_terms).mean()
    kl_loss = torch.stack(kl_terms).mean()
    # Total ELBO: reconstruction + weighted KL
    total = recon_loss + kl_weight * kl_loss

    if return_breakdown:
        return total, recon_loss, kl_loss
    return total
