"""Loss functions for world model training."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from lwp.data.normalization import NormStats, normalize, denormalize


def _compute_dim_weights(
    dim_weights: str | None,
    norm_stats: NormStats,
) -> torch.Tensor | None:
    """Compute per-dimension loss weights."""
    if dim_weights is None:
        return None
    if dim_weights == "inv_var":
        inv_var = 1.0 / (norm_stats.delta_std ** 2 + 1e-8)
        weights = inv_var / inv_var.mean()
        return weights
    raise ValueError(f"Unknown dim_weights: {dim_weights}. Use 'inv_var' or None.")


def _weighted_mse(pred: torch.Tensor, target: torch.Tensor,
                  weights: torch.Tensor | None) -> torch.Tensor:
    """MSE loss with optional per-dim weighting."""
    if weights is None:
        return F.mse_loss(pred, target)
    sq_err = (pred - target) ** 2
    weighted = sq_err * weights
    return weighted.mean()


def single_step_loss(model, batch, norm_stats: NormStats,
                     dim_weights: str | None = None) -> torch.Tensor:
    """MSE on single-step delta prediction.

    Model contract: receives state-normalized obs, outputs delta-normalized delta.

    Args:
        batch: (obs, actions, true_deltas) each [batch, dim]
        norm_stats: normalization statistics
    """
    obs, actions, true_deltas = batch
    obs_n = normalize(obs, norm_stats.state_mean, norm_stats.state_std)
    pred_deltas_n, _ = model.step(obs_n, actions)
    true_deltas_n = normalize(true_deltas, norm_stats.delta_mean, norm_stats.delta_std)
    w = _compute_dim_weights(dim_weights, norm_stats)
    return _weighted_mse(pred_deltas_n, true_deltas_n, w)


def multi_step_loss(model, batch, norm_stats: NormStats, k: int,
                    dim_weights: str | None = None) -> torch.Tensor:
    """MSE on k-step autoregressive rollout.

    Rollout in NORMALIZED space: the model receives and produces normalized
    values. State accumulation stays in normalized space throughout, avoiding
    the raw-space round-trip that creates unstable gradients.

    State update: s_{t+1}_normalized = s_t_normalized + (delta_n * delta_std + delta_mean) / state_std

    Args:
        batch: (state_seq, action_seq) where state_seq is [batch, T+1, state_dim]
               and action_seq is [batch, T, action_dim]
        norm_stats: normalization statistics
        k: number of rollout steps
        dim_weights: "inv_var" for per-dim weighting, None for uniform MSE
    """
    state_seq, action_seq = batch
    s0 = state_seq[:, 0]
    actions_k = action_seq[:, :k]
    true_states_k = state_seq[:, :k + 1]

    delta_std = norm_stats.delta_std
    delta_mean = norm_stats.delta_mean
    state_std = norm_stats.state_std
    state_mean = norm_stats.state_mean

    # Rollout in normalized space
    pred_deltas_n = []
    s_n = normalize(s0, state_mean, state_std)
    model_state = None
    for t in range(k):
        delta_n, model_state = model.step(s_n, actions_k[:, t], model_state)
        pred_deltas_n.append(delta_n)
        # Accumulate in normalized space:
        delta_raw = delta_n * (delta_std + 1e-8) + delta_mean
        s_n = s_n + delta_raw / (state_std + 1e-8)
    pred_deltas_n = torch.stack(pred_deltas_n, dim=1)

    # Loss in delta-normalized space
    true_deltas = true_states_k[:, 1:] - true_states_k[:, :-1]
    true_deltas_n = normalize(true_deltas, delta_mean, delta_std)

    w = _compute_dim_weights(dim_weights, norm_stats)
    return _weighted_mse(pred_deltas_n, true_deltas_n, w)


def scheduled_sampling_loss(model, batch, norm_stats: NormStats,
                            k: int, sampling_prob: float,
                            dim_weights: str | None = None) -> torch.Tensor:
    """MSE on k-step rollout with scheduled sampling."""
    state_seq, action_seq = batch
    s0 = state_seq[:, 0]
    actions_k = action_seq[:, :k]
    true_states_k = state_seq[:, :k + 1]

    delta_std = norm_stats.delta_std
    delta_mean = norm_stats.delta_mean
    state_std = norm_stats.state_std
    state_mean = norm_stats.state_mean

    pred_deltas_n = []
    s_n = normalize(s0, state_mean, state_std)
    model_state = model.initial_state(s0.shape[0], device=s0.device)

    for t in range(k):
        delta_n, model_state = model.step(s_n, actions_k[:, t], model_state)
        pred_deltas_n.append(delta_n)
        # Predicted next state in normalized space
        delta_raw = delta_n * (delta_std + 1e-8) + delta_mean
        s_n_pred = s_n + delta_raw / (state_std + 1e-8)
        # Scheduled sampling: use true or predicted
        if t + 1 < k:
            use_pred = torch.rand(1).item() < sampling_prob
            if use_pred:
                s_n = s_n_pred
            else:
                s_n = normalize(true_states_k[:, t + 1], state_mean, state_std)

    pred_deltas_n = torch.stack(pred_deltas_n, dim=1)
    true_deltas = true_states_k[:, 1:] - true_states_k[:, :-1]
    true_deltas_n = normalize(true_deltas, delta_mean, delta_std)

    w = _compute_dim_weights(dim_weights, norm_stats)
    return _weighted_mse(pred_deltas_n, true_deltas_n, w)


def elbo_loss(model, batch, norm_stats: NormStats, k: int,
              kl_weight: float = 1.0,
              dim_weights: str | None = None) -> torch.Tensor:
    """ELBO loss for RSSM: reconstruction + KL divergence."""
    state_seq, action_seq = batch
    s0 = state_seq[:, 0]
    actions_k = action_seq[:, :k]
    true_states_k = state_seq[:, :k + 1]

    delta_std = norm_stats.delta_std
    delta_mean = norm_stats.delta_mean
    state_std = norm_stats.state_std
    state_mean = norm_stats.state_mean

    pred_deltas_n = []
    kl_terms = []
    s_n = normalize(s0, state_mean, state_std)
    model_state = model.initial_state(s0.shape[0], device=s0.device)

    for t in range(k):
        delta_n, model_state = model.step(s_n, actions_k[:, t], model_state)
        pred_deltas_n.append(delta_n)
        kl_terms.append(model.kl_loss(model_state))
        delta_raw = delta_n * (delta_std + 1e-8) + delta_mean
        s_n = s_n + delta_raw / (state_std + 1e-8)

    pred_deltas_n = torch.stack(pred_deltas_n, dim=1)
    true_deltas = true_states_k[:, 1:] - true_states_k[:, :-1]
    true_deltas_n = normalize(true_deltas, delta_mean, delta_std)

    w = _compute_dim_weights(dim_weights, norm_stats)
    recon_loss = _weighted_mse(pred_deltas_n, true_deltas_n, w)
    kl_loss = torch.stack(kl_terms).mean()

    return recon_loss + kl_weight * kl_loss
