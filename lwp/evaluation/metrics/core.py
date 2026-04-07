"""Core evaluation metrics for world models."""
from __future__ import annotations

import numpy as np
import torch

from lwp.data.normalization import NormStats, normalize, denormalize
from lwp.training.integration import hybrid_state_update


@torch.no_grad()
def per_dim_mse(model, data_loader, norm_stats: NormStats,
                device: str = "cpu") -> torch.Tensor:
    """1-step per-dimension MSE (Eval A).

    Returns: [state_dim] tensor of per-dimension MSE.
    """
    model.eval()
    ns = norm_stats.to(device)
    all_sq_errors = []
    for batch in data_loader:
        obs, actions, true_deltas = (t.to(device) for t in batch)
        obs_n = normalize(obs, ns.state_mean, ns.state_std)
        pred_deltas_n, _ = model.step(obs_n, actions)
        pred_deltas = denormalize(pred_deltas_n, ns.delta_mean, ns.delta_std)
        sq_err = (pred_deltas - true_deltas).pow(2)
        all_sq_errors.append(sq_err.cpu())
    return torch.cat(all_sq_errors, dim=0).mean(dim=0)


def _rollout_raw_space(model, s0, actions, norm_stats,
                       warmup_states=None, warmup_actions=None,
                       subsample: int = 1):
    """Open-loop rollout in raw space with optional warmup for recurrent models.

    If warmup_states/warmup_actions are provided, teacher-forces through them
    first to build hidden state before the open-loop rollout.
    """
    states = [s0]
    s = s0
    model_state = None

    # Warmup: teacher-force through warmup steps to build hidden state
    if warmup_states is not None and warmup_actions is not None:
        for t in range(warmup_actions.shape[0]):
            ws_n = normalize(warmup_states[t:t+1], norm_stats.state_mean, norm_stats.state_std)
            _, model_state = model.step(ws_n, warmup_actions[t:t+1], model_state)

    for t in range(actions.shape[1]):
        s_n = normalize(s, norm_stats.state_mean, norm_stats.state_std)
        delta_n, model_state = model.step(s_n, actions[:, t], model_state)
        delta = denormalize(delta_n, norm_stats.delta_mean, norm_stats.delta_std)
        s = hybrid_state_update(s, delta, subsample=subsample)
        states.append(s)
    return torch.stack(states, dim=1)  # [batch, T+1, state_dim]


@torch.no_grad()
def horizon_error_curve(model, dataset, norm_stats: NormStats,
                        horizons: list[int] = (1, 5, 10, 20, 50, 100),
                        n_rollouts: int = 50, device: str = "cpu",
                        warmup_steps: int = 0,
                        subsample: int = 1) -> dict:
    """MSE at each horizon via autoregressive rollout (Eval B).

    Rolls out in raw space (normalize/denormalize at each step) to
    avoid conflating state and delta normalization.

    Returns: dict mapping horizon -> [state_dim] tensor of per-dim MSE.
    """
    model.eval()
    max_h = max(horizons)
    ns = norm_stats.to(device)

    step_errors = {h: [] for h in horizons}
    n_done = 0

    for ep_idx in range(dataset.n_episodes):
        if n_done >= n_rollouts:
            break
        states = torch.from_numpy(dataset.states[ep_idx]).to(device)
        actions = torch.from_numpy(dataset.actions[ep_idx]).to(device)
        T = len(actions)
        if T < warmup_steps + max_h:
            continue

        s0 = states[warmup_steps].unsqueeze(0)
        acts = actions[warmup_steps:warmup_steps + max_h].unsqueeze(0)

        warmup_s = states[:warmup_steps] if warmup_steps > 0 else None
        warmup_a = actions[:warmup_steps] if warmup_steps > 0 else None
        pred_states = _rollout_raw_space(model, s0, acts, ns,
                                         warmup_states=warmup_s,
                                         warmup_actions=warmup_a,
                                         subsample=subsample)

        for h in horizons:
            if h > T - warmup_steps:
                continue
            pred_raw = pred_states[0, h]
            true_raw = states[warmup_steps + h]
            sq_err = (pred_raw - true_raw).pow(2)
            step_errors[h].append(sq_err.cpu())

        n_done += 1

    result = {}
    for h in horizons:
        if step_errors[h]:
            result[h] = torch.stack(step_errors[h]).mean(dim=0)
        else:
            result[h] = torch.zeros(norm_stats.state_mean.shape[0])
    return result


@torch.no_grad()
def cumulative_trajectory_mse(model, dataset, norm_stats: NormStats,
                               horizons: list[int] = (1, 5, 10, 20, 50),
                               n_rollouts: int = 50, device: str = "cpu",
                               warmup_steps: int = 0,
                               subsample: int = 1) -> dict:
    """Average MSE across all steps 1..h for each horizon (Eval B').

    Unlike horizon_error_curve which measures error at step h only,
    this measures mean error across the entire trajectory up to h.
    Captures trajectory-level divergence that endpoint-only measurement misses.

    Returns: dict mapping horizon -> [state_dim] tensor of per-dim mean MSE.
    """
    model.eval()
    max_h = max(horizons)
    ns = norm_stats.to(device)

    step_errors = {h: [] for h in horizons}
    n_done = 0

    for ep_idx in range(dataset.n_episodes):
        if n_done >= n_rollouts:
            break
        states = torch.from_numpy(dataset.states[ep_idx]).to(device)
        actions = torch.from_numpy(dataset.actions[ep_idx]).to(device)
        T = len(actions)
        if T < warmup_steps + max_h:
            continue

        s0 = states[warmup_steps].unsqueeze(0)
        acts = actions[warmup_steps:warmup_steps + max_h].unsqueeze(0)

        warmup_s = states[:warmup_steps] if warmup_steps > 0 else None
        warmup_a = actions[:warmup_steps] if warmup_steps > 0 else None
        pred_states = _rollout_raw_space(model, s0, acts, ns,
                                         warmup_states=warmup_s,
                                         warmup_actions=warmup_a,
                                         subsample=subsample)

        for h in horizons:
            if h > T - warmup_steps:
                continue
            errs = []
            for t in range(1, h + 1):
                sq_err = (pred_states[0, t] - states[warmup_steps + t]).pow(2)
                errs.append(sq_err)
            cumul_mse = torch.stack(errs).mean(dim=0)
            step_errors[h].append(cumul_mse.cpu())

        n_done += 1

    result = {}
    for h in horizons:
        if step_errors[h]:
            result[h] = torch.stack(step_errors[h]).mean(dim=0)
        else:
            result[h] = torch.zeros(norm_stats.state_mean.shape[0])
    return result


@torch.no_grad()
def rollout_error_metrics(model, dataset, norm_stats: NormStats,
                          horizons: list[int] = (1, 5, 10, 20, 50),
                          n_rollouts: int = 50, device: str = "cpu",
                          warmup_steps: int = 0,
                          subsample: int = 1) -> tuple[dict, dict]:
    """Combined endpoint and cumulative trajectory MSE from one rollout pass.

    Does the same rollouts as horizon_error_curve + cumulative_trajectory_mse
    but only runs the model once per episode instead of twice.

    Returns:
        (endpoint_errors, cumulative_errors): each maps horizon -> [state_dim] tensor.
    """
    model.eval()
    max_h = max(horizons)
    ns = norm_stats.to(device)

    endpoint_errs = {h: [] for h in horizons}
    cumul_errs = {h: [] for h in horizons}
    n_done = 0

    for ep_idx in range(dataset.n_episodes):
        if n_done >= n_rollouts:
            break
        states = torch.from_numpy(dataset.states[ep_idx]).to(device)
        actions = torch.from_numpy(dataset.actions[ep_idx]).to(device)
        T = len(actions)
        if T < warmup_steps + max_h:
            continue

        s0 = states[warmup_steps].unsqueeze(0)
        acts = actions[warmup_steps:warmup_steps + max_h].unsqueeze(0)

        warmup_s = states[:warmup_steps] if warmup_steps > 0 else None
        warmup_a = actions[:warmup_steps] if warmup_steps > 0 else None
        pred_states = _rollout_raw_space(model, s0, acts, ns,
                                         warmup_states=warmup_s,
                                         warmup_actions=warmup_a,
                                         subsample=subsample)

        for h in horizons:
            if h > T - warmup_steps:
                continue
            # Endpoint MSE at step h
            sq_err = (pred_states[0, h] - states[warmup_steps + h]).pow(2)
            endpoint_errs[h].append(sq_err.cpu())
            # Cumulative MSE averaged across steps 1..h
            step_sq = torch.stack(
                [(pred_states[0, t] - states[warmup_steps + t]).pow(2) for t in range(1, h + 1)]
            )
            cumul_errs[h].append(step_sq.mean(dim=0).cpu())

        n_done += 1

    def _agg(d):
        result = {}
        for h in horizons:
            if d[h]:
                result[h] = torch.stack(d[h]).mean(dim=0)
            else:
                result[h] = torch.zeros(norm_stats.state_mean.shape[0])
        return result

    return _agg(endpoint_errs), _agg(cumul_errs)


def divergence_exponent(horizon_errors: dict) -> float:
    """Fit lambda in error(h) ~ e^(lambda * h) (Eval B).

    Args:
        horizon_errors: dict mapping horizon -> scalar mean MSE

    Returns:
        lambda: divergence rate (higher = faster divergence)
    """
    horizons = sorted(horizon_errors.keys())
    errors = [horizon_errors[h] if isinstance(horizon_errors[h], (int, float))
              else float(horizon_errors[h].mean()) for h in horizons]

    # Filter out zero/negative errors (can't log)
    valid = [(h, e) for h, e in zip(horizons, errors) if e > 0]
    if len(valid) < 2:
        return 0.0

    h_arr = np.array([v[0] for v in valid], dtype=np.float64)
    log_err = np.log(np.array([v[1] for v in valid], dtype=np.float64))

    # Least-squares fit: log(error) = lambda * h + c
    A = np.column_stack([h_arr, np.ones_like(h_arr)])
    result = np.linalg.lstsq(A, log_err, rcond=None)
    lam = float(result[0][0])
    return lam


def horizon_to_failure(horizon_errors: dict, threshold: float) -> int:
    """Steps until mean MSE exceeds threshold (Eval B).

    Returns the last horizon where error is below threshold.
    """
    horizons = sorted(horizon_errors.keys())
    last_good = horizons[0]
    for h in horizons:
        err = horizon_errors[h]
        if isinstance(err, torch.Tensor):
            err = float(err.mean())
        if err < threshold:
            last_good = h
    return last_good
