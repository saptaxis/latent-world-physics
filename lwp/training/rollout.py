"""Rollout utilities built on model.step().

Single-step transition is the primitive. All rollout variants are loops over step().
"""
from __future__ import annotations

import torch


def rollout_open_loop(model, s0, actions, model_state=None):
    """Autoregressive rollout using model's own predictions.

    Args:
        model: WorldModel instance
        s0: initial state [batch, state_dim]
        actions: action sequence [batch, T, action_dim]
        model_state: initial model state (None for stateless)

    Returns:
        states: [batch, T+1, state_dim] (includes s0)
        deltas: [batch, T, state_dim]
        model_state: final model state
    """
    states = [s0]
    deltas = []
    s = s0
    ms = model_state if model_state is not None else model.initial_state(
        batch_size=s0.shape[0], device=s0.device
    )
    for t in range(actions.shape[1]):
        delta, ms = model.step(s, actions[:, t], ms)
        s = s + delta
        deltas.append(delta)
        states.append(s)
    return torch.stack(states, dim=1), torch.stack(deltas, dim=1), ms


def rollout_teacher_forced(model, states, actions, model_state=None):
    """Teacher-forced rollout: model sees true states at every step.

    Args:
        model: WorldModel instance
        states: true state sequence [batch, T, state_dim]
        actions: action sequence [batch, T, action_dim]
        model_state: initial model state

    Returns:
        deltas: [batch, T, state_dim]
        model_state: final model state
    """
    deltas = []
    ms = model_state if model_state is not None else model.initial_state(
        batch_size=states.shape[0], device=states.device
    )
    for t in range(actions.shape[1]):
        delta, ms = model.step(states[:, t], actions[:, t], ms)
        deltas.append(delta)
    return torch.stack(deltas, dim=1), ms


def rollout_warmup_then_branch(model, prefix_states, prefix_actions,
                                branch_actions, model_state=None):
    """Warmup on true prefix, then open-loop rollout.

    For stateless models, warmup is a no-op (model_state stays None),
    and the branch starts from the last prefix state.

    Args:
        prefix_states: [batch, T_pre, state_dim]
        prefix_actions: [batch, T_pre, action_dim]
        branch_actions: [batch, T_branch, action_dim]

    Returns:
        branch_states: [batch, T_branch+1, state_dim]
        branch_deltas: [batch, T_branch, state_dim]
        model_state: final
    """
    _, ms = rollout_teacher_forced(model, prefix_states, prefix_actions, model_state)
    s0 = prefix_states[:, -1]
    return rollout_open_loop(model, s0, branch_actions, ms)


def rollout_scheduled_sampling(model, true_states, actions, sampling_prob,
                                model_state=None):
    """Rollout with scheduled sampling for recurrent model training.

    At each step, with probability sampling_prob, use the model's own
    prediction instead of the true state.

    Args:
        true_states: [batch, T, state_dim]
        actions: [batch, T, action_dim]
        sampling_prob: probability of using model prediction (0.0 = teacher forced)

    Returns:
        deltas: [batch, T, state_dim]
        model_state: final
    """
    deltas = []
    ms = model_state if model_state is not None else model.initial_state(
        batch_size=true_states.shape[0], device=true_states.device
    )
    s_prev = true_states[:, 0]
    for t in range(actions.shape[1]):
        delta, ms = model.step(s_prev, actions[:, t], ms)
        deltas.append(delta)
        s_pred = s_prev + delta
        if t + 1 < true_states.shape[1]:
            use_pred = torch.rand(1).item() < sampling_prob
            s_prev = s_pred if use_pred else true_states[:, t + 1]
    return torch.stack(deltas, dim=1), ms
