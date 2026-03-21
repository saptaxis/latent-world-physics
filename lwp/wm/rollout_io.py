"""Rollout IO for world model evaluation.

Defines the canonical format for WM rollout comparisons and provides
run/save/load functions. This is the WM analogue of episode_io.py —
all evaluation scripts, metrics, and visualization consume rollout dicts
produced by this module.

Rollout format (dict / .npz):
    predicted_states:  (T+1, D) float32 — model's autoregressive predictions,
                       starting from the known initial state at t=0.
    actual_states:     (T+1, D) float32 — ground truth states (with supervision
                       masking applied, matching what the model saw in training).
    actions:           (T, A) float32 — action sequence used for rollout.
    context:           (K, input_dim) float32 — context transitions fed to encoder.
                       None for non-context models.
    raw_physics:       (7,) float32 — unmasked physics params (dims 8:15 of
                       the original state at t=0). Always present for correlation
                       analysis, even in blind mode.
    metadata_json:     str — JSON dict with run_name, context_k, prediction_target,
                       supervision, episode_idx, rollout_steps.

Convention: T is the number of rollout steps. predicted_states and actual_states
have T+1 entries (initial state + one per step). actions has T entries.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


def run_rollout(
    model,
    episode: dict,
    context_k: int | None = None,
    prediction_target: str = "delta",
    supervision: str = "labeled",
    rollout_steps: int | None = None,
    episode_idx: int = 0,
) -> dict:
    """Run a single rollout comparison on one episode.

    Builds context from the first K timesteps, then rolls out the model
    autoregressively from timestep K. Returns a rollout dict in the
    canonical format.

    Args:
        model: WorldModel with forward(s, a, context=...).
        episode: Dict with "states" (T+1, D) and "actions" (T, A) arrays.
            States are raw (unmasked) — this function applies supervision masking.
        context_k: Number of context transitions. None for non-context models.
        prediction_target: "delta" or "absolute".
        supervision: "blind" or "labeled".
        rollout_steps: Max steps to roll out. None = use all available steps
            after context.
        episode_idx: Episode index for metadata.

    Returns:
        Rollout dict with keys: predicted_states, actual_states, actions,
        context (or None), raw_physics, metadata.
    """
    model.eval()
    raw_states = episode["states"]
    actions = episode["actions"]
    T = len(actions)
    K = context_k or 0

    # How many steps we can roll out after the context window.
    max_available = T - K
    if rollout_steps is None:
        rollout_steps = max_available
    else:
        rollout_steps = min(rollout_steps, max_available)

    if rollout_steps < 1:
        raise ValueError(
            f"Episode too short for rollout: T={T}, K={K}, "
            f"need at least K+1={K+1} timesteps."
        )

    # Apply supervision masking (blind: slice to kinematic dims 0:8).
    if supervision == "blind":
        states = raw_states[:, :8].copy()
    else:
        states = raw_states.copy()

    # Extract raw physics params (always from raw data, for correlation analysis).
    raw_physics = raw_states[0, 8:15].copy()

    # Build context from first K transitions.
    context_array = None
    context_tensor = None
    if context_k is not None and context_k > 0:
        ctx_rows = []
        for t in range(context_k):
            row = np.concatenate([states[t], actions[t], states[t + 1]])
            ctx_rows.append(row)
        context_array = np.stack(ctx_rows).astype(np.float32)
        context_tensor = torch.tensor(context_array).unsqueeze(0)

    # Autoregressive rollout from timestep K.
    s0 = torch.tensor(states[K], dtype=torch.float32).unsqueeze(0)
    rollout_actions = actions[K:K + rollout_steps]
    rollout_actions_tensor = torch.tensor(
        rollout_actions, dtype=torch.float32,
    ).unsqueeze(0)

    with torch.no_grad():
        s_pred = s0
        pred_list = [s0.squeeze(0).numpy()]
        for h in range(rollout_steps):
            kwargs = {}
            if context_tensor is not None:
                kwargs["context"] = context_tensor
            output = model(s_pred, rollout_actions_tensor[:, h], **kwargs)
            if prediction_target == "delta":
                s_pred = s_pred + output
            else:
                s_pred = output
            pred_list.append(s_pred.squeeze(0).numpy())

    predicted_states = np.stack(pred_list).astype(np.float32)
    actual_states = states[K:K + rollout_steps + 1].astype(np.float32)

    metadata = {
        "context_k": context_k,
        "prediction_target": prediction_target,
        "supervision": supervision,
        "episode_idx": episode_idx,
        "rollout_steps": rollout_steps,
    }

    result = {
        "predicted_states": predicted_states,
        "actual_states": actual_states,
        "actions": rollout_actions.astype(np.float32),
        "raw_physics": raw_physics.astype(np.float32),
        "metadata": metadata,
    }
    if context_array is not None:
        result["context"] = context_array

    return result


def save_rollout(path: str | Path, rollout: dict) -> Path:
    """Save a rollout comparison to .npz format.

    Validates shape consistency before writing.

    Args:
        path: Output file path.
        rollout: Rollout dict from run_rollout().

    Returns:
        Path to the saved file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    pred = rollout["predicted_states"]
    actual = rollout["actual_states"]
    actions = rollout["actions"]

    # Validate shapes.
    assert pred.shape == actual.shape, (
        f"predicted_states shape {pred.shape} != actual_states shape {actual.shape}"
    )
    assert pred.shape[0] == actions.shape[0] + 1, (
        f"predicted_states has {pred.shape[0]} entries but actions has "
        f"{actions.shape[0]} steps (expected {pred.shape[0] - 1})"
    )

    save_dict = {
        "predicted_states": pred.astype(np.float32),
        "actual_states": actual.astype(np.float32),
        "actions": actions.astype(np.float32),
        "raw_physics": rollout.get("raw_physics", np.zeros(7, dtype=np.float32)),
        "metadata_json": json.dumps(rollout.get("metadata", {})),
    }
    if "context" in rollout and rollout["context"] is not None:
        save_dict["context"] = rollout["context"].astype(np.float32)

    np.savez_compressed(str(path), **save_dict)
    return path


def load_rollout(path: str | Path) -> dict:
    """Load a rollout comparison from .npz format.

    Args:
        path: Path to .npz file saved by save_rollout().

    Returns:
        Rollout dict with keys: predicted_states, actual_states, actions,
        context (or None), raw_physics, metadata.
    """
    data = np.load(str(path), allow_pickle=False)
    result = {
        "predicted_states": data["predicted_states"],
        "actual_states": data["actual_states"],
        "actions": data["actions"],
        "raw_physics": data["raw_physics"],
        "metadata": json.loads(str(data["metadata_json"])),
    }
    if "context" in data:
        result["context"] = data["context"]
    return result
