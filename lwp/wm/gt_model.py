# lwp/wm/gt_model.py
"""Ground truth 'model' for validating physics extraction.

Returns actual episode deltas instead of learned predictions. Used to
verify that the physics constant extraction pipeline recovers correct
values from perfect data — a sanity check on the methodology itself.

If extraction gives wrong numbers on GT data, then ALL model evaluation
results (Finding 05 etc.) are suspect.
"""
from __future__ import annotations

import warnings

import numpy as np
import torch
import torch.nn as nn


def prepare_episodes_for_norm(episodes: list[dict]) -> list[dict]:
    """Preprocess raw episodes for compute_norm_stats.

    compute_norm_stats expects dicts with 'states' and 'deltas' as
    torch Tensors. Raw npz episodes have numpy arrays and no 'deltas'
    key. This function adds deltas and converts to tensors.

    Args:
        episodes: List of dicts with 'states' (numpy, T+1, >=6) and
            'actions' (numpy, T, 2).

    Returns:
        List of dicts with 'states' (Tensor, T+1, 6) and
        'deltas' (Tensor, T, 6) ready for compute_norm_stats.
    """
    prepared = []
    for ep in episodes:
        states = torch.tensor(ep["states"][:, :6], dtype=torch.float32)
        deltas = states[1:] - states[:-1]
        prepared.append({"states": states, "deltas": deltas})
    return prepared


class GroundTruthModel(nn.Module):
    """A 'model' that returns ground truth deltas from pre-loaded episodes.

    Implements the WorldModel.step() interface so it can be plugged into
    the physics extraction pipeline (physics_understanding.py). When
    step() is called with a normalized state + action, it finds the
    matching transition in the episode data and returns the actual
    normalized delta.

    Matching is by nearest-neighbor in (state, action) space — both
    state and action are included because the same state with different
    actions produces different deltas (e.g., hovering at the same
    position with or without thrust). Since oracle extraction feeds GT
    states from the same episodes, matches should be exact.

    Args:
        episodes: List of episode dicts with 'states' (T+1, >=6) and
            'actions' (T, 2). Raw numpy — not preprocessed.
        norm_stats: NormStats for state/delta normalization. Must be the
            same NormStats used by the extraction pipeline.
    """

    def __init__(self, episodes: list[dict], norm_stats):
        super().__init__()
        self.norm_stats = norm_stats
        self.n_mismatches = 0  # track inexact lookups

        # Pre-compute all (state, action, delta) triples from episodes.
        all_states = []
        all_actions = []
        all_deltas_norm = []

        for ep in episodes:
            states = ep["states"][:, :6].astype(np.float32)
            actions = ep["actions"].astype(np.float32)
            n = min(len(states) - 1, len(actions))
            for t in range(n):
                delta_raw = states[t + 1] - states[t]
                # Normalize delta using the same convention as
                # _predict_oracle_delta: (delta - mean) / std, NO epsilon.
                delta_norm = torch.nan_to_num(
                    (torch.tensor(delta_raw) - norm_stats.delta_mean.squeeze())
                    / norm_stats.delta_std.squeeze()
                ).numpy()
                all_states.append(states[t])
                all_actions.append(actions[t])
                all_deltas_norm.append(delta_norm)

        self._states = np.array(all_states, dtype=np.float32)
        self._actions = np.array(all_actions, dtype=np.float32)
        self._deltas_norm = np.array(all_deltas_norm, dtype=np.float32)

        # Precompute normalized states for matching (same normalization
        # as _predict_oracle_delta line 203: no epsilon)
        self._states_norm = torch.nan_to_num(
            (torch.tensor(self._states) - norm_stats.state_mean.squeeze())
            / norm_stats.state_std.squeeze()
        ).numpy()

        # Normalize actions for matching (scale to similar magnitude as states)
        self._actions_norm = self._actions  # actions are already in [0,1] range

        # Concatenate state + action for joint NN matching
        # This ensures same-state-different-action transitions are distinguished
        self._match_keys = np.concatenate(
            [self._states_norm, self._actions_norm], axis=1,
        )  # (N, 8)

        # Dummy buffer so _get_model_device() can detect our device
        self.register_buffer("_device_dummy", torch.zeros(1))

    def step(
        self, obs: torch.Tensor, action: torch.Tensor, model_state
    ) -> tuple[torch.Tensor, None]:
        """Return GT normalized delta for the closest matching transition.

        Args:
            obs: (B, 6) normalized state
            action: (B, 2) action
            model_state: ignored (stateless model)

        Returns:
            (delta_norm, None): (B, 6) normalized delta, no model state
        """
        obs_np = np.nan_to_num(obs.detach().cpu().numpy())
        act_np = action.detach().cpu().numpy()
        B = obs_np.shape[0]
        deltas = np.zeros((B, 6), dtype=np.float32)

        for b in range(B):
            # NN match in (state_norm, action) space
            query = np.concatenate([obs_np[b], act_np[b]])
            dists = np.sum((self._match_keys - query) ** 2, axis=1)
            idx = np.argmin(dists)
            deltas[b] = self._deltas_norm[idx]

            # Warn on inexact matches — for oracle extraction these should
            # be near-exact since the pipeline feeds states from the same
            # episodes. Large distances indicate a bug or data mismatch.
            if dists[idx] > 1e-4:
                self.n_mismatches += 1
                if self.n_mismatches <= 5:
                    warnings.warn(
                        f"GroundTruthModel: inexact match (dist={dists[idx]:.6f}). "
                        f"Query state may not exist in episode data."
                    )

        return torch.tensor(deltas, device=obs.device), None

    def report_match_quality(self):
        """Print summary of matching quality. Call after extraction."""
        if self.n_mismatches > 0:
            warnings.warn(
                f"GroundTruthModel: {self.n_mismatches} inexact matches "
                f"(dist > 1e-4). Results may not be exact GT."
            )
        else:
            print(f"  GT model: all lookups matched exactly (0 mismatches)")
