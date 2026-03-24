"""Per-dimension z-score normalization for world model inputs and targets."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class NormStats:
    state_mean: torch.Tensor   # [state_dim]
    state_std: torch.Tensor    # [state_dim]
    delta_mean: torch.Tensor   # [state_dim]
    delta_std: torch.Tensor    # [state_dim]

    def to_dict(self) -> dict:
        return {k: v.clone() for k, v in {
            "state_mean": self.state_mean, "state_std": self.state_std,
            "delta_mean": self.delta_mean, "delta_std": self.delta_std,
        }.items()}

    @classmethod
    def from_dict(cls, d: dict) -> NormStats:
        return cls(**{k: v.clone() for k, v in d.items()})

    def to(self, device) -> NormStats:
        return NormStats(
            state_mean=self.state_mean.to(device),
            state_std=self.state_std.to(device),
            delta_mean=self.delta_mean.to(device),
            delta_std=self.delta_std.to(device),
        )


def normalize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x - mean) / (std + 1e-8)


def denormalize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return x * (std + 1e-8) + mean


def compute_norm_stats(episodes: list[dict]) -> NormStats:
    """Compute per-dim mean and std from episode dicts.

    Each episode dict has 'states' (T+1, state_dim) and 'deltas' (T, state_dim)
    as torch Tensors.
    """
    all_states = torch.cat([ep["states"] for ep in episodes], dim=0)
    all_deltas = torch.cat([ep["deltas"] for ep in episodes], dim=0)
    return NormStats(
        state_mean=all_states.mean(dim=0),
        state_std=all_states.std(dim=0),
        delta_mean=all_deltas.mean(dim=0),
        delta_std=all_deltas.std(dim=0),
    )
