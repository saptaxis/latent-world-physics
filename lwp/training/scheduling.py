"""Training schedules for curriculum learning and scheduled sampling."""
from __future__ import annotations


def curriculum_schedule(epoch: int, total_epochs: int,
                        k_min: int = 1, k_max: int = 20) -> int:
    """Anneal rollout horizon k from k_min to k_max over training.

    Linear interpolation, clamped to [k_min, k_max].

    Args:
        epoch: current epoch (0-indexed)
        total_epochs: total number of epochs
        k_min: starting rollout horizon
        k_max: ending rollout horizon

    Returns:
        k value for this epoch
    """
    if total_epochs <= 0:
        return k_max
    progress = min(epoch / total_epochs, 1.0)
    return int(k_min + (k_max - k_min) * progress)


def sampling_schedule(epoch: int, total_epochs: int,
                      start: float = 0.0, end: float = 0.5) -> float:
    """Anneal scheduled sampling probability from start to end.

    Linear interpolation, clamped to [start, end].

    Args:
        epoch: current epoch (0-indexed)
        total_epochs: total number of epochs
        start: initial sampling probability (0.0 = pure teacher forcing)
        end: final sampling probability

    Returns:
        sampling probability for this epoch
    """
    if total_epochs <= 0:
        return end
    progress = min(epoch / total_epochs, 1.0)
    return start + (end - start) * progress
