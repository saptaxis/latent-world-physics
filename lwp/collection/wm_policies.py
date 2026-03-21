"""Policy sources for world model data collection.

Provides callables with signature policy_fn(obs) -> action for each
source type: random, heuristic, and noisy-expert. RL agent policies
are loaded separately via SB3 and don't need wrappers here.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def make_random_policy(seed: int = 0) -> Callable[[np.ndarray], np.ndarray]:
    """Create a random policy that samples uniform actions in [-1, 1].

    Actions are independent of observation — purely exploratory. Useful for
    maximizing state-action coverage in world model training data.

    Args:
        seed: RNG seed for reproducibility.

    Returns:
        Callable taking (15,) obs, returning (2,) action in [-1, 1].
    """
    rng = np.random.default_rng(seed)

    def policy_fn(obs: np.ndarray) -> np.ndarray:
        return rng.uniform(-1.0, 1.0, size=(2,)).astype(np.float32)

    return policy_fn


def make_noisy_expert_policy(
    predict_fn: Callable,
    noise_sigma: float = 0.1,
    seed: int = 0,
) -> Callable[[np.ndarray], np.ndarray]:
    """Wrap a trained agent's predict with additive Gaussian action noise.

    Produces near-competent trajectories with exploration — representative of
    what an inner RL agent generates as it learns (starts noisy, gradually
    improves). The base action comes from predict_fn; noise is added and
    the result is clipped to [-1, 1].

    Args:
        predict_fn: Callable with SB3 interface: predict(obs, deterministic) -> (action, _).
            The obs passed is the raw env observation (not VecNormalize'd).
            For SB3 models, this is model.predict. For VecNormalize'd models,
            the caller must handle normalization before passing predict_fn.
        noise_sigma: Std of Gaussian noise added to each action dimension.
        seed: RNG seed.

    Returns:
        Callable taking (15,) obs, returning (2,) noisy action in [-1, 1].
    """
    rng = np.random.default_rng(seed)

    def policy_fn(obs: np.ndarray) -> np.ndarray:
        base_action, _ = predict_fn(obs, deterministic=True)
        noise = rng.normal(0.0, noise_sigma, size=base_action.shape).astype(np.float32)
        noisy_action = np.clip(base_action + noise, -1.0, 1.0)
        return noisy_action

    return policy_fn
