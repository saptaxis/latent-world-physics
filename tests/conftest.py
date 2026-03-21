"""Shared test fixtures. Synthetic episodes for fast tests."""
import json
import sys
from pathlib import Path

import numpy as np

# Add repo root to sys.path so tests can import scripts/*
sys.path.insert(0, str(Path(__file__).parent.parent))
import pytest


@pytest.fixture
def episode_dir(tmp_path):
    """Create a directory with 10 small synthetic .npz episodes.

    Each episode: 50 steps, 15D states (8 kinematic + 7 constant physics),
    2D actions in [-1, 1], random rewards.
    """
    rng = np.random.default_rng(42)
    for i in range(10):
        T = 50
        states = rng.standard_normal((T + 1, 8)).astype(np.float32) * 0.1
        # Make states a random walk so deltas are small
        states = np.cumsum(states * 0.02, axis=0).astype(np.float32)
        actions = rng.uniform(-1, 1, (T, 2)).astype(np.float32)
        rewards = rng.standard_normal(T).astype(np.float32)
        metadata = json.dumps({"seed": i})
        np.savez(
            tmp_path / f"episode_{i:03d}.npz",
            states=states, actions=actions, rewards=rewards,
            metadata_json=metadata,
        )
    return tmp_path
