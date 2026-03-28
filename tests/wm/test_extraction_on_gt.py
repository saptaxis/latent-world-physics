# tests/wm/test_extraction_on_gt.py
"""Integration test: extraction pipeline on GT data."""
import numpy as np
import torch
import pytest
import glob

from lwp.wm.gt_model import GroundTruthModel, prepare_episodes_for_norm
from lwp.wm.physics_understanding import (
    extract_gravity_oracle, passes_general_filter, passes_gravity_filter,
    VY,
)
from lwp.data.normalization import compute_norm_stats


def load_test_episodes(data_path: str, max_episodes: int = 50) -> list[dict]:
    """Load a few real episodes for testing."""
    npz_files = sorted(glob.glob(f"{data_path}/**/*.npz", recursive=True))[:max_episodes]
    episodes = []
    for f in npz_files:
        data = np.load(f)
        episodes.append({
            "states": data["states"].astype(np.float32),
            "actions": data["actions"].astype(np.float32),
        })
    return episodes


@pytest.fixture(scope="module")
def real_episodes():
    """Load real gym-default episodes."""
    data_path = "/media/hdd1/physics-priors-latent-space/lunar-lander-data/world_model_data/gym-default"
    eps = load_test_episodes(data_path)
    if len(eps) == 0:
        pytest.skip("No gym-default episodes found at data_path")
    return eps


@pytest.fixture(scope="module")
def norm_stats(real_episodes):
    prepared = prepare_episodes_for_norm(real_episodes)
    return compute_norm_stats(prepared)


class TestGTGravityExtraction:
    def test_gt_gravity_delta_is_consistent(self, real_episodes):
        """Check raw GT delta_vy during free fall (no thrust, upright).

        The mean GT delta_vy across qualifying transitions should be
        a consistent negative value (gravity pulls lander down).
        The std should be small (gravity is constant).
        """
        gt_dvys = []
        for ep in real_episodes:
            states = ep["states"][:, :6]
            actions = ep["actions"]
            for t in range(min(len(states) - 1, len(actions))):
                if not passes_general_filter(states[t], t):
                    continue
                if not passes_gravity_filter(states[t], actions[t]):
                    continue
                gt_dvy = states[t + 1][VY] - states[t][VY]
                gt_dvys.append(gt_dvy)

        gt_dvys = np.array(gt_dvys)
        print(f"\nGT gravity (delta_vy per step):")
        print(f"  n_samples: {len(gt_dvys)}")
        print(f"  mean: {gt_dvys.mean():.6f}")
        print(f"  std:  {gt_dvys.std():.6f}")
        print(f"  min:  {gt_dvys.min():.6f}")
        print(f"  max:  {gt_dvys.max():.6f}")

        # Gravity should produce consistent negative delta_vy
        assert gt_dvys.mean() < 0, "Gravity should be negative"
        # Should be very consistent (gravity is a constant)
        assert gt_dvys.std() < abs(gt_dvys.mean()) * 0.1, \
            f"GT gravity std ({gt_dvys.std():.4f}) too high relative to mean ({gt_dvys.mean():.4f})"

    def test_extraction_on_gt_model_matches_gt_values(self, real_episodes, norm_stats):
        """When model IS ground truth, model_mean should equal gt_mean."""
        gt_model = GroundTruthModel(real_episodes, norm_stats)
        result = extract_gravity_oracle(gt_model, norm_stats, real_episodes)

        print(f"\nExtraction on GT model:")
        print(f"  n_samples: {result['n_samples']}")
        print(f"  model_mean: {result['model_mean']:.6f}")
        print(f"  gt_mean:    {result['gt_mean']:.6f}")
        print(f"  rel_error:  {result['relative_error']:.6f}")

        assert result['n_samples'] > 0, "No qualifying gravity samples"
        # Model IS GT, so model_mean should match gt_mean within floating point
        assert abs(result['model_mean'] - result['gt_mean']) < 1e-3, \
            f"GT model extraction ({result['model_mean']:.6f}) doesn't match " \
            f"GT values ({result['gt_mean']:.6f})"
        # Relative error should be near zero
        assert result['relative_error'] < 0.01, \
            f"Relative error {result['relative_error']:.4f} too high for GT model"
