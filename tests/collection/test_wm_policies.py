"""Tests for world model data collection policy sources."""

import numpy as np

from lwp.collection.wm_policies import make_random_policy


class TestRandomPolicy:
    """Random policy: uniform samples from action space."""

    def test_returns_2d_action(self):
        policy_fn = make_random_policy(seed=42)
        obs = np.zeros(15, dtype=np.float32)
        action = policy_fn(obs)
        assert action.shape == (2,)
        assert action.dtype == np.float32

    def test_actions_within_bounds(self):
        policy_fn = make_random_policy(seed=42)
        obs = np.zeros(15, dtype=np.float32)
        for _ in range(100):
            action = policy_fn(obs)
            assert np.all(action >= -1.0)
            assert np.all(action <= 1.0)

    def test_different_seeds_different_actions(self):
        obs = np.zeros(15, dtype=np.float32)
        a1 = make_random_policy(seed=42)(obs)
        a2 = make_random_policy(seed=99)(obs)
        assert not np.allclose(a1, a2)

    def test_ignores_observation(self):
        """Random policy doesn't depend on obs — actions are purely random."""
        policy_fn = make_random_policy(seed=42)
        a1 = policy_fn(np.zeros(15, dtype=np.float32))
        policy_fn2 = make_random_policy(seed=42)
        a2 = policy_fn2(np.ones(15, dtype=np.float32))
        np.testing.assert_array_equal(a1, a2)


from lwp.collection.wm_policies import make_noisy_expert_policy


class TestNoisyExpertPolicy:
    """Noisy-expert: trained agent action + Gaussian noise, clipped."""

    def _mock_predict(self, obs, deterministic=False):
        """Deterministic mock that returns [0.5, -0.3]."""
        return np.array([0.5, -0.3], dtype=np.float32), None

    def test_returns_2d_action(self):
        policy_fn = make_noisy_expert_policy(
            predict_fn=self._mock_predict, noise_sigma=0.1, seed=42
        )
        obs = np.zeros(15, dtype=np.float32)
        action = policy_fn(obs)
        assert action.shape == (2,)
        assert action.dtype == np.float32

    def test_adds_noise_to_base_action(self):
        """Noisy actions should differ from base [0.5, -0.3]."""
        policy_fn = make_noisy_expert_policy(
            predict_fn=self._mock_predict, noise_sigma=0.5, seed=42
        )
        obs = np.zeros(15, dtype=np.float32)
        actions = [policy_fn(obs) for _ in range(20)]
        # Not all identical to base action
        base = np.array([0.5, -0.3])
        diffs = [np.linalg.norm(a - base) for a in actions]
        assert max(diffs) > 0.01

    def test_actions_clipped_to_bounds(self):
        """Even with large noise, actions stay in [-1, 1]."""
        policy_fn = make_noisy_expert_policy(
            predict_fn=self._mock_predict, noise_sigma=10.0, seed=42
        )
        obs = np.zeros(15, dtype=np.float32)
        for _ in range(100):
            action = policy_fn(obs)
            assert np.all(action >= -1.0)
            assert np.all(action <= 1.0)

    def test_zero_noise_matches_base(self):
        policy_fn = make_noisy_expert_policy(
            predict_fn=self._mock_predict, noise_sigma=0.0, seed=42
        )
        obs = np.zeros(15, dtype=np.float32)
        action = policy_fn(obs)
        np.testing.assert_allclose(action, [0.5, -0.3])
