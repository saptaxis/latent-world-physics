"""Tests for rl_common generic wrappers."""

import numpy as np
import pytest
import gymnasium
from gymnasium import spaces

from lwp.rl.wrappers import HistoryStackWrapper


class SimpleBoxEnv(gymnasium.Env):
    """Minimal env with a flat Box obs space for testing wrappers.

    Each reset returns a fixed obs; each step returns an incrementing counter
    so we can verify history ordering.
    """

    def __init__(self, obs_dim: int = 4):
        super().__init__()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32,
        )
        self.action_space = spaces.Discrete(2)
        self._step_count = 0

    def reset(self, **kwargs):
        self._step_count = 0
        # Initial obs: all ones (distinguishable from step obs)
        obs = np.ones(self.observation_space.shape[0], dtype=np.float32)
        return obs, {}

    def step(self, action):
        self._step_count += 1
        # Each step returns [step_count, step_count, ...] so we can verify ordering
        obs = np.full(
            self.observation_space.shape[0], self._step_count, dtype=np.float32,
        )
        return obs, 0.0, False, False, {}


class TestHistoryStackWrapper:
    def test_obs_shape_default_k(self):
        """With k=8 and obs_dim=4, stacked obs should be (32,)."""
        env = HistoryStackWrapper(SimpleBoxEnv(obs_dim=4), k=8)
        obs, _ = env.reset()
        assert obs.shape == (32,)

    def test_obs_shape_custom_k(self):
        """Different K values produce correct shapes."""
        for k in [1, 4, 16]:
            env = HistoryStackWrapper(SimpleBoxEnv(obs_dim=3), k=k)
            obs, _ = env.reset()
            assert obs.shape == (3 * k,), f"Failed for k={k}"

    def test_reset_fills_with_initial_obs(self):
        """On reset, all K slots should contain the initial observation."""
        env = HistoryStackWrapper(SimpleBoxEnv(obs_dim=4), k=4)
        obs, _ = env.reset()
        # Initial obs from SimpleBoxEnv is all-ones, repeated 4 times
        expected = np.ones(16, dtype=np.float32)
        np.testing.assert_array_equal(obs, expected)

    def test_history_after_k_steps(self):
        """After K steps, history should contain K distinct observations."""
        k = 4
        obs_dim = 3
        env = HistoryStackWrapper(SimpleBoxEnv(obs_dim=obs_dim), k=k)
        env.reset()

        # Take 4 steps — step obs values are [1,1,1], [2,2,2], [3,3,3], [4,4,4]
        for i in range(k):
            obs, _, _, _, _ = env.step(0)

        # After 4 steps, history should be [step1, step2, step3, step4]
        # (initial obs pushed out by the 4th step)
        expected = np.concatenate([
            np.full(obs_dim, i + 1, dtype=np.float32) for i in range(k)
        ])
        np.testing.assert_array_equal(obs, expected)

    def test_history_ordering_oldest_first(self):
        """Oldest observation should be at the start of the stacked vector."""
        k = 3
        obs_dim = 2
        env = HistoryStackWrapper(SimpleBoxEnv(obs_dim=obs_dim), k=k)
        env.reset()

        # Take 2 steps: history should be [init, step1, step2]
        env.step(0)
        obs, _, _, _, _ = env.step(0)

        # Oldest (init = [1,1]) at start, newest (step2 = [2,2]) at end
        assert obs[0] == 1.0  # oldest: init obs
        assert obs[-1] == 2.0  # newest: step 2

    def test_history_slides_as_window(self):
        """After more than K steps, oldest observations drop off."""
        k = 2
        obs_dim = 2
        env = HistoryStackWrapper(SimpleBoxEnv(obs_dim=obs_dim), k=k)
        env.reset()

        # Step 1: history = [init, step1] = [[1,1], [1,1]] wait no
        # init = [1,1], step1 = [1,1] ... SimpleBoxEnv returns ones for init
        # and step_count for steps. step1=[1,1], step2=[2,2], step3=[3,3]
        env.step(0)  # step1: [1,1]
        env.step(0)  # step2: [2,2]
        obs, _, _, _, _ = env.step(0)  # step3: [3,3]

        # With k=2, history should be [step2, step3] = [[2,2], [3,3]]
        expected = np.array([2.0, 2.0, 3.0, 3.0], dtype=np.float32)
        np.testing.assert_array_equal(obs, expected)

    def test_observation_space_bounds(self):
        """Stacked observation space should tile the original bounds."""
        base_env = SimpleBoxEnv(obs_dim=3)
        env = HistoryStackWrapper(base_env, k=4)
        assert env.observation_space.shape == (12,)
        # Low/high should be tiled
        np.testing.assert_array_equal(
            env.observation_space.low,
            np.tile(base_env.observation_space.low, 4),
        )

    def test_reset_clears_previous_history(self):
        """A second reset should not carry over state from the first episode."""
        env = HistoryStackWrapper(SimpleBoxEnv(obs_dim=2), k=3)
        env.reset()
        env.step(0)  # step1: [1,1]
        env.step(0)  # step2: [2,2]

        # Reset again — history should be fresh
        obs, _ = env.reset()
        expected = np.ones(6, dtype=np.float32)  # all init obs
        np.testing.assert_array_equal(obs, expected)

    def test_rejects_non_box_space(self):
        """Should raise if obs space is not Box."""
        env = gymnasium.make("CartPole-v1")
        # CartPole has Box space, so let's make a Dict space env
        class DictObsEnv(gymnasium.Env):
            def __init__(self):
                super().__init__()
                self.observation_space = spaces.Dict({
                    "a": spaces.Box(low=0, high=1, shape=(2,)),
                })
                self.action_space = spaces.Discrete(2)

        with pytest.raises(AssertionError, match="Box obs space"):
            HistoryStackWrapper(DictObsEnv())
        env.close()
