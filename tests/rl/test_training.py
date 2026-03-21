"""Smoke tests for rl_common training library.

These tests verify basic functionality — that training runs without
errors, produces model files, and that saved models can be loaded.
They use a simple Box env (not the platformer) to keep tests fast.
"""

import os
import tempfile

import numpy as np
import pytest
import gymnasium
from gymnasium import spaces

from lwp.rl.training import train, load_trained_agent


class SimpleBoxEnv(gymnasium.Env):
    """Minimal continuous-action env for testing the training pipeline.

    Observation: Box(4,) — random features.
    Action: Box(2,) — matches platformer wrapper output shape.
    Reward: negative L2 distance from obs to zero (trivial objective).
    Episodes terminate after 50 steps.
    """

    def __init__(self, seed: int = 0):
        super().__init__()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32,
        )
        self._step_count = 0
        self._rng = np.random.default_rng(seed)

    def reset(self, **kwargs):
        self._step_count = 0
        obs = self._rng.standard_normal(4).astype(np.float32)
        return obs, {}

    def step(self, action):
        self._step_count += 1
        obs = self._rng.standard_normal(4).astype(np.float32)
        reward = -float(np.linalg.norm(obs))
        terminated = False
        truncated = self._step_count >= 50
        return obs, reward, terminated, truncated, {}


def _make_simple_env(seed: int) -> gymnasium.Env:
    """Factory matching the signature expected by train()."""
    return SimpleBoxEnv(seed=seed)


class TestTrainPPO:
    def test_ppo_produces_model_file(self):
        """PPO training for 1000 steps should produce a saved model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_dir = os.path.join(tmpdir, "ckpt")
            log_dir = os.path.join(tmpdir, "logs")

            model, vec_env = train(
                env_fn=_make_simple_env,
                algo="ppo",
                total_steps=1024,  # at least one rollout (n_steps=2048 / n_envs=2)
                n_envs=2,
                seed=42,
                checkpoint_dir=ckpt_dir,
                log_dir=log_dir,
                checkpoint_freq=10_000,  # don't checkpoint mid-test
                eval_freq=10_000,        # don't eval mid-test
                algo_kwargs=dict(n_steps=512),  # smaller rollout for fast test
            )

            assert os.path.exists(os.path.join(ckpt_dir, "model.zip"))
            assert os.path.exists(os.path.join(ckpt_dir, "vec_normalize.pkl"))
            vec_env.close()


class TestTrainSAC:
    def test_sac_produces_model_file(self):
        """SAC training for 1000 steps should produce a saved model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_dir = os.path.join(tmpdir, "ckpt")
            log_dir = os.path.join(tmpdir, "logs")

            model, vec_env = train(
                env_fn=_make_simple_env,
                algo="sac",
                total_steps=500,
                n_envs=2,
                seed=42,
                checkpoint_dir=ckpt_dir,
                log_dir=log_dir,
                checkpoint_freq=10_000,
                eval_freq=10_000,
                algo_kwargs=dict(learning_starts=100),  # less warmup for test
            )

            assert os.path.exists(os.path.join(ckpt_dir, "model.zip"))
            assert os.path.exists(os.path.join(ckpt_dir, "vec_normalize.pkl"))
            vec_env.close()


class TestLoadAgent:
    def test_load_produces_valid_actions(self):
        """Saved model should load and produce deterministic actions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_dir = os.path.join(tmpdir, "ckpt")
            log_dir = os.path.join(tmpdir, "logs")

            # Train a quick model
            model, vec_env = train(
                env_fn=_make_simple_env,
                algo="ppo",
                total_steps=1024,
                n_envs=2,
                seed=42,
                checkpoint_dir=ckpt_dir,
                log_dir=log_dir,
                checkpoint_freq=10_000,
                eval_freq=10_000,
                algo_kwargs=dict(n_steps=512),
            )
            vec_env.close()

            # Load and predict
            loaded_model, loaded_env = load_trained_agent(
                ckpt_dir, algo="ppo", env_fn=_make_simple_env,
            )
            assert loaded_env is not None

            obs = loaded_env.reset()
            action, _ = loaded_model.predict(obs, deterministic=True)
            assert action.shape == (1, 2)  # (n_envs=1, action_dim=2)
            loaded_env.close()

    def test_load_without_env_fn(self):
        """Loading without env_fn should return model only."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_dir = os.path.join(tmpdir, "ckpt")
            log_dir = os.path.join(tmpdir, "logs")

            model, vec_env = train(
                env_fn=_make_simple_env,
                algo="ppo",
                total_steps=1024,
                n_envs=2,
                seed=42,
                checkpoint_dir=ckpt_dir,
                log_dir=log_dir,
                checkpoint_freq=10_000,
                eval_freq=10_000,
                algo_kwargs=dict(n_steps=512),
            )
            vec_env.close()

            loaded_model, loaded_env = load_trained_agent(
                ckpt_dir, algo="ppo",
            )
            assert loaded_model is not None
            assert loaded_env is None
