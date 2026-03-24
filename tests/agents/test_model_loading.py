"""Tests for model loading utilities."""
import json
import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from lwp.agents.model_loading import load_model, load_eval_env


# Use real checkpoints for integration tests. Skip if not available.
FROZEN_DIR = (
    "/media/hdd1/physics-priors-latent-space/lunar-lander-networks/"
    "visual_rl_agents/gym-default/"
    "visual-ppo-gym-default-128px-impala-lrdecay-pretrained-frozen-lowlr/s42"
)
BLIND_DIR = (
    "/media/hdd1/physics-priors-latent-space/lunar-lander-networks/"
    "rl_agents/parametric-vs-behavioral/full-variation/"
    "blind-ppo-easy-128-lowent/s42"
)

has_frozen = os.path.exists(FROZEN_DIR)
has_blind = os.path.exists(BLIND_DIR)


class TestLoadModel:
    @pytest.mark.skipif(not has_frozen, reason="Frozen agent checkpoint not available")
    def test_load_visual_agent(self):
        """Load a visual agent with old lunar_lander imports."""
        model, config = load_model(FROZEN_DIR)
        assert config["variant"].startswith("visual")
        assert config["algo"] == "ppo"
        fe_name = type(model.policy.features_extractor).__name__
        assert fe_name == "ImpalaCNN"

    @pytest.mark.skipif(not has_blind, reason="Blind agent checkpoint not available")
    def test_load_state_vector_agent(self):
        """Load a state-vector agent (no compat needed)."""
        model, config = load_model(BLIND_DIR)
        assert not config["variant"].startswith("visual")
        assert config["algo"] == "ppo"

    @pytest.mark.skipif(not has_frozen, reason="Frozen agent checkpoint not available")
    def test_load_model_returns_config(self):
        """Config dict has expected keys from config.json."""
        _, config = load_model(FROZEN_DIR)
        assert "variant" in config
        assert "algo" in config

    def test_load_model_missing_dir(self):
        """Raises FileNotFoundError for nonexistent directory."""
        with pytest.raises(FileNotFoundError):
            load_model("/nonexistent/path")


class TestLoadEvalEnv:
    @pytest.mark.skipif(not has_frozen, reason="Frozen agent checkpoint not available")
    def test_visual_env_obs_shape(self):
        """Visual eval env has (H, W, n_stack) observation space after frame stacking.

        VecFrameStack stacks along the channel axis (last axis for HWC envs),
        producing (H, W, n_stack). SB3's model.predict() handles HWC→CHW
        transposition internally via maybe_transpose().
        """
        _, config = load_model(FROZEN_DIR)
        env = load_eval_env(config, FROZEN_DIR)
        obs_shape = env.observation_space.shape
        n_stack = config.get("n_stack", 4)
        H = config.get("frame_size", 128)
        W = config.get("frame_size", 128)
        assert obs_shape == (H, W, n_stack), f"Expected ({H}, {W}, {n_stack}), got {obs_shape}"
        env.close()

    @pytest.mark.skipif(not has_blind, reason="Blind agent checkpoint not available")
    def test_state_env_no_frame_stack(self):
        """State-vector env has flat observation space, no frame stacking."""
        _, config = load_model(BLIND_DIR)
        env = load_eval_env(config, BLIND_DIR)
        obs_shape = env.observation_space.shape
        assert len(obs_shape) == 1, f"Expected 1D obs, got shape {obs_shape}"
        env.close()

    @pytest.mark.skipif(not has_frozen, reason="Frozen agent checkpoint not available")
    def test_visual_env_with_vecnormalize(self):
        """Visual eval env loads VecNormalize stats when available."""
        _, config = load_model(FROZEN_DIR)
        env = load_eval_env(config, FROZEN_DIR)
        from stable_baselines3.common.vec_env import VecNormalize
        inner = env
        found_vecnorm = False
        while hasattr(inner, "venv"):
            if isinstance(inner, VecNormalize):
                found_vecnorm = True
                assert not inner.training
                assert not inner.norm_reward
                break
            inner = inner.venv
        assert found_vecnorm, "Expected VecNormalize in env stack"
        env.close()
