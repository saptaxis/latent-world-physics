# lunar_lander/tests/test_visual_wrappers.py
"""Tests for visual observation wrappers."""

import numpy as np
import pytest
from parametric_lunar_lander.wrappers import make_lunar_lander_env


class TestVisualWrapperStack:
    """Verify the visual variant produces correct observation shape and dtype."""

    def test_visual_obs_shape_single_frame(self):
        """Single env (before VecFrameStack) should output (84, 84, 1) HWC."""
        env = make_lunar_lander_env(variant="visual", seed=0, profile="gym-default")
        obs, _ = env.reset()
        # HWC format — SB3's VecTransposeImage converts to CHW at VecEnv level
        assert obs.shape == (84, 84, 1), f"Expected (84, 84, 1), got {obs.shape}"
        env.close()

    def test_visual_obs_dtype(self):
        """Visual obs should be uint8 (raw pixels, not normalized floats)."""
        env = make_lunar_lander_env(variant="visual", seed=0, profile="gym-default")
        obs, _ = env.reset()
        assert obs.dtype == np.uint8, f"Expected uint8, got {obs.dtype}"
        env.close()

    def test_visual_obs_step(self):
        """Step should also return correct shape and dtype."""
        env = make_lunar_lander_env(variant="visual", seed=0, profile="gym-default")
        env.reset()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (84, 84, 1)
        assert obs.dtype == np.uint8
        env.close()

    def test_visual_has_render_mode(self):
        """Visual env must have render_mode='rgb_array' set on base env."""
        env = make_lunar_lander_env(variant="visual", seed=0, profile="gym-default")
        assert env.unwrapped.render_mode == "rgb_array"
        env.close()

    def test_visual_obs_not_all_zeros(self):
        """Visual obs should contain actual rendered content, not blank frame."""
        env = make_lunar_lander_env(variant="visual", seed=0, profile="gym-default")
        obs, _ = env.reset()
        assert obs.max() > 0, "Frame is all zeros — render pipeline broken"
        env.close()

    def test_visual_domain_randomization(self):
        """Visual variant still gets domain randomization."""
        env = make_lunar_lander_env(variant="visual", seed=0, profile="easy")
        configs = []
        for _ in range(5):
            env.reset()
            configs.append(env.unwrapped._physics_config.gravity)
        # With 'easy' profile, gravity should vary across resets
        assert len(set(configs)) > 1, "Domain randomization not working"
        env.close()

    def test_visual_info_has_physics(self):
        """Info dict should still contain physics_config and outcome."""
        env = make_lunar_lander_env(variant="visual", seed=0, profile="gym-default")
        env.reset()
        _, _, _, _, info = env.step(env.action_space.sample())
        assert "physics_config" in info
        assert "outcome" in info
        env.close()

    def test_visual_invalid_variant_rejected(self):
        """Truly invalid variants are still rejected."""
        with pytest.raises(ValueError, match="variant must be one of"):
            make_lunar_lander_env(variant="invalid", seed=0)

    def test_blind_still_works(self):
        """Existing blind variant unchanged."""
        env = make_lunar_lander_env(variant="blind", seed=0, profile="gym-default")
        obs, _ = env.reset()
        # blind = 8D kinematic + 7 rays = 15D
        assert obs.shape == (15,)
        assert obs.dtype == np.float32
        env.close()
