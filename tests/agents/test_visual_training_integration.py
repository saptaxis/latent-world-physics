# lunar_lander/tests/test_visual_training_integration.py
"""Integration test: visual agent training pipeline end-to-end."""

import os
import numpy as np
import pytest

from parametric_lunar_lander.wrappers import make_lunar_lander_env
from lwp.rl.training import train


@pytest.fixture
def tmp_run_dir(tmp_path):
    """Temporary directory for training outputs."""
    return str(tmp_path / "test_visual_run")


class TestVisualTrainingIntegration:
    """Verify the full visual training pipeline works end-to-end."""

    @pytest.mark.slow
    def test_visual_train_1k_steps(self, tmp_run_dir):
        """Train a visual agent for 1K steps — smoke test for the full pipeline."""
        def env_fn(seed):
            return make_lunar_lander_env(
                variant="visual", seed=seed, profile="gym-default",
            )

        checkpoint_dir = os.path.join(tmp_run_dir, "checkpoints")
        log_dir = os.path.join(tmp_run_dir, "logs")

        model, vec_env = train(
            env_fn=env_fn,
            algo="ppo",
            total_steps=1024,  # exactly 1 rollout buffer (n_steps=1024)
            n_envs=2,
            seed=42,
            log_dir=log_dir,
            checkpoint_dir=checkpoint_dir,
            checkpoint_freq=10_000,  # won't fire at 1K steps
            eval_freq=0,  # skip eval for speed
            policy_class="CnnPolicy",
            norm_obs=False,
            n_stack=4,
        )

        # Verify model saved
        assert os.path.exists(os.path.join(checkpoint_dir, "model.zip"))

        # Verify observation shape through the full pipeline.
        # VecFrameStack stacks 4 grayscale frames along channel dim: (84, 84, 4).
        # SB3 internally transposes to CHW but obs from vec_env is still HWC.
        obs = vec_env.reset()
        # Shape: (n_envs, H, W, n_stack) = (2, 84, 84, 4)
        assert obs.shape == (2, 84, 84, 4), f"Unexpected obs shape: {obs.shape}"
        assert obs.dtype == np.uint8

        # Verify we can run inference
        action, _ = model.predict(obs, deterministic=True)
        assert action.shape == (2, 2)  # n_envs=2, action_dim=2

        vec_env.close()

    @pytest.mark.slow
    def test_visual_eval_after_train(self, tmp_run_dir):
        """Train, then eval — tests evaluate_agent with VecFrameStack."""
        from lwp.agents.eval_utils import evaluate_agent

        def env_fn(seed):
            return make_lunar_lander_env(
                variant="visual", seed=seed, profile="gym-default",
            )

        checkpoint_dir = os.path.join(tmp_run_dir, "checkpoints")
        log_dir = os.path.join(tmp_run_dir, "logs")

        model, vec_env = train(
            env_fn=env_fn,
            algo="ppo",
            total_steps=1024,
            n_envs=2,
            seed=42,
            log_dir=log_dir,
            checkpoint_dir=checkpoint_dir,
            checkpoint_freq=10_000,
            eval_freq=0,
            policy_class="CnnPolicy",
            norm_obs=False,
            n_stack=4,
        )
        vec_env.close()

        # Evaluate using the saved VecNormalize stats
        result = evaluate_agent(
            model=model,
            env_fn=env_fn,
            n_episodes=3,
            seed=99,
            vec_normalize_path=os.path.join(checkpoint_dir, "vec_normalize.pkl"),
            n_stack=4,
        )

        assert result["summary"]["n_episodes"] == 3
        # Every episode should have a valid outcome
        for ep in result["episodes"]:
            assert ep["outcome"] in ("landed", "crashed", "out_of_bounds", "timeout")
