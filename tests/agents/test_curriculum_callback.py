"""Tests for CurriculumCallback — profile switching during training."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

REPO_ROOT = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, REPO_ROOT)

from parametric_lunar_lander.sampling_profiles import CurriculumSchedule


class TestCurriculumCallback:
    """Tests for CurriculumCallback using mocked training env."""

    def _make_callback(self, schedule_str="easy:0,medium:500K,hard:1M"):
        """Create a CurriculumCallback with a mocked training env.

        SB3's BaseCallback.training_env is a read-only property that
        delegates to self.model.get_env(). We mock the model so that
        get_env() returns our mock vec env.
        """
        from scripts.agents.train_rl import CurriculumCallback

        schedule = CurriculumSchedule.from_string(schedule_str)
        cb = CurriculumCallback(schedule=schedule, verbose=1)

        # Mock model.get_env() -> mock_vec_env (this is what training_env returns)
        mock_vec_env = MagicMock()
        mock_model = MagicMock()
        mock_model.get_env.return_value = mock_vec_env
        cb.model = mock_model
        cb.num_timesteps = 0

        return cb, mock_vec_env

    def test_on_training_start_sets_initial_profile(self):
        """_on_training_start sets the first profile."""
        cb, mock_env = self._make_callback()
        cb.num_timesteps = 0

        cb._on_training_start()

        mock_env.env_method.assert_called_with("set_profile", "easy")

    def test_on_training_start_resume_sets_correct_profile(self):
        """When resuming at step 600K, should set medium (not easy)."""
        cb, mock_env = self._make_callback()
        cb.num_timesteps = 600_000

        cb._on_training_start()

        mock_env.env_method.assert_called_with("set_profile", "medium")

    def test_on_step_transitions_at_threshold(self):
        """Profile switches exactly at the threshold step."""
        cb, mock_env = self._make_callback()

        # Start training
        cb.num_timesteps = 0
        cb._on_training_start()
        mock_env.env_method.reset_mock()

        # Step just before threshold — no switch
        cb.num_timesteps = 499_999
        cb._on_step()
        mock_env.env_method.assert_not_called()

        # Step at threshold — switch happens
        cb.num_timesteps = 500_000
        cb._on_step()
        mock_env.env_method.assert_called_once_with("set_profile", "medium")

    def test_no_redundant_calls(self):
        """set_profile is NOT called when the profile hasn't changed."""
        cb, mock_env = self._make_callback()

        cb.num_timesteps = 0
        cb._on_training_start()
        mock_env.env_method.reset_mock()

        # Multiple steps within same stage — no calls
        for step in [100, 1000, 10_000, 100_000, 499_999]:
            cb.num_timesteps = step
            cb._on_step()

        mock_env.env_method.assert_not_called()

    def test_multiple_transitions(self):
        """Track transitions across the full schedule."""
        cb, mock_env = self._make_callback("easy:0,medium:500K,hard:1M,full:2M")

        # Simulate training progression
        cb.num_timesteps = 0
        cb._on_training_start()
        assert cb._current_profile == "easy"

        cb.num_timesteps = 500_000
        cb._on_step()
        assert cb._current_profile == "medium"

        cb.num_timesteps = 1_000_000
        cb._on_step()
        assert cb._current_profile == "hard"

        cb.num_timesteps = 2_000_000
        cb._on_step()
        assert cb._current_profile == "full"

    def test_returns_true(self):
        """_on_step always returns True (don't halt training)."""
        cb, mock_env = self._make_callback()
        cb.num_timesteps = 0
        cb._on_training_start()

        assert cb._on_step() is True
