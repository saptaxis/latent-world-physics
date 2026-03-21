"""Tests for OutcomeEvalCallback — training-time outcome metrics."""

from unittest.mock import MagicMock, patch


class TestOutcomeEvalCallback:
    """Tests for OutcomeEvalCallback using mocked evaluate_agent."""

    def _make_callback(self, eval_freq=1000, n_eval_episodes=10):
        from scripts.agents.train_rl import OutcomeEvalCallback

        env_fn = MagicMock()
        cb = OutcomeEvalCallback(
            env_fn=env_fn,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            verbose=0,
        )

        # Mock model — BaseCallback.logger is a property that reads self.model.logger
        mock_model = MagicMock()
        cb.model = mock_model
        cb.num_timesteps = 0
        cb.n_calls = 0

        return cb

    @patch("scripts.agents.train_rl.evaluate_agent")
    def test_logs_outcome_metrics(self, mock_eval):
        """Callback logs landed_pct, crashed_pct, etc. to logger."""
        mock_eval.return_value = {
            "episodes": [
                {"outcome": "landed", "reward": 200, "steps": 300}
            ] * 8 + [
                {"outcome": "crashed", "reward": -100, "steps": 50}
            ] * 2,
            "summary": {
                "n_episodes": 10,
                "mean_reward": 140.0,
                "std_reward": 50.0,
                "landed_pct": 80.0,
                "crashed_pct": 20.0,
                "out_of_bounds_pct": 0.0,
                "timeout_pct": 0.0,
                "mean_steps": 250.0,
                "n_landed": 8,
                "n_crashed": 2,
                "n_out_of_bounds": 0,
                "n_timeout": 0,
            },
        }

        cb = self._make_callback(eval_freq=1000)
        mock_logger = MagicMock()
        # logger is a read-only property that delegates to model.logger
        cb.model.logger = mock_logger

        cb.num_timesteps = 1000
        cb.n_calls = 1000
        cb._on_step()

        # Verify tensorboard metrics were logged
        calls = {c[0][0]: c[0][1] for c in mock_logger.record.call_args_list}
        assert calls["eval/mean_reward"] == 140.0
        assert calls["eval/landed_pct"] == 80.0
        assert calls["eval/crashed_pct"] == 20.0
        assert calls["eval/out_of_bounds_pct"] == 0.0
        assert calls["eval/timeout_pct"] == 0.0

    @patch("scripts.agents.train_rl.evaluate_agent")
    def test_respects_eval_freq(self, mock_eval):
        """Callback only fires at eval_freq intervals."""
        cb = self._make_callback(eval_freq=1000)

        # Step 500 — should NOT trigger eval
        cb.num_timesteps = 500
        cb.n_calls = 500
        cb._on_step()
        mock_eval.assert_not_called()

    @patch("scripts.agents.train_rl.evaluate_agent")
    def test_returns_true(self, mock_eval):
        """_on_step always returns True (don't halt training)."""
        cb = self._make_callback()
        cb.num_timesteps = 0
        cb.n_calls = 0
        # num_timesteps=0 doesn't match eval_freq=1000, so no eval call
        assert cb._on_step() is True
