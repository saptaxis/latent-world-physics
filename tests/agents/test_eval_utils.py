"""Tests for eval_utils — shared evaluation logic."""

import numpy as np
import pytest

from lwp.agents.eval_utils import (
    compute_episode_metrics,
    compute_summary,
    evaluate_agent,
    plot_eval_summary,
)


# ---------------------------------------------------------------------------
# compute_episode_metrics
# ---------------------------------------------------------------------------

class TestComputeEpisodeMetrics:
    """Test per-episode metric extraction from eval records."""

    def _make_record(self, outcome="landed", reward=200.0, steps=300,
                     gravity=-5.0):
        return {
            "reward": reward, "steps": steps,
            "info": {
                "outcome": outcome,
                "physics_config": {
                    "gravity": gravity, "main_engine_power": 15.0,
                    "side_engine_power": 0.8, "lander_density": 4.0,
                    "angular_damping": 2.0, "wind_power": 0.0,
                    "turbulence_power": 0.0,
                },
            },
        }

    def test_extracts_outcome(self):
        """Reads outcome from info dict."""
        metrics = compute_episode_metrics(self._make_record(), episode_idx=0)
        assert metrics["outcome"] == "landed"
        assert metrics["reward"] == 200.0
        assert metrics["steps"] == 300
        assert metrics["gravity"] == -5.0
        assert "twr" in metrics

    def test_timeout_from_none_outcome(self):
        """When outcome is None (truncated), classify as timeout."""
        metrics = compute_episode_metrics(
            self._make_record(outcome=None, reward=-50.0, steps=1000),
            episode_idx=0,
        )
        assert metrics["outcome"] == "timeout"

    def test_out_of_bounds_outcome(self):
        """out_of_bounds passes through directly."""
        metrics = compute_episode_metrics(
            self._make_record(outcome="out_of_bounds", reward=-100.0, steps=80),
            episode_idx=0,
        )
        assert metrics["outcome"] == "out_of_bounds"

    def test_profile_tag(self):
        """Profile name is included in metrics."""
        metrics = compute_episode_metrics(
            self._make_record(), episode_idx=0, profile="easy",
        )
        assert metrics["profile"] == "easy"

    def test_twr_computed(self):
        """TWR should be a positive float."""
        metrics = compute_episode_metrics(self._make_record(), episode_idx=0)
        assert metrics["twr"] is not None
        assert metrics["twr"] > 0


# ---------------------------------------------------------------------------
# compute_summary
# ---------------------------------------------------------------------------

class TestComputeSummary:
    """Test aggregate summary computation."""

    def test_summary_stats(self):
        episodes = [
            {"outcome": "landed", "reward": 200.0, "steps": 300},
            {"outcome": "landed", "reward": 180.0, "steps": 350},
            {"outcome": "crashed", "reward": -120.0, "steps": 50},
            {"outcome": "timeout", "reward": -30.0, "steps": 1000},
        ]
        summary = compute_summary(episodes)
        assert summary["n_episodes"] == 4
        assert summary["landed_pct"] == 50.0
        assert summary["crashed_pct"] == 25.0
        assert summary["timeout_pct"] == 25.0
        assert "mean_reward" in summary
        assert "std_reward" in summary
        assert "mean_steps" in summary

    def test_out_of_bounds_counted(self):
        episodes = [
            {"outcome": "out_of_bounds", "reward": -100.0, "steps": 80},
            {"outcome": "landed", "reward": 200.0, "steps": 300},
        ]
        summary = compute_summary(episodes)
        assert summary["n_out_of_bounds"] == 1
        assert summary["out_of_bounds_pct"] == 50.0

    def test_empty_episodes(self):
        summary = compute_summary([])
        assert summary["n_episodes"] == 0


# ---------------------------------------------------------------------------
# evaluate_agent (integration)
# ---------------------------------------------------------------------------

class TestEvaluateAgent:
    """Integration test: evaluate_agent runs episodes and returns results."""

    def test_returns_episodes_and_summary(self):
        """Smoke test with random agent."""
        from stable_baselines3 import PPO
        from parametric_lunar_lander.wrappers import make_lunar_lander_env

        env_fn = lambda seed: make_lunar_lander_env("blind", seed=seed)
        model = PPO("MlpPolicy", make_lunar_lander_env("blind", seed=0),
                     n_steps=64, device="cpu")

        result = evaluate_agent(model, env_fn, n_episodes=3, seed=0)

        assert "episodes" in result
        assert "summary" in result
        assert len(result["episodes"]) == 3
        assert result["summary"]["n_episodes"] == 3
        # Each episode should have a valid outcome
        for ep in result["episodes"]:
            assert ep["outcome"] in ("landed", "crashed", "out_of_bounds", "timeout")

        model.get_env().close()


# ---------------------------------------------------------------------------
# plot_eval_summary
# ---------------------------------------------------------------------------

class TestPlotEvalSummary:
    """Test plot generation doesn't crash and produces files."""

    def test_generates_png_files(self, tmp_path):
        episodes = [
            {"outcome": "landed", "reward": 200.0, "steps": 300,
             "twr": 8.0, "gravity": -5.0, "profile": "easy"},
            {"outcome": "crashed", "reward": -100.0, "steps": 50,
             "twr": 2.0, "gravity": -10.0, "profile": "easy"},
            {"outcome": "timeout", "reward": -30.0, "steps": 1000,
             "twr": 5.0, "gravity": -7.0, "profile": "easy"},
        ] * 5  # 15 episodes

        plot_eval_summary(episodes, str(tmp_path))

        assert (tmp_path / "outcome_counts.png").exists()
        assert (tmp_path / "reward_by_outcome.png").exists()
        assert (tmp_path / "twr_vs_outcome.png").exists()
        # No profile_breakdown.png (single profile / no per_profile_summaries)
        assert not (tmp_path / "profile_breakdown.png").exists()

    def test_multi_profile_generates_breakdown(self, tmp_path):
        episodes = [
            {"outcome": "landed", "reward": 200.0, "steps": 300,
             "twr": 8.0, "gravity": -5.0, "profile": "easy"},
            {"outcome": "crashed", "reward": -100.0, "steps": 50,
             "twr": 2.0, "gravity": -10.0, "profile": "hard"},
        ]
        per_profile = {
            "easy": {"landed_pct": 100, "crashed_pct": 0,
                     "out_of_bounds_pct": 0, "timeout_pct": 0},
            "hard": {"landed_pct": 0, "crashed_pct": 100,
                     "out_of_bounds_pct": 0, "timeout_pct": 0},
        }
        plot_eval_summary(episodes, str(tmp_path),
                          per_profile_summaries=per_profile)
        assert (tmp_path / "profile_breakdown.png").exists()
