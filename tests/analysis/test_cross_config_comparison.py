"""Tests for cross-config comparison module."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _make_metrics_csv(
    dir_path: Path,
    n_episodes: int = 10,
    landed_pct: float = 0.8,
    mean_reward: float = 220.0,
    seed: int = 42,
) -> Path:
    """Create a fake metrics.csv with controllable outcomes.

    Args:
        dir_path: Directory to write into (creates trajectories/ subdir).
        n_episodes: Number of episode rows.
        landed_pct: Fraction that should have outcome='landed'.
        mean_reward: Average total_reward (with small noise).
        seed: RNG seed for reproducibility.
    """
    rng = np.random.RandomState(seed)
    traj_dir = dir_path / "trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)

    n_landed = int(n_episodes * landed_pct)
    outcomes = ["landed"] * n_landed + ["crashed"] * (n_episodes - n_landed)
    rewards = rng.normal(mean_reward, 10.0, n_episodes)

    rows = []
    for i in range(n_episodes):
        rows.append(
            {
                "npz_path": f"episode_{i:04d}.npz",
                "outcome": outcomes[i],
                "episode_steps": rng.randint(50, 500),
                "total_reward": rewards[i],
                "profile": "easy",
                "gravity": -10.0,
                "main_engine_power": 13.0,
                "side_engine_power": 0.6,
                "lander_density": 5.0,
                "angular_damping": 0.5,
                "wind_power": 0.0,
                "turbulence_power": 0.0,
                "twr": 0.5,
                "max_altitude": rng.uniform(0.5, 1.5),
                "max_lateral_drift": rng.uniform(0.0, 0.5),
                "landing_x_error": rng.uniform(0.0, 0.2),
                "landing_vy": rng.uniform(-0.5, 0.0),
                "mean_main_thrust": rng.uniform(0.2, 0.6),
                "mean_abs_side_thrust": rng.uniform(0.1, 0.4),
                "thrust_duty_cycle": rng.uniform(0.3, 0.8),
                "side_thrust_reversals": rng.randint(5, 30),
                "total_fuel": rng.uniform(20, 80),
                "fuel_efficiency": rng.uniform(2.0, 8.0),
                "mean_abs_angular_vel": rng.uniform(0.0, 0.3),
                "angle_at_landing": rng.uniform(-0.1, 0.1),
                "hover_time": rng.randint(0, 50),
                "time_to_first_contact": rng.randint(50, 300),
                "std_main_thrust": rng.uniform(0.2, 0.5),
                "std_side_thrust": rng.uniform(0.2, 0.5),
                "main_thrust_frac_full": rng.uniform(0.0, 0.2),
                "main_thrust_frac_zero": rng.uniform(0.3, 0.7),
                "frac_descending": rng.uniform(0.3, 0.7),
                "frac_hovering": rng.uniform(0.05, 0.3),
                "frac_approaching": rng.uniform(0.2, 0.6),
                "frac_correcting": rng.uniform(0.3, 0.8),
                "thrust_autocorr_lag1": rng.uniform(0.5, 0.9),
                "side_thrust_autocorr_lag1": rng.uniform(0.8, 1.0),
            }
        )

    df = pd.DataFrame(rows)
    csv_path = traj_dir / "metrics.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def _make_behavioral_summary(
    dir_path: Path,
    landed_pct: float = 80.0,
    adaptation_score: float = 0.15,
) -> Path:
    """Create a fake behavioral_summary.json."""
    ba_dir = dir_path / "trajectories" / "behavioral_analysis"
    ba_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "n_episodes": 100,
        "landed_pct": landed_pct,
        "adaptation_score": adaptation_score,
        "mean_std_main_thrust": 0.45,
        "mean_std_side_thrust": 0.35,
        "mean_main_thrust_frac_full": 0.05,
        "mean_main_thrust_frac_zero": 0.55,
        "mean_frac_descending": 0.52,
        "mean_frac_hovering": 0.15,
        "mean_frac_approaching": 0.40,
        "mean_frac_correcting": 0.60,
        "mean_thrust_autocorr_lag1": 0.72,
        "mean_side_thrust_autocorr_lag1": 0.95,
    }

    path = ba_dir / "behavioral_summary.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    return path


def _make_comparison_data(
    tmp_path,
    n_seeds=3,
    labeled_landed=0.8,
    blind_landed=0.9,
    n_episodes=10,
):
    """Create fake seed directories with metrics for a comparison.

    Returns a comparison dict in manifest format.
    """
    comp = {
        "condition": "full-variation",
        "profile": "easy",
        "configs": {},
    }

    for variant, landed_pct in [
        ("labeled", labeled_landed),
        ("blind", blind_landed),
    ]:
        seed_base = tmp_path / variant
        seeds = list(range(n_seeds))
        seed_dirs = []

        for i, seed in enumerate(seeds):
            seed_dir = seed_base / f"s{seed}"
            # Vary landed_pct slightly per seed for realism.
            seed_landed = landed_pct + (i - 1) * 0.05
            seed_landed = max(0.0, min(1.0, seed_landed))
            _make_metrics_csv(
                seed_dir,
                n_episodes=n_episodes,
                landed_pct=seed_landed,
                mean_reward=200 + landed_pct * 50,
                seed=seed + i * 100,
            )
            _make_behavioral_summary(
                seed_dir,
                landed_pct=seed_landed * 100,
                adaptation_score=0.1 + i * 0.02,
            )
            seed_dirs.append(str(seed_dir))

        comp["configs"][variant] = {
            "seed_base": str(seed_base),
            "seeds": seeds,
            "seed_dirs": seed_dirs,
        }

    return comp


# ============================================================
# Task 2: load_comparison_metrics + compute_variant_stats
# ============================================================


class TestLoadComparisonMetrics:
    """Test load_comparison_metrics(): loading per-seed DataFrames."""

    def test_loads_all_seeds(self, tmp_path):
        from lwp.analysis.cross_config_comparison import (
            load_comparison_metrics,
        )

        comp = _make_comparison_data(tmp_path, n_seeds=2)
        result = load_comparison_metrics(comp)

        assert "labeled" in result
        assert "blind" in result
        assert len(result["labeled"]) == 2
        assert len(result["blind"]) == 2

    def test_each_seed_is_dataframe(self, tmp_path):
        from lwp.analysis.cross_config_comparison import (
            load_comparison_metrics,
        )

        comp = _make_comparison_data(tmp_path, n_seeds=2)
        result = load_comparison_metrics(comp)

        for df in result["labeled"]:
            assert isinstance(df, pd.DataFrame)
            assert "total_reward" in df.columns
            assert "outcome" in df.columns

    def test_missing_metrics_csv_raises(self, tmp_path):
        from lwp.analysis.cross_config_comparison import (
            load_comparison_metrics,
        )

        comp = {
            "configs": {
                "labeled": {
                    "seed_dirs": [str(tmp_path / "nonexistent" / "s0")],
                    "seeds": [0],
                },
            },
        }

        with pytest.raises(FileNotFoundError, match="metrics.csv"):
            load_comparison_metrics(comp)


class TestComputeVariantStats:
    """Test compute_variant_stats(): per-seed aggregation."""

    def test_returns_per_seed_means(self, tmp_path):
        from lwp.analysis.cross_config_comparison import (
            compute_variant_stats,
            load_comparison_metrics,
        )

        comp = _make_comparison_data(tmp_path, n_seeds=3)
        metrics = load_comparison_metrics(comp)
        stats = compute_variant_stats(metrics["labeled"])

        # Should have per_seed list with one value per seed.
        assert "landed_pct" in stats
        assert len(stats["landed_pct"]["per_seed"]) == 3
        assert "mean" in stats["landed_pct"]
        assert "std" in stats["landed_pct"]

    def test_computes_mean_reward(self, tmp_path):
        from lwp.analysis.cross_config_comparison import (
            compute_variant_stats,
            load_comparison_metrics,
        )

        comp = _make_comparison_data(tmp_path, n_seeds=2)
        metrics = load_comparison_metrics(comp)
        stats = compute_variant_stats(metrics["blind"])

        assert "mean_reward" in stats
        assert stats["mean_reward"]["mean"] > 0

    def test_computes_behavioral_metrics(self, tmp_path):
        from lwp.analysis.cross_config_comparison import (
            compute_variant_stats,
            load_comparison_metrics,
        )

        comp = _make_comparison_data(tmp_path, n_seeds=2)
        metrics = load_comparison_metrics(comp)
        stats = compute_variant_stats(metrics["labeled"])

        # Fuel efficiency and thrust duty cycle should be present.
        assert "fuel_efficiency" in stats
        assert "thrust_duty_cycle" in stats


# ============================================================
# Task 3: run_statistical_tests
# ============================================================


class TestRunStatisticalTests:
    """Test run_statistical_tests(): Mann-Whitney U + Cohen's d."""

    def test_returns_test_results_per_metric(self, tmp_path):
        from lwp.analysis.cross_config_comparison import (
            compute_variant_stats,
            load_comparison_metrics,
            run_statistical_tests,
        )

        comp = _make_comparison_data(
            tmp_path, n_seeds=3, labeled_landed=0.7, blind_landed=0.95
        )
        metrics = load_comparison_metrics(comp)
        stats = {v: compute_variant_stats(dfs) for v, dfs in metrics.items()}
        result = run_statistical_tests(stats, ["labeled", "blind"])

        assert "landed_pct" in result
        assert "p_value" in result["landed_pct"]
        assert "effect_size_cohens_d" in result["landed_pct"]
        assert "u_stat" in result["landed_pct"]

    def test_winner_field_set(self, tmp_path):
        from lwp.analysis.cross_config_comparison import (
            compute_variant_stats,
            load_comparison_metrics,
            run_statistical_tests,
        )

        # blind has higher landed_pct, so should be the winner.
        comp = _make_comparison_data(
            tmp_path, n_seeds=3, labeled_landed=0.5, blind_landed=0.95
        )
        metrics = load_comparison_metrics(comp)
        stats = {v: compute_variant_stats(dfs) for v, dfs in metrics.items()}
        result = run_statistical_tests(stats, ["labeled", "blind"])

        assert result["landed_pct"]["winner"] == "blind"

    def test_handles_n_equals_2(self, tmp_path):
        """With N=2, test still runs but may not be meaningful."""
        from lwp.analysis.cross_config_comparison import (
            compute_variant_stats,
            load_comparison_metrics,
            run_statistical_tests,
        )

        comp = _make_comparison_data(tmp_path, n_seeds=2)
        metrics = load_comparison_metrics(comp)
        stats = {v: compute_variant_stats(dfs) for v, dfs in metrics.items()}
        result = run_statistical_tests(stats, ["labeled", "blind"])

        # Should not crash, p_value may be NaN or 1.0.
        assert "landed_pct" in result
        assert "p_value" in result["landed_pct"]

    def test_per_variant_stats_included(self, tmp_path):
        """Result includes per_variant summary for each metric."""
        from lwp.analysis.cross_config_comparison import (
            compute_variant_stats,
            load_comparison_metrics,
            run_statistical_tests,
        )

        comp = _make_comparison_data(tmp_path, n_seeds=3)
        metrics = load_comparison_metrics(comp)
        stats = {v: compute_variant_stats(dfs) for v, dfs in metrics.items()}
        result = run_statistical_tests(stats, ["labeled", "blind"])

        assert "per_variant" in result["landed_pct"]
        assert "labeled" in result["landed_pct"]["per_variant"]
        assert "blind" in result["landed_pct"]["per_variant"]


# ============================================================
# Task 4: write_comparison_outputs
# ============================================================


class TestWriteComparisonOutputs:
    """Test output writers: comparison_table.txt, .csv, stat_tests.json."""

    def _make_all_comparison_results(self, tmp_path):
        """Run full pipeline to get results dict for writing."""
        from lwp.analysis.cross_config_comparison import (
            compute_variant_stats,
            load_comparison_metrics,
            run_statistical_tests,
        )

        comp = _make_comparison_data(
            tmp_path / "data", n_seeds=3, labeled_landed=0.7, blind_landed=0.9
        )
        metrics = load_comparison_metrics(comp)
        variant_stats = {
            v: compute_variant_stats(dfs) for v, dfs in metrics.items()
        }
        test_results = run_statistical_tests(variant_stats, ["labeled", "blind"])

        return {
            "full-variation-easy": {
                "condition": "full-variation",
                "profile": "easy",
                "variants": ["labeled", "blind"],
                "variant_stats": variant_stats,
                "test_results": test_results,
                "n_seeds": {"labeled": 3, "blind": 3},
                "n_episodes": {"labeled": 30, "blind": 30},
                "seed_dfs": metrics,
                "seeds": {"labeled": [0, 1, 2], "blind": [0, 1, 2]},
            },
        }

    def test_writes_comparison_table_txt(self, tmp_path):
        from lwp.analysis.cross_config_comparison import (
            write_comparison_outputs,
        )

        results = self._make_all_comparison_results(tmp_path)
        out_dir = tmp_path / "output"
        write_comparison_outputs("test-experiment", results, str(out_dir))

        txt_path = out_dir / "comparison_table.txt"
        assert txt_path.exists()
        content = txt_path.read_text()
        assert "full-variation-easy" in content
        assert "labeled" in content
        assert "blind" in content

    def test_writes_comparison_table_csv(self, tmp_path):
        from lwp.analysis.cross_config_comparison import (
            write_comparison_outputs,
        )

        results = self._make_all_comparison_results(tmp_path)
        out_dir = tmp_path / "output"
        write_comparison_outputs("test-experiment", results, str(out_dir))

        csv_path = out_dir / "comparison_table.csv"
        assert csv_path.exists()
        df = pd.read_csv(csv_path)
        assert "comparison" in df.columns
        assert "variant" in df.columns
        assert len(df) == 2  # labeled + blind

    def test_writes_stat_tests_json(self, tmp_path):
        from lwp.analysis.cross_config_comparison import (
            write_comparison_outputs,
        )

        results = self._make_all_comparison_results(tmp_path)
        out_dir = tmp_path / "output"
        write_comparison_outputs("test-experiment", results, str(out_dir))

        json_path = out_dir / "stat_tests.json"
        assert json_path.exists()
        with open(json_path) as f:
            data = json.load(f)
        assert "full-variation-easy" in data
        assert "landed_pct" in data["full-variation-easy"]

    def test_writes_per_comparison_csv(self, tmp_path):
        from lwp.analysis.cross_config_comparison import (
            write_comparison_outputs,
        )

        results = self._make_all_comparison_results(tmp_path)
        out_dir = tmp_path / "output"
        write_comparison_outputs("test-experiment", results, str(out_dir))

        per_comp = out_dir / "full-variation-easy" / "metrics_by_variant.csv"
        assert per_comp.exists()
        df = pd.read_csv(per_comp)
        assert "variant" in df.columns
        assert "seed" in df.columns


# ============================================================
# Task 5: plots
# ============================================================


class TestPlots:
    """Test plot generation (smoke tests -- verify PNGs are created)."""

    def _make_all_results_with_seed_dfs(self, tmp_path):
        """Full pipeline results including seed_dfs for plots."""
        from lwp.analysis.cross_config_comparison import (
            compute_variant_stats,
            load_comparison_metrics,
            run_statistical_tests,
        )

        comp = _make_comparison_data(
            tmp_path / "data", n_seeds=3, labeled_landed=0.7, blind_landed=0.9
        )
        metrics = load_comparison_metrics(comp)
        variant_stats = {
            v: compute_variant_stats(dfs) for v, dfs in metrics.items()
        }
        test_results = run_statistical_tests(variant_stats, ["labeled", "blind"])

        return {
            "full-variation-easy": {
                "condition": "full-variation",
                "profile": "easy",
                "variants": ["labeled", "blind"],
                "variant_stats": variant_stats,
                "test_results": test_results,
                "n_seeds": {"labeled": 3, "blind": 3},
                "n_episodes": {"labeled": 30, "blind": 30},
            },
        }

    def test_plot_performance_bars(self, tmp_path):
        from lwp.analysis.cross_config_comparison import (
            plot_performance_bars,
        )

        results = self._make_all_results_with_seed_dfs(tmp_path)
        out_path = tmp_path / "output" / "performance_bars.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plot_performance_bars(results, str(out_path))

        assert out_path.exists()
        assert out_path.stat().st_size > 0

    def test_plot_outcome_breakdown(self, tmp_path):
        from lwp.analysis.cross_config_comparison import (
            plot_outcome_breakdown,
        )

        results = self._make_all_results_with_seed_dfs(tmp_path)
        comp = results["full-variation-easy"]
        out_path = tmp_path / "output" / "outcome_breakdown.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plot_outcome_breakdown(
            comp["variant_stats"],
            comp["variants"],
            "full-variation-easy",
            str(out_path),
        )

        assert out_path.exists()
        assert out_path.stat().st_size > 0

    def test_plot_behavioral_comparison(self, tmp_path):
        from lwp.analysis.cross_config_comparison import (
            plot_behavioral_comparison,
        )

        results = self._make_all_results_with_seed_dfs(tmp_path)
        out_path = tmp_path / "output" / "behavioral_comparison.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plot_behavioral_comparison(results, str(out_path))

        assert out_path.exists()
        assert out_path.stat().st_size > 0


# ============================================================
# trajectory_subdir support
# ============================================================


class TestTrajectorySubdir:
    """Test trajectory_subdir override for corruption experiments."""

    def _make_metrics_in_subdir(self, seed_dir, subdir_name, n_episodes=10, seed=42):
        """Create metrics.csv in a custom subdir (not 'trajectories')."""
        rng = np.random.RandomState(seed)
        traj_dir = seed_dir / subdir_name
        traj_dir.mkdir(parents=True, exist_ok=True)

        n_landed = int(n_episodes * 0.5)
        outcomes = ["landed"] * n_landed + ["crashed"] * (n_episodes - n_landed)
        rewards = rng.normal(100.0, 10.0, n_episodes)

        rows = []
        for i in range(n_episodes):
            rows.append({
                "npz_path": f"episode_{i:04d}.npz",
                "outcome": outcomes[i],
                "episode_steps": rng.randint(50, 500),
                "total_reward": rewards[i],
                "profile": "easy",
                "gravity": -10.0,
                "main_engine_power": 13.0,
                "side_engine_power": 0.6,
                "lander_density": 5.0,
                "angular_damping": 0.5,
                "wind_power": 0.0,
                "turbulence_power": 0.0,
                "twr": 0.5,
                "fuel_efficiency": rng.uniform(2.0, 8.0),
                "thrust_duty_cycle": rng.uniform(0.3, 0.8),
                "mean_main_thrust": rng.uniform(0.2, 0.6),
                "mean_abs_side_thrust": rng.uniform(0.1, 0.4),
            })

        df = pd.DataFrame(rows)
        df.to_csv(traj_dir / "metrics.csv", index=False)

    def test_load_metrics_custom_subdir(self, tmp_path):
        """Should resolve metrics from custom trajectory_subdir."""
        from lwp.analysis.cross_config_comparison import (
            load_comparison_metrics,
        )

        seed_dir = tmp_path / "s42"
        self._make_metrics_in_subdir(seed_dir, "trajectories-zero", n_episodes=5)

        comparison_data = {
            "configs": {
                "labeled-zeroed": {
                    "seed_dirs": [str(seed_dir)],
                    "seeds": [42],
                    "trajectory_subdir": "trajectories-zero",
                }
            }
        }
        result = load_comparison_metrics(comparison_data)
        assert "labeled-zeroed" in result
        assert len(result["labeled-zeroed"]) == 1
        assert len(result["labeled-zeroed"][0]) == 5

    def test_default_subdir_is_trajectories(self, tmp_path):
        """Without trajectory_subdir, should use 'trajectories' (backwards-compatible)."""
        from lwp.analysis.cross_config_comparison import (
            load_comparison_metrics,
        )

        comp = _make_comparison_data(tmp_path, n_seeds=1)
        result = load_comparison_metrics(comp)
        assert "labeled" in result

    def test_load_behavioral_summaries_custom_subdir(self, tmp_path):
        """Should resolve behavioral summaries from custom subdir."""
        from lwp.analysis.cross_config_comparison import (
            load_behavioral_summaries,
        )

        seed_dir = tmp_path / "s42"
        ba_dir = seed_dir / "trajectories-zero" / "behavioral_analysis"
        ba_dir.mkdir(parents=True, exist_ok=True)
        with open(ba_dir / "behavioral_summary.json", "w") as f:
            json.dump({"adaptation_score": 0.12, "n_episodes": 50}, f)

        comparison_data = {
            "configs": {
                "labeled-zeroed": {
                    "seed_dirs": [str(seed_dir)],
                    "seeds": [42],
                    "trajectory_subdir": "trajectories-zero",
                }
            }
        }
        result = load_behavioral_summaries(comparison_data)
        assert result["labeled-zeroed"][0]["adaptation_score"] == 0.12


# ============================================================
# Multi-group comparison (k>2)
# ============================================================


class TestRunMultiGroupTests:
    """Tests for k>2 group comparison with Kruskal-Wallis + pairwise MW."""

    def _make_variant_stats(self):
        """Create 3 groups: A and B similar, C different."""
        return {
            "group_a": {
                "landed_pct": {
                    "mean": 90.0, "std": 3.0,
                    "per_seed": [88.0, 90.0, 92.0, 91.0, 89.0],
                },
                "frac_upright": {
                    "mean": 0.72, "std": 0.03,
                    "per_seed": [0.70, 0.73, 0.71, 0.74, 0.72],
                },
            },
            "group_b": {
                "landed_pct": {
                    "mean": 91.0, "std": 2.5,
                    "per_seed": [89.0, 92.0, 93.0, 90.0, 91.0],
                },
                "frac_upright": {
                    "mean": 0.73, "std": 0.02,
                    "per_seed": [0.72, 0.74, 0.73, 0.75, 0.71],
                },
            },
            "group_c": {
                "landed_pct": {
                    "mean": 60.0, "std": 5.0,
                    "per_seed": [55.0, 58.0, 62.0, 65.0, 60.0],
                },
                "frac_upright": {
                    "mean": 0.45, "std": 0.05,
                    "per_seed": [0.42, 0.44, 0.48, 0.46, 0.45],
                },
            },
        }

    def test_returns_expected_structure(self):
        from lwp.analysis.cross_config_comparison import run_multi_group_tests

        stats = self._make_variant_stats()
        names = ["group_a", "group_b", "group_c"]
        result = run_multi_group_tests(stats, names)

        assert "landed_pct" in result
        assert "omnibus_p" in result["landed_pct"]
        assert "omnibus_test" in result["landed_pct"]
        assert result["landed_pct"]["omnibus_test"] == "kruskal_wallis"
        assert "pairwise" in result["landed_pct"]

    def test_pairwise_count(self):
        """k=3 groups -> 3 pairwise comparisons."""
        from lwp.analysis.cross_config_comparison import run_multi_group_tests

        stats = self._make_variant_stats()
        names = ["group_a", "group_b", "group_c"]
        result = run_multi_group_tests(stats, names)

        pairs = result["landed_pct"]["pairwise"]
        assert len(pairs) == 3  # (a,b), (a,c), (b,c)

    def test_pairwise_keys_are_tuples(self):
        from lwp.analysis.cross_config_comparison import run_multi_group_tests

        stats = self._make_variant_stats()
        names = ["group_a", "group_b", "group_c"]
        result = run_multi_group_tests(stats, names)

        pairs = result["landed_pct"]["pairwise"]
        for key in pairs:
            assert isinstance(key, tuple)
            assert len(key) == 2

    def test_bonferroni_correction_applied(self):
        """Corrected p-values should be >= uncorrected."""
        from lwp.analysis.cross_config_comparison import run_multi_group_tests

        stats = self._make_variant_stats()
        names = ["group_a", "group_b", "group_c"]
        result = run_multi_group_tests(stats, names)

        for pair, pr in result["landed_pct"]["pairwise"].items():
            if pr["p_value"] is not None and pr["p_corrected"] is not None:
                assert pr["p_corrected"] >= pr["p_value"]

    def test_detects_different_group(self):
        """Group C is clearly different -- omnibus should be significant."""
        from lwp.analysis.cross_config_comparison import run_multi_group_tests

        stats = self._make_variant_stats()
        names = ["group_a", "group_b", "group_c"]
        result = run_multi_group_tests(stats, names)

        # Kruskal-Wallis should detect the difference
        assert result["landed_pct"]["omnibus_p"] < 0.05

        # Pairwise: A vs C and B vs C should be significant
        pairs = result["landed_pct"]["pairwise"]
        assert pairs[("group_a", "group_c")]["p_corrected"] < 0.05
        assert pairs[("group_b", "group_c")]["p_corrected"] < 0.05

    def test_similar_groups_not_significant(self):
        """Groups A and B are similar -- pairwise should NOT be significant."""
        from lwp.analysis.cross_config_comparison import run_multi_group_tests

        stats = self._make_variant_stats()
        names = ["group_a", "group_b", "group_c"]
        result = run_multi_group_tests(stats, names)

        pairs = result["landed_pct"]["pairwise"]
        ab_p = pairs[("group_a", "group_b")]["p_corrected"]
        assert ab_p is None or ab_p > 0.05

    def test_effect_sizes_present(self):
        from lwp.analysis.cross_config_comparison import run_multi_group_tests

        stats = self._make_variant_stats()
        names = ["group_a", "group_b", "group_c"]
        result = run_multi_group_tests(stats, names)

        for pair, pr in result["landed_pct"]["pairwise"].items():
            assert "cohens_d" in pr

    def test_per_variant_stats_included(self):
        """Each metric result should include per-variant means."""
        from lwp.analysis.cross_config_comparison import run_multi_group_tests

        stats = self._make_variant_stats()
        names = ["group_a", "group_b", "group_c"]
        result = run_multi_group_tests(stats, names)

        per_v = result["landed_pct"]["per_variant"]
        assert "group_a" in per_v
        assert "group_b" in per_v
        assert "group_c" in per_v
        assert "mean" in per_v["group_a"]

    def test_two_groups_degrades_gracefully(self):
        """With k=2, should still work (Kruskal-Wallis degrades to MW)."""
        from lwp.analysis.cross_config_comparison import run_multi_group_tests

        stats = self._make_variant_stats()
        names = ["group_a", "group_c"]
        result = run_multi_group_tests(stats, names)

        assert "landed_pct" in result
        assert len(result["landed_pct"]["pairwise"]) == 1

    def test_auto_discovers_metrics(self):
        """With metrics=None, uses all metrics present in all variants."""
        from lwp.analysis.cross_config_comparison import run_multi_group_tests

        stats = self._make_variant_stats()
        names = ["group_a", "group_b", "group_c"]
        result = run_multi_group_tests(stats, names)

        assert "landed_pct" in result
        assert "frac_upright" in result
