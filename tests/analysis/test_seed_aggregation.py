"""Tests for seed aggregation logic."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


def _write_fake_tb_events(log_dir, scalars: dict[str, list[tuple[int, float]]]):
    """Write synthetic TensorBoard scalar events."""
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(log_dir=str(log_dir))
    for tag, pairs in scalars.items():
        for step, value in pairs:
            writer.add_scalar(tag, value, global_step=step)
    writer.close()


def _make_seed_dir(base_path, seed: int, landed_pct_curve: list[float],
                   reward_curve: list[float], eval_freq: int = 50000):
    """Create a fake seed run directory with TB events.

    Args:
        base_path: Parent directory for seed dirs.
        seed: Seed number (creates base_path/s{seed}/).
        landed_pct_curve: Values for eval/landed_pct at each eval step.
        reward_curve: Values for eval/mean_reward at each eval step.
        eval_freq: Steps between eval checkpoints.
    """
    seed_dir = base_path / f"s{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    scalars = {
        "eval/landed_pct": [
            (eval_freq * (i + 1), v) for i, v in enumerate(landed_pct_curve)
        ],
        "eval/mean_reward": [
            (eval_freq * (i + 1), v) for i, v in enumerate(reward_curve)
        ],
        "eval/crashed_pct": [
            (eval_freq * (i + 1), 100.0 - v) for i, v in enumerate(landed_pct_curve)
        ],
        "train/entropy_loss": [
            (eval_freq * (i + 1), 1.0 - 0.01 * i) for i in range(len(landed_pct_curve))
        ],
    }
    _write_fake_tb_events(str(seed_dir), scalars)
    return seed_dir


class TestAggregateSeedMetrics:
    """Test aggregate_seed_metrics(): cross-seed statistics."""

    def test_computes_mean_and_std(self, tmp_path):
        """Final metrics have mean, std, median, per_seed across seeds."""
        from lwp.analysis.seed_aggregation import aggregate_seed_metrics

        # Three seeds with slightly different final performance.
        _make_seed_dir(tmp_path, 42,
                       landed_pct_curve=[20, 50, 70, 80, 85, 88, 90, 92],
                       reward_curve=[50, 100, 150, 180, 200, 210, 220, 225])
        _make_seed_dir(tmp_path, 123,
                       landed_pct_curve=[15, 45, 65, 75, 82, 86, 88, 90],
                       reward_curve=[40, 90, 140, 170, 195, 205, 215, 222])
        _make_seed_dir(tmp_path, 456,
                       landed_pct_curve=[25, 55, 72, 82, 87, 89, 91, 94],
                       reward_curve=[55, 105, 155, 185, 205, 215, 225, 230])

        seed_dirs = [str(tmp_path / f"s{s}") for s in [42, 123, 456]]
        result = aggregate_seed_metrics(seed_dirs, seeds=[42, 123, 456])

        # Check structure.
        assert "training_metrics" in result
        assert "landed_pct" in result["training_metrics"]
        landed = result["training_metrics"]["landed_pct"]
        assert "mean" in landed
        assert "std" in landed
        assert "median" in landed
        assert "per_seed" in landed
        assert len(landed["per_seed"]) == 3

    def test_learning_curves_aligned(self, tmp_path):
        """Learning curves have aligned steps across seeds."""
        from lwp.analysis.seed_aggregation import aggregate_seed_metrics

        _make_seed_dir(tmp_path, 42,
                       landed_pct_curve=[20, 50, 80],
                       reward_curve=[50, 100, 200])
        _make_seed_dir(tmp_path, 123,
                       landed_pct_curve=[15, 45, 75],
                       reward_curve=[40, 90, 190])

        seed_dirs = [str(tmp_path / f"s{s}") for s in [42, 123]]
        result = aggregate_seed_metrics(seed_dirs, seeds=[42, 123])

        curves = result["learning_curves"]
        assert "steps" in curves
        assert len(curves["steps"]) == 3
        assert len(curves["landed_pct"]["mean"]) == 3
        assert len(curves["landed_pct"]["per_seed"]["42"]) == 3

    def test_seed_consistency_flags_outlier(self, tmp_path):
        """Seed consistency detects when one seed is an outlier."""
        from lwp.analysis.seed_aggregation import aggregate_seed_metrics

        # Two seeds agree, one is way off.
        _make_seed_dir(tmp_path, 42,
                       landed_pct_curve=[80, 85, 88, 90, 92],
                       reward_curve=[200, 210, 220, 225, 230])
        _make_seed_dir(tmp_path, 123,
                       landed_pct_curve=[78, 83, 87, 89, 91],
                       reward_curve=[195, 208, 218, 223, 228])
        _make_seed_dir(tmp_path, 456,
                       landed_pct_curve=[20, 25, 30, 35, 40],  # way worse
                       reward_curve=[50, 60, 70, 80, 90])

        seed_dirs = [str(tmp_path / f"s{s}") for s in [42, 123, 456]]
        result = aggregate_seed_metrics(seed_dirs, seeds=[42, 123, 456])

        consistency = result["seed_consistency"]
        assert not consistency["consistent"]
        assert len(consistency["flags"]) > 0

    def test_consistent_seeds_pass(self, tmp_path):
        """Close seeds are flagged as consistent."""
        from lwp.analysis.seed_aggregation import aggregate_seed_metrics

        _make_seed_dir(tmp_path, 42,
                       landed_pct_curve=[88, 90, 92],
                       reward_curve=[220, 225, 230])
        _make_seed_dir(tmp_path, 123,
                       landed_pct_curve=[86, 89, 91],
                       reward_curve=[218, 223, 228])
        _make_seed_dir(tmp_path, 456,
                       landed_pct_curve=[87, 90, 93],
                       reward_curve=[222, 227, 232])

        seed_dirs = [str(tmp_path / f"s{s}") for s in [42, 123, 456]]
        result = aggregate_seed_metrics(seed_dirs, seeds=[42, 123, 456])

        assert result["seed_consistency"]["consistent"]
        assert len(result["seed_consistency"]["flags"]) == 0


class TestWriteConfigOutputs:
    """Test write_config_outputs(): JSON artifact writing."""

    def test_writes_metrics_json(self, tmp_path):
        """metrics.json contains training_metrics + metadata."""
        from lwp.analysis.seed_aggregation import write_config_outputs

        config_data = {
            "variant": "blind",
            "condition": "full-variation",
            "profile": "easy",
            "net_arch": [128, 128],
            "ent_coef": 0.001,
            "seeds": [42, 123, 456],
        }
        agg_result = {
            "training_metrics": {
                "landed_pct": {"mean": 88.0, "std": 2.0, "median": 89.0, "per_seed": [86, 89, 89]},
            },
            "training_dynamics": {},
            "learning_curves": {"steps": [50000], "landed_pct": {"mean": [88.0], "std": [2.0], "per_seed": {}}},
            "seed_consistency": {"consistent": True, "flags": []},
        }
        output_dir = tmp_path / "output"
        write_config_outputs(
            config_name="blind-ppo-easy-128-lowent",
            config_data=config_data,
            agg_result=agg_result,
            output_dir=str(output_dir),
        )

        metrics_path = output_dir / "metrics.json"
        assert metrics_path.exists()
        with open(metrics_path) as f:
            data = json.load(f)
        assert data["config_name"] == "blind-ppo-easy-128-lowent"
        assert data["variant"] == "blind"
        assert data["n_seeds"] == 3
        assert data["training_metrics"]["landed_pct"]["mean"] == 88.0

    def test_writes_learning_curves_json(self, tmp_path):
        """learning_curves.json written with step-aligned data."""
        from lwp.analysis.seed_aggregation import write_config_outputs

        agg_result = {
            "training_metrics": {},
            "training_dynamics": {},
            "learning_curves": {
                "steps": [50000, 100000],
                "landed_pct": {"mean": [50.0, 80.0], "std": [5.0, 3.0], "per_seed": {"42": [55, 82]}},
            },
            "seed_consistency": {"consistent": True, "flags": []},
        }
        output_dir = tmp_path / "output"
        write_config_outputs(
            config_name="test",
            config_data={"variant": "blind", "condition": "test", "seeds": [42]},
            agg_result=agg_result,
            output_dir=str(output_dir),
        )

        curves_path = output_dir / "learning_curves.json"
        assert curves_path.exists()
        with open(curves_path) as f:
            data = json.load(f)
        assert data["steps"] == [50000, 100000]

    def test_writes_seed_consistency_json(self, tmp_path):
        """seed_consistency.json written."""
        from lwp.analysis.seed_aggregation import write_config_outputs

        agg_result = {
            "training_metrics": {},
            "training_dynamics": {},
            "learning_curves": {"steps": []},
            "seed_consistency": {"consistent": False, "flags": ["outlier detected"]},
        }
        output_dir = tmp_path / "output"
        write_config_outputs(
            config_name="test",
            config_data={"variant": "blind", "condition": "test", "seeds": [42]},
            agg_result=agg_result,
            output_dir=str(output_dir),
        )

        consistency_path = output_dir / "seed_consistency.json"
        assert consistency_path.exists()
        with open(consistency_path) as f:
            data = json.load(f)
        assert not data["consistent"]


class TestWriteExperimentSummary:
    """Test write_experiment_summary(): txt + csv summary tables."""

    def test_writes_summary_txt(self, tmp_path):
        """summary_table.txt is human-readable."""
        from lwp.analysis.seed_aggregation import write_experiment_summary

        configs = {
            "full-variation/blind-ppo-easy-128-lowent": {
                "n_seeds": 3,
                "training_metrics": {
                    "landed_pct": {"mean": 88.0, "std": 2.0},
                    "mean_reward": {"mean": 225.0, "std": 3.0},
                },
            },
        }
        write_experiment_summary(
            experiment_name="test-experiment",
            configs_results=configs,
            output_dir=str(tmp_path),
        )

        txt_path = tmp_path / "summary_table.txt"
        assert txt_path.exists()
        content = txt_path.read_text()
        assert "test-experiment" in content
        assert "88.0" in content

    def test_writes_summary_csv(self, tmp_path):
        """summary_table.csv has correct columns."""
        from lwp.analysis.seed_aggregation import write_experiment_summary

        configs = {
            "full-variation/blind-ppo-easy-128-lowent": {
                "n_seeds": 3,
                "training_metrics": {
                    "landed_pct": {"mean": 88.0, "std": 2.0},
                    "mean_reward": {"mean": 225.0, "std": 3.0},
                },
            },
        }
        write_experiment_summary(
            experiment_name="test-experiment",
            configs_results=configs,
            output_dir=str(tmp_path),
        )

        import csv
        csv_path = tmp_path / "summary_table.csv"
        assert csv_path.exists()
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 1
        assert "config" in reader.fieldnames
        assert "landed_pct_mean" in reader.fieldnames
