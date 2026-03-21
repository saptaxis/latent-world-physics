"""Tests for post-training reporting utilities."""
from pathlib import Path

import torch

from lwp.utils.reporting import (
    format_horizon_table,
    format_per_dim_table,
    generate_eval_report,
)


def test_format_per_dim_table():
    per_dim = {"x": 0.001, "y": 0.002, "vx": 0.01, "vy": 0.005,
               "angle": 0.003, "angular_vel": 0.008, "left_leg": 0.0001, "right_leg": 0.0002}
    table = format_per_dim_table(per_dim)
    assert "x" in table
    assert "0.001" in table
    assert "---" in table


def test_format_horizon_table():
    horizon_data = {
        1: {"x": 0.001, "y": 0.002},
        5: {"x": 0.01, "y": 0.02},
        10: {"x": 0.1, "y": 0.2},
    }
    table = format_horizon_table(horizon_data)
    assert "h=1" in table or "1" in table
    assert "---" in table


def test_generate_eval_report():
    results = {
        "per_dim_mse": {"x": 0.001, "y": 0.002, "vx": 0.01, "vy": 0.005,
                        "angle": 0.003, "angular_vel": 0.008,
                        "left_leg": 0.0001, "right_leg": 0.0002},
        "horizon_mean_mse": {1: 0.001, 5: 0.01, 10: 0.1, 20: 0.5},
        "divergence_exponent": 0.05,
        "horizon_to_failure": 10,
    }
    report = generate_eval_report("mlp-test", results)
    assert "mlp-test" in report
    assert "divergence" in report.lower() or "Divergence" in report
    assert "0.05" in report


def test_generate_eval_report_saves_to_file(tmp_path):
    results = {
        "per_dim_mse": {"x": 0.001, "y": 0.002},
        "horizon_mean_mse": {1: 0.001, 5: 0.01},
        "divergence_exponent": 0.05,
        "horizon_to_failure": 5,
    }
    out_path = tmp_path / "report.md"
    report = generate_eval_report("test-run", results, output_path=str(out_path))
    assert out_path.exists()
    content = out_path.read_text()
    assert "test-run" in content
