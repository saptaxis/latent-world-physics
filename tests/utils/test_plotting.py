"""Tests for plotting utilities."""
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from lwp.utils.plotting import plot_horizon_curve, plot_per_dim_bars, export_plots


def test_plot_horizon_curve(tmp_path):
    horizon_data = {1: 0.001, 5: 0.01, 10: 0.1, 20: 0.5, 50: 2.0}
    out = tmp_path / "horizon.png"
    plot_horizon_curve(horizon_data, str(out), title="Test Model")
    assert out.exists()


def test_plot_per_dim_bars(tmp_path):
    per_dim = {"x": 0.001, "y": 0.002, "vx": 0.01, "vy": 0.005,
               "angle": 0.003, "angular_vel": 0.008,
               "left_leg": 0.0001, "right_leg": 0.0002}
    out = tmp_path / "per_dim.png"
    plot_per_dim_bars(per_dim, str(out), title="Test Model")
    assert out.exists()


def test_plot_horizon_curve_log_scale(tmp_path):
    horizon_data = {1: 0.001, 5: 0.1, 10: 10.0, 20: 1000.0}
    out = tmp_path / "horizon_log.png"
    plot_horizon_curve(horizon_data, str(out), log_scale=True)
    assert out.exists()


def test_export_plots_creates_pngs(tmp_path):
    tb_dir = tmp_path / "tb"
    writer = SummaryWriter(log_dir=str(tb_dir))
    for i in range(10):
        writer.add_scalar("train/loss", 1.0 / (i + 1), i)
        writer.add_scalar("val/loss", 1.5 / (i + 1), i)
    writer.close()

    plot_dir = tmp_path / "plots"
    export_plots(str(tb_dir), str(plot_dir))

    assert plot_dir.exists()
    png_files = list(plot_dir.glob("*.png"))
    assert len(png_files) >= 2


def test_export_plots_skips_short_series(tmp_path):
    tb_dir = tmp_path / "tb"
    writer = SummaryWriter(log_dir=str(tb_dir))
    writer.add_scalar("single/point", 1.0, 0)
    writer.add_scalar("multi/points", 1.0, 0)
    writer.add_scalar("multi/points", 0.5, 1)
    writer.close()

    plot_dir = tmp_path / "plots"
    export_plots(str(tb_dir), str(plot_dir))

    png_files = list(plot_dir.glob("*.png"))
    assert len(png_files) == 1


def test_export_plots_empty_dir(tmp_path):
    tb_dir = tmp_path / "tb"
    tb_dir.mkdir()
    plot_dir = tmp_path / "plots"
    export_plots(str(tb_dir), str(plot_dir))
