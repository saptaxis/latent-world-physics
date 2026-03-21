"""Tests for TensorBoard event file parsing."""

from __future__ import annotations

import numpy as np
import pytest


def _write_fake_tb_events(log_dir, scalars: dict[str, list[tuple[int, float]]]):
    """Write synthetic TensorBoard scalar events for testing.

    Creates real TB event files in log_dir so the parser reads them
    with the actual EventAccumulator — no mocking needed.

    Args:
        log_dir: Directory to write events into.
        scalars: Mapping of tag_name -> list of (step, value) pairs.
    """
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(log_dir=str(log_dir))
    for tag, pairs in scalars.items():
        for step, value in pairs:
            writer.add_scalar(tag, value, global_step=step)
    writer.close()


class TestParseTBEvents:
    """Test parse_tb_events(): reads event files into step/value dicts."""

    def test_reads_all_scalar_tags(self, tmp_path):
        """Parser returns all scalar tags present in the event file."""
        _write_fake_tb_events(tmp_path, {
            "eval/landed_pct": [(100, 10.0), (200, 50.0)],
            "eval/mean_reward": [(100, -50.0), (200, 100.0)],
            "train/entropy_loss": [(50, 1.2), (100, 0.8)],
        })

        from lwp.analysis.tb_parser import parse_tb_events
        result = parse_tb_events(str(tmp_path))

        assert "eval/landed_pct" in result
        assert "eval/mean_reward" in result
        assert "train/entropy_loss" in result

    def test_returns_step_value_tuples(self, tmp_path):
        """Each tag maps to a list of (step, value) tuples, sorted by step."""
        _write_fake_tb_events(tmp_path, {
            "eval/landed_pct": [(200, 80.0), (100, 40.0), (300, 90.0)],
        })

        from lwp.analysis.tb_parser import parse_tb_events
        result = parse_tb_events(str(tmp_path))

        steps = [s for s, v in result["eval/landed_pct"]]
        values = [v for s, v in result["eval/landed_pct"]]
        assert steps == [100, 200, 300], "Steps should be sorted ascending"
        assert values == [40.0, 80.0, 90.0]

    def test_empty_directory_returns_empty_dict(self, tmp_path):
        """No event files -> empty dict, not an error."""
        from lwp.analysis.tb_parser import parse_tb_events
        result = parse_tb_events(str(tmp_path))
        assert result == {}

    def test_nonexistent_directory_raises(self):
        """Missing directory should raise FileNotFoundError."""
        from lwp.analysis.tb_parser import parse_tb_events
        with pytest.raises(FileNotFoundError):
            parse_tb_events("/nonexistent/path")


class TestExtractLastKMetrics:
    """Test extract_last_k_metrics(): last-K averaging from parsed scalars."""

    def test_averages_last_k_values(self):
        """Final metric = mean of last K eval checkpoints."""
        from lwp.analysis.tb_parser import extract_last_k_metrics

        scalars = {
            "eval/landed_pct": [(i * 50000, float(v)) for i, v in
                                enumerate([10, 30, 50, 70, 80, 85, 90, 92, 95, 88])],
        }
        result = extract_last_k_metrics(scalars, tags=["eval/landed_pct"], last_k=5)
        # Last 5: 85, 90, 92, 95, 88
        expected = np.mean([85.0, 90.0, 92.0, 95.0, 88.0])
        assert abs(result["eval/landed_pct"] - expected) < 0.01

    def test_fewer_than_k_uses_all(self):
        """If fewer than K checkpoints exist, use all of them."""
        from lwp.analysis.tb_parser import extract_last_k_metrics

        scalars = {
            "eval/landed_pct": [(50000, 70.0), (100000, 80.0)],
        }
        result = extract_last_k_metrics(scalars, tags=["eval/landed_pct"], last_k=5)
        assert abs(result["eval/landed_pct"] - 75.0) < 0.01

    def test_missing_tag_returns_nan(self):
        """Requested tag not in scalars -> NaN, not an error."""
        from lwp.analysis.tb_parser import extract_last_k_metrics

        result = extract_last_k_metrics({}, tags=["eval/landed_pct"], last_k=5)
        assert np.isnan(result["eval/landed_pct"])

    def test_extracts_multiple_tags(self):
        """Can extract multiple metrics in one call."""
        from lwp.analysis.tb_parser import extract_last_k_metrics

        scalars = {
            "eval/landed_pct": [(100000, 80.0), (200000, 90.0)],
            "eval/mean_reward": [(100000, 200.0), (200000, 250.0)],
        }
        result = extract_last_k_metrics(
            scalars,
            tags=["eval/landed_pct", "eval/mean_reward"],
            last_k=5,
        )
        assert abs(result["eval/landed_pct"] - 85.0) < 0.01
        assert abs(result["eval/mean_reward"] - 225.0) < 0.01
