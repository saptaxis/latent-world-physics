from unittest.mock import MagicMock
from lwp.utils.logging import TrainLogger, get_dim_names


def test_logger_log_scalar():
    writer = MagicMock()
    logger = TrainLogger(writer)
    logger.log_scalar("train/loss", 0.5, step=10)
    writer.add_scalar.assert_called_once_with("train/loss", 0.5, 10)


def test_logger_log_per_dim():
    writer = MagicMock()
    logger = TrainLogger(writer)
    values = [0.1, 0.2, 0.3]
    dim_names = ["x", "y", "vx"]
    logger.log_per_dim("eval/mse", values, dim_names, step=5)
    assert writer.add_scalar.call_count == 3
    writer.add_scalar.assert_any_call("eval/mse/x", 0.1, 5)
    writer.add_scalar.assert_any_call("eval/mse/y", 0.2, 5)
    writer.add_scalar.assert_any_call("eval/mse/vx", 0.3, 5)


def test_logger_log_dict():
    writer = MagicMock()
    logger = TrainLogger(writer)
    logger.log_dict({"a": 1.0, "b": 2.0}, prefix="metrics", step=3)
    assert writer.add_scalar.call_count == 2
    writer.add_scalar.assert_any_call("metrics/a", 1.0, 3)


def test_get_dim_names_none_returns_generic():
    names = get_dim_names(None, state_dim=4)
    assert names == ["dim_0", "dim_1", "dim_2", "dim_3"]


def test_get_dim_names_custom():
    names = get_dim_names(["x", "y", "vx"], state_dim=3)
    assert names == ["x", "y", "vx"]
