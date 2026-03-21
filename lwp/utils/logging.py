"""TensorBoard logging helpers."""
from __future__ import annotations

DIM_NAMES_8D = ["x", "y", "vx", "vy", "angle", "angular_vel", "left_leg", "right_leg"]


def get_dim_names(dim_names: list[str] | None, state_dim: int) -> list[str]:
    """Return dimension names, falling back to generic names."""
    if dim_names is not None:
        return dim_names[:state_dim]
    return [f"dim_{i}" for i in range(state_dim)]


class TrainLogger:
    """Thin wrapper around TensorBoard SummaryWriter."""

    def __init__(self, writer):
        self.writer = writer

    def log_scalar(self, tag: str, value: float, step: int):
        self.writer.add_scalar(tag, value, step)

    def log_per_dim(self, tag: str, values, dim_names: list[str], step: int):
        for name, val in zip(dim_names, values):
            self.writer.add_scalar(f"{tag}/{name}", val, step)

    def log_dict(self, d: dict, prefix: str, step: int):
        for key, val in d.items():
            self.writer.add_scalar(f"{prefix}/{key}", val, step)
