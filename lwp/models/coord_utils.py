# lwp/models/coord_utils.py
"""Physics-to-grid coordinate transforms for STN compositing.

Lunar Lander physics coordinates → PyTorch grid coordinates.
Grid coords: both axes [-1, 1], y-axis flipped (image y-down).
"""
from __future__ import annotations

import torch


def physics_to_grid(
    gt_x: torch.Tensor,
    gt_y: torch.Tensor,
    x_range: tuple[float, float] = (-1.0, 1.0),
    y_range: tuple[float, float] = (0.0, 1.5),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert physics (x, y) to PyTorch grid coordinates [-1, 1].

    Args:
        gt_x: physics x positions
        gt_y: physics y positions
        x_range: (min, max) of physics x in dataset
        y_range: (min, max) of physics y in dataset

    Returns:
        (tx, ty) in [-1, 1] grid coordinates. y is flipped (physics y-up → image y-down).
    """
    # Normalize to [-1, 1]
    tx = 2.0 * (gt_x - x_range[0]) / (x_range[1] - x_range[0]) - 1.0
    # Normalize then negate for y-flip (physics y-up, image y-down)
    ty = -(2.0 * (gt_y - y_range[0]) / (y_range[1] - y_range[0]) - 1.0)
    return tx, ty


def angle_to_sincos(angle: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert angle (radians) to (sin, cos) pair on unit circle."""
    return torch.sin(angle), torch.cos(angle)
