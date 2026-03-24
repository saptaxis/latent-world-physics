# tests/models/test_coord_utils.py
import torch
import pytest
from lwp.models.coord_utils import physics_to_grid, angle_to_sincos


class TestPhysicsToGrid:
    def test_center_maps_to_origin(self):
        """Physics center should map to grid origin (0, 0)."""
        tx, ty = physics_to_grid(
            gt_x=torch.tensor([0.0]),
            gt_y=torch.tensor([0.75]),  # vertical center
            x_range=(-1.0, 1.0),
            y_range=(0.0, 1.5),
        )
        assert tx.item() == pytest.approx(0.0, abs=1e-5)
        assert ty.item() == pytest.approx(0.0, abs=1e-5)

    def test_y_flip(self):
        """Higher physics y (top of screen) → negative grid y (top of image)."""
        _, ty_high = physics_to_grid(
            gt_x=torch.tensor([0.0]),
            gt_y=torch.tensor([1.5]),  # top in physics
            x_range=(-1.0, 1.0), y_range=(0.0, 1.5),
        )
        _, ty_low = physics_to_grid(
            gt_x=torch.tensor([0.0]),
            gt_y=torch.tensor([0.0]),  # bottom in physics
            x_range=(-1.0, 1.0), y_range=(0.0, 1.5),
        )
        assert ty_high < ty_low  # y-flip: high physics y → low grid y

    def test_output_range(self):
        """Outputs should be in [-1, 1]."""
        tx, ty = physics_to_grid(
            gt_x=torch.tensor([-1.0, 0.0, 1.0]),
            gt_y=torch.tensor([0.0, 0.75, 1.5]),
            x_range=(-1.0, 1.0), y_range=(0.0, 1.5),
        )
        assert tx.min() >= -1.0 and tx.max() <= 1.0
        assert ty.min() >= -1.0 and ty.max() <= 1.0

    def test_batch(self):
        """Should work with batched inputs."""
        tx, ty = physics_to_grid(
            gt_x=torch.randn(16),
            gt_y=torch.rand(16) * 1.5,
            x_range=(-1.0, 1.0), y_range=(0.0, 1.5),
        )
        assert tx.shape == (16,)
        assert ty.shape == (16,)


class TestAngleToSincos:
    def test_unit_circle(self):
        """sin²+cos² should equal 1."""
        angles = torch.tensor([0.0, 0.5, -1.0, 3.14])
        sin_t, cos_t = angle_to_sincos(angles)
        norms = sin_t ** 2 + cos_t ** 2
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_zero_angle(self):
        sin_t, cos_t = angle_to_sincos(torch.tensor([0.0]))
        assert sin_t.item() == pytest.approx(0.0, abs=1e-5)
        assert cos_t.item() == pytest.approx(1.0, abs=1e-5)
