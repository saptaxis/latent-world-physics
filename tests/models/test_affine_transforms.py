# tests/models/test_affine_transforms.py
"""Tests for STN affine transform math.

Validates: inverse affine construction, round-trip identity,
grid_sample behavior with known transforms, align_corners=False.
"""
import torch
import torch.nn.functional as F
import pytest


def build_inverse_affine(tx, ty, sin_t, cos_t, s):
    """Build inverse affine matrix for affine_grid (output→input mapping).

    Forward: frame_pos = s * R @ canon_pos + t
    Inverse: canon_pos = (1/s) * R^T @ (frame_pos - t)
    """
    inv_s = 1.0 / s
    # Row 0: [inv_s*cos, inv_s*sin, inv_s*(-cos*tx - sin*ty)]
    # Row 1: [-inv_s*sin, inv_s*cos, inv_s*(sin*tx - cos*ty)]
    theta = torch.zeros(tx.shape[0], 2, 3)
    theta[:, 0, 0] = inv_s * cos_t
    theta[:, 0, 1] = inv_s * sin_t
    theta[:, 0, 2] = inv_s * (-cos_t * tx - sin_t * ty)
    theta[:, 1, 0] = -inv_s * sin_t
    theta[:, 1, 1] = inv_s * cos_t
    theta[:, 1, 2] = inv_s * (sin_t * tx - cos_t * ty)
    return theta


class TestInverseAffine:
    def test_identity_transform(self):
        """No rotation, no translation, scale=1 → identity grid."""
        theta = build_inverse_affine(
            tx=torch.tensor([0.0]), ty=torch.tensor([0.0]),
            sin_t=torch.tensor([0.0]), cos_t=torch.tensor([1.0]),
            s=torch.tensor([1.0]),
        )
        expected = torch.tensor([[[1, 0, 0], [0, 1, 0]]], dtype=torch.float)
        assert torch.allclose(theta, expected, atol=1e-5)

    def test_translation_only(self):
        """Pure translation: object at (0.5, -0.3), no rotation, scale=1."""
        theta = build_inverse_affine(
            tx=torch.tensor([0.5]), ty=torch.tensor([-0.3]),
            sin_t=torch.tensor([0.0]), cos_t=torch.tensor([1.0]),
            s=torch.tensor([1.0]),
        )
        # Inverse translation: shift by (-tx, -ty)
        assert theta[0, 0, 2].item() == pytest.approx(-0.5, abs=1e-5)
        assert theta[0, 1, 2].item() == pytest.approx(0.3, abs=1e-5)

    def test_round_trip(self):
        """Forward then inverse should recover original coordinates."""
        B = 4
        tx = torch.randn(B)
        ty = torch.randn(B)
        angles = torch.randn(B)
        sin_t, cos_t = torch.sin(angles), torch.cos(angles)
        s = torch.rand(B) * 0.5 + 0.1  # scale in [0.1, 0.6]

        # Forward affine: y = s * R @ x + t
        forward = torch.zeros(B, 2, 3)
        forward[:, 0, 0] = s * cos_t
        forward[:, 0, 1] = -s * sin_t
        forward[:, 0, 2] = tx
        forward[:, 1, 0] = s * sin_t
        forward[:, 1, 1] = s * cos_t
        forward[:, 1, 2] = ty

        # Inverse
        inverse = build_inverse_affine(tx, ty, sin_t, cos_t, s)

        # Product should be identity (2x2 block) with zero translation
        for b in range(B):
            F_mat = forward[b, :, :2]  # 2x2
            I_mat = inverse[b, :, :2]  # 2x2
            product = I_mat @ F_mat
            assert torch.allclose(product, torch.eye(2), atol=1e-4), \
                f"Batch {b}: rotation block not identity"

    def test_scale_shrinks_object(self):
        """Small scale → inverse has large values → samples from wider region."""
        theta_small = build_inverse_affine(
            tx=torch.tensor([0.0]), ty=torch.tensor([0.0]),
            sin_t=torch.tensor([0.0]), cos_t=torch.tensor([1.0]),
            s=torch.tensor([0.1]),
        )
        theta_large = build_inverse_affine(
            tx=torch.tensor([0.0]), ty=torch.tensor([0.0]),
            sin_t=torch.tensor([0.0]), cos_t=torch.tensor([1.0]),
            s=torch.tensor([0.5]),
        )
        # Smaller scale → larger inverse scale values
        assert theta_small[0, 0, 0].item() > theta_large[0, 0, 0].item()


class TestGridSampleWarp:
    def test_centered_identity_preserves_patch(self):
        """Identity warp at center should preserve a centered patch."""
        patch = torch.zeros(1, 1, 8, 8)
        patch[0, 0, 2:6, 2:6] = 1.0  # centered square

        theta = torch.tensor([[[1, 0, 0], [0, 1, 0]]], dtype=torch.float)
        grid = F.affine_grid(theta, size=(1, 1, 8, 8), align_corners=False)
        warped = F.grid_sample(patch, grid, padding_mode='zeros', align_corners=False)

        assert torch.allclose(warped, patch, atol=0.1)

    def test_translation_moves_object(self):
        """Translating by (0.5, 0) should shift object right in output."""
        patch = torch.zeros(1, 1, 16, 16)
        patch[0, 0, 6:10, 6:10] = 1.0  # centered square

        # To move object right: inverse transform shifts left
        # (output pixel at right → sample from left in input)
        theta = build_inverse_affine(
            tx=torch.tensor([0.5]), ty=torch.tensor([0.0]),
            sin_t=torch.tensor([0.0]), cos_t=torch.tensor([1.0]),
            s=torch.tensor([1.0]),
        )
        grid = F.affine_grid(theta, size=(1, 1, 16, 16), align_corners=False)
        warped = F.grid_sample(patch, grid, padding_mode='zeros', align_corners=False)

        # Object should be right of center
        left_energy = warped[0, 0, :, :8].sum()
        right_energy = warped[0, 0, :, 8:].sum()
        assert right_energy > left_energy
