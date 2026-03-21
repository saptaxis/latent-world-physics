# tests/test_pixel_metrics.py
"""Tests for pixel-space evaluation metrics."""
import torch
import numpy as np
import pytest
from lwp.evaluation.pixel_metrics import pixel_mse, compute_ssim, recognizable_horizon


class TestPixelMetrics:
    def test_pixel_mse_identical(self):
        a = torch.rand(4, 1, 84, 84)
        assert pixel_mse(a, a).item() < 1e-6

    def test_pixel_mse_different(self):
        a = torch.rand(4, 1, 84, 84)
        b = torch.rand(4, 1, 84, 84)
        assert pixel_mse(a, b).item() > 0

    def test_ssim_identical(self):
        a = torch.rand(1, 1, 84, 84)
        val = compute_ssim(a, a)
        assert val > 0.99

    def test_ssim_different(self):
        a = torch.rand(1, 1, 84, 84)
        b = torch.rand(1, 1, 84, 84)
        val = compute_ssim(a, b)
        assert val < 0.5

    def test_recognizable_horizon(self):
        pred = [torch.rand(1, 1, 84, 84) for _ in range(10)]
        gt = pred.copy()
        for i in range(5, 10):
            pred[i] = torch.rand(1, 1, 84, 84)
        pred_seq = torch.cat(pred, dim=0)
        gt_seq = torch.cat(gt, dim=0)
        horizon = recognizable_horizon(pred_seq, gt_seq, threshold=0.5)
        assert 3 <= horizon <= 7
