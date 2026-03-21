# evaluation/pixel_metrics.py
"""Pixel-space evaluation metrics for visual world models.

Provides MSE, SSIM, and recognizable horizon metrics for comparing
predicted frames to ground truth.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def pixel_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-frame MSE averaged over batch. Inputs: (B, C, H, W) in [0, 1]."""
    return F.mse_loss(pred, target)


def _ssim_single(pred: torch.Tensor, target: torch.Tensor,
                 window_size: int = 11, C1: float = 0.01**2,
                 C2: float = 0.03**2) -> float:
    """Compute SSIM for a single pair of images. Inputs: (1, C, H, W)."""
    channels = pred.size(1)
    kernel = torch.ones(channels, 1, window_size, window_size,
                        device=pred.device) / (window_size * window_size)
    pad = window_size // 2

    mu_p = F.conv2d(pred, kernel, padding=pad, groups=channels)
    mu_t = F.conv2d(target, kernel, padding=pad, groups=channels)
    mu_p_sq = mu_p * mu_p
    mu_t_sq = mu_t * mu_t
    mu_pt = mu_p * mu_t

    sigma_p_sq = F.conv2d(pred * pred, kernel, padding=pad, groups=channels) - mu_p_sq
    sigma_t_sq = F.conv2d(target * target, kernel, padding=pad, groups=channels) - mu_t_sq
    sigma_pt = F.conv2d(pred * target, kernel, padding=pad, groups=channels) - mu_pt

    ssim_map = ((2 * mu_pt + C1) * (2 * sigma_pt + C2)) / \
               ((mu_p_sq + mu_t_sq + C1) * (sigma_p_sq + sigma_t_sq + C2))
    return ssim_map.mean().item()


def compute_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    """SSIM between predicted and target frames. Inputs: (B, C, H, W) in [0, 1].

    Returns mean SSIM across batch. Range: [-1, 1], 1 = identical.
    """
    with torch.no_grad():
        total = 0.0
        B = pred.size(0)
        for i in range(B):
            total += _ssim_single(pred[i:i+1], target[i:i+1])
        return total / B


def recognizable_horizon(pred_seq: torch.Tensor, gt_seq: torch.Tensor,
                         threshold: float = 0.5) -> int:
    """Find the number of steps before SSIM drops below threshold.

    Args:
        pred_seq: (T, C, H, W) predicted frame sequence
        gt_seq: (T, C, H, W) ground truth frame sequence
        threshold: SSIM threshold for "recognizable"

    Returns:
        Number of steps where SSIM >= threshold (0 to T)
    """
    T = pred_seq.size(0)
    with torch.no_grad():
        for t in range(T):
            ssim = _ssim_single(pred_seq[t:t+1], gt_seq[t:t+1])
            if ssim < threshold:
                return t
    return T
