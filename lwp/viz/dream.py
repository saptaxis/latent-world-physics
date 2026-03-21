# viz/dream.py
"""Dream sequence generation and video export.

Generates dream sequences from seed frames + actions, with side-by-side
comparison to ground truth and multi-dream fan-out visualization.
Exports as GIF or MP4 via imageio.
"""
from __future__ import annotations

from pathlib import Path

import imageio
import numpy as np
import torch

from lwp.models.pixel_world_model import PixelWorldModel


class DreamGenerator:
    """Generate and export dream sequences from a pixel world model."""

    def __init__(self, model: PixelWorldModel, device: str = "cpu"):
        self.device = torch.device(device)
        # Move model to target device once at init, not per-call, to avoid
        # repeated GPU transfers during batch generation.
        self.model = model.to(self.device)
        # eval() disables dropout and batchnorm updates — dreams should use
        # the learned statistics, not running estimates.
        self.model.eval()

    @torch.no_grad()
    def generate(self, seed_frames: torch.Tensor, actions: torch.Tensor
                 ) -> np.ndarray:
        """Generate dream sequence.

        Returns:
            frames: (T+1, H, W) uint8 numpy array (grayscale) or (T+1, H, W, 3) (RGB)
        """
        # Add batch dim if caller passed unbatched tensors — model.dream
        # expects (B, T, C, H, W) for seed_frames and (B, T, action_dim)
        # for actions.
        if seed_frames.dim() == 3:
            seed_frames = seed_frames.unsqueeze(0)
        if actions.dim() == 2:
            actions = actions.unsqueeze(0)

        seed_frames = seed_frames.to(self.device)
        actions = actions.to(self.device)

        # dream() returns (B, T+1, C, H, W) — the seed frame plus T
        # autoregressively predicted frames.
        dream_frames = self.model.dream(seed_frames, actions)
        # Remove batch dim (we only process one episode at a time)
        frames = dream_frames.squeeze(0).cpu()

        if frames.size(1) == 1:
            # Grayscale: (T+1, 1, H, W) → (T+1, H, W) uint8.
            # squeeze(1) removes the singleton channel dim.
            frames_np = (frames.squeeze(1).numpy() * 255).clip(0, 255).astype(np.uint8)
        else:
            # Stacked frames: take the last channel slice (most recent frame)
            # for visualization — earlier channels are history frames.
            frames_np = (frames[:, -1].numpy() * 255).clip(0, 255).astype(np.uint8)

        return frames_np

    def comparison(self, seed_frames: torch.Tensor, actions: torch.Tensor,
                   gt_frames: torch.Tensor) -> np.ndarray:
        """Generate side-by-side: GT | Predicted."""
        pred = self.generate(seed_frames, actions)

        # Convert GT frames to uint8, handling both grayscale (B,1,H,W) and
        # stacked-frame (B,C,H,W) cases, same logic as generate().
        if gt_frames.dim() == 4 and gt_frames.size(1) == 1:
            gt_np = (gt_frames.squeeze(1).numpy() * 255).clip(0, 255).astype(np.uint8)
        else:
            gt_np = (gt_frames[:, -1].numpy() * 255).clip(0, 255).astype(np.uint8)

        # Concatenate along width (axis=2) to produce [GT | Predicted] layout.
        # Truncate to the shorter sequence in case dream length differs.
        T = min(len(pred), len(gt_np))
        combined = np.concatenate([gt_np[:T], pred[:T]], axis=2)
        return combined

    def save_gif(self, frames: np.ndarray, path: str, fps: int = 15):
        """Save frames as animated GIF."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        # imageio requires RGB (H, W, 3) frames — replicate grayscale across
        # all three channels to produce a visually identical greyscale GIF.
        if frames.ndim == 3:
            frames_rgb = np.stack([frames] * 3, axis=-1)
        else:
            frames_rgb = frames
        # loop=0 means infinite looping, standard for inspection GIFs
        imageio.mimsave(path, frames_rgb, fps=fps, loop=0)

    def save_mp4(self, frames: np.ndarray, path: str, fps: int = 15):
        """Save frames as MP4 video."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        if frames.ndim == 3:
            frames_rgb = np.stack([frames] * 3, axis=-1)
        else:
            frames_rgb = frames
        # libx264 + yuv420p is the most compatible H.264 combo — plays in
        # browsers, Slack, QuickTime, and TensorBoard without re-encoding.
        writer = imageio.get_writer(path, fps=fps, codec="libx264",
                                    pixelformat="yuv420p")
        for frame in frames_rgb:
            writer.append_data(frame)
        writer.close()
