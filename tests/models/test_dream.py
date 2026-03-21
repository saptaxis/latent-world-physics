# tests/test_dream.py
"""Tests for DreamGenerator."""
import tempfile
from pathlib import Path

import numpy as np
import torch
import pytest

from lwp.models.pixel_vae import PixelVAE
from lwp.models.pixel_dynamics import LatentDynamicsModel
from lwp.models.pixel_world_model import PixelWorldModel
from lwp.viz.dream import DreamGenerator


@pytest.fixture
def dream_gen():
    vae = PixelVAE(in_channels=1, latent_dim=16, frame_size=84,
                   channels=[8, 16, 32, 64])
    dynamics = LatentDynamicsModel(latent_dim=16, action_dim=2, hidden_size=32)
    model = PixelWorldModel(vae, dynamics)
    return DreamGenerator(model, device="cpu")


class TestDreamGenerator:
    def test_generate_returns_numpy(self, dream_gen):
        seed = torch.rand(1, 1, 84, 84)
        actions = torch.randn(1, 5, 2)
        frames = dream_gen.generate(seed, actions)
        assert isinstance(frames, np.ndarray)
        assert frames.shape == (6, 84, 84)
        assert frames.dtype == np.uint8

    def test_comparison_side_by_side(self, dream_gen):
        seed = torch.rand(1, 1, 84, 84)
        actions = torch.randn(1, 5, 2)
        gt = torch.rand(6, 1, 84, 84)
        combined = dream_gen.comparison(seed, actions, gt)
        assert combined.shape == (6, 84, 168)

    def test_save_gif(self, dream_gen, tmp_path):
        frames = np.random.randint(0, 255, (10, 84, 84), dtype=np.uint8)
        path = str(tmp_path / "test.gif")
        dream_gen.save_gif(frames, path)
        assert Path(path).exists()
