# tests/test_pixel_callbacks.py
"""Tests for pixel-specific callbacks."""
import tempfile
from pathlib import Path

import numpy as np
import torch
import pytest

from lwp.models.pixel_vae import PixelVAE
from lwp.models.pixel_dynamics import LatentDynamicsModel
from lwp.training.callbacks import CallbackContext
from lwp.training.pixel_callbacks import (
    PixelVAEValidationCallback,
    ReconGridCallback,
    PixelDynamicsValidationCallback,
    DreamGridCallback,
    KinematicsValidationCallback,
    DreamComparisonVideoCallback,
    RSSMDiagnosticCallback,
)


@pytest.fixture
def vae():
    return PixelVAE(in_channels=1, latent_dim=16, frame_size=84,
                    channels=[8, 16, 32, 64])


@pytest.fixture
def ctx(vae, tmp_path):
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
    return CallbackContext(
        model=vae, optimizer=optimizer, writer=None,
        global_step=0, epoch=0, run_dir=str(tmp_path),
        device="cpu", extras={"config": {"test": True}},
    )


class TestPixelVAEValidationCallback:
    def test_triggers_at_interval(self, vae, ctx, tmp_path):
        """Callback runs validation at specified step interval."""
        val_data = [torch.rand(4, 1, 84, 84) for _ in range(2)]
        val_loader = val_data

        cb = PixelVAEValidationCallback(
            val_loader=val_loader, beta=0.0001,
            every_n_steps=10, patience=5,
            checkpoint_dir=str(tmp_path / "ckpt"),
        )
        cb.on_train_start(ctx)

        ctx.global_step = 1
        ctx.extras["train_loss_step"] = 0.5
        result = cb.on_step(ctx)
        assert result is True
        assert "val_loss" not in ctx.extras

        ctx.global_step = 10
        result = cb.on_step(ctx)
        assert result is True
        assert "val_loss" in ctx.extras

    def test_early_stopping(self, vae, ctx, tmp_path):
        """Callback returns False after patience exhausted."""
        val_data = [torch.rand(4, 1, 84, 84) for _ in range(2)]

        cb = PixelVAEValidationCallback(
            val_loader=val_data, beta=0.0001,
            every_n_steps=1, patience=2,
            checkpoint_dir=str(tmp_path / "ckpt"),
        )
        cb.on_train_start(ctx)

        cb.best_val_loss = 0.0
        for step in range(1, 10):
            ctx.global_step = step
            ctx.extras["train_loss_step"] = 1.0
            result = cb.on_step(ctx)
            if result is False:
                assert step <= 4
                return
        pytest.fail("Early stopping did not trigger")


class TestReconGridCallback:
    def test_logs_grid(self, vae, ctx):
        """ReconGridCallback produces reconstruction grid without error."""
        val_loader = [torch.rand(8, 1, 84, 84)]
        cb = ReconGridCallback(val_loader=val_loader, every_n_steps=5)

        ctx.global_step = 5
        result = cb.on_step(ctx)
        assert result is True


# ---------------------------------------------------------------------------
# PixelDynamicsValidationCallback tests
# ---------------------------------------------------------------------------

class TestPixelDynamicsValidationCallback:
    def test_latent_mse_mode(self):
        """Validation works with default latent_mse mode."""
        vae = PixelVAE(in_channels=1, latent_dim=16, frame_size=64,
                       channels=[8, 16, 32, 64])
        vae.eval()
        dynamics = LatentDynamicsModel(latent_dim=16, action_dim=2,
                                       hidden_size=32)
        optimizer = torch.optim.Adam(dynamics.parameters(), lr=1e-3)
        ctx = CallbackContext(
            model=dynamics, optimizer=optimizer, writer=None,
            global_step=0, epoch=0, run_dir="/tmp", device="cpu",
            extras={"config": {}},
        )
        # val_loader yields (frames_batch, actions_batch)
        # latent_mse uses predict_sequence which expects T actions for T+1 frames
        val_data = [(torch.rand(2, 10, 1, 64, 64), torch.randn(2, 10, 2))]
        cb = PixelDynamicsValidationCallback(
            val_loader=val_data, vae=vae,
            every_n_steps=1, patience=5,
        )
        ctx.global_step = 1
        result = cb.on_step(ctx)
        assert result is True
        assert "val_loss" in ctx.extras

    def test_multi_step_mode(self):
        """Validation works with multi_step_latent mode."""
        vae = PixelVAE(in_channels=1, latent_dim=16, frame_size=64,
                       channels=[8, 16, 32, 64])
        vae.eval()
        dynamics = LatentDynamicsModel(latent_dim=16, action_dim=2,
                                       hidden_size=32)
        optimizer = torch.optim.Adam(dynamics.parameters(), lr=1e-3)
        ctx = CallbackContext(
            model=dynamics, optimizer=optimizer, writer=None,
            global_step=0, epoch=0, run_dir="/tmp", device="cpu",
            extras={"config": {}},
        )
        # multi_step expects T-1 actions for T frames
        val_data = [(torch.rand(2, 10, 1, 64, 64), torch.randn(2, 9, 2))]
        cb = PixelDynamicsValidationCallback(
            val_loader=val_data, vae=vae,
            every_n_steps=1, patience=5,
            training_mode="multi_step_latent",
            rollout_k=4,
        )
        ctx.global_step = 1
        result = cb.on_step(ctx)
        assert result is True
        assert "val_loss" in ctx.extras

    def test_elbo_mode_with_rssm(self):
        """Validation works with latent_elbo mode and RSSM dynamics."""
        from lwp.models.pixel_rssm import LatentRSSM
        vae = PixelVAE(in_channels=1, latent_dim=16, frame_size=64,
                       channels=[8, 16, 32, 64])
        vae.eval()
        dynamics = LatentRSSM(latent_dim=16, action_dim=2, deter_dim=32,
                              stoch_dim=8, hidden_dim=32)
        optimizer = torch.optim.Adam(dynamics.parameters(), lr=1e-3)
        ctx = CallbackContext(
            model=dynamics, optimizer=optimizer, writer=None,
            global_step=0, epoch=0, run_dir="/tmp", device="cpu",
            extras={"config": {}},
        )
        val_data = [(torch.rand(2, 10, 1, 64, 64), torch.randn(2, 9, 2))]
        cb = PixelDynamicsValidationCallback(
            val_loader=val_data, vae=vae,
            every_n_steps=1, patience=5,
            training_mode="latent_elbo",
            rollout_k=4, kl_weight=1.0,
        )
        ctx.global_step = 1
        result = cb.on_step(ctx)
        assert result is True
        assert "val_loss" in ctx.extras

    def test_checkpoint_includes_model_type(self, tmp_path):
        """Saved checkpoint includes model_type from extras."""
        vae = PixelVAE(in_channels=1, latent_dim=16, frame_size=64,
                       channels=[8, 16, 32, 64])
        vae.eval()
        dynamics = LatentDynamicsModel(latent_dim=16, action_dim=2,
                                       hidden_size=32)
        optimizer = torch.optim.Adam(dynamics.parameters(), lr=1e-3)
        ckpt_dir = str(tmp_path / "ckpt")
        ctx = CallbackContext(
            model=dynamics, optimizer=optimizer, writer=None,
            global_step=0, epoch=0, run_dir="/tmp", device="cpu",
            extras={"config": {}, "model_type": "rssm"},
        )
        val_data = [(torch.rand(2, 10, 1, 64, 64), torch.randn(2, 10, 2))]
        cb = PixelDynamicsValidationCallback(
            val_loader=val_data, vae=vae,
            every_n_steps=1, patience=5,
            checkpoint_dir=ckpt_dir,
        )
        cb.on_train_start(ctx)
        ctx.global_step = 1
        cb.on_step(ctx)
        # Load checkpoint and verify model_type is saved
        ckpt = torch.load(Path(ckpt_dir) / "best.pt", weights_only=False)
        assert ckpt["model_type"] == "rssm"


# ---------------------------------------------------------------------------
# DreamGridCallback tests
# ---------------------------------------------------------------------------

class _FakeDataset:
    """Minimal dataset returning (frames, actions) tuples for testing."""
    def __init__(self, T=10, C=1, H=64, W=64, action_dim=2):
        self.T = T
        self.C = C
        self.H = H
        self.W = W
        self.action_dim = action_dim

    def __len__(self):
        return 10

    def __getitem__(self, idx):
        # (T, C, H, W), (T-1, action_dim) — matches episode dataset format
        return torch.rand(self.T, self.C, self.H, self.W), \
               torch.randn(self.T - 1, self.action_dim)


class TestDreamGridCallback:
    def test_dream_grid_fires(self):
        """DreamGridCallback fires without error (no writer = no-op image)."""
        vae = PixelVAE(in_channels=1, latent_dim=16, frame_size=64,
                       channels=[8, 16, 32, 64])
        vae.eval()
        dynamics = LatentDynamicsModel(latent_dim=16, action_dim=2,
                                       hidden_size=32)
        optimizer = torch.optim.Adam(dynamics.parameters(), lr=1e-3)
        ctx = CallbackContext(
            model=dynamics, optimizer=optimizer, writer=None,
            global_step=0, epoch=0, run_dir="/tmp", device="cpu",
            extras={},
        )
        cb = DreamGridCallback(vae=vae, val_dataset=_FakeDataset(),
                               n_episodes=2, every_n_steps=1)
        ctx.global_step = 1
        result = cb.on_step(ctx)
        # No writer, so grid is skipped — but should still return True
        assert result is True

    def test_dream_grid_with_rssm(self):
        """DreamGridCallback works with RSSM dynamics model."""
        from lwp.models.pixel_rssm import LatentRSSM
        vae = PixelVAE(in_channels=1, latent_dim=16, frame_size=64,
                       channels=[8, 16, 32, 64])
        vae.eval()
        dynamics = LatentRSSM(latent_dim=16, action_dim=2, deter_dim=32,
                              stoch_dim=8, hidden_dim=32)
        optimizer = torch.optim.Adam(dynamics.parameters(), lr=1e-3)
        ctx = CallbackContext(
            model=dynamics, optimizer=optimizer, writer=None,
            global_step=0, epoch=0, run_dir="/tmp", device="cpu",
            extras={},
        )
        cb = DreamGridCallback(vae=vae, val_dataset=_FakeDataset(),
                               n_episodes=2, every_n_steps=1)
        ctx.global_step = 1
        result = cb.on_step(ctx)
        assert result is True

    def test_dream_grid_produces_image(self):
        """DreamGridCallback produces image grid when writer is provided."""
        vae = PixelVAE(in_channels=1, latent_dim=16, frame_size=64,
                       channels=[8, 16, 32, 64])
        vae.eval()
        dynamics = LatentDynamicsModel(latent_dim=16, action_dim=2,
                                       hidden_size=32)
        optimizer = torch.optim.Adam(dynamics.parameters(), lr=1e-3)

        # Mock writer to capture add_image calls
        class MockWriter:
            def __init__(self):
                self.images = {}
            def add_image(self, tag, img, step):
                self.images[tag] = (img, step)

        writer = MockWriter()
        ctx = CallbackContext(
            model=dynamics, optimizer=optimizer, writer=writer,
            global_step=0, epoch=0, run_dir="/tmp", device="cpu",
            extras={},
        )
        cb = DreamGridCallback(vae=vae, val_dataset=_FakeDataset(),
                               n_episodes=2, every_n_steps=1)
        ctx.global_step = 1
        result = cb.on_step(ctx)
        assert result is True
        # Verify image was logged
        assert "dynamics/dream_grid" in writer.images
        img, step = writer.images["dynamics/dream_grid"]
        assert step == 1
        # Grid should be a 3D tensor (C, H, W)
        assert img.ndim == 3


# ---------------------------------------------------------------------------
# Helpers for new callback tests
# ---------------------------------------------------------------------------

def _make_npz(tmp_dir, T=20, H=64, W=64, state_dim=8, with_metadata=False):
    """Create a temporary npz file with random episode data."""
    path = str(Path(tmp_dir) / f"ep_{np.random.randint(100000)}.npz")
    kwargs = dict(
        rgb_frames=np.random.randint(0, 255, (T, H, W, 3), dtype=np.uint8),
        actions=np.random.randn(T - 1, 2).astype(np.float32),
        states=np.random.randn(T, state_dim).astype(np.float32),
    )
    if with_metadata:
        import json
        kwargs["metadata_json"] = json.dumps({
            "source_type": "heuristic",
        })
    np.savez(path, **kwargs)
    return path


class _MockWriter:
    """Captures add_scalar / add_image calls for test assertions."""
    def __init__(self):
        self.scalars = {}
        self.images = {}

    def add_scalar(self, tag, value, step):
        self.scalars[tag] = (value, step)

    def add_image(self, tag, img, step):
        self.images[tag] = (img, step)


def _make_dynamics_ctx(dynamics, writer=None, tmp_path="/tmp", global_step=0,
                       extras=None):
    """Build a CallbackContext for dynamics-model callbacks."""
    optimizer = torch.optim.Adam(dynamics.parameters(), lr=1e-3)
    return CallbackContext(
        model=dynamics, optimizer=optimizer, writer=writer,
        global_step=global_step, epoch=0, run_dir=str(tmp_path),
        device="cpu", extras=extras or {},
    )


# ---------------------------------------------------------------------------
# KinematicsValidationCallback tests
# ---------------------------------------------------------------------------

class TestKinematicsValidationCallback:
    def test_kinematics_callback_logs_per_dim_mse(self, tmp_path):
        """Verify add_scalar called with kinematics/mse_h1_x, h1_y, etc."""
        # VAE with state_dim=6 so the state head is active
        vae = PixelVAE(in_channels=1, latent_dim=16, frame_size=64,
                       channels=[8, 16, 32, 64], state_dim=6)
        vae.eval()

        dynamics = LatentDynamicsModel(latent_dim=16, action_dim=2,
                                       hidden_size=32)
        writer = _MockWriter()
        ctx = _make_dynamics_ctx(dynamics, writer=writer,
                                tmp_path=tmp_path, global_step=10)

        npz_paths = [_make_npz(str(tmp_path), T=15, state_dim=8)]
        cb = KinematicsValidationCallback(
            vae=vae, episode_paths=npz_paths,
            every_n_steps=10, horizons=[1, 5], frame_size=64,
        )

        result = cb.on_step(ctx)
        assert result is True

        # Should have logged per-dim MSE for horizon 1 at minimum
        expected_tags = [f"kinematics/mse_h1_{d}" for d in
                         ["x", "y", "vx", "vy", "angle", "ang_vel"]]
        for tag in expected_tags:
            assert tag in writer.scalars, f"Missing scalar: {tag}"

    def test_kinematics_callback_logs_with_rssm(self, tmp_path):
        """Verify kinematics callback works with RSSM dynamics."""
        from lwp.models.pixel_rssm import LatentRSSM
        vae = PixelVAE(in_channels=1, latent_dim=16, frame_size=64,
                       channels=[8, 16, 32, 64], state_dim=6)
        vae.eval()

        dynamics = LatentRSSM(latent_dim=16, action_dim=2, deter_dim=32,
                              stoch_dim=8, hidden_dim=32)
        writer = _MockWriter()
        ctx = _make_dynamics_ctx(dynamics, writer=writer,
                                tmp_path=tmp_path, global_step=10)

        npz_paths = [_make_npz(str(tmp_path), T=15, state_dim=8)]
        cb = KinematicsValidationCallback(
            vae=vae, episode_paths=npz_paths,
            every_n_steps=10, horizons=[1], frame_size=64,
        )

        result = cb.on_step(ctx)
        assert result is True
        assert "kinematics/mse_h1_x" in writer.scalars

    def test_kinematics_callback_noop_without_state_head(self, tmp_path):
        """VAE with state_dim=0 should no-op: no errors, no scalars."""
        vae = PixelVAE(in_channels=1, latent_dim=16, frame_size=64,
                       channels=[8, 16, 32, 64], state_dim=0)
        vae.eval()

        dynamics = LatentDynamicsModel(latent_dim=16, action_dim=2,
                                       hidden_size=32)
        writer = _MockWriter()
        ctx = _make_dynamics_ctx(dynamics, writer=writer,
                                tmp_path=tmp_path, global_step=10)

        npz_paths = [_make_npz(str(tmp_path))]
        cb = KinematicsValidationCallback(
            vae=vae, episode_paths=npz_paths,
            every_n_steps=10, horizons=[1], frame_size=64,
        )

        result = cb.on_step(ctx)
        assert result is True
        # No scalars should have been logged
        assert len(writer.scalars) == 0


# ---------------------------------------------------------------------------
# DreamComparisonVideoCallback tests
# ---------------------------------------------------------------------------

class TestDreamComparisonVideoCallback:
    def test_dream_video_callback_writes_mp4(self, tmp_path):
        """Fire once, verify mp4 files exist with expected names."""
        vae = PixelVAE(in_channels=1, latent_dim=16, frame_size=64,
                       channels=[8, 16, 32, 64])
        vae.eval()

        dynamics = LatentDynamicsModel(latent_dim=16, action_dim=2,
                                       hidden_size=32)
        ctx = _make_dynamics_ctx(dynamics, tmp_path=tmp_path, global_step=100)

        video_dir = str(tmp_path / "videos")
        npz_paths = [
            _make_npz(str(tmp_path), T=12, with_metadata=True),
            _make_npz(str(tmp_path), T=12, with_metadata=False),
        ]
        cb = DreamComparisonVideoCallback(
            vae=vae, episode_paths=npz_paths,
            video_dir=video_dir, every_n_steps=100, fps=5, frame_size=64,
        )

        result = cb.on_step(ctx)
        assert result is True

        # Check that mp4 files were written
        step_dir = Path(video_dir) / "step_100"
        assert step_dir.exists()
        mp4s = list(step_dir.glob("*.mp4"))
        assert len(mp4s) == 2, f"Expected 2 mp4 files, got {len(mp4s)}"

    def test_dream_video_callback_with_rssm(self, tmp_path):
        """Video callback works with RSSM dynamics."""
        from lwp.models.pixel_rssm import LatentRSSM
        vae = PixelVAE(in_channels=1, latent_dim=16, frame_size=64,
                       channels=[8, 16, 32, 64])
        vae.eval()

        dynamics = LatentRSSM(latent_dim=16, action_dim=2, deter_dim=32,
                              stoch_dim=8, hidden_dim=32)
        ctx = _make_dynamics_ctx(dynamics, tmp_path=tmp_path, global_step=50)

        video_dir = str(tmp_path / "videos")
        npz_paths = [_make_npz(str(tmp_path), T=10)]
        cb = DreamComparisonVideoCallback(
            vae=vae, episode_paths=npz_paths,
            video_dir=video_dir, every_n_steps=50, fps=5, frame_size=64,
        )

        result = cb.on_step(ctx)
        assert result is True
        mp4s = list((Path(video_dir) / "step_50").glob("*.mp4"))
        assert len(mp4s) == 1


# ---------------------------------------------------------------------------
# RSSMDiagnosticCallback tests
# ---------------------------------------------------------------------------

class TestRSSMDiagnosticCallback:
    def test_rssm_diagnostic_skips_for_gru(self, tmp_path):
        """GRU dynamics => no-op, no scalars logged."""
        vae = PixelVAE(in_channels=1, latent_dim=16, frame_size=64,
                       channels=[8, 16, 32, 64])
        vae.eval()

        dynamics = LatentDynamicsModel(latent_dim=16, action_dim=2,
                                       hidden_size=32)
        writer = _MockWriter()
        ctx = _make_dynamics_ctx(dynamics, writer=writer,
                                tmp_path=tmp_path, global_step=10,
                                extras={"kl_loss": 0.5})

        npz_paths = [_make_npz(str(tmp_path))]
        cb = RSSMDiagnosticCallback(
            dynamics=dynamics, episode_paths=npz_paths,
            vae=vae, every_n_steps=10, frame_size=64,
        )

        result = cb.on_step(ctx)
        assert result is True
        # No scalars should be logged for GRU
        assert len(writer.scalars) == 0

    def test_rssm_diagnostic_logs_kl_and_divergence(self, tmp_path):
        """RSSM dynamics => logs kl_per_step and prior_post_divergence."""
        from lwp.models.pixel_rssm import LatentRSSM
        vae = PixelVAE(in_channels=1, latent_dim=16, frame_size=64,
                       channels=[8, 16, 32, 64])
        vae.eval()

        dynamics = LatentRSSM(latent_dim=16, action_dim=2, deter_dim=32,
                              stoch_dim=8, hidden_dim=32)
        writer = _MockWriter()
        ctx = _make_dynamics_ctx(
            dynamics, writer=writer, tmp_path=tmp_path, global_step=10,
            extras={"kl_loss": 0.42, "prior_post_div": 1.23},
        )

        npz_paths = [_make_npz(str(tmp_path), T=15)]
        cb = RSSMDiagnosticCallback(
            dynamics=dynamics, episode_paths=npz_paths,
            vae=vae, every_n_steps=10, frame_size=64,
        )

        result = cb.on_step(ctx)
        assert result is True

        # Per-step metrics from extras
        assert "rssm/kl_per_step" in writer.scalars
        assert writer.scalars["rssm/kl_per_step"][0] == pytest.approx(0.42)
        assert "rssm/prior_post_divergence" in writer.scalars
        assert writer.scalars["rssm/prior_post_divergence"][0] == pytest.approx(1.23)

        # Periodic validation metrics
        assert "rssm/prior_mse" in writer.scalars
        assert "rssm/posterior_mse" in writer.scalars
