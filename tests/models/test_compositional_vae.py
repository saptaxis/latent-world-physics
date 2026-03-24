# tests/models/test_compositional_vae.py
import torch
import pytest
from lwp.models.compositional_vae import CompositionalPixelVAE


class TestCompositionalVAEFlat:
    """Tests for flat latent mode."""

    @pytest.fixture
    def model(self):
        return CompositionalPixelVAE(
            latent_dim=45, frame_size=84, latent_mode='flat',
            canonical_size=16, beta=0.0001,
        )

    def test_forward_shape(self, model):
        x = torch.randn(4, 1, 84, 84)
        gt_state = torch.randn(4, 3)  # x, y, angle
        recon, mu, logvar, pose_pred = model(x, gt_state=gt_state)
        assert recon.shape == (4, 1, 84, 84)
        assert mu.shape == (4, 45)
        assert logvar.shape == (4, 45)
        assert pose_pred.shape == (4, 5)

    def test_encode_shape(self, model):
        x = torch.randn(4, 1, 84, 84)
        z = model.encode(x)
        assert z.shape == (4, 45)

    def test_decode_shape(self, model):
        z = torch.randn(4, 45)
        recon = model.decode(z)
        assert recon.shape == (4, 1, 84, 84)

    def test_predict_state(self, model):
        z = torch.randn(4, 45)
        pose = model.predict_state(z)
        assert pose.shape == (4, 5)

    def test_state_dim(self, model):
        assert model.state_dim == 5

    def test_forward_no_gt_state(self, model):
        """forward without gt_state should still work (pose_pred returned,
        just can't compute pose loss)."""
        x = torch.randn(4, 1, 84, 84)
        recon, mu, logvar, pose_pred = model(x)
        assert recon.shape == (4, 1, 84, 84)
        assert pose_pred.shape == (4, 5)

    def test_backward(self, model):
        """Gradients should flow through the full pipeline."""
        x = torch.randn(2, 1, 84, 84)
        recon, mu, logvar, pose_pred = model(x)
        loss = recon.sum() + mu.sum() + pose_pred.sum()
        loss.backward()
        # Check encoder got gradients
        assert model.fc_mu.weight.grad is not None
        # Check pose head got gradients
        for p in model.pose_head.parameters():
            assert p.grad is not None

    def test_recon_in_01(self, model):
        """Output should be in [0, 1] (sigmoid)."""
        x = torch.randn(2, 1, 84, 84)
        recon, _, _, _ = model(x)
        assert recon.min() >= 0.0
        assert recon.max() <= 1.0


class TestCompositionalVAESplit:
    """Tests for split latent mode."""

    @pytest.fixture
    def model(self):
        return CompositionalPixelVAE(
            latent_dim=45, frame_size=84, latent_mode='split',
            bg_dim=8, obj_dim=32, pose_dim=5, canonical_size=16,
        )

    def test_forward_shape(self, model):
        x = torch.randn(4, 1, 84, 84)
        gt_state = torch.randn(4, 3)
        recon, mu, logvar, pose_pred = model(x, gt_state=gt_state)
        assert recon.shape == (4, 1, 84, 84)
        # mu/logvar: bg(8) + obj(32) = 40 (no KL on pose_raw)
        assert mu.shape == (4, 40)
        assert logvar.shape == (4, 40)
        assert pose_pred.shape == (4, 5)

    def test_encode_shape(self, model):
        x = torch.randn(4, 1, 84, 84)
        z = model.encode(x)
        # Full z: bg(8) + pose(5) + obj(32) = 45
        assert z.shape == (4, 45)

    def test_decode_shape(self, model):
        z = torch.randn(4, 45)
        recon = model.decode(z)
        assert recon.shape == (4, 1, 84, 84)

    def test_latent_dim_property(self, model):
        assert model.latent_dim == 45
        assert model.bg_dim == 8
        assert model.obj_dim == 32
        assert model.pose_dim == 5


class TestCompositionalVAEDecomposition:
    """Tests for decomposition outputs (canonical patch, mask, background)."""

    @pytest.fixture
    def model(self):
        return CompositionalPixelVAE(
            latent_dim=45, frame_size=84, latent_mode='flat',
            canonical_size=16,
        )

    def test_decode_decomposed(self, model):
        """decode_decomposed should return all intermediate outputs."""
        z = torch.randn(2, 45)
        result = model.decode_decomposed(z)
        assert result['x_hat'].shape == (2, 1, 84, 84)
        assert result['O_hat'].shape == (2, 1, 16, 16)
        assert result['A_hat'].shape == (2, 1, 16, 16)
        assert result['B_hat'].shape == (2, 1, 84, 84)
        assert result['O_warp'].shape == (2, 1, 84, 84)
        assert result['A_warp'].shape == (2, 1, 84, 84)
        assert result['pose_params'].shape == (2, 5)

    def test_mask_in_01(self, model):
        z = torch.randn(2, 45)
        result = model.decode_decomposed(z)
        assert result['A_hat'].min() >= 0.0
        assert result['A_hat'].max() <= 1.0
