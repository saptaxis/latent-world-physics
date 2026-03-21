# tests/test_pixel_vae.py
"""Tests for PixelVAE model."""
import torch
import pytest
from lwp.models.pixel_vae import PixelVAE


class TestPixelVAE:
    """Test PixelVAE encoder-decoder."""

    def test_encode_returns_mu_logvar(self):
        """Encoder produces mu and logvar of correct shape."""
        vae = PixelVAE(in_channels=1, latent_dim=64, frame_size=84)
        x = torch.randn(2, 1, 84, 84)
        mu, logvar = vae.encode_params(x)
        assert mu.shape == (2, 64)
        assert logvar.shape == (2, 64)

    def test_decode_reconstructs_shape(self):
        """Decoder output matches input spatial dims and channels."""
        vae = PixelVAE(in_channels=1, latent_dim=64, frame_size=84)
        z = torch.randn(2, 64)
        recon = vae.decode(z)
        assert recon.shape == (2, 1, 84, 84)

    def test_forward_returns_recon_mu_logvar(self):
        """Forward pass returns reconstruction, mu, logvar."""
        vae = PixelVAE(in_channels=1, latent_dim=64, frame_size=84)
        x = torch.randn(2, 1, 84, 84)
        recon, mu, logvar, _ = vae(x)
        assert recon.shape == x.shape
        assert mu.shape == (2, 64)
        assert logvar.shape == (2, 64)

    def test_encode_deterministic_at_eval(self):
        """encode() returns mu (no sampling) for deterministic inference."""
        vae = PixelVAE(in_channels=1, latent_dim=64, frame_size=84)
        vae.eval()
        x = torch.randn(1, 1, 84, 84)
        z = vae.encode(x)
        assert z.shape == (1, 64)
        # Should be deterministic
        z2 = vae.encode(x)
        assert torch.allclose(z, z2)

    def test_4channel_input(self):
        """VAE works with 4-channel (stacked frames) input."""
        vae = PixelVAE(in_channels=4, latent_dim=64, frame_size=84)
        x = torch.randn(2, 4, 84, 84)
        recon, mu, logvar, _ = vae(x)
        assert recon.shape == (2, 4, 84, 84)

    def test_128_resolution(self):
        """VAE works at 128x128 resolution."""
        vae = PixelVAE(in_channels=1, latent_dim=64, frame_size=128)
        x = torch.randn(2, 1, 128, 128)
        recon, mu, logvar, _ = vae(x)
        assert recon.shape == (2, 1, 128, 128)

    def test_custom_channels(self):
        """VAE works with custom channel config."""
        vae = PixelVAE(in_channels=1, latent_dim=32,
                       frame_size=84, channels=[16, 32, 64, 128])
        x = torch.randn(2, 1, 84, 84)
        recon, mu, logvar, _ = vae(x)
        assert recon.shape == x.shape
        assert mu.shape == (2, 32)

    def test_output_range_sigmoid(self):
        """Decoder output is in [0, 1] due to sigmoid."""
        vae = PixelVAE(in_channels=1, latent_dim=64, frame_size=84)
        vae.eval()
        x = torch.randn(2, 1, 84, 84).clamp(0, 1)
        recon, _, _, _ = vae(x)
        assert recon.min() >= 0.0
        assert recon.max() <= 1.0


class TestPixelVAECoordConv:
    """Tests for CoordConv option — x,y coordinate channels for encoder."""

    def test_forward_shape_unchanged(self):
        """Output shapes identical with or without coord_conv."""
        model = PixelVAE(in_channels=1, latent_dim=32, frame_size=64,
                         channels=[4, 8, 16, 32], coord_conv=True)
        x = torch.rand(2, 1, 64, 64)
        recon, mu, logvar, state_pred = model(x)
        assert recon.shape == (2, 1, 64, 64)
        assert mu.shape == (2, 32)
        assert logvar.shape == (2, 32)

    def test_encode_decode_round_trip(self):
        """Encode-decode produces correct shapes with coord_conv."""
        model = PixelVAE(in_channels=1, latent_dim=32, frame_size=64,
                         channels=[4, 8, 16, 32], coord_conv=True)
        x = torch.rand(2, 1, 64, 64)
        z = model.encode(x)
        assert z.shape == (2, 32)
        recon = model.decode(z)
        assert recon.shape == (2, 1, 64, 64)

    def test_coord_conv_more_params_than_standard(self):
        """coord_conv adds parameters to the first encoder conv layer."""
        kwargs = dict(in_channels=1, latent_dim=32, frame_size=64,
                      channels=[4, 8, 16, 32])
        standard = PixelVAE(**kwargs, coord_conv=False)
        coordconv = PixelVAE(**kwargs, coord_conv=True)
        n_std = sum(p.numel() for p in standard.parameters())
        n_cc = sum(p.numel() for p in coordconv.parameters())
        # Only the first conv layer changes: in_channels+2 vs in_channels.
        # Extra params = channels[0] * 2 * kernel_size^2 = 4 * 2 * 16 = 128
        assert n_cc > n_std
        assert n_cc - n_std == 4 * 2 * 4 * 4  # channels[0] * 2 * k * k

    def test_coord_conv_with_rgb(self):
        """coord_conv works with 3-channel RGB input (3+2=5 encoder channels)."""
        model = PixelVAE(in_channels=3, latent_dim=32, frame_size=64,
                         channels=[4, 8, 16, 32], coord_conv=True)
        x = torch.rand(2, 3, 64, 64)
        recon, mu, logvar, _ = model(x)
        assert recon.shape == (2, 3, 64, 64)

    def test_coord_conv_with_state_head(self):
        """coord_conv + state_dim work together."""
        model = PixelVAE(in_channels=1, latent_dim=32, frame_size=64,
                         channels=[4, 8, 16, 32], coord_conv=True,
                         state_dim=6)
        x = torch.rand(2, 1, 64, 64)
        recon, mu, logvar, state_pred = model(x)
        assert state_pred.shape == (2, 6)

    def test_coord_conv_128_resolution(self):
        """Coordinate grids scale to 128x128."""
        model = PixelVAE(in_channels=1, latent_dim=32, frame_size=128,
                         channels=[4, 8, 16, 32], coord_conv=True)
        x = torch.rand(2, 1, 128, 128)
        recon, mu, logvar, _ = model(x)
        assert recon.shape == (2, 1, 128, 128)

    def test_coord_conv_default_off(self):
        """coord_conv defaults to False — no change to existing behavior."""
        model = PixelVAE(in_channels=1, latent_dim=32, frame_size=64,
                         channels=[4, 8, 16, 32])
        assert not model.coord_conv
