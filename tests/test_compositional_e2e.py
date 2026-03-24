# tests/test_compositional_e2e.py
"""End-to-end smoke test for compositional STN VAE pipeline."""
import torch
import pytest
from lwp.models.compositional_vae import CompositionalPixelVAE
from lwp.models.pixel_dynamics import LatentDynamicsModel
from lwp.models.pixel_world_model import PixelWorldModel
from lwp.training.pixel_losses import compositional_vae_loss


@pytest.mark.parametrize("latent_mode", ["flat", "split"])
def test_vae_train_step(latent_mode):
    """One forward + backward pass through the VAE."""
    model = CompositionalPixelVAE(
        latent_dim=45, frame_size=84, latent_mode=latent_mode,
        bg_dim=8, obj_dim=32, canonical_size=16,
    )
    x = torch.randn(4, 1, 84, 84).sigmoid()
    gt_state = torch.randn(4, 3)

    recon, mu, logvar, pose_pred = model(x, gt_state=gt_state)

    from lwp.models.coord_utils import physics_to_grid, angle_to_sincos
    tx, ty = physics_to_grid(gt_state[:, 0], gt_state[:, 1])
    sin_t, cos_t = angle_to_sincos(gt_state[:, 2])
    gt_pose = torch.stack([tx, ty, sin_t, cos_t], dim=-1)

    # Use cached decomposition from forward() — same z, no double encode
    decomposed = model.get_last_decomposed()
    losses = compositional_vae_loss(
        recon, x, mu, logvar, pose_pred, gt_pose,
        decomposed['A_hat'],
        split_kl=(latent_mode == 'split'),
        bg_dim=8,
    )
    losses['total'].backward()
    # No NaN in loss
    assert not torch.isnan(losses['total'])


def test_dream_flat():
    """Dream through compositional VAE + dynamics."""
    vae = CompositionalPixelVAE(
        latent_dim=45, frame_size=84, latent_mode='flat', canonical_size=16,
    )
    dynamics = LatentDynamicsModel(
        latent_dim=45, hidden_size=64, action_dim=2,
    )
    wm = PixelWorldModel(vae=vae, dynamics=dynamics)

    z_seed = torch.randn(2, 45)
    actions = torch.randn(2, 5, 2)  # 5 steps
    frames, z_seq = wm.dream_from_latent(z_seed, actions)
    assert frames.shape[1] == 6  # seed + 5 steps
    assert z_seq.shape == (2, 6, 45)
