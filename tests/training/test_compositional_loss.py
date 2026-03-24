# tests/training/test_compositional_loss.py
import torch
import pytest
from lwp.training.pixel_losses import compositional_vae_loss


class TestCompositionalVAELoss:
    def test_returns_dict(self):
        losses = compositional_vae_loss(
            recon=torch.randn(4, 1, 84, 84).sigmoid(),
            target=torch.randn(4, 1, 84, 84).sigmoid(),
            mu=torch.randn(4, 45),
            logvar=torch.randn(4, 45),
            pose_pred=torch.randn(4, 5),
            gt_pose=torch.randn(4, 4),  # tx, ty, sin, cos
            A_hat=torch.randn(4, 1, 16, 16).sigmoid(),
        )
        assert isinstance(losses, dict)
        assert 'total' in losses
        assert 'recon' in losses
        assert 'pose' in losses
        assert 'kl' in losses
        assert 'mask_area' in losses

    def test_fg_weight_increases_loss(self):
        recon = torch.zeros(4, 1, 84, 84)
        target = torch.ones(4, 1, 84, 84) * 0.5  # all foreground intensity
        common = dict(mu=torch.zeros(4, 45), logvar=torch.zeros(4, 45),
                      pose_pred=torch.zeros(4, 5), gt_pose=torch.zeros(4, 4),
                      A_hat=torch.ones(4, 1, 16, 16) * 0.5)
        loss_noweight = compositional_vae_loss(recon, target, fg_weight=1.0, **common)
        loss_weighted = compositional_vae_loss(recon, target, fg_weight=50.0, **common)
        assert loss_weighted['recon'] > loss_noweight['recon']

    def test_mask_area_penalty(self):
        common = dict(
            recon=torch.randn(4, 1, 84, 84).sigmoid(),
            target=torch.randn(4, 1, 84, 84).sigmoid(),
            mu=torch.zeros(4, 45), logvar=torch.zeros(4, 45),
            pose_pred=torch.zeros(4, 5), gt_pose=torch.zeros(4, 4),
        )
        # Mask at target area → low penalty
        loss_good = compositional_vae_loss(
            A_hat=torch.ones(4, 1, 16, 16) * 0.35, mask_target=0.35, **common)
        # Mask all-ones → high penalty
        loss_bad = compositional_vae_loss(
            A_hat=torch.ones(4, 1, 16, 16), mask_target=0.35, **common)
        assert loss_bad['mask_area'] > loss_good['mask_area']

    def test_split_kl(self):
        """Split mode: separate KL for bg and obj."""
        losses = compositional_vae_loss(
            recon=torch.randn(4, 1, 84, 84).sigmoid(),
            target=torch.randn(4, 1, 84, 84).sigmoid(),
            mu=torch.randn(4, 40),  # bg(8) + obj(32)
            logvar=torch.randn(4, 40),
            pose_pred=torch.randn(4, 5),
            gt_pose=torch.randn(4, 4),
            A_hat=torch.randn(4, 1, 16, 16).sigmoid(),
            split_kl=True, bg_dim=8,
            beta_bg=0.0001, beta_obj=0.0001,
        )
        assert 'kl_bg' in losses
        assert 'kl_obj' in losses
        assert 'kl' not in losses

    def test_gradient_flows(self):
        recon_leaf = torch.randn(4, 1, 84, 84, requires_grad=True)
        recon = recon_leaf.sigmoid()
        mu = torch.randn(4, 45, requires_grad=True)
        logvar = torch.randn(4, 45, requires_grad=True)
        pose_pred = torch.randn(4, 5, requires_grad=True)
        A_hat_leaf = torch.randn(4, 1, 16, 16, requires_grad=True)
        A_hat = A_hat_leaf.sigmoid()
        losses = compositional_vae_loss(
            recon, torch.randn(4, 1, 84, 84).sigmoid(),
            mu, logvar, pose_pred, torch.randn(4, 4), A_hat,
        )
        losses['total'].backward()
        assert recon_leaf.grad is not None
        assert mu.grad is not None
        assert pose_pred.grad is not None
        assert A_hat_leaf.grad is not None
