# lwp/models/compositional_vae.py
"""Compositional STN VAE for pixel world modeling.

Solves the rotation rendering problem by architectural decomposition:
instead of a single deconv decoder that must memorize angle->pixel
mappings, this model decodes a canonical upright lander patch, applies
an affine transform (rotation + translation + scale) via grid_sample,
and composites over a separately decoded background.

Design spec: traitful-docs/.../specs/compositional-stn-vae.md

Two latent modes:
- 'flat': single z, all decoders read full z.
- 'split': z = [z_bg, z_pose, z_obj], each decoder reads only its slice.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_inverse_affine(
    tx: torch.Tensor, ty: torch.Tensor,
    sin_t: torch.Tensor, cos_t: torch.Tensor,
    s: torch.Tensor,
) -> torch.Tensor:
    """Build inverse affine matrix for F.affine_grid.

    PyTorch convention: affine_grid expects the INVERSE transform
    (output->input mapping). Given our forward transform that places
    the canonical patch into the frame:

        Forward:  frame_pos = s * R @ canon_pos + t
        Inverse:  canon_pos = (1/s) * R^T @ (frame_pos - t)

    where R = [[cos, -sin], [sin, cos]]. The inverse matrix is:

        [[ (1/s)*cos,  (1/s)*sin, (1/s)*(-cos*tx - sin*ty) ],
         [-(1/s)*sin,  (1/s)*cos, (1/s)*( sin*tx - cos*ty) ]]

    Args:
        tx, ty: translation in PyTorch grid coords [-1, 1]. Shape (B,).
        sin_t, cos_t: rotation on unit circle. Shape (B,).
        s: positive scale. Shape (B,).

    Returns:
        (B, 2, 3) inverse affine matrix for F.affine_grid.
    """
    B = tx.shape[0]
    inv_s = 1.0 / s
    # Allocate on same device/dtype as input
    theta = tx.new_zeros(B, 2, 3)
    # Rotation block: (1/s) * R^T
    theta[:, 0, 0] = inv_s * cos_t
    theta[:, 0, 1] = inv_s * sin_t
    theta[:, 1, 0] = -inv_s * sin_t
    theta[:, 1, 1] = inv_s * cos_t
    # Translation column: -(1/s) * R^T @ t
    theta[:, 0, 2] = inv_s * (-cos_t * tx - sin_t * ty)
    theta[:, 1, 2] = inv_s * (sin_t * tx - cos_t * ty)
    return theta


def _make_small_decoder(
    in_dim: int, out_channels: int, spatial: int,
    hidden_ch: int = 64, mid_ch: int = 32,
) -> nn.Sequential:
    """Build a tiny 2-layer deconv decoder.

    Architecture: Linear -> reshape -> Upsample+Conv -> ReLU -> Upsample+Conv -> Sigmoid.
    Uses Upsample(size=...)+Conv instead of ConvTranspose2d to avoid spatial dimension mismatches.

    Used for three decoders with different spatial targets:
    - Object decoder: spatial=16 (canonical patch). Reshape 4x4 -> 8 -> 16.
    - Mask decoder: spatial=16 (same). Separate weights from object decoder.
    - Background decoder: spatial=84 (full frame). Reshape 7x7 -> 14 -> 84.
    """
    if spatial <= 16:
        init_spatial = 4
        mid_spatial = spatial // 2  # 8 for spatial=16
    else:
        init_spatial = 7
        mid_spatial = 14

    flat_dim = hidden_ch * init_spatial * init_spatial
    return nn.Sequential(
        nn.Linear(in_dim, flat_dim),
        nn.Unflatten(1, (hidden_ch, init_spatial, init_spatial)),
        nn.Upsample(size=mid_spatial, mode='bilinear', align_corners=False),
        nn.Conv2d(hidden_ch, mid_ch, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.Upsample(size=spatial, mode='bilinear', align_corners=False),
        nn.Conv2d(mid_ch, out_channels, kernel_size=3, stride=1, padding=1),
        nn.Sigmoid(),
    )


class CompositionalPixelVAE(nn.Module):
    """Compositional STN VAE with canonical lander + affine warp + background.

    Rendering pipeline:
        z -> pose_head -> affine params (tx, ty, sin, cos, scale)
        z -> obj_decoder -> O_hat (16x16 canonical upright lander)
        z -> mask_decoder -> A_hat (16x16 alpha mask)
        z -> bg_decoder -> B_hat (84x84 background)
        affine_grid + grid_sample -> warp O_hat and A_hat into frame
        x_hat = A_warp * O_warp + (1 - A_warp) * B_hat
    """

    def __init__(
        self,
        in_channels: int = 1,
        latent_dim: int = 45,
        frame_size: int = 84,
        channels: list[int] | None = None,
        latent_mode: str = 'flat',
        bg_dim: int = 8,
        obj_dim: int = 32,
        pose_dim: int = 5,
        canonical_size: int = 16,
        beta: float = 0.0001,
        fixed_scale: float | None = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.frame_size = frame_size
        self.latent_mode = latent_mode
        self.canonical_size = canonical_size
        self.beta = beta
        # When fixed_scale is set, pose head outputs 4 dims (tx, ty, sin, cos)
        # instead of 5. Scale is a constant — no learning, no collapse.
        # Use when canonical template is at the correct pixel proportions
        # and only translation + rotation need to be learned.
        self.fixed_scale = fixed_scale
        self.pose_dim = 4 if fixed_scale is not None else pose_dim

        if channels is None:
            channels = [32, 64, 128, 256]
        self.channels = channels

        # --- Dimension bookkeeping ---
        if latent_mode == 'split':
            self.bg_dim = bg_dim
            self.obj_dim = obj_dim
            self.latent_dim = bg_dim + pose_dim + obj_dim
            self._kl_dim = bg_dim + obj_dim
        else:
            self.bg_dim = 0
            self.obj_dim = 0
            self.latent_dim = latent_dim
            self._kl_dim = latent_dim

        # --- Encoder ---
        enc_layers = []
        prev_ch = in_channels
        for ch in channels:
            enc_layers.append(nn.Conv2d(prev_ch, ch, kernel_size=4, stride=2, padding=1))
            enc_layers.append(nn.ReLU(inplace=True))
            prev_ch = ch
        self.encoder_conv = nn.Sequential(*enc_layers)

        # Compute spatial size after encoder convolutions
        spatial = frame_size
        for _ in channels:
            spatial = spatial // 2
        self._spatial = spatial
        self._flat_dim = channels[-1] * spatial * spatial

        # --- Latent projections ---
        if latent_mode == 'flat':
            self.fc_mu = nn.Linear(self._flat_dim, latent_dim)
            self.fc_logvar = nn.Linear(self._flat_dim, latent_dim)
        else:
            self.fc_mu_bg = nn.Linear(self._flat_dim, bg_dim)
            self.fc_logvar_bg = nn.Linear(self._flat_dim, bg_dim)
            self.fc_mu_obj = nn.Linear(self._flat_dim, obj_dim)
            self.fc_logvar_obj = nn.Linear(self._flat_dim, obj_dim)
            self.fc_pose_raw = nn.Linear(self._flat_dim, pose_dim)

        # --- Pose head ---
        # Outputs (tx, ty, sin, cos) when fixed_scale is set (4 dims),
        # or (tx, ty, sin, cos, scale_raw) when scale is learned (5 dims).
        pose_output_dim = 4 if fixed_scale is not None else 5
        pose_input_dim = latent_dim if latent_mode == 'flat' else self.pose_dim
        self.pose_head = nn.Sequential(
            nn.Linear(pose_input_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, pose_output_dim),
        )
        if fixed_scale is None:
            # Initialize scale bias so initial scale ~ 0.1
            # softplus^{-1}(0.09) = ln(e^0.09 - 1) ~ -2.35
            with torch.no_grad():
                self.pose_head[-1].bias[4] = -2.35

        # --- Decoders ---
        obj_input_dim = latent_dim if latent_mode == 'flat' else obj_dim
        self.obj_decoder = _make_small_decoder(obj_input_dim, 1, canonical_size)
        self.mask_decoder = _make_small_decoder(obj_input_dim, 1, canonical_size)
        self._init_mask_center_bias()

        bg_input_dim = latent_dim if latent_mode == 'flat' else bg_dim
        self.bg_decoder = _make_small_decoder(bg_input_dim, 1, frame_size)

    def _init_mask_center_bias(self):
        """Initialize mask decoder's final conv bias for center-biased output."""
        # Sequential is: Linear, Unflatten, Upsample, Conv2d, ReLU, Upsample, Conv2d, Sigmoid
        # The final Conv2d is at index -2 (Sigmoid is last)
        last_conv = self.mask_decoder[-2]
        with torch.no_grad():
            last_conv.bias.data[0] = 1.0

    @property
    def state_dim(self) -> int:
        """Returns 5 (pose dims). For callback compatibility."""
        return 5

    def _encode_features(self, x: torch.Tensor) -> torch.Tensor:
        """Run encoder CNN, return flattened features."""
        h = self.encoder_conv(x)
        return h.reshape(h.size(0), -1)

    def encode_params(self, x: torch.Tensor):
        """Encode frames to distribution parameters."""
        feat = self._encode_features(x)
        if self.latent_mode == 'flat':
            return self.fc_mu(feat), self.fc_logvar(feat)
        else:
            mu_bg = self.fc_mu_bg(feat)
            logvar_bg = self.fc_logvar_bg(feat)
            mu_obj = self.fc_mu_obj(feat)
            logvar_obj = self.fc_logvar_obj(feat)
            pose_raw = self.fc_pose_raw(feat)
            return mu_bg, logvar_bg, mu_obj, logvar_obj, pose_raw

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample z using reparameterization trick."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + std * eps
        return mu

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode frame to latent z."""
        if self.latent_mode == 'flat':
            mu, logvar = self.encode_params(x)
            return self.reparameterize(mu, logvar)
        else:
            mu_bg, logvar_bg, mu_obj, logvar_obj, pose_raw = self.encode_params(x)
            z_bg = self.reparameterize(mu_bg, logvar_bg)
            z_obj = self.reparameterize(mu_obj, logvar_obj)
            return torch.cat([z_bg, pose_raw, z_obj], dim=-1)

    def _apply_pose_head(self, pose_input: torch.Tensor) -> torch.Tensor:
        """Run pose head and apply geometry constraints.

        Returns (B, 5): (tx, ty, sin_t, cos_t, s) regardless of whether
        scale is learned or fixed — downstream code always gets 5 dims.
        """
        raw = self.pose_head(pose_input)

        if self.fixed_scale is not None:
            # Fixed scale: pose head outputs 4 dims only
            tx, ty, sin_raw, cos_raw = raw.unbind(-1)
            s = torch.full_like(tx, self.fixed_scale)
        else:
            # Learned scale: pose head outputs 5 dims
            tx, ty, sin_raw, cos_raw, s_raw = raw.unbind(-1)
            # Softplus scale: s = softplus(raw) + 0.01
            s = F.softplus(s_raw) + 0.01

        # Unit-circle normalization — guarantees rigid rotation
        norm = torch.sqrt(sin_raw**2 + cos_raw**2 + 1e-6)
        sin_t = sin_raw / norm
        cos_t = cos_raw / norm

        return torch.stack([tx, ty, sin_t, cos_t, s], dim=-1)

    def _split_z(self, z: torch.Tensor):
        """Split concatenated z into (z_bg, z_pose_raw, z_obj)."""
        z_bg = z[:, :self.bg_dim]
        z_pose_raw = z[:, self.bg_dim:self.bg_dim + self.pose_dim]
        z_obj = z[:, self.bg_dim + self.pose_dim:]
        return z_bg, z_pose_raw, z_obj

    def decode_decomposed(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        """Full decode pipeline with all intermediate outputs."""
        if self.latent_mode == 'flat':
            pose_input = z
            obj_input = z
            bg_input = z
        else:
            z_bg, z_pose_raw, z_obj = self._split_z(z)
            pose_input = z_pose_raw
            obj_input = z_obj
            bg_input = z_bg

        # Pose head -> affine transform parameters
        pose_params = self._apply_pose_head(pose_input)
        tx, ty, sin_t, cos_t, s = pose_params.unbind(-1)

        # Decode canonical patches and background
        O_hat = self.obj_decoder(obj_input)
        A_hat = self.mask_decoder(obj_input)
        B_hat = self.bg_decoder(bg_input)

        # Affine warp: place canonical patch into full frame
        inv_affine = build_inverse_affine(tx, ty, sin_t, cos_t, s)
        B_size = z.size(0)
        grid = F.affine_grid(
            inv_affine,
            size=(B_size, 1, self.frame_size, self.frame_size),
            align_corners=False,
        )
        O_warp = F.grid_sample(O_hat, grid, padding_mode='zeros', align_corners=False)
        A_warp = F.grid_sample(A_hat, grid, padding_mode='zeros', align_corners=False)

        # Alpha compositing
        x_hat = A_warp * O_warp + (1 - A_warp) * B_hat

        return {
            'x_hat': x_hat,
            'O_hat': O_hat,
            'A_hat': A_hat,
            'B_hat': B_hat,
            'O_warp': O_warp,
            'A_warp': A_warp,
            'pose_params': pose_params,
        }

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent z to reconstructed frame."""
        return self.decode_decomposed(z)['x_hat']

    def predict_state(self, z: torch.Tensor) -> torch.Tensor:
        """Predict pose from z. Drop-in for PixelVAE.predict_state interface."""
        if self.latent_mode == 'flat':
            return self._apply_pose_head(z)
        else:
            _, z_pose_raw, _ = self._split_z(z)
            return self._apply_pose_head(z_pose_raw)

    def forward(
        self, x: torch.Tensor, gt_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass: encode -> sample -> decode -> compose."""
        if self.latent_mode == 'flat':
            mu, logvar = self.encode_params(x)
            z = self.reparameterize(mu, logvar)
            decomposed = self.decode_decomposed(z)
        else:
            mu_bg, logvar_bg, mu_obj, logvar_obj, pose_raw = self.encode_params(x)
            z_bg = self.reparameterize(mu_bg, logvar_bg)
            z_obj = self.reparameterize(mu_obj, logvar_obj)
            z = torch.cat([z_bg, pose_raw, z_obj], dim=-1)
            decomposed = self.decode_decomposed(z)
            mu = torch.cat([mu_bg, mu_obj], dim=-1)
            logvar = torch.cat([logvar_bg, logvar_obj], dim=-1)

        # Cache decomposition for training loop access
        self._last_decomposed = decomposed

        return decomposed['x_hat'], mu, logvar, decomposed['pose_params']

    def get_last_decomposed(self) -> dict[str, torch.Tensor]:
        """Return decomposition dict from last forward() call."""
        if not hasattr(self, '_last_decomposed') or self._last_decomposed is None:
            raise RuntimeError(
                "No cached decomposition -- call forward() before get_last_decomposed()"
            )
        return self._last_decomposed
