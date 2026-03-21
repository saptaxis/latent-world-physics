# models/pixel_vae.py
"""Convolutional VAE for frame compression.

Encodes frames (B, C, H, W) to a latent vector z of size latent_dim.
Decoder reconstructs frames from z. Supports configurable resolution
(84, 128) and input channels (1 for single frame, 4 for stacked).

Architecture follows ArcadeDreamer: 4 conv layers with increasing
channels, kernel 4, stride 2, ReLU. Decoder mirrors with transposed
convolutions and sigmoid output.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class PixelVAE(nn.Module):
    """Convolutional VAE for pixel-space world modeling.

    Args:
        in_channels: number of input channels (1=grayscale, 4=stacked frames)
        latent_dim: dimensionality of latent vector z
        frame_size: spatial resolution (84 or 128)
        channels: list of channel sizes for conv layers
        beta: KL divergence weight (used externally in loss, stored for reference)
        state_dim: if > 0, add auxiliary state prediction head (z → kinematic state).
            Forces latent space to encode physical state, giving z spatial/physical
            meaning. 6 = (x, y, vx, vy, angle, ang_vel).
    """

    def __init__(
        self,
        in_channels: int = 1,
        latent_dim: int = 64,
        frame_size: int = 84,
        channels: list[int] | None = None,
        beta: float = 0.0001,
        state_dim: int = 0,
        coord_conv: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.frame_size = frame_size
        self.beta = beta
        self.coord_conv = coord_conv

        # Default channel progression doubles at each layer following the
        # ArcadeDreamer convention — 32→64→128→256 gives 4x capacity growth
        # that balances representation power against parameter count
        if channels is None:
            channels = [32, 64, 128, 256]
        self.channels = channels

        # CoordConv (Liu et al., 2018): append fixed x,y coordinate meshgrids
        # as extra input channels. Gives the CNN explicit spatial information
        # so it doesn't have to learn positional encoding from scratch. Helps
        # with rotation/position which require spatial reasoning.
        # Only affects encoder — decoder gets spatial info via the latent vector.
        encoder_in_ch = in_channels + 2 if coord_conv else in_channels

        # Build encoder: sequence of Conv2d(kernel=4, stride=2, pad=1) + ReLU
        # Each layer halves spatial dims: 84->42->21->10->5, 128->64->32->16->8
        enc_layers = []
        prev_ch = encoder_in_ch
        for ch in channels:
            enc_layers.append(nn.Conv2d(prev_ch, ch, kernel_size=4, stride=2, padding=1))
            enc_layers.append(nn.ReLU(inplace=True))
            prev_ch = ch
        self.encoder_conv = nn.Sequential(*enc_layers)

        # Register fixed coordinate grids as buffers — they move with model.to(device)
        # and are included in state_dict for checkpoint round-trips. Normalized
        # to [-1, 1] so the coordinate signal has similar scale to pixel values.
        if coord_conv:
            # x varies across columns (left=-1 to right=+1)
            x_grid = torch.linspace(-1, 1, frame_size).view(1, 1, 1, frame_size).expand(
                1, 1, frame_size, frame_size)
            # y varies across rows (top=-1 to bottom=+1)
            y_grid = torch.linspace(-1, 1, frame_size).view(1, 1, frame_size, 1).expand(
                1, 1, frame_size, frame_size)
            self.register_buffer("_coord_x", x_grid)
            self.register_buffer("_coord_y", y_grid)

        # Compute spatial size after encoder convolutions.
        # Each conv with k=4, s=2, p=1: out = floor((in + 2*1 - 4) / 2 + 1) = floor(in/2)
        spatial = frame_size
        for _ in channels:
            spatial = spatial // 2
        self._spatial = spatial
        self._flat_dim = channels[-1] * spatial * spatial

        # Latent projections — separate heads for mu and logvar so the
        # network can independently control the mean and uncertainty of
        # each latent dimension
        self.fc_mu = nn.Linear(self._flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self._flat_dim, latent_dim)

        # Decoder: linear -> reshape -> upsample+conv layers
        # Uses Upsample+Conv2d instead of ConvTranspose2d to avoid spatial
        # dimension mismatches (84->42->21->10->5, transposed conv from 5
        # produces 80, not 84). Upsample to exact target size each layer.
        self.fc_decode = nn.Linear(latent_dim, self._flat_dim)

        # Compute target spatial sizes for each decoder layer (reverse of encoder)
        self._dec_sizes = []
        s = frame_size
        for _ in channels:
            self._dec_sizes.append(s)
            s = s // 2
        self._dec_sizes.reverse()  # [5, 10, 21, 42, 84] for frame_size=84

        dec_layers = nn.ModuleList()
        rev_channels = list(reversed(channels))  # [256, 128, 64, 32] — mirror encoder order
        for i in range(len(rev_channels) - 1):
            # Each decoder block: upsample to the exact target spatial size,
            # then refine with a conv layer. Using Upsample+Conv instead of
            # ConvTranspose2d avoids checkerboard artifacts and guarantees
            # exact spatial dimensions (critical for non-power-of-2 sizes like 84)
            dec_layers.append(nn.Sequential(
                nn.Upsample(size=self._dec_sizes[i + 1], mode='bilinear', align_corners=False),
                nn.Conv2d(rev_channels[i], rev_channels[i + 1],
                          kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
            ))
        # Final layer: upsample to original frame_size, project to input
        # channels, sigmoid clamps output to [0,1] matching normalized pixel range
        dec_layers.append(nn.Sequential(
            nn.Upsample(size=frame_size, mode='bilinear', align_corners=False),
            nn.Conv2d(rev_channels[-1], in_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        ))
        self.decoder_layers = dec_layers

        # Optional auxiliary state prediction head — forces the latent space
        # to encode physical state (position, velocity, angle) by adding a
        # supervised regression loss. Small hidden layer (64) is intentional:
        # the mapping from a well-structured z to kinematic state should be
        # near-linear, so a small head avoids overfitting while still providing
        # gradient signal to shape the latent space
        self.state_dim = state_dim
        if state_dim > 0:
            self.state_head = nn.Sequential(
                nn.Linear(latent_dim, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, state_dim),
            )
        else:
            self.state_head = None

    def predict_state(self, z: torch.Tensor) -> torch.Tensor | None:
        """Predict kinematic state from latent z. Returns None if no state head."""
        if self.state_head is None:
            return None
        return self.state_head(z)

    def encode_params(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode frames to (mu, logvar) parameters."""
        if self.coord_conv:
            B = x.size(0)
            # Expand fixed grids to batch size and concatenate with input.
            # (B, C, H, W) + (B, 1, H, W) + (B, 1, H, W) -> (B, C+2, H, W)
            coords = torch.cat([
                self._coord_x.expand(B, -1, -1, -1),
                self._coord_y.expand(B, -1, -1, -1),
            ], dim=1)
            x = torch.cat([x, coords], dim=1)
        h = self.encoder_conv(x)
        # Flatten spatial dims: (B, C_last, S, S) -> (B, C_last * S * S)
        # so the linear projection can map to latent_dim
        h = h.reshape(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample z using reparameterization trick."""
        if self.training:
            # exp(0.5 * logvar) = sqrt(var) = std; log-space parameterisation
            # ensures std is always positive without constrained optimisation
            std = torch.exp(0.5 * logvar)
            # Reparameterization: z = mu + std * eps allows gradients to flow
            # through mu and std while eps ~ N(0,1) provides stochasticity
            eps = torch.randn_like(std)
            return mu + std * eps
        # At eval, return the mode (mu) for deterministic predictions
        return mu

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode frames to latent z. Deterministic at eval (returns mu)."""
        mu, logvar = self.encode_params(x)
        return self.reparameterize(mu, logvar)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent z to reconstructed frames."""
        h = self.fc_decode(z)
        # Reshape flat vector back to spatial feature map: (B, flat_dim) ->
        # (B, last_enc_channels, spatial, spatial) to feed into conv decoder
        h = h.reshape(h.size(0), self.channels[-1], self._spatial, self._spatial)
        for layer in self.decoder_layers:
            h = layer(h)
        return h

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Full forward pass: encode, sample, decode, optionally predict state.

        Returns:
            recon: (B, C, H, W) reconstruction
            mu: (B, latent_dim)
            logvar: (B, latent_dim)
            state_pred: (B, state_dim) or None if no state head
        """
        mu, logvar = self.encode_params(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        state_pred = self.predict_state(z)
        return recon, mu, logvar, state_pred
