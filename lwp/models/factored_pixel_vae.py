"""Factored Pixel VAE — explicit [z_kin, z_ctx] latent split.

Inherits from PixelVAE. Same encoder architecture. The latent space is
split into z_kin (supervised kinematic dims) and z_ctx (unsupervised
context dims). Two decoder variants:

  concat: decode(concat(z_kin, z_ctx)) — same as parent. Split is loss-only.
  film: z_ctx → features, FiLM(features, z_kin) → frame. Kinematics
        multiplicatively modulates appearance, forcing the encoder to
        put rendering-relevant kinematics into z_kin.

The key difference from PixelVAE with a state_head: predict_state(z)
returns a SLICE of z (zero perception noise from the head itself), not
a neural network output. The encoder must learn to place GT kinematic
values directly into the designated z dims.

Design spec: traitful-docs/.../specs/factored-pixel-world-model.md
E3 Finding 04: angle is the load-bearing kinematic — kin_targets enables
ablation of which dims to supervise (e.g., angle-only = [4, 5]).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from lwp.models.pixel_vae import PixelVAE


class FactoredPixelVAE(PixelVAE):
    """VAE with factored latent: z = [z_kin, z_ctx].

    z_kin dims are at positions specified by kin_targets (default [0..5]).
    z_ctx dims are all remaining dims. The encoder places supervised
    kinematic values into z_kin positions; the decoder uses them via
    concat (baseline) or FiLM (architectural separation).

    Args:
        kin_targets: list of latent dim indices to designate as z_kin.
            Default [0,1,2,3,4,5] = all 6 kinematics.
            [4,5] = angle + ang_vel only (from E3 Finding 04).
            The dataset must return states with matching dims.
        decoder_type: "concat" or "film".
            concat: same decoder as PixelVAE (split is loss-only)
            film: z_ctx → features → FiLM(features, z_kin) → frame
    """

    def __init__(
        self,
        in_channels: int = 1,
        latent_dim: int = 64,
        frame_size: int = 84,
        channels: list[int] | None = None,
        beta: float = 0.0001,
        kin_targets: list[int] | None = None,
        decoder_type: str = "concat",
        coord_conv: bool = False,
    ):
        # kin_targets determines which z dims are kinematic.
        # state_dim for the parent = len(kin_targets) so the state_head
        # gets created (though we override predict_state to bypass it).
        kin_targets = kin_targets or [0, 1, 2, 3, 4, 5]
        # Initialize parent with state_dim=0 — we handle state prediction
        # ourselves via z slice, not via the parent's MLP state_head.
        # coord_conv passed through so encoder gets coordinate channels.
        super().__init__(
            in_channels=in_channels,
            latent_dim=latent_dim,
            frame_size=frame_size,
            channels=channels,
            beta=beta,
            state_dim=0,  # no MLP state head — we use z slice
            coord_conv=coord_conv,
        )

        self.kin_targets = kin_targets
        self.kin_dim = len(kin_targets)
        self.ctx_dim = latent_dim - self.kin_dim

        # Validate: kin_targets must be valid indices into latent_dim
        if max(kin_targets) >= latent_dim:
            raise ValueError(
                f"kin_targets max index {max(kin_targets)} >= latent_dim {latent_dim}")

        # Register as buffers so they move with model.to(device) and are
        # included in state_dict — avoids re-creating tensors per forward pass
        self.register_buffer("_kin_indices",
                             torch.tensor(kin_targets, dtype=torch.long))
        self.register_buffer("_ctx_indices",
                             torch.tensor([i for i in range(latent_dim)
                                           if i not in kin_targets], dtype=torch.long))
        self.decoder_type = decoder_type

        # Override state_dim property value — PixelVAE stores it as self.state_dim
        # but we set state_dim=0 in super().__init__. We need to fix this.
        self._factored_state_dim = self.kin_dim

        if decoder_type == "film":
            self._build_film_decoder()

    def _build_film_decoder(self):
        """Build FiLM decoder modules. Called only when decoder_type='film'.

        Architecture:
          z_ctx → Linear → spatial features (same flat_dim as parent)
          z_kin → Linear → (gamma, beta) per channel
          features * gamma + beta → existing conv upsampling layers

        The conv layers (decoder_layers from parent) are reused — only
        the initial projection changes from Linear(latent_dim → flat_dim)
        to ctx_to_features(ctx_dim → flat_dim) + FiLM modulation.
        """
        ctx_dim = len(self._ctx_indices)
        # Project z_ctx to the same spatial feature shape as parent's fc_decode
        self.ctx_to_features = nn.Linear(ctx_dim, self._flat_dim)
        # FiLM generator: z_kin → (gamma, beta) for per-channel modulation.
        # channels[-1] is the first decoder layer's channel count (256 by default).
        # gamma and beta each have this many dims.
        n_channels = self.channels[-1]
        self.kin_to_film = nn.Sequential(
            nn.Linear(self.kin_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, n_channels * 2),  # gamma + beta concatenated
        )

    @property
    def state_dim(self):
        """Number of kinematic dims in z_kin. Used by eval pipeline for
        compatibility checks (if vae.state_dim > 0: run kinematics eval)."""
        return self._factored_state_dim

    @state_dim.setter
    def state_dim(self, value):
        """Allow parent __init__ to set state_dim=0 without error."""
        pass  # Parent sets self.state_dim = state_dim; we ignore it

    def predict_state(self, z: torch.Tensor) -> torch.Tensor:
        """Return z_kin — the kinematic slice of z. No neural network.

        Unlike PixelVAE.predict_state which runs an MLP head, this is a
        direct index into z. Zero perception noise from the head itself.
        The only error is how well the encoder placed kinematics into
        the designated z dims during VAE training.
        """
        return z[:, self._kin_indices]

    def _split_z(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Split z into z_kin and z_ctx using configured indices."""
        return z[:, self._kin_indices], z[:, self._ctx_indices]

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode z to frame. Accepts single concatenated z (same signature
        as PixelVAE.decode) — splits internally for FiLM variant.

        Called by PixelWorldModel.dream(), DreamGridCallback, dream_compare.py,
        eval pipeline, etc. — all pass a single flat z tensor.
        """
        if self.decoder_type == "concat":
            # Same as parent — the split is only in the loss, not architecture
            return super().decode(z)
        else:
            return self._decode_film(z)

    def _decode_film(self, z: torch.Tensor) -> torch.Tensor:
        """FiLM decoder: z_ctx provides appearance, z_kin modulates.

        z_ctx → Linear → spatial features (B, C, H, W)
        z_kin → FiLM generator → gamma, beta (B, C)
        modulated = gamma * features + beta
        modulated → conv upsampling layers → frame
        """
        z_kin, z_ctx = self._split_z(z)

        # z_ctx → base appearance features
        h = self.ctx_to_features(z_ctx)
        # Reshape to spatial: (B, channels[-1], spatial, spatial)
        h = h.reshape(h.size(0), self.channels[-1], self._spatial, self._spatial)

        # z_kin → FiLM modulation parameters
        film_params = self.kin_to_film(z_kin)  # (B, channels[-1] * 2)
        gamma, beta = film_params.chunk(2, dim=-1)  # each (B, channels[-1])
        # Center gamma at 1 so default modulation is identity —
        # untrained model passes features through unchanged
        gamma = gamma + 1.0

        # Modulate: kinematics controls how appearance features render.
        # Per-channel (not per-pixel) because z_kin is only kin_dim numbers.
        # gamma.unsqueeze(-1).unsqueeze(-1) broadcasts over H, W dims.
        h = gamma.unsqueeze(-1).unsqueeze(-1) * h + beta.unsqueeze(-1).unsqueeze(-1)

        # Standard conv upsampling layers (reused from parent)
        for layer in self.decoder_layers:
            h = layer(h)
        return h

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward: encode, sample, decode, return z_kin as state_pred.

        Returns same 4-tuple as PixelVAE for training loop compatibility:
        (recon, mu, logvar, state_pred) where state_pred = z_kin slice.
        """
        mu, logvar = self.encode_params(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        # state_pred is z_kin — a slice of z, not a neural network output
        state_pred = self.predict_state(z)
        return recon, mu, logvar, state_pred
