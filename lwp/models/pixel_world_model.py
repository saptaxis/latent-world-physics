# models/pixel_world_model.py
"""Combined VAE + dynamics pixel world model.

Top-level interface for encoding, predicting, and dreaming.
Supports both single-frame (in_channels=1) and stacked-frame
(in_channels=4) modes. For stacked frames, dreaming maintains
a frame buffer to construct stacked inputs from decoded predictions.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from lwp.models.pixel_vae import PixelVAE


class PixelWorldModel(nn.Module):
    """Combined pixel world model: VAE encoder-decoder + latent dynamics.

    Accepts any dynamics module that implements forward(z, action, state)
    and rollout(z_start, actions) — both LatentDynamicsModel (GRU) and
    LatentRSSM satisfy this interface.
    """

    def __init__(self, vae: PixelVAE, dynamics: nn.Module):
        super().__init__()
        self.vae = vae
        self.dynamics = dynamics

    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        """Encode frames to latent z (deterministic — returns mu)."""
        # Temporarily switch to eval mode so encode() returns mu (deterministic)
        # rather than a stochastic sample — we want consistent latent codes
        # when encoding observations for dynamics training or dreaming
        was_training = self.vae.training
        self.vae.eval()
        with torch.no_grad():
            z = self.vae.encode(frames)
        # Restore original training mode to avoid side effects on the VAE
        self.vae.train(was_training)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent z to frames."""
        return self.vae.decode(z)

    def predict_next(self, frames: torch.Tensor, action: torch.Tensor,
                     hidden: torch.Tensor | None = None
                     ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Teacher-forced single-step: encode real frame, predict next, decode."""
        # Encode with training mode active — stochastic sampling during training
        # acts as regularisation for the dynamics model
        z = self.vae.encode(frames)
        # Dynamics predicts next latent from current latent + action
        z_next, hidden = self.dynamics(z, action, hidden)
        # Decode back to pixel space for reconstruction loss or visualisation
        pred_frame = self.vae.decode(z_next)
        return pred_frame, z_next, hidden

    @torch.no_grad()
    def dream(self, seed_frames: torch.Tensor, actions: torch.Tensor
              ) -> torch.Tensor:
        """Autoregressive dream: encode seed, roll out feeding own predictions.

        For in_channels=1: operates purely in latent space.
        For in_channels>1: maintains frame buffer, re-encodes stacked frames.

        Returns:
            frames: (B, T+1, C, H, W) — seed frame + T predicted frames
        """
        was_training = self.training
        self.eval()

        B, C, H, W = seed_frames.shape
        T = actions.size(1)

        if C == 1:
            # Single-frame mode: pure latent-space rollout is efficient —
            # encode once, rollout in latent space, batch-decode all at once
            z = self.vae.encode(seed_frames)
            z_seq, _ = self.dynamics.rollout(z, actions)
            # Batch-decode all T+1 latents at once for GPU efficiency
            all_z = z_seq.reshape(B * (T + 1), -1)
            all_frames = self.vae.decode(all_z)
            frames = all_frames.reshape(B, T + 1, 1, H, W)
        else:
            # Stacked-frame mode (C>1): cannot do pure latent rollout because
            # the encoder expects C stacked channels. Instead, maintain a
            # sliding buffer of individual frames, decode each predicted
            # frame, push it into the buffer, and re-encode the new stack.
            frame_stack = C
            # Split seed (B, C, H, W) into C individual (B, 1, H, W) frames
            buffer = list(seed_frames.split(1, dim=1))

            z = self.vae.encode(seed_frames)
            hidden = None
            all_frames = [seed_frames]

            for t in range(T):
                z_next, hidden = self.dynamics(z, actions[:, t], hidden)
                # Decode to pixels, take only channel 0 — the decoder outputs
                # C channels but we only need the newest single frame prediction
                pred_single = self.vae.decode(z_next)[:, :1]
                # Slide the buffer: drop oldest frame, append newest prediction
                buffer = buffer[1:] + [pred_single]
                # Re-stack and re-encode so the next dynamics step sees the
                # updated frame history, maintaining temporal context
                stacked = torch.cat(buffer, dim=1)
                all_frames.append(stacked)
                z = self.vae.encode(stacked)

            # Stack along time: (B, T+1, C, H, W)
            frames = torch.stack(all_frames, dim=1)

        self.train(was_training)
        return frames

    @torch.no_grad()
    def dream_from_latent(self, z_seed: torch.Tensor, actions: torch.Tensor
                          ) -> tuple[torch.Tensor, torch.Tensor]:
        """Dream from a latent seed. Handles compositional split-mode z_bg copy."""
        was_training = self.training
        self.eval()

        B, T = z_seed.shape[0], actions.shape[1]

        # Check if compositional split mode — need to strip z_bg before dynamics
        if hasattr(self.vae, 'latent_mode') and self.vae.latent_mode == 'split':
            bg_dim = self.vae.bg_dim
            z_bg = z_seed[:, :bg_dim]                    # (B, bg_dim) — static
            z_dynamic = z_seed[:, bg_dim:]                # (B, pose_dim + obj_dim)
            z_dynamic_seq, _ = self.dynamics.rollout(z_dynamic, actions)  # (B, T+1, dyn_dim)
            # Re-attach z_bg for all timesteps
            z_bg_expanded = z_bg.unsqueeze(1).expand(-1, T + 1, -1)  # (B, T+1, bg_dim)
            z_seq = torch.cat([z_bg_expanded, z_dynamic_seq], dim=-1)  # (B, T+1, full_dim)
        else:
            # Flat mode or standard model — dynamics on full z
            z_seq, _ = self.dynamics.rollout(z_seed, actions)  # (B, T+1, latent_dim)

        # Decode all timesteps
        B, Tp1, D = z_seq.shape
        all_z = z_seq.reshape(B * Tp1, D)
        all_frames = self.vae.decode(all_z)
        C, H, W = all_frames.shape[1:]
        frames = all_frames.reshape(B, Tp1, C, H, W)

        self.train(was_training)
        return frames, z_seq
