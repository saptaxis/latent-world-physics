"""Factored dynamics model — separate kinematic and context prediction heads.

Maintains the factored VAE's [z_kin, z_ctx] separation through prediction.
f_kin predicts z_kin_next from z_kin + action (physics pathway).
f_ctx predicts z_ctx_next from z_ctx + z_kin (appearance pathway).
Both share a GRU hidden state for temporal context.

External interface is identical to LatentDynamicsModel (forward, rollout,
predict_sequence, initial_state) — factorization is internal. All callbacks,
PixelWorldModel.dream(), and loss functions work unchanged.

Design spec: traitful-docs/.../specs/factored-pixel-world-model.md
"""
from __future__ import annotations

import torch
import torch.nn as nn


class FactoredDynamicsModel(nn.Module):
    """Dynamics with separate f_kin and f_ctx prediction heads.

    Architecture:
        Shared GRU: maintains temporal context for both heads.
        Input: concat(z_kin, z_ctx, action) → GRU → hidden

        f_kin head: Linear(hidden → hidden_kin) → ReLU → Linear(hidden_kin → kin_dims)
            Predicts z_kin_next from GRU hidden. Action effects flow through
            the shared GRU input where action is concatenated.

        f_ctx head: Linear(hidden + z_kin → hidden_ctx) → ReLU → Linear(hidden_ctx → ctx_dims)
            Predicts z_ctx_next from GRU hidden AND z_kin (one-directional:
            z_kin informs appearance, but f_ctx cannot write to z_kin dims).

    Args:
        latent_dim: total latent dim (kin_dims + ctx_dims)
        action_dim: action vector dim
        hidden_size: GRU hidden state size
        kin_dims: number of leading z dims that are kinematic
    """

    def __init__(self, latent_dim: int = 64, action_dim: int = 2,
                 hidden_size: int = 256, kin_dims: int = 6):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.kin_dims = kin_dims
        self.ctx_dims = latent_dim - kin_dims

        # Shared GRU: sees full z + action for temporal context.
        # Both heads read from the GRU hidden state.
        self.input_proj = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_size),
            nn.ReLU(inplace=True),
        )
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )

        # f_kin: physics head. Predicts z_kin_next from GRU hidden.
        # Small output (kin_dims, typically 6) — this is the physics model.
        self.f_kin = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, kin_dims),
        )

        # f_ctx: appearance head. Predicts z_ctx_next from GRU hidden + z_kin.
        # z_kin input because appearance depends on kinematics (where the
        # lander is determines what pixels look like). One-directional:
        # z_kin flows in, but f_ctx output is z_ctx only.
        self.f_ctx = nn.Sequential(
            nn.Linear(hidden_size + kin_dims, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, self.ctx_dims),
        )

    def initial_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Zero-initialized GRU hidden state."""
        return torch.zeros(1, batch_size, self.hidden_size, device=device)

    def forward(self, z: torch.Tensor, action: torch.Tensor,
                hidden: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """Single-step prediction. Returns full concatenated z_next."""
        if hidden is None:
            hidden = self.initial_state(z.size(0), z.device)

        # Split input z into kinematic and context parts
        z_kin = z[:, :self.kin_dims]

        # Shared GRU step: sees full z + action for temporal context
        x = torch.cat([z, action], dim=-1)
        x = self.input_proj(x)
        x = x.unsqueeze(1)
        output, hidden = self.gru(x, hidden)
        h = output.squeeze(1)  # (B, hidden_size)

        # f_kin: predict z_kin_next from GRU hidden
        z_kin_next = self.f_kin(h)

        # f_ctx: predict z_ctx_next from GRU hidden + z_kin
        # z_kin flows into f_ctx so appearance can track kinematics
        ctx_input = torch.cat([h, z_kin], dim=-1)
        z_ctx_next = self.f_ctx(ctx_input)

        # Concatenate back to full latent_dim — external interface unchanged
        z_next = torch.cat([z_kin_next, z_ctx_next], dim=-1)
        return z_next, hidden

    def rollout(self, z_start: torch.Tensor, actions: torch.Tensor,
                hidden: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """Multi-step autoregressive rollout. Returns (B, T+1, latent_dim)."""
        T = actions.size(1)
        if hidden is None:
            hidden = self.initial_state(z_start.size(0), z_start.device)

        z_seq = [z_start]
        z = z_start
        for t in range(T):
            z, hidden = self.forward(z, actions[:, t], hidden)
            z_seq.append(z)

        return torch.stack(z_seq, dim=1), hidden

    def predict_sequence(self, z_sequence: torch.Tensor, actions: torch.Tensor,
                         hidden: torch.Tensor | None = None,
                         teacher_forcing: float = 1.0
                         ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict next latents for a sequence (scheduled sampling)."""
        B, T, _ = z_sequence.shape
        if hidden is None:
            hidden = self.initial_state(B, z_sequence.device)

        z_preds = []
        z = z_sequence[:, 0]

        for t in range(T):
            z_next, hidden = self.forward(z, actions[:, t], hidden)
            z_preds.append(z_next)

            if t < T - 1:
                if self.training and torch.rand(1).item() < teacher_forcing:
                    z = z_sequence[:, t + 1]
                else:
                    z = z_next.detach() if self.training else z_next

        return torch.stack(z_preds, dim=1), hidden
