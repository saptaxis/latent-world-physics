"""Level 3: GRU encoder-decoder world model with recurrent hidden state."""
from __future__ import annotations

import torch
import torch.nn as nn

from lwp.models.base import WorldModel


class GRUModel(WorldModel):
    """GRU world model: encoder -> GRU -> decoder.

    Architecture:
        encoder MLP: [obs, action] -> GRU input
        GRU: processes encoded input, maintains hidden state
        decoder MLP: GRU output -> delta prediction

    Args:
        state_dim: observation dimension
        action_dim: action dimension
        hidden_dim: GRU hidden state dimension
        num_layers: number of stacked GRU layers
        encoder_dims: hidden layer sizes for encoder MLP (empty = linear projection)
        decoder_dims: hidden layer sizes for decoder MLP (empty = linear projection)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 1,
        encoder_dims: list[int] = None,
        decoder_dims: list[int] = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Encoder: [obs, action] -> gru_input_dim
        if encoder_dims is None:
            encoder_dims = []
        enc_layers = []
        in_dim = state_dim + action_dim
        for h in encoder_dims:
            enc_layers.append(nn.Linear(in_dim, h))
            enc_layers.append(nn.ReLU())
            in_dim = h
        # Final projection to hidden_dim (GRU input size)
        enc_layers.append(nn.Linear(in_dim, hidden_dim))
        self.encoder = nn.Sequential(*enc_layers)

        # GRU
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=False,
        )

        # Decoder: gru_output -> delta
        if decoder_dims is None:
            decoder_dims = []
        dec_layers = []
        in_dim = hidden_dim
        for h in decoder_dims:
            dec_layers.append(nn.Linear(in_dim, h))
            dec_layers.append(nn.ReLU())
            in_dim = h
        dec_layers.append(nn.Linear(in_dim, state_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def initial_state(self, batch_size: int, device=None):
        """Return zero hidden state [num_layers, batch, hidden_dim]."""
        return torch.zeros(self.num_layers, batch_size, self.hidden_dim,
                           device=device)

    def step(self, obs, action, model_state=None):
        if model_state is None:
            model_state = self.initial_state(obs.shape[0], device=obs.device)

        # Encode [obs, action] -> [1, batch, hidden_dim] for GRU
        x = torch.cat([obs, action], dim=-1)
        encoded = self.encoder(x).unsqueeze(0)  # [1, batch, hidden_dim]

        # GRU step
        gru_out, new_hidden = self.gru(encoded, model_state)

        # Decode GRU output -> delta
        delta = self.decoder(gru_out.squeeze(0))  # [batch, state_dim]

        return delta, new_hidden
