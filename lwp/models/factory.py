"""Model construction from RunConfig."""
from __future__ import annotations

from lwp.utils.config import RunConfig
from lwp.models.copy import CopyStateModel
from lwp.models.linear import LinearModel
from lwp.models.mlp import MLPModel
from lwp.models.gru import GRUModel
from lwp.models.rssm import RSSMModel


def build_model(config: RunConfig):
    """Construct a WorldModel from config."""
    if config.arch == "copy":
        return CopyStateModel(state_dim=config.state_dim)
    elif config.arch == "linear":
        return LinearModel(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
        )
    elif config.arch == "mlp":
        params = config.arch_params
        return MLPModel(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            hidden_dims=params.get("hidden_dims", [256, 256]),
            activation=params.get("activation", "relu"),
            dropout=params.get("dropout", 0.0),
        )
    elif config.arch == "gru":
        params = config.arch_params
        return GRUModel(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            hidden_dim=params.get("hidden_dim", 128),
            num_layers=params.get("num_layers", 1),
            encoder_dims=params.get("encoder_dims"),
            decoder_dims=params.get("decoder_dims"),
        )
    elif config.arch == "rssm":
        params = config.arch_params
        return RSSMModel(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            deter_dim=params.get("deter_dim", 200),
            stoch_dim=params.get("stoch_dim", 30),
            hidden_dim=params.get("hidden_dim", 200),
            encoder_dims=params.get("encoder_dims"),
        )
    else:
        raise ValueError(
            f"Unknown architecture: {config.arch}. "
            f"Available: copy, linear, mlp, gru, rssm"
        )
