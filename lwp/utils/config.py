"""RunConfig dataclass, YAML serialization, and run name generation."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class RunConfig:
    # Architecture
    arch: str                           # "linear" | "mlp" | "gru" | "rssm"
    arch_params: dict = field(default_factory=dict)

    # Prediction
    prediction: str = "delta"           # "absolute" | "delta"

    # Training
    training_mode: str = "single_step"  # "single_step" | "multi_step" | "scheduled_sampling"
    rollout_k: int = 1
    sampling_start: float = 0.0
    sampling_end: float = 0.5
    curriculum: bool = False
    kl_weight: float = 1.0              # KL weight for ELBO loss
    dim_weights: str | None = None      # "inv_var" for 1/delta_std^2, None for uniform

    # Callback frequencies
    val_every: int = 500
    patience: int = 20
    ckpt_every: int = 2000
    plot_every: int = 5000
    grad_norm_every: int = 50
    rollout_every: int = 2000
    rollout_n_rollouts: int = 10
    grad_clip: float = 1.0

    # Data
    data_mix: str = "policy"            # "policy" | "primitives" | "policy_primitives"
    data_path: str | list[str] = ""     # single path or list of paths — validated in __post_init__
    subsample: int = 1                  # take every Nth frame (1 = no subsampling)

    # Environment
    state_dim: int = 8
    action_dim: int = 2

    # Training hyperparams
    lr: float = 1e-3
    batch_size: int = 64
    epochs: int = 100
    seq_len: int = 50
    val_fraction: float = 0.1

    # Dimension names for per-dim logging (None = generic dim_0, dim_1, ...)
    dim_names: list[str] | None = None

    # Output
    run_dir: str = "runs/"
    suffix: str = ""

    def __post_init__(self):
        # Normalize data_path: single string → list of one
        if isinstance(self.data_path, str):
            if not self.data_path:
                raise ValueError("data_path is required (got empty string)")
            self.data_path = [self.data_path]
        elif isinstance(self.data_path, list):
            if not self.data_path:
                raise ValueError("data_path is required (got empty list)")
        else:
            raise TypeError(f"data_path must be str or list[str], got {type(self.data_path)}")

    def save(self, path: str | Path):
        """Save config as YAML."""
        d = dataclasses.asdict(self)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(d, f, default_flow_style=False, sort_keys=False)

    @classmethod
    def load(cls, path: str | Path) -> RunConfig:
        """Load config from YAML."""
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls(**d)


def generate_run_name(config: RunConfig) -> str:
    """Auto-generate run name from config axes.

    Format: {arch}-{prediction}-{training_mode}_k{rollout_k}-{data_mix}[--{suffix}]
    """
    name = f"{config.arch}-{config.prediction}-{config.training_mode}_k{config.rollout_k}-{config.data_mix}"
    if config.suffix:
        name += f"--{config.suffix}"
    return name


# Architectures known to implement kl_loss()
_KL_ARCHS = {"rssm"}


def validate_config(config: RunConfig) -> None:
    """Validate config consistency. Raises ValueError on problems."""
    if config.training_mode == "elbo" and config.arch not in _KL_ARCHS:
        raise ValueError(
            f"training_mode='elbo' requires a model with kl_loss() "
            f"(arch must be one of {_KL_ARCHS}, got '{config.arch}')"
        )
    if config.training_mode in ("multi_step", "scheduled_sampling", "elbo"):
        if config.rollout_k > config.seq_len:
            raise ValueError(
                f"rollout_k ({config.rollout_k}) exceeds seq_len ({config.seq_len}). "
                f"Multi-step training can only roll out within the available sequence window."
            )
    if config.dim_names is not None and len(config.dim_names) != config.state_dim:
        raise ValueError(
            f"dim_names has {len(config.dim_names)} entries but state_dim={config.state_dim}"
        )


def load_config(path: str, overrides: dict | None = None) -> RunConfig:
    """Load config from YAML with optional field overrides."""
    cfg = RunConfig.load(path)
    if overrides:
        for key, value in overrides.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
            else:
                raise ValueError(f"Unknown config field: {key}")
    return cfg
