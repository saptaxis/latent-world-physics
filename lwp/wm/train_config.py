"""YAML config parsing for world model training runs.

One config = one reproducible training run. The YAML has a nested structure
(data/model/training/output sections); this parser flattens it into a
dataclass with all fields directly accessible.

Follows the same patterns as wm_collection_config.py — from_dict() for
programmatic creation, load() for file-based loading.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class TrainConfig:
    """Parsed training config for a world model run.

    Flattened from the nested YAML structure. All fields needed to
    reproduce a training run are accessible as top-level attributes.

    Attributes:
        run_name: Unique identifier for this run. Used as output subdirectory name.
        mix_config_path: Path to the mix config YAML (data composition + split rules).
        profiles: Which profiles from the mix to include (e.g., ["full-variation"]).
        observation: Observation type ("full_kinematic" or "position_only").
        supervision: "blind" (physics dims zeroed) or "labeled" (full state).
        policy_holdout: Source type to hold out for eval, or None.
        architecture: Model architecture name ("context_mlp", "feedforward_mlp", etc.).
        dynamics_hidden: Hidden layer dimensions for the dynamics MLP.
        dynamics_activation: Activation function name ("relu", "gelu").
        encoder_type: Context encoder type ("mean_pool", etc.) or None.
        encoder_hidden: Hidden dims for encoder MLP, or None.
        context_k: Number of context transitions for context-conditioned models, or None.
        z_dim: Latent dimension for context encoder, or None.
        conditioning: How z is fed to dynamics ("concatenation" or "film").
        prediction_target: "absolute" (predict s_{t+1}) or "delta" (predict Δs).
        loss_weights: Optional per-dimension weight vector (length 15), or None.
        variance_normalize: If True, weight per-dim loss by 1/var(target_i).
        rollout_horizon: Number of autoregressive prediction steps (1 = single-step).
        batch_size: Training batch size.
        learning_rate: Peak learning rate for Adam.
        lr_scheduler: LR schedule type ("cosine" or "constant").
        warmup_steps: Linear warmup steps before cosine decay starts.
        max_epochs: Maximum training epochs.
        steps_per_epoch: Gradient steps per epoch (random sampling, not data passes).
        early_stopping_patience: Stop if val loss doesn't improve for N epochs.
        grad_clip: Max gradient norm for clipping.
        val_every_n_steps: Run validation every N global steps.
        checkpoint_every_n_epochs: Save periodic checkpoint every N epochs.
        checkpoint_every_n_steps: Save periodic checkpoint every N global steps.
        seed: Random seed for reproducibility.
        output_base: Root directory for training outputs.
    """

    # Identity
    run_name: str

    # Data
    mix_config_path: str
    profiles: list[str]
    observation: str
    supervision: str
    policy_holdout: str | None

    # Model
    architecture: str
    dynamics_hidden: list[int]
    dynamics_activation: str
    encoder_type: str | None
    encoder_hidden: list[int] | None
    context_k: int | None
    z_dim: int | None
    conditioning: str

    # Training
    prediction_target: str
    loss_weights: list[float] | None
    variance_normalize: bool
    input_normalize: bool
    rollout_horizon: int
    batch_size: int
    learning_rate: float
    lr_scheduler: str
    warmup_steps: int
    max_epochs: int
    steps_per_epoch: int
    early_stopping_patience: int
    grad_clip: float
    val_every_n_steps: int
    checkpoint_every_n_epochs: int
    checkpoint_every_n_steps: int
    seed: int
    device: str | None

    # Output
    output_base: str

    # Model config (new pattern for temporal architectures)
    # Path to architecture-specific config YAML. Loaded at parse time
    # and stored as a dict. The model class receives this dict — TrainConfig
    # doesn't need to know the schema.
    model_config_path: str | None = None
    model_config: dict | None = None

    # Training — temporal model fields (optional, only used by GRU/RSSM/Transformer)
    seq_len: int | None = None  # Training sequence length

    @property
    def output_dir(self) -> str:
        """Full output directory: {output_base}/{run_name}/."""
        return str(Path(self.output_base) / self.run_name)

    @classmethod
    def from_dict(cls, d: dict) -> TrainConfig:
        """Parse from a YAML-loaded dict.

        Flattens the nested data/model/training/output sections into
        a flat dataclass. Provides sensible defaults for optional fields.
        """
        data = d.get("data", {})
        model = d.get("model", {})
        training = d.get("training", {})
        output = d.get("output", {})
        encoder = model.get("encoder", {})
        dynamics = model.get("dynamics", {})

        # Model config file: load architecture-specific hyperparameters.
        # If model.config is set, load the referenced YAML file and store
        # the dict. The model class receives this — TrainConfig is just
        # a pass-through.
        model_config_path = model.get("config")
        model_config = None
        if model_config_path is not None:
            with open(model_config_path) as f:
                model_config = yaml.safe_load(f)

        return cls(
            run_name=d["run_name"],
            # Data
            mix_config_path=data.get("mix_config", ""),
            profiles=data.get("profiles", []),
            observation=data.get("observation", "full_kinematic"),
            supervision=data.get("supervision", "blind"),
            policy_holdout=data.get("policy_holdout"),
            # Model
            architecture=model.get("architecture", "context_mlp"),
            dynamics_hidden=dynamics.get("hidden_dims", [256, 256]),
            dynamics_activation=dynamics.get("activation", "relu"),
            encoder_type=encoder.get("type"),
            encoder_hidden=encoder.get("hidden_dims"),
            context_k=encoder.get("context_k"),
            z_dim=encoder.get("z_dim"),
            conditioning=model.get("conditioning", "concatenation"),
            model_config_path=model_config_path,
            model_config=model_config,
            # Training
            seq_len=training.get("seq_len"),
            prediction_target=training.get("prediction_target", "absolute"),
            loss_weights=training.get("loss_weights"),
            variance_normalize=training.get("variance_normalize", False),
            input_normalize=training.get("input_normalize", False),
            rollout_horizon=training.get("rollout_horizon", 1),
            batch_size=training.get("batch_size", 256),
            learning_rate=training.get("learning_rate", 3e-4),
            lr_scheduler=training.get("lr_scheduler", "cosine"),
            warmup_steps=training.get("warmup_steps", 1000),
            max_epochs=training.get("max_epochs", 100),
            steps_per_epoch=training.get("steps_per_epoch", 500),
            early_stopping_patience=training.get("early_stopping_patience", 10),
            grad_clip=training.get("grad_clip", 1.0),
            val_every_n_steps=training.get("val_every_n_steps", 500),
            checkpoint_every_n_epochs=training.get("checkpoint_every_n_epochs", 10),
            checkpoint_every_n_steps=training.get("checkpoint_every_n_steps", 2000),
            seed=training.get("seed", 42),
            device=training.get("device"),
            # Output
            output_base=output.get("base_dir", "."),
        )

    @classmethod
    def load(cls, path: str | Path) -> TrainConfig:
        """Load from a YAML file on disk."""
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f))

    @classmethod
    def from_flat_dict(cls, d: dict) -> TrainConfig:
        """Reconstruct from a flat dict (e.g., config.json from training runs).

        config.json is saved by train_world_model.py using dataclasses.asdict(),
        so all keys are top-level. Extra metadata keys (started_at, git_commit,
        completed_at, etc.) are silently ignored.
        """
        # Filter to only the fields that TrainConfig actually has.
        import dataclasses
        field_names = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in d.items() if k in field_names}
        # Defaults for fields added after initial release — old config.json
        # files may not have them.
        filtered.setdefault("checkpoint_every_n_steps", 2000)
        filtered.setdefault("model_config_path", None)
        filtered.setdefault("model_config", None)
        filtered.setdefault("seq_len", None)
        filtered.setdefault("input_normalize", False)
        return cls(**filtered)
