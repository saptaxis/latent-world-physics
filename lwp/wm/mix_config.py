"""Mix config for world model training data.

Defines which collected episode directories to include, how to split
them into train/val/test, and optional holdout rules (OOD physics
corner, policy-type holdout).

The mix config is a pointer-based composition — it references existing
collection directories. No data is copied.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class MixConfig:
    """Parsed mix configuration for world model training data.

    Attributes:
        profiles: List of dicts with 'name' and 'path' keys. Each profile
            is a directory of collection subdirs (e.g., full-variation/).
        data_base: Base directory for resolving relative profile paths.
        split_method: Splitting strategy ('quantile_grid').
        split_axes: Physics param names to bin on for quantile grid.
        bins: Number of quantile bins per axis.
        train_ratio: Fraction of regions assigned to train.
        val_ratio: Fraction of regions assigned to val.
            Remaining (1 - train - val) goes to in-distribution test.
        ood_holdout: Optional dict mapping param names to (lo, hi) ranges.
            Episodes matching ALL conditions go to OOD test set.
        policy_holdout: Optional source type name to hold out entirely
            for policy-generalization evaluation.
    """
    profiles: list[dict]
    data_base: str
    split_method: str = "quantile_grid"
    split_axes: list[str] = field(default_factory=lambda: ["gravity", "main_engine_power", "lander_density"])
    bins: int = 3
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    ood_holdout: dict[str, tuple[float, float]] | None = None
    policy_holdout: str | None = None

    @classmethod
    def from_dict(cls, d: dict) -> MixConfig:
        split = d.get("split", {})
        ood = split.get("ood_holdout")
        if ood is not None:
            ood = {k: tuple(v) for k, v in ood.items()}

        return cls(
            profiles=d["profiles"],
            data_base=d.get("data_base", "."),
            split_method=split.get("method", "quantile_grid"),
            split_axes=split.get("axes", ["gravity", "main_engine_power", "lander_density"]),
            bins=split.get("bins", 3),
            train_ratio=split.get("train_ratio", 0.8),
            val_ratio=split.get("val_ratio", 0.1),
            ood_holdout=ood,
            policy_holdout=split.get("policy_holdout"),
        )

    @classmethod
    def load(cls, path: str | Path) -> MixConfig:
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f))
