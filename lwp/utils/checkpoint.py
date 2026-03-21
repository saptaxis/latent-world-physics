"""Checkpoint save/load for world model training."""
from __future__ import annotations

import dataclasses
import subprocess
from pathlib import Path

import torch

from lwp.data.normalization import NormStats
from lwp.utils.config import RunConfig


def get_git_hash() -> str | None:
    """Return current git commit hash, or None if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def save_checkpoint(path, model, optimizer, norm_stats: NormStats,
                    config: RunConfig, epoch: int, metrics: dict | None = None,
                    global_step: int | None = None):
    """Save everything needed to resume training or run evaluation."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "norm_stats": norm_stats.to_dict(),
        "config": dataclasses.asdict(config),
        "epoch": epoch,
        "metrics": metrics or {},
    }
    if global_step is not None:
        ckpt["global_step"] = global_step
    git_hash = get_git_hash()
    if git_hash:
        ckpt["git_hash"] = git_hash
    torch.save(ckpt, path)


def load_checkpoint(path, device="cpu") -> dict:
    """Load checkpoint. Returns dict with all saved components.

    Returns:
        dict with keys: model_state_dict, optimizer_state_dict,
        norm_stats (dict), config (RunConfig), epoch, metrics
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    ckpt["config"] = RunConfig(**ckpt["config"])
    return ckpt
