"""Training callback system for world models.

Provides a CallbackContext (shared state across callbacks) and a TrainCallback
base class with hooks dispatched at each gradient step, epoch end, and
training start/end.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer

from lwp.data.normalization import NormStats, normalize, denormalize
from lwp.evaluation.metrics.core import per_dim_mse, rollout_error_metrics
from lwp.training.loop import validate
from lwp.utils.checkpoint import save_checkpoint
from lwp.utils.plotting import export_plots


@dataclass
class CallbackContext:
    """Shared state passed to all callbacks."""
    model: nn.Module
    optimizer: Optimizer
    writer: Any  # SummaryWriter or None
    global_step: int
    epoch: int
    run_dir: str
    device: str
    extras: dict = field(default_factory=dict)


class TrainCallback:
    """Base class for training callbacks."""

    def on_train_start(self, ctx: CallbackContext) -> None:
        pass

    def on_step(self, ctx: CallbackContext) -> bool:
        return True

    def on_epoch_end(self, ctx: CallbackContext) -> bool:
        return True

    def on_train_end(self, ctx: CallbackContext) -> None:
        pass


class ValidationCallback(TrainCallback):
    """Runs validation every N steps with early stopping and best-checkpoint saving."""

    def __init__(self, val_loader, norm_stats: NormStats,
                 training_mode: str = "single_step",
                 every_n_steps: int = 500, patience: int = 10,
                 checkpoint_dir: str | None = None,
                 rollout_k: int = 1, sampling_prob: float = 0.0,
                 kl_weight: float = 1.0,
                 dim_weights: str | None = None):
        self.val_loader = val_loader
        self.norm_stats = norm_stats
        self.training_mode = training_mode
        self.every_n_steps = every_n_steps
        self.patience = patience
        self.checkpoint_dir = checkpoint_dir
        self.rollout_k = rollout_k
        self.sampling_prob = sampling_prob
        self.kl_weight = kl_weight
        self.dim_weights = dim_weights
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.last_val_step = -1

    def on_train_start(self, ctx):
        if self.checkpoint_dir:
            Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def on_step(self, ctx):
        if ctx.global_step == 0:
            return True
        if ctx.global_step % self.every_n_steps != 0:
            return True
        if ctx.global_step == self.last_val_step:
            return True
        self.last_val_step = ctx.global_step

        # For single-step mode, compute per-dim MSE in the same forward pass
        # so PerDimLossCallback can skip its redundant data iteration
        val_metrics = validate(
            ctx.model, self.val_loader, self.norm_stats,
            training_mode=self.training_mode, rollout_k=self.rollout_k,
            device=ctx.device, sampling_prob=self.sampling_prob,
            kl_weight=self.kl_weight,
            dim_weights=self.dim_weights,
            compute_per_dim=(self.training_mode == "single_step"),
        )
        val_loss = val_metrics["val_loss"]
        ctx.extras["val_loss"] = val_loss
        if "per_dim_mse" in val_metrics:
            ctx.extras["per_dim_mse"] = val_metrics["per_dim_mse"].tolist()
            ctx.extras["_per_dim_step"] = ctx.global_step

        if ctx.writer:
            ctx.writer.add_scalar("val/loss", val_loss, ctx.global_step)

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            if self.checkpoint_dir and "config" in ctx.extras:
                save_checkpoint(
                    Path(self.checkpoint_dir) / "best.pt",
                    ctx.model, ctx.optimizer, self.norm_stats,
                    ctx.extras["config"], ctx.epoch,
                    {"val_loss": val_loss, "step": ctx.global_step},
                    global_step=ctx.global_step,
                )
        else:
            self.patience_counter += 1

        if self.patience_counter >= self.patience:
            print(f"Early stopping at step {ctx.global_step} "
                  f"(patience {self.patience} exhausted, best val_loss={self.best_val_loss:.6f})")
            return False

        return True


class CheckpointCallback(TrainCallback):
    """Saves periodic checkpoints and a latest checkpoint at each epoch end."""

    def __init__(self, checkpoint_dir: str, every_n_steps: int = 2000):
        self.checkpoint_dir = checkpoint_dir
        self.every_n_steps = every_n_steps

    def on_train_start(self, ctx):
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def _save(self, ctx, filename):
        path = Path(self.checkpoint_dir) / filename
        metrics = {}
        if "val_loss" in ctx.extras:
            metrics["val_loss"] = ctx.extras["val_loss"]
        norm_stats = ctx.extras.get("norm_stats")
        config = ctx.extras.get("config")
        if norm_stats is None or config is None:
            # Minimal save without full checkpoint infrastructure
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state_dict": ctx.model.state_dict(),
                "optimizer_state_dict": ctx.optimizer.state_dict(),
                "epoch": ctx.epoch,
                "global_step": ctx.global_step,
                "metrics": metrics,
            }, path)
        else:
            save_checkpoint(
                path, ctx.model, ctx.optimizer,
                norm_stats, config, ctx.epoch, metrics,
                global_step=ctx.global_step,
            )

    def on_step(self, ctx):
        if ctx.global_step > 0 and ctx.global_step % self.every_n_steps == 0:
            self._save(ctx, f"step_{ctx.global_step:05d}.pt")
        return True

    def on_epoch_end(self, ctx):
        self._save(ctx, "latest.pt")
        return True


class PerDimLossCallback(TrainCallback):
    """Logs per-dimension MSE at regular intervals."""

    def __init__(self, val_loader, norm_stats: NormStats,
                 every_n_steps: int = 500,
                 dim_names: list[str] | None = None):
        self.val_loader = val_loader
        self.norm_stats = norm_stats
        self.every_n_steps = every_n_steps
        self.dim_names = dim_names

    def on_step(self, ctx):
        if ctx.global_step == 0 or ctx.global_step % self.every_n_steps != 0:
            return True

        # Skip if ValidationCallback already computed per-dim MSE this step
        # (happens in single-step mode where both metrics share a forward pass)
        if ctx.extras.get("_per_dim_step") == ctx.global_step:
            pdm = ctx.extras["per_dim_mse"]
        else:
            pdm = per_dim_mse(ctx.model, self.val_loader, self.norm_stats, device=ctx.device)
            pdm = pdm.tolist()
            ctx.extras["per_dim_mse"] = pdm

        names = self.dim_names or [f"dim_{i}" for i in range(len(pdm))]
        if ctx.writer:
            for i, name in enumerate(names[:len(pdm)]):
                val = pdm[i] if isinstance(pdm[i], float) else pdm[i].item()
                ctx.writer.add_scalar(f"loss_dim/{name}", val, ctx.global_step)
        return True


class RolloutMetricsCallback(TrainCallback):
    """Computes horizon error curves at regular intervals."""

    def __init__(self, dataset, norm_stats: NormStats,
                 horizons: list[int] | None = None,
                 every_n_steps: int = 2000, n_rollouts: int = 10,
                 warmup_steps: int = 0):
        self.dataset = dataset
        self.norm_stats = norm_stats
        self.horizons = horizons or [1, 5, 10, 20, 50]
        self.every_n_steps = every_n_steps
        self.n_rollouts = n_rollouts
        self.warmup_steps = warmup_steps

    def on_step(self, ctx):
        if ctx.global_step == 0 or ctx.global_step % self.every_n_steps != 0:
            return True
        # Single rollout pass computes both endpoint and cumulative MSE
        endpoint, cumul = rollout_error_metrics(
            ctx.model, self.dataset, self.norm_stats,
            horizons=self.horizons, n_rollouts=self.n_rollouts,
            device=ctx.device, warmup_steps=self.warmup_steps,
        )
        ctx.extras["horizon_errors"] = {h: float(v.mean()) for h, v in endpoint.items()}
        ctx.extras["cumul_horizon_errors"] = {h: float(v.mean()) for h, v in cumul.items()}
        if ctx.writer:
            for h, err_tensor in endpoint.items():
                ctx.writer.add_scalar(f"rollout/mse_h{h:02d}", float(err_tensor.mean()), ctx.global_step)
            for h, err_tensor in cumul.items():
                ctx.writer.add_scalar(f"rollout/cumul_h{h:02d}", float(err_tensor.mean()), ctx.global_step)
        return True


class GradNormCallback(TrainCallback):
    """Tracks per-module gradient norms."""

    def __init__(self, every_n_steps: int = 50):
        self.every_n_steps = every_n_steps

    def on_step(self, ctx):
        if ctx.global_step == 0 or ctx.global_step % self.every_n_steps != 0:
            return True
        module_norms = {}
        for name, param in ctx.model.named_parameters():
            if param.grad is None:
                continue
            module_name = name.split(".")[0]
            if module_name not in module_norms:
                module_norms[module_name] = 0.0
            module_norms[module_name] += param.grad.data.norm(2).item() ** 2
        grad_norms = {k: v ** 0.5 for k, v in module_norms.items()}
        ctx.extras["grad_norms"] = grad_norms
        total_norm = sum(v ** 2 for v in grad_norms.values()) ** 0.5
        grad_norms["total"] = total_norm
        if ctx.writer:
            for module_name, norm in grad_norms.items():
                ctx.writer.add_scalar(f"grad/{module_name}", norm, ctx.global_step)
        return True


class PlotExportCallback(TrainCallback):
    """Periodically export TensorBoard scalars as PNG plots."""

    def __init__(self, tb_dir: str, plot_dir: str, every_n_steps: int = 5000):
        self.tb_dir = tb_dir
        self.plot_dir = plot_dir
        self.every_n_steps = every_n_steps

    def on_train_start(self, ctx):
        Path(self.plot_dir).mkdir(parents=True, exist_ok=True)

    def on_step(self, ctx):
        if ctx.global_step == 0 or ctx.global_step % self.every_n_steps != 0:
            return True
        try:
            if ctx.writer:
                ctx.writer.flush()
            export_plots(self.tb_dir, self.plot_dir)
        except Exception:
            pass
        return True

    def on_train_end(self, ctx):
        try:
            if ctx.writer:
                ctx.writer.flush()
            export_plots(self.tb_dir, self.plot_dir)
        except Exception:
            pass


class PerTimestepLossCallback(TrainCallback):
    """Log per-timestep MSE within multi-step rollouts."""

    def __init__(self, val_loader, norm_stats: NormStats,
                 every_n_steps: int = 500,
                 positions: list[int] | None = None):
        self.val_loader = val_loader
        self.norm_stats = norm_stats
        self.every_n_steps = every_n_steps
        self.positions = positions or [0, 4, 9, 24, 49]

    def on_step(self, ctx):
        if ctx.global_step == 0 or ctx.global_step % self.every_n_steps != 0:
            return True

        ctx.model.eval()
        all_sq_errors = []
        ns = self.norm_stats.to(ctx.device)

        with torch.no_grad():
            for batch in self.val_loader:
                batch = tuple(t.to(ctx.device) for t in batch)
                state_seq, action_seq = batch

                s = state_seq[:, 0]
                model_state = None
                step_errors = []

                T = action_seq.shape[1]
                for t in range(T):
                    s_n = normalize(s, ns.state_mean, ns.state_std)
                    delta_n, model_state = ctx.model.step(s_n, action_seq[:, t], model_state)
                    delta_raw = denormalize(delta_n, ns.delta_mean, ns.delta_std)
                    s = s + delta_raw

                    true_delta = state_seq[:, t + 1] - state_seq[:, t]
                    sq_err = (delta_raw - true_delta).pow(2).mean(dim=-1)
                    step_errors.append(sq_err.mean().item())

                all_sq_errors.append(step_errors)

        ctx.model.train()

        if not all_sq_errors:
            return True

        avg_errors = np.mean(all_sq_errors, axis=0)
        max_t = len(avg_errors)

        per_ts = {}
        for p in self.positions:
            if p < max_t:
                per_ts[p] = float(avg_errors[p])
                if ctx.writer:
                    ctx.writer.add_scalar(f"seq_loss/t{p + 1:02d}", avg_errors[p], ctx.global_step)

        ctx.extras["per_timestep_mse"] = per_ts
        return True


class HiddenStateHealthCallback(TrainCallback):
    """Monitor GRU/RSSM hidden state health during training."""

    def __init__(self, dataset, norm_stats: NormStats,
                 every_n_steps: int = 500, n_episodes: int = 16):
        self.dataset = dataset
        self.norm_stats = norm_stats
        self.every_n_steps = every_n_steps
        self.n_episodes = n_episodes

    def on_step(self, ctx):
        if ctx.global_step == 0 or ctx.global_step % self.every_n_steps != 0:
            return True

        model = ctx.model
        test_state = model.initial_state(1, device=ctx.device)
        if test_state is None:
            return True

        model.eval()
        ns = self.norm_stats.to(ctx.device)
        all_hidden = []

        with torch.no_grad():
            n_done = 0
            for ep_idx in range(self.dataset.n_episodes):
                if n_done >= self.n_episodes:
                    break
                states_np = self.dataset.states[ep_idx]
                actions_np = self.dataset.actions[ep_idx]
                T = len(actions_np)
                if T < 5:
                    continue

                states = torch.from_numpy(states_np).to(ctx.device)
                actions = torch.from_numpy(actions_np).to(ctx.device)

                model_state = model.initial_state(1, device=ctx.device)
                for t in range(T):
                    s_n = normalize(states[t].unsqueeze(0), ns.state_mean, ns.state_std)
                    _, model_state = model.step(s_n, actions[t].unsqueeze(0), model_state)

                    if isinstance(model_state, torch.Tensor):
                        h = model_state[-1, 0]
                    elif hasattr(model_state, 'deter'):
                        h = model_state.deter[0]
                    else:
                        break
                    all_hidden.append(h.cpu())

                n_done += 1

        model.train()

        if not all_hidden:
            return True

        hidden_matrix = torch.stack(all_hidden).numpy()

        norms = np.linalg.norm(hidden_matrix, axis=1)
        magnitude = float(np.mean(norms))
        saturation = float(np.mean(np.abs(hidden_matrix) > 0.95))

        centered = hidden_matrix - hidden_matrix.mean(axis=0)
        if centered.shape[0] > 1 and centered.shape[1] > 1:
            try:
                cov = np.cov(centered.T)
                eigvals = np.linalg.eigvalsh(cov)
                eigvals = np.maximum(eigvals[::-1], 0)
                total_var = eigvals.sum()
                if total_var > 0:
                    cumvar = np.cumsum(eigvals) / total_var
                    effective_dim = int(np.searchsorted(cumvar, 0.95) + 1)
                else:
                    effective_dim = 0
            except np.linalg.LinAlgError:
                effective_dim = 0
        else:
            effective_dim = 0

        health = {"magnitude": magnitude, "saturation": saturation, "effective_dim": effective_dim}
        ctx.extras["hidden_health"] = health

        if ctx.writer:
            ctx.writer.add_scalar("hidden/magnitude", magnitude, ctx.global_step)
            ctx.writer.add_scalar("hidden/saturation", saturation, ctx.global_step)
            ctx.writer.add_scalar("hidden/effective_dim", effective_dim, ctx.global_step)

        return True


class WarmupRolloutCallback(TrainCallback):
    """Rollout evaluation with hidden state warmup for recurrent models."""

    def __init__(self, dataset, norm_stats: NormStats,
                 warmup_steps: int = 10,
                 horizons: list[int] | None = None,
                 every_n_steps: int = 2000, n_rollouts: int = 10):
        self.dataset = dataset
        self.norm_stats = norm_stats
        self.warmup_steps = warmup_steps
        self.horizons = horizons or [1, 5, 10, 20]
        self.every_n_steps = every_n_steps
        self.n_rollouts = n_rollouts

    def on_step(self, ctx):
        if ctx.global_step == 0 or ctx.global_step % self.every_n_steps != 0:
            return True

        model = ctx.model
        test_state = model.initial_state(1, device=ctx.device)
        if test_state is None:
            return True

        model.eval()
        ns = self.norm_stats.to(ctx.device)
        max_h = max(self.horizons)
        min_ep_len = self.warmup_steps + max_h

        step_errors = {h: [] for h in self.horizons}
        n_done = 0

        with torch.no_grad():
            for ep_idx in range(self.dataset.n_episodes):
                if n_done >= self.n_rollouts:
                    break
                states_np = self.dataset.states[ep_idx]
                actions_np = self.dataset.actions[ep_idx]
                T = len(actions_np)
                if T < min_ep_len:
                    continue

                states = torch.from_numpy(states_np).to(ctx.device)
                actions = torch.from_numpy(actions_np).to(ctx.device)

                model_state = model.initial_state(1, device=ctx.device)
                for t in range(self.warmup_steps):
                    s_n = normalize(states[t].unsqueeze(0), ns.state_mean, ns.state_std)
                    _, model_state = model.step(s_n, actions[t].unsqueeze(0), model_state)

                s = states[self.warmup_steps].unsqueeze(0)
                for t in range(max_h):
                    act_idx = self.warmup_steps + t
                    if act_idx >= T:
                        break
                    s_n = normalize(s, ns.state_mean, ns.state_std)
                    delta_n, model_state = model.step(s_n, actions[act_idx].unsqueeze(0), model_state)
                    delta_raw = denormalize(delta_n, ns.delta_mean, ns.delta_std)
                    s = s + delta_raw

                    h = t + 1
                    if h in step_errors:
                        true_state = states[self.warmup_steps + h]
                        sq_err = (s[0] - true_state).pow(2).mean()
                        step_errors[h].append(sq_err.item())

                n_done += 1

        model.train()

        warmup_errors = {}
        for h in self.horizons:
            if step_errors[h]:
                mean_err = float(np.mean(step_errors[h]))
                warmup_errors[h] = mean_err
                if ctx.writer:
                    ctx.writer.add_scalar(f"warmup_rollout/mse_h{h:02d}", mean_err, ctx.global_step)

        if warmup_errors:
            ctx.extras["warmup_horizon_errors"] = warmup_errors

        return True


class NaNDetectionCallback(TrainCallback):
    """Halts training when NaN or Inf loss is detected."""

    def on_step(self, ctx):
        loss = ctx.extras.get("train_loss_step")
        if loss is not None and (math.isnan(loss) or math.isinf(loss)):
            print(f"NaN/Inf detected at step {ctx.global_step} (loss={loss}). Halting training.")
            return False
        return True


class ProgressCallback(TrainCallback):
    """Prints training progress at regular intervals."""

    def __init__(self, every_n_steps: int = 100, total_epochs: int = 0):
        self.every_n_steps = every_n_steps
        self.total_epochs = total_epochs
        self.start_time = None

    def on_train_start(self, ctx):
        self.start_time = time.time()

    def on_step(self, ctx):
        if ctx.global_step == 0 or ctx.global_step % self.every_n_steps != 0:
            return True

        train_loss = ctx.extras.get("train_loss_step", float("nan"))
        val_loss = ctx.extras.get("val_loss", float("nan"))

        elapsed = time.time() - self.start_time if self.start_time else 0
        steps_per_sec = ctx.global_step / elapsed if elapsed > 0 else 0

        epoch_str = f"epoch={ctx.epoch}"
        if self.total_epochs > 0:
            epoch_str = f"epoch={ctx.epoch}/{self.total_epochs}"

        print(f"  [{epoch_str}  step={ctx.global_step}]  "
              f"train={train_loss:.6f}  val={val_loss:.6f}  "
              f"({steps_per_sec:.1f} steps/s)")

        if ctx.writer:
            ctx.writer.add_scalar("perf/steps_per_sec", steps_per_sec, ctx.global_step)

        return True
