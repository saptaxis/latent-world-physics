# training/pixel_callbacks.py
"""Pixel-specific training callbacks.

Reuses the CallbackContext and TrainCallback base from training/callbacks.py.
Model-agnostic callbacks (CheckpointCallback, GradNormCallback,
NaNDetectionCallback, ProgressCallback, PlotExportCallback) work directly
with pixel training loops -- no changes needed.

These callbacks handle pixel-specific concerns: VAE validation with
reconstruction+KL loss, reconstruction grid logging, dream sequence logging,
kinematics validation, dream comparison video export, and RSSM diagnostics.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

from lwp.training.callbacks import TrainCallback, CallbackContext
from lwp.training.pixel_losses import (
    vae_loss, compositional_vae_loss, latent_dynamics_loss, multi_step_latent_loss, latent_elbo_loss,
)


class PixelVAEValidationCallback(TrainCallback):
    """VAE validation with early stopping and best-checkpoint saving."""

    def __init__(self, val_loader, beta: float = 0.0001,
                 fg_weight: float = 1.0, state_weight: float = 0.0,
                 every_n_steps: int = 500, patience: int = 10,
                 checkpoint_dir: str | None = None):
        self.val_loader = val_loader
        # Loss hyperparams must match training loop so val loss is comparable
        self.beta = beta
        self.fg_weight = fg_weight
        self.state_weight = state_weight
        self.every_n_steps = every_n_steps
        # patience: number of val checks without improvement before stopping
        self.patience = patience
        self.checkpoint_dir = checkpoint_dir
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        # Guard against double-firing when a step aligns with multiple triggers
        self.last_val_step = -1

    def on_train_start(self, ctx: CallbackContext):
        if self.checkpoint_dir:
            Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def on_step(self, ctx: CallbackContext) -> bool:
        # Skip step 0 — model hasn't seen any data yet
        if ctx.global_step == 0:
            return True
        if ctx.global_step % self.every_n_steps != 0:
            return True
        # Prevent double-firing if multiple callbacks trigger at the same step
        if ctx.global_step == self.last_val_step:
            return True
        self.last_val_step = ctx.global_step

        # Switch to eval mode for correct BatchNorm/Dropout behavior
        ctx.model.eval()
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        total_state = 0.0
        n = 0
        with torch.no_grad():
            for batch in self.val_loader:
                # Handle both frames-only and (frames, states) datasets
                if isinstance(batch, torch.Tensor):
                    x = batch.to(ctx.device)
                    state_target = None
                else:
                    x = batch[0].to(ctx.device)
                    state_target = batch[1].to(ctx.device) if len(batch) > 1 else None
                if hasattr(ctx.model, 'latent_mode'):
                    recon, mu, logvar, pose_pred = ctx.model(x, gt_state=state_target)
                    decomposed = ctx.model.get_last_decomposed()
                    from lwp.models.coord_utils import physics_to_grid, angle_to_sincos
                    gt_tx, gt_ty = physics_to_grid(state_target[:, 0], state_target[:, 1])
                    gt_sin, gt_cos = angle_to_sincos(state_target[:, 2])
                    gt_pose = torch.stack([gt_tx, gt_ty, gt_sin, gt_cos], dim=-1).to(ctx.device)
                    loss_dict = compositional_vae_loss(
                        recon, x, mu, logvar, pose_pred, gt_pose,
                        decomposed['A_hat'],
                        beta=self.beta, fg_weight=self.fg_weight,
                        pose_weight=self.state_weight,
                        split_kl=getattr(ctx.model, 'latent_mode', 'flat') == 'split',
                        bg_dim=getattr(ctx.model, 'bg_dim', 8),
                    )
                    loss = loss_dict['total']
                    recon_l = loss_dict['recon']
                    kl_l = loss_dict.get('kl', loss_dict.get('kl_bg', torch.tensor(0.0)))
                    state_l = loss_dict['pose']
                else:
                    recon, mu, logvar, state_pred = ctx.model(x)
                    # Use same loss function and hyperparams as training
                    # so val loss is directly comparable to train loss
                    loss, recon_l, kl_l, state_l = vae_loss(
                        recon, x, mu, logvar, self.beta,
                        fg_weight=self.fg_weight,
                        state_pred=state_pred, state_target=state_target,
                        state_weight=self.state_weight)
                total_loss += loss.item()
                total_recon += recon_l.item()
                total_kl += kl_l.item()
                total_state += state_l.item()
                n += 1
        # Restore training mode for subsequent forward passes
        ctx.model.train()

        val_loss = total_loss / max(n, 1)
        # Expose val_loss in extras so other callbacks can read it
        ctx.extras["val_loss"] = val_loss

        # Step LR scheduler immediately when val_loss is computed — not
        # at epoch level. lr_patience=N means N val checks, not N epochs.
        scheduler = ctx.extras.get("scheduler")
        if scheduler is not None:
            scheduler.step(val_loss)

        if ctx.writer:
            # Log all loss components to TensorBoard for train/val comparison
            ctx.writer.add_scalar("val/loss", val_loss, ctx.global_step)
            ctx.writer.add_scalar("val/recon_loss", total_recon / max(n, 1), ctx.global_step)
            ctx.writer.add_scalar("val/kl_loss", total_kl / max(n, 1), ctx.global_step)
            # Only log state loss when active — avoids cluttering TB with zeros
            if self.state_weight > 0:
                ctx.writer.add_scalar("val/state_loss", total_state / max(n, 1), ctx.global_step)

        if val_loss < self.best_val_loss:
            # New best — save checkpoint and reset patience counter
            self.best_val_loss = val_loss
            self.patience_counter = 0
            if self.checkpoint_dir:
                # Save full training state for resumption, plus config for reproducibility
                torch.save({
                    "model_state_dict": ctx.model.state_dict(),
                    "optimizer_state_dict": ctx.optimizer.state_dict(),
                    "epoch": ctx.epoch,
                    "global_step": ctx.global_step,
                    "val_loss": val_loss,
                    "config": ctx.extras.get("config", {}),
                }, Path(self.checkpoint_dir) / "best.pt")
        else:
            self.patience_counter += 1

        # Early stopping — return False to signal the training loop to break
        if self.patience_counter >= self.patience:
            print(f"Early stopping at step {ctx.global_step} "
                  f"(patience {self.patience}, best={self.best_val_loss:.6f})")
            return False

        return True


class ReconGridCallback(TrainCallback):
    """Log reconstruction grid (original vs reconstructed) to TensorBoard."""

    def __init__(self, val_loader, every_n_steps: int = 500):
        self.val_loader = val_loader
        self.every_n_steps = every_n_steps

    def on_step(self, ctx: CallbackContext) -> bool:
        if ctx.global_step == 0 or ctx.global_step % self.every_n_steps != 0:
            return True
        if ctx.writer is None:
            return True

        # Sample random frames from val dataset — random sampling each firing
        # shows different frames for qualitative diversity across training
        dataset = self.val_loader.dataset
        indices = torch.randperm(len(dataset))[:8]
        frames = [dataset[i] for i in indices]
        # Handle both frames-only and (frames, states) datasets
        if isinstance(frames[0], tuple):
            frames = [f[0] for f in frames]
        x = torch.stack(frames).to(ctx.device)

        ctx.model.eval()
        with torch.no_grad():
            recon, _, _, _ = ctx.model(x)
            # Interleave original and reconstruction: [orig1, recon1, orig2, recon2, ...]
            # nrow=2 puts each orig-recon pair side by side in the grid
            pairs = torch.stack([x, recon], dim=1).reshape(-1, *x.shape[1:])
            try:
                from torchvision.utils import make_grid
                grid = make_grid(pairs, nrow=2, normalize=False)
                ctx.writer.add_image("vae/recon_grid", grid, ctx.global_step)
            except ImportError:
                # torchvision not installed — silently skip grid logging
                pass
        # Compositional model: show decomposition grid
        if hasattr(ctx.model, 'latent_mode'):
            with torch.no_grad():
                z = ctx.model.encode(x)
                decomp = ctx.model.decode_decomposed(z)

            import torch.nn.functional as F_cb
            # Upscale canonical patches to frame_size for grid alignment
            fs = ctx.model.frame_size
            O_up = F_cb.interpolate(decomp['O_hat'], size=fs, mode='nearest')
            A_up = F_cb.interpolate(decomp['A_hat'], size=fs, mode='nearest')

            # Build grid: rows = samples, cols = GT | Recon | O_hat | A_hat | B_hat
            n = min(8, x.size(0))
            rows = []
            for i in range(n):
                rows.append(torch.cat([
                    x[i],                    # GT
                    decomp['x_hat'][i],      # composited reconstruction
                    O_up[i],                 # canonical lander (upscaled)
                    A_up[i],                 # alpha mask (upscaled)
                    decomp['B_hat'][i],      # background
                ], dim=-1))  # concat along width
            grid = torch.cat(rows, dim=-2)  # concat along height
            ctx.writer.add_image("vae/decomposition_grid", grid, ctx.global_step)

        ctx.model.train()
        return True


class PixelDynamicsValidationCallback(TrainCallback):
    """Dynamics validation with early stopping.

    Supports three training modes that mirror the training loop's loss
    dispatch, so validation loss matches training loss semantics:
    - latent_mse: single-step predict_sequence + MSE (original GRU default)
    - multi_step_latent: k-step autoregressive rollout + MSE (GRU or RSSM)
    - latent_elbo: ELBO = reconstruction MSE + KL (RSSM only)
    """

    def __init__(self, val_loader, vae: nn.Module,
                 every_n_steps: int = 500, patience: int = 10,
                 checkpoint_dir: str | None = None,
                 training_mode: str = "latent_mse",
                 rollout_k: int = 1,
                 kl_weight: float = 1.0,
                 sampling_prob: float = 0.0):
        self.val_loader = val_loader
        self.vae = vae
        self.every_n_steps = every_n_steps
        self.patience = patience
        self.checkpoint_dir = checkpoint_dir
        # Loss dispatch params — must match training loop config so
        # val loss is comparable to train loss
        self.training_mode = training_mode
        self.rollout_k = rollout_k
        self.kl_weight = kl_weight
        self.sampling_prob = sampling_prob
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.last_val_step = -1

    def on_train_start(self, ctx):
        if self.checkpoint_dir:
            Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def _compute_loss(self, ctx, z_seq: torch.Tensor,
                      actions: torch.Tensor) -> torch.Tensor:
        """Dispatch to correct loss function based on training_mode.

        Mirrors the training loop's loss dispatch so validation loss
        is directly comparable to training loss.
        """
        if self.training_mode == "multi_step_latent":
            # k-step autoregressive rollout — works for both GRU and RSSM
            return multi_step_latent_loss(ctx.model, z_seq, actions,
                                          k=self.rollout_k)
        elif self.training_mode == "latent_elbo":
            # ELBO loss — RSSM only (needs step() and kl_loss())
            return latent_elbo_loss(ctx.model, z_seq, actions,
                                    k=self.rollout_k,
                                    kl_weight=self.kl_weight)
        else:
            # Default: single-step predict_sequence + MSE (original behavior)
            z_pred, _ = ctx.model.predict_sequence(
                z_seq, actions, teacher_forcing=self.sampling_prob)
            return latent_dynamics_loss(z_pred[:, :-1], z_seq[:, 1:])

    def on_step(self, ctx) -> bool:
        # Skip step 0 — no data seen yet
        if ctx.global_step == 0:
            return True
        if ctx.global_step % self.every_n_steps != 0:
            return True
        # Prevent double-firing within the same step
        if ctx.global_step == self.last_val_step:
            return True
        self.last_val_step = ctx.global_step

        ctx.model.eval()
        total_loss = 0.0
        n = 0
        with torch.no_grad():
            for frames_batch, actions_batch in self.val_loader:
                # Flatten frames for batch VAE encoding — same reshape as training loop
                B, T, C, H, W = frames_batch.shape
                frames_flat = frames_batch.reshape(B * T, C, H, W).to(ctx.device)
                actions = actions_batch.to(ctx.device)
                # Encode with frozen VAE — must match training loop encoding
                z_all = self.vae.encode(frames_flat)
                latent_dim = z_all.shape[-1]
                z_seq = z_all.reshape(B, T, latent_dim)
                # Dispatch to correct loss function matching training loop
                loss = self._compute_loss(ctx, z_seq, actions)
                total_loss += loss.item()
                n += 1
        ctx.model.train()

        val_loss = total_loss / max(n, 1)
        ctx.extras["val_loss"] = val_loss

        # Step LR scheduler immediately when val_loss is computed
        scheduler = ctx.extras.get("scheduler")
        if scheduler is not None:
            scheduler.step(val_loss)

        if ctx.writer:
            ctx.writer.add_scalar("val/loss", val_loss, ctx.global_step)

        if val_loss < self.best_val_loss:
            # New best — save checkpoint and reset patience
            self.best_val_loss = val_loss
            self.patience_counter = 0
            if self.checkpoint_dir:
                # Save full training state plus metadata needed to reload:
                # vae_checkpoint path so we know which VAE to pair with,
                # model_type so loader knows GRU vs RSSM
                torch.save({
                    "model_state_dict": ctx.model.state_dict(),
                    "optimizer_state_dict": ctx.optimizer.state_dict(),
                    "epoch": ctx.epoch,
                    "global_step": ctx.global_step,
                    "val_loss": val_loss,
                    "config": ctx.extras.get("config", {}),
                    "vae_checkpoint": ctx.extras.get("vae_checkpoint", ""),
                    # Track model architecture so checkpoint loader knows
                    # which dynamics class to instantiate
                    "model_type": ctx.extras.get("model_type", "gru"),
                }, Path(self.checkpoint_dir) / "best.pt")
        else:
            self.patience_counter += 1

        # Early stopping — return False to signal training loop to break
        if self.patience_counter >= self.patience:
            print(f"Early stopping at step {ctx.global_step} "
                  f"(patience {self.patience}, best={self.best_val_loss:.6f})")
            return False

        return True


class DreamGridCallback(TrainCallback):
    """Log GT|Dream side-by-side grids to TensorBoard during dynamics training.

    Each firing samples random episodes from val_dataset, encodes GT frames,
    rolls out dreamed latents via model.rollout(), decodes both, and shows
    GT (top) vs Dream (bottom) for each episode. Random sampling means each
    firing shows different episodes for qualitative diversity.
    """

    def __init__(self, vae: nn.Module, val_dataset,
                 n_episodes: int = 4, every_n_steps: int = 1000):
        self.vae = vae
        # Store dataset (not loader) so we can sample random episodes each firing
        self.val_dataset = val_dataset
        self.n_episodes = n_episodes
        self.every_n_steps = every_n_steps

    def on_step(self, ctx) -> bool:
        if ctx.global_step == 0 or ctx.global_step % self.every_n_steps != 0:
            return True
        if ctx.writer is None:
            return True

        ctx.model.eval()
        with torch.no_grad():
            # Sample random episodes each firing for diversity
            indices = torch.randperm(len(self.val_dataset))[:self.n_episodes]
            episodes = [self.val_dataset[i] for i in indices]

            rows = []
            for frames_ep, actions_ep in episodes:
                # frames_ep: (T, C, H, W), actions_ep: (T-1, action_dim)
                frames = frames_ep.to(ctx.device)
                actions = actions_ep.to(ctx.device).unsqueeze(0)  # (1, T-1, A)
                T = frames.size(0)

                # Encode all GT frames to latent space
                z_gt = self.vae.encode(frames)  # (T, latent_dim)

                # Dream: rollout from first frame's latent using model.rollout()
                # rollout() works for both GRU and RSSM — returns (1, T, D)
                z_dream, _ = ctx.model.rollout(z_gt[0:1], actions)
                z_dream = z_dream.squeeze(0)  # (T, latent_dim)

                # Pick evenly-spaced frames to show (up to 6)
                show_indices = list(
                    range(0, T, max(1, T // 6))
                )[:6]

                # Decode GT and dream latents at selected timesteps
                gt_frames = self.vae.decode(z_gt[show_indices])
                dream_frames = self.vae.decode(z_dream[show_indices])

                # Stack GT on top, dream on bottom for this episode
                rows.append(gt_frames)
                rows.append(dream_frames)

            try:
                from torchvision.utils import make_grid
                # Each row has len(show_indices) frames; nrow controls columns
                n_cols = len(show_indices)
                grid = make_grid(torch.cat(rows, dim=0), nrow=n_cols,
                                 normalize=False)
                ctx.writer.add_image("dynamics/dream_grid", grid, ctx.global_step)
            except ImportError:
                pass
        ctx.model.train()
        return True


# ---------------------------------------------------------------------------
# Helpers for loading raw npz episodes
# ---------------------------------------------------------------------------

def _load_npz_episode(
    path: str,
    frame_size: int = 64,
) -> dict:
    """Load a raw npz episode file and preprocess for callback use.

    Returns a dict with tensors ready for VAE consumption:
      - frames: (T+1, 1, frame_size, frame_size) float32 [0,1]
      - actions: (T, 2) float32
      - states: (T+1, state_dim) float32
      - source_label: str describing the episode source (e.g. "heuristic")
    """
    data = np.load(path, allow_pickle=True)

    # --- Preprocess frames to match VAE training: grayscale, 64x64, [0,1] ---
    raw_frames = data["rgb_frames"]  # (T+1, H, W, 3) uint8
    processed = []
    for frame in raw_frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (frame_size, frame_size),
                             interpolation=cv2.INTER_AREA)
        # Shape: (1, H, W) float32 normalised to [0,1]
        tensor = torch.from_numpy(resized).float().unsqueeze(0) / 255.0
        processed.append(tensor)
    frames = torch.stack(processed)  # (T+1, 1, frame_size, frame_size)

    actions = torch.from_numpy(data["actions"].astype(np.float32))
    states = torch.from_numpy(data["states"].astype(np.float32))

    # --- Parse source metadata for file naming ---
    source_label = "unknown"
    try:
        if "metadata_json" in data:
            meta = json.loads(str(data["metadata_json"]))
            source_type = meta.get("source_type", "unknown")
            if source_type == "primitive":
                maneuver = meta.get("maneuver_type", "unknown")
                source_label = f"primitive_{maneuver}"
            else:
                source_label = source_type
    except (json.JSONDecodeError, KeyError, TypeError):
        # Gracefully handle missing or malformed metadata
        pass

    return {
        "frames": frames,
        "actions": actions,
        "states": states,
        "source_label": source_label,
    }


# ---------------------------------------------------------------------------
# KinematicsValidationCallback
# ---------------------------------------------------------------------------

# Canonical kinematic dimension names for the first 6 state dims
_KINEMATICS_DIMS = ["x", "y", "vx", "vy", "angle", "ang_vel"]


class KinematicsValidationCallback(TrainCallback):
    """Validate dreamed kinematics against ground-truth states.

    Loads raw npz episodes at init, encodes GT frames via the VAE, rolls
    out dreamed latents via the dynamics model, predicts kinematic state
    from the dreamed latents using the VAE's state head, and logs per-dim
    MSE at specified horizons.

    No-ops silently when the VAE has no state head (state_dim == 0) --
    this lets the callback be unconditionally wired into the training
    script without conditional logic.
    """

    def __init__(
        self,
        vae: nn.Module,
        episode_paths: list[str],
        every_n_steps: int = 2000,
        horizons: list[int] | None = None,
        frame_size: int = 64,
    ):
        self.vae = vae
        self.every_n_steps = every_n_steps
        self.horizons = horizons or [1, 5, 10]
        self.frame_size = frame_size

        # Early exit marker -- checked in on_step to skip all work
        self.active = getattr(vae, "state_dim", 0) > 0

        # Pre-load episodes so on_step is fast (no I/O)
        self.episodes: list[dict] = []
        if self.active:
            for p in episode_paths:
                self.episodes.append(_load_npz_episode(p, frame_size))

    def on_step(self, ctx: CallbackContext) -> bool:
        # No-op when VAE has no state head — allows unconditional wiring
        if not self.active:
            return True
        if ctx.global_step == 0 or ctx.global_step % self.every_n_steps != 0:
            return True
        if ctx.writer is None:
            return True

        device = ctx.device
        self.vae.eval()
        ctx.model.eval()

        with torch.no_grad():
            # Accumulate per-dim squared errors at each horizon across episodes.
            # Separate lists per horizon because not all episodes are long enough
            # for all horizons — we only include valid ones.
            errors: dict[int, list[torch.Tensor]] = {h: [] for h in self.horizons}

            for ep in self.episodes:
                frames = ep["frames"].to(device)   # (T+1, 1, H, W)
                actions = ep["actions"].to(device)  # (T, 2)
                gt_states = ep["states"].to(device)  # (T+1, state_dim)
                # Only use the first 6 kinematic dims: x, y, vx, vy, angle, ang_vel
                gt_kin = gt_states[:, :6]

                # Encode all GT frames to latent space
                z_gt = self.vae.encode(frames)  # (T+1, latent_dim)

                # Dream: rollout from z_0 using actions — tests whether the
                # dynamics model preserves physical state information over time
                z_dream, _ = ctx.model.rollout(
                    z_gt[0:1], actions.unsqueeze(0)
                )  # (1, T+1, latent_dim)
                z_dream = z_dream.squeeze(0)  # (T+1, latent_dim)

                # Predict kinematic state from dreamed latents — uses the VAE's
                # auxiliary state head to decode z back to physical quantities
                pred_kin = self.vae.predict_state(z_dream)  # (T+1, state_dim)
                if pred_kin is None:
                    continue
                pred_kin = pred_kin[:, :6]

                T_actions = actions.shape[0]
                for h in self.horizons:
                    if h <= T_actions:
                        # Per-dim squared error at horizon h — measures how
                        # accurately the dream preserves each kinematic quantity
                        err = (pred_kin[h] - gt_kin[h]).pow(2)  # (6,)
                        errors[h].append(err)

            # Log mean per-dim MSE at each horizon — e.g. kinematics/mse_h5_x
            # tracks x-position error 5 steps into a dream. This reveals which
            # kinematic dimensions degrade fastest during rollout.
            for h in self.horizons:
                if not errors[h]:
                    continue
                mean_err = torch.stack(errors[h]).mean(dim=0)  # (6,)
                for i, dim_name in enumerate(_KINEMATICS_DIMS):
                    if i < mean_err.shape[0]:
                        ctx.writer.add_scalar(
                            f"kinematics/mse_h{h}_{dim_name}",
                            mean_err[i].item(),
                            ctx.global_step,
                        )

        ctx.model.train()
        return True


# ---------------------------------------------------------------------------
# DreamComparisonVideoCallback
# ---------------------------------------------------------------------------


class DreamComparisonVideoCallback(TrainCallback):
    """Export GT|Dream side-by-side MP4 videos for qualitative evaluation.

    Loads diverse npz episodes at init (with metadata for naming).
    At each firing, dreams each episode, creates a side-by-side
    comparison frame array, and saves as MP4 to the video directory.
    """

    def __init__(
        self,
        vae: nn.Module,
        episode_paths: list[str],
        video_dir: str,
        every_n_steps: int = 5000,
        fps: int = 10,
        frame_size: int = 64,
    ):
        self.vae = vae
        self.video_dir = video_dir
        self.every_n_steps = every_n_steps
        self.fps = fps
        self.frame_size = frame_size

        # Pre-load episodes with source labels for file naming
        self.episodes: list[dict] = []
        for p in episode_paths:
            self.episodes.append(_load_npz_episode(p, frame_size))

    def on_step(self, ctx: CallbackContext) -> bool:
        if ctx.global_step == 0 or ctx.global_step % self.every_n_steps != 0:
            return True

        device = ctx.device
        self.vae.eval()
        ctx.model.eval()

        # Organise videos by training step for easy comparison across training
        step_dir = Path(self.video_dir) / f"step_{ctx.global_step}"
        step_dir.mkdir(parents=True, exist_ok=True)

        with torch.no_grad():
            for ep_idx, ep in enumerate(self.episodes):
                frames = ep["frames"].to(device)
                actions = ep["actions"].to(device)
                # source_label (e.g. "heuristic", "primitive_hover") used in
                # the filename so you can tell which episode type each video shows
                label = ep["source_label"]

                # Encode GT frames to latent space
                z_gt = self.vae.encode(frames)

                # Dream from z_0 — full autoregressive rollout using only actions.
                # This is the prior-only path, showing what the model predicts
                # without any future observations.
                z_dream, _ = ctx.model.rollout(
                    z_gt[0:1], actions.unsqueeze(0)
                )
                z_dream = z_dream.squeeze(0)

                # Decode both GT latents and dreamed latents to pixel space
                # for visual comparison — GT shows what VAE reconstruction
                # looks like, dream shows accumulated prediction error
                gt_decoded = self.vae.decode(z_gt)    # (T+1, 1, H, W)
                dream_decoded = self.vae.decode(z_dream)  # (T+1, 1, H, W)

                # Convert to uint8 numpy for video export
                gt_np = (gt_decoded.squeeze(1).cpu().numpy() * 255).clip(
                    0, 255
                ).astype(np.uint8)  # (T+1, H, W)
                dream_np = (dream_decoded.squeeze(1).cpu().numpy() * 255).clip(
                    0, 255
                ).astype(np.uint8)

                # Side-by-side: GT on left, Dream on right — makes divergence
                # between dream and reality immediately visible
                T = gt_np.shape[0]
                combined = np.concatenate(
                    [gt_np[:T], dream_np[:T]], axis=2
                )  # (T, H, 2*W)

                # Save as MP4 — filename includes source label for identification
                video_path = str(step_dir / f"{label}_{ep_idx}.mp4")
                self._save_mp4(combined, video_path)

        ctx.model.train()
        return True

    def _save_mp4(self, frames: np.ndarray, path: str):
        """Save grayscale frames as MP4 by converting to RGB."""
        import imageio
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        # Grayscale (T, H, W) -> RGB (T, H, W, 3) for codec compatibility
        frames_rgb = np.stack([frames] * 3, axis=-1)
        writer = imageio.get_writer(
            path, fps=self.fps, codec="libx264", pixelformat="yuv420p"
        )
        for frame in frames_rgb:
            writer.append_data(frame)
        writer.close()


# ---------------------------------------------------------------------------
# RSSMDiagnosticCallback
# ---------------------------------------------------------------------------


class RSSMDiagnosticCallback(TrainCallback):
    """RSSM-specific diagnostics: KL tracking and prior/posterior MSE.

    Only active when the dynamics model is a LatentRSSM instance.
    Silently no-ops for GRU dynamics, so it can be unconditionally
    wired into the callback list.

    Per-step (cheap -- reads from ctx.extras already computed by training loop):
      - rssm/kl_per_step: mean KL divergence
      - rssm/prior_post_divergence: L2 between prior and posterior means

    Per-validation (every_n_steps -- requires forward passes):
      - rssm/prior_mse: MSE of prior-only dreamed latents vs GT
      - rssm/posterior_mse: MSE of posterior-stepped latents vs GT
    """

    def __init__(
        self,
        dynamics: nn.Module,
        episode_paths: list[str],
        vae: nn.Module,
        every_n_steps: int = 2000,
        frame_size: int = 64,
    ):
        from lwp.models.pixel_rssm import LatentRSSM
        # Type-check at init — silently no-ops for GRU dynamics so this
        # callback can be unconditionally wired into the callback list
        self.is_rssm = isinstance(dynamics, LatentRSSM)
        self.every_n_steps = every_n_steps
        self.vae = vae
        self.frame_size = frame_size

        # Pre-load episodes at init so on_step has no I/O — only loaded
        # for RSSM since GRU skips all diagnostic work
        self.episodes: list[dict] = []
        if self.is_rssm:
            for p in episode_paths:
                self.episodes.append(_load_npz_episode(p, frame_size))

    def on_step(self, ctx: CallbackContext) -> bool:
        # No-op for GRU dynamics — allows unconditional wiring
        if not self.is_rssm:
            return True

        # --- Per-step cheap metrics from ctx.extras ---
        # These are computed by the training loop already, so reading them
        # here adds zero compute cost — just TensorBoard logging.
        if ctx.writer and ctx.global_step > 0:
            # KL divergence per step — tracks posterior collapse (KL -> 0 means
            # the posterior isn't using observations) and KL explosion
            kl = ctx.extras.get("kl_loss")
            if kl is not None:
                val = kl if isinstance(kl, float) else kl.item()
                ctx.writer.add_scalar("rssm/kl_per_step", val, ctx.global_step)

            # L2 between prior and posterior means — measures how much the
            # posterior corrects the prior. If near zero, the posterior isn't
            # learning anything from observations (collapsed); if very large,
            # prior and posterior disagree strongly (dreaming will be poor).
            pp_div = ctx.extras.get("prior_post_div")
            if pp_div is not None:
                val = pp_div if isinstance(pp_div, float) else pp_div.item()
                ctx.writer.add_scalar(
                    "rssm/prior_post_divergence", val, ctx.global_step
                )

        # --- Periodic validation-time diagnostics ---
        # These require forward passes and are more expensive, so they run
        # at a lower frequency (every_n_steps, typically 2000)
        if ctx.global_step == 0 or ctx.global_step % self.every_n_steps != 0:
            return True
        if ctx.writer is None:
            return True

        device = ctx.device
        self.vae.eval()
        ctx.model.eval()

        prior_mse_sum = 0.0
        posterior_mse_sum = 0.0
        n_episodes = 0

        with torch.no_grad():
            for ep in self.episodes:
                frames = ep["frames"].to(device)
                actions = ep["actions"].to(device)
                T = actions.shape[0]
                # Skip very short episodes — need at least 2 transitions
                # for meaningful prior/posterior comparison
                if T < 2:
                    continue

                # Encode all GT frames
                z_gt = self.vae.encode(frames)  # (T+1, latent_dim)

                # --- Prior MSE: dream with rollout() (prior only) ---
                # rollout() uses imagine_step() internally — no observations.
                # This is the quality the model achieves at test time.
                z_prior, _ = ctx.model.rollout(
                    z_gt[0:1], actions.unsqueeze(0)
                )  # (1, T+1, latent_dim)
                z_prior = z_prior.squeeze(0)  # (T+1, latent_dim)
                prior_mse = (z_prior[1:] - z_gt[1:]).pow(2).mean().item()
                prior_mse_sum += prior_mse

                # --- Posterior MSE: step through with model.step() ---
                # step() uses the posterior — it observes each GT z_t before
                # predicting z_{t+1}. This should always be lower than prior MSE;
                # if the gap is small, the posterior isn't helping much.
                state = None
                post_preds = []
                for t in range(T):
                    z_next, state = ctx.model.step(
                        z_gt[t].unsqueeze(0),
                        actions[t].unsqueeze(0),
                        state,
                    )
                    post_preds.append(z_next.squeeze(0))
                z_post = torch.stack(post_preds)  # (T, latent_dim)
                posterior_mse = (z_post - z_gt[1:]).pow(2).mean().item()
                posterior_mse_sum += posterior_mse

                n_episodes += 1

        ctx.model.train()

        if n_episodes > 0 and ctx.writer:
            # prior_mse: dream quality (test-time performance)
            ctx.writer.add_scalar(
                "rssm/prior_mse",
                prior_mse_sum / n_episodes,
                ctx.global_step,
            )
            # posterior_mse: observation-guided prediction quality (training upper bound)
            # Gap between prior and posterior MSE indicates room for improvement
            ctx.writer.add_scalar(
                "rssm/posterior_mse",
                posterior_mse_sum / n_episodes,
                ctx.global_step,
            )

        return True


class CompositionalDiagnosticsCallback(TrainCallback):
    """Log decomposition diagnostics for compositional VAE every val pass.

    Metrics: pose R² per dim, angle MAE, scale stats, mask area stats.
    """

    def __init__(self, val_loader, log_every_n_steps=500):
        self.val_loader = val_loader
        self.log_every_n_steps = log_every_n_steps

    def on_step(self, ctx):
        if ctx.global_step % self.log_every_n_steps != 0:
            return True
        if not hasattr(ctx.model, 'latent_mode'):
            return True  # not compositional
        if ctx.writer is None:
            return True

        ctx.model.eval()
        all_pose_pred, all_gt_pose, all_mask_area, all_scale = [], [], [], []

        from lwp.models.coord_utils import physics_to_grid, angle_to_sincos

        with torch.no_grad():
            for batch in self.val_loader:
                if isinstance(batch, torch.Tensor):
                    continue  # need states
                x, state = batch[0].to(ctx.device), batch[1].to(ctx.device)
                z = ctx.model.encode(x)
                decomp = ctx.model.decode_decomposed(z)
                pose = decomp['pose_params']  # (B, 5)

                # GT pose
                gt_tx, gt_ty = physics_to_grid(state[:, 0], state[:, 1])
                gt_sin, gt_cos = angle_to_sincos(state[:, 2])
                gt_pose = torch.stack([gt_tx, gt_ty, gt_sin, gt_cos], dim=-1)

                all_pose_pred.append(pose[:, :4].cpu())
                all_gt_pose.append(gt_pose.cpu())
                all_scale.append(pose[:, 4].cpu())
                all_mask_area.append(decomp['A_hat'].mean(dim=[1, 2, 3]).cpu())

        ctx.model.train()

        if not all_pose_pred:
            return True

        pose_pred = torch.cat(all_pose_pred)
        gt_pose = torch.cat(all_gt_pose)
        scales = torch.cat(all_scale)
        mask_areas = torch.cat(all_mask_area)

        # Per-dim pose R²
        dim_names = ['tx', 'ty', 'sin', 'cos']
        for i, name in enumerate(dim_names):
            pred, gt = pose_pred[:, i], gt_pose[:, i]
            ss_res = ((pred - gt) ** 2).sum()
            ss_tot = ((gt - gt.mean()) ** 2).sum()
            r2 = 1 - ss_res / (ss_tot + 1e-8)
            ctx.writer.add_scalar(f"val/decomp/pose_r2_{name}", r2.item(), ctx.global_step)
            ctx.writer.add_scalar(f"val/decomp/pose_mae_{name}", (pred - gt).abs().mean().item(), ctx.global_step)

        # Angle MAE in degrees
        pred_angle = torch.atan2(pose_pred[:, 2], pose_pred[:, 3])
        gt_angle = torch.atan2(gt_pose[:, 2], gt_pose[:, 3])
        angle_mae_deg = ((pred_angle - gt_angle).abs() * 180 / 3.14159).mean()
        ctx.writer.add_scalar("val/decomp/angle_mae_deg", angle_mae_deg.item(), ctx.global_step)

        # Scale stats
        ctx.writer.add_scalar("val/decomp/scale_mean", scales.mean().item(), ctx.global_step)
        ctx.writer.add_scalar("val/decomp/scale_std", scales.std().item(), ctx.global_step)

        # Mask area stats
        ctx.writer.add_scalar("val/decomp/mask_area_mean", mask_areas.mean().item(), ctx.global_step)
        ctx.writer.add_scalar("val/decomp/mask_area_std", mask_areas.std().item(), ctx.global_step)

        return True
