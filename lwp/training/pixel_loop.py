# training/pixel_loop.py
"""Training loops for pixel world models.

Follows the same callback dispatch pattern as training/loop.py so that
existing model-agnostic callbacks (CheckpointCallback, GradNormCallback,
NaNDetectionCallback, ProgressCallback, PlotExportCallback) work
without modification.
"""
from __future__ import annotations

import time

import torch

from lwp.training.pixel_losses import (
    vae_loss, latent_dynamics_loss, multi_step_latent_loss, latent_elbo_loss,
)


def pixel_vae_train_epoch(model, train_loader, optimizer, beta: float = 0.0001,
                          fg_weight: float = 1.0, state_weight: float = 0.0,
                          device: str = "cpu", max_grad_norm: float = 1.0,
                          ctx=None, callbacks=None) -> dict:
    """One VAE training epoch with callback dispatch."""
    model.train()
    # Running accumulators for epoch-level averaging — reported back to caller
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    total_state = 0.0
    n_batches = 0
    # Early-stopping flag — set by callbacks (e.g. NaN detection, patience)
    stop_requested = False

    for batch in train_loader:
        # Increment global step before forward pass so callbacks see the
        # correct step number when they fire on_step
        if ctx is not None:
            ctx.global_step += 1

        # batch is either a tensor (frames only) or tuple (frames, states).
        # The state branch supports auxiliary kinematic supervision when
        # the dataset provides ground-truth state alongside frames.
        if isinstance(batch, torch.Tensor):
            x = batch.to(device)
            state_target = None
        else:
            x = batch[0].to(device)
            state_target = batch[1].to(device) if len(batch) > 1 else None

        # Forward pass — model returns reconstruction, posterior params,
        # and optional state prediction from the latent
        recon, mu, logvar, state_pred = model(x)
        # Composite loss: weighted recon + beta*KL + state_weight*state_MSE
        loss, recon_loss, kl_loss, state_loss = vae_loss(
            recon, x, mu, logvar, beta, fg_weight=fg_weight,
            state_pred=state_pred, state_target=state_target,
            state_weight=state_weight)

        optimizer.zero_grad()
        loss.backward()
        # Clip gradients before step — VAE losses can spike early in training
        # when the KL term collapses or fg_weight amplifies pixel errors
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        if ctx is not None and callbacks:
            # Expose step-level losses in ctx.extras so callbacks can read them
            # (e.g. NaNDetectionCallback checks train_loss_step for NaN)
            ctx.extras["train_loss_step"] = loss.item()
            ctx.extras["recon_loss_step"] = recon_loss.item()
            ctx.extras["kl_loss_step"] = kl_loss.item()
            ctx.extras["state_loss_step"] = state_loss.item()
            if ctx.writer:
                # Log all loss components every step for fine-grained TensorBoard curves
                ctx.writer.add_scalar("train/loss", loss.item(), ctx.global_step)
                ctx.writer.add_scalar("train/recon_loss", recon_loss.item(), ctx.global_step)
                ctx.writer.add_scalar("train/kl_loss", kl_loss.item(), ctx.global_step)
                # State loss only logged when active — avoids cluttering TB with zeros
                if state_weight > 0:
                    ctx.writer.add_scalar("train/state_loss", state_loss.item(), ctx.global_step)
            # Dispatch callbacks after gradient computation but before optimizer.step()
            # — this lets GradNormCallback inspect pre-step gradient norms
            for cb in callbacks:
                if cb.on_step(ctx) is False:
                    stop_requested = True
                    break

        # Step after callbacks so gradient-inspecting callbacks see the raw grads
        optimizer.step()

        # Accumulate for epoch-level averages reported to caller
        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()
        total_state += state_loss.item()
        n_batches += 1

        if stop_requested:
            break

    # max(n_batches, 1) prevents division by zero if loader was empty
    return {
        "train_loss": total_loss / max(n_batches, 1),
        "recon_loss": total_recon / max(n_batches, 1),
        "kl_loss": total_kl / max(n_batches, 1),
        "state_loss": total_state / max(n_batches, 1),
        "stop_requested": stop_requested,
    }


def pixel_dynamics_train_epoch(model_dynamics, vae, train_loader, optimizer,
                               sampling_prob: float = 0.0,
                               device: str = "cpu", max_grad_norm: float = 1.0,
                               ctx=None, callbacks=None,
                               training_mode: str = "latent_mse",
                               rollout_k: int = 1,
                               kl_weight: float = 1.0,
                               ms_weight: float = 1.0,
                               free_bits: float = 0.0,
                               kin_weight: float = 1.0,
                               kin_dims: int = 6) -> dict:
    """One dynamics training epoch with frozen VAE and callback dispatch.

    Supports three training modes via loss dispatch:
    - latent_mse: teacher-forced predict_sequence + MSE (original path)
    - multi_step_latent: autoregressive rollout with full gradient flow
    - latent_elbo: posterior-guided RSSM training with KL divergence

    Args:
        model_dynamics: dynamics model (LatentDynamicsModel or LatentRSSM)
        vae: frozen VAE encoder for mapping frames -> latent codes
        train_loader: yields (frames, actions) batches
        optimizer: optimizer for dynamics parameters
        sampling_prob: scheduled sampling probability (latent_mse only)
        device: torch device string
        max_grad_norm: gradient clipping threshold
        ctx: CallbackContext for logging and step tracking
        callbacks: list of TrainCallback instances
        training_mode: loss dispatch key — "latent_mse", "multi_step_latent",
            or "latent_elbo"
        rollout_k: rollout horizon for multi_step_latent and latent_elbo modes
        kl_weight: KL divergence weight for latent_elbo mode
        ms_weight: scalar multiplier for multi_step_latent loss
    """
    # VAE stays frozen throughout — only dynamics params are updated
    model_dynamics.train()
    total_loss = 0.0
    n_batches = 0
    stop_requested = False

    for frames_batch, actions_batch in train_loader:
        if ctx is not None:
            ctx.global_step += 1

        # Flatten frames for batch VAE encoding: (B, T, C, H, W) -> (B*T, C, H, W)
        B, T, C, H, W = frames_batch.shape
        frames_flat = frames_batch.reshape(B * T, C, H, W).to(device)
        actions = actions_batch.to(device)

        # Encode all frames with frozen VAE — no gradients through encoder
        with torch.no_grad():
            z_all = vae.encode(frames_flat)
        latent_dim = z_all.shape[-1]
        # Reshape back to sequence: (B, T, latent_dim)
        z_seq = z_all.reshape(B, T, latent_dim)

        # --- Loss dispatch based on training_mode ---
        if training_mode == "latent_mse":
            # Original path: teacher-forced predict_sequence + MSE
            # predict_sequence handles T actions for T frames (legacy convention)
            z_pred, _ = model_dynamics.predict_sequence(
                z_seq, actions, teacher_forcing=1.0 - sampling_prob)
            # Weighted MSE on latent dims — same logic as multi_step_latent.
            # Upweights first kin_dims dimensions when kin_weight > 1, so
            # the dynamics model prioritizes getting kinematics right.
            if kin_weight != 1.0 and kin_dims > 0:
                latent_dim = z_seq.size(-1)
                dim_weights = torch.ones(latent_dim, device=z_seq.device)
                dim_weights[:kin_dims] = kin_weight
                # Normalize so mean weight = 1 — keeps loss magnitude comparable
                dim_weights = dim_weights / dim_weights.mean()
                sq_err = (z_pred[:, :-1] - z_seq[:, 1:]).pow(2)
                loss = (sq_err * dim_weights).mean()
            else:
                loss = latent_dynamics_loss(z_pred[:, :-1], z_seq[:, 1:])

        elif training_mode == "multi_step_latent":
            # Autoregressive rollout with full gradient flow through k steps.
            # NOTE: For RSSM, this trains the prior directly via rollout()
            # (which calls imagine_step(), no posterior). This skips the
            # RSSM's posterior/KL machinery entirely — useful as a
            # prior-only ablation. Use latent_elbo for full RSSM training.
            # sampling_prob = P(use own prediction). For multi-step:
            # 1.0 (default after warmup) = pure autoregressive with full
            # gradient flow. Lower values mix in GT z at some steps,
            # acting as a gradient chain curriculum (Bengio et al. 2015).
            # teacher_forcing = P(use GT) = 1 - sampling_prob.
            loss = multi_step_latent_loss(
                model_dynamics, z_seq, actions, k=rollout_k,
                teacher_forcing=1.0 - sampling_prob,
                kin_weight=kin_weight, kin_dims=kin_dims) * ms_weight

        elif training_mode == "latent_elbo":
            # Full RSSM ELBO: posterior-guided step + KL(posterior || prior)
            # Returns breakdown for detailed logging of recon vs KL components
            loss, recon_loss, kl_loss = latent_elbo_loss(
                model_dynamics, z_seq, actions, k=rollout_k,
                kl_weight=kl_weight, free_bits=free_bits,
                return_breakdown=True)

        else:
            raise ValueError(
                f"Unknown training_mode={training_mode!r}. "
                f"Expected one of: latent_mse, multi_step_latent, latent_elbo")

        optimizer.zero_grad()
        loss.backward()
        # Clip gradients — multi-step rollouts can produce large gradients
        # because errors compound across the autoregressive chain
        torch.nn.utils.clip_grad_norm_(model_dynamics.parameters(), max_grad_norm)

        if ctx is not None and callbacks:
            # Store step-level loss for callbacks to inspect
            # (NaNDetectionCallback reads train_loss_step for early abort)
            ctx.extras["train_loss_step"] = loss.item()

            # ELBO mode: store recon/KL breakdown so RSSMDiagnosticCallback
            # can log per-step KL without recomputing it
            if training_mode == "latent_elbo":
                ctx.extras["kl_loss"] = kl_loss.item()
                ctx.extras["recon_loss"] = recon_loss.item()

            if ctx.writer:
                ctx.writer.add_scalar("train/loss", loss.item(), ctx.global_step)
                # ELBO mode: log recon + KL breakdown to TensorBoard —
                # crucial for diagnosing posterior collapse (KL -> 0)
                # or KL domination (recon stays high)
                if training_mode == "latent_elbo":
                    ctx.writer.add_scalar(
                        "train/recon_loss", recon_loss.item(), ctx.global_step)
                    ctx.writer.add_scalar(
                        "train/kl_loss", kl_loss.item(), ctx.global_step)

            # Dispatch callbacks after gradient computation but before step —
            # same convention as VAE loop for GradNormCallback consistency
            for cb in callbacks:
                if cb.on_step(ctx) is False:
                    stop_requested = True
                    break

        # Step after callbacks so gradient-inspecting callbacks see raw grads
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        if stop_requested:
            break

    return {
        "train_loss": total_loss / max(n_batches, 1),
        "stop_requested": stop_requested,
    }
