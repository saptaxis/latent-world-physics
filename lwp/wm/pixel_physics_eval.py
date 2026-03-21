"""Pixel world model physics evaluation.

4-layer evaluation pipeline:
  Layer 1: Perception baseline (state head fidelity, VAE reconstruction)
  Layer 2: Oracle physics extraction (1-step re-grounded)
  Layer 3: Rollout kinematics + pixel metrics (full dream)
  Layer 4: Action conditioning (sensitivity + ablation)

This module mirrors the state-space physics_understanding.py but operates
on pixel observations via a trained PixelVAE + dynamics model. The key
question: does the pixel world model learn the same physics that the
state-space model does, purely from visual inputs?

Spec: traitful-docs/.../specs/pixel-wm-physics-eval.md
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# Reuse filter functions and ConstantResult from state-space physics
# understanding — the physics tests themselves are identical, only the
# source of predictions differs (pixel encoder vs state-space model).
from lwp.wm.physics_understanding import (
    passes_general_filter, passes_gravity_filter, passes_main_thrust_filter,
    passes_side_thrust_filter, passes_kinematic_filter,
    passes_angle_thrust_filter, passes_angular_damping_filter,
    ConstantResult, X, Y, VX, VY, ANGLE, ANGULAR_VEL,
)

# Canonical kinematic dimension names for the first 6 state dims.
# These match the LunarLander observation layout: (x, y, vx, vy, angle, ang_vel).
_KIN_DIMS = ["x", "y", "vx", "vy", "angle", "ang_vel"]


def load_eval_episodes(
    paths: list[str],
    frame_size: int = 84,
) -> list[dict]:
    """Load raw npz episode files and preprocess for evaluation.

    For each npz file:
    - rgb_frames: resize to frame_size, convert to grayscale, normalize to
      [0,1] float32 tensor (1, H, W) per frame
    - states: keep as float32 tensor (T+1, state_dim)
    - actions: keep as float32 tensor (T, action_dim)

    Why grayscale? The PixelVAE is trained on single-channel grayscale frames
    (in_channels=1). Converting here ensures eval uses the same format.

    Why resize? Episodes are collected at varying resolutions (e.g., 150x100)
    but the VAE expects a fixed square input (e.g., 64x64 or 84x84).

    Returns list of dicts with keys: frames, states, actions.
    """
    episodes = []
    for path in paths:
        data = np.load(path, allow_pickle=True)
        raw_frames = data["rgb_frames"]  # (T+1, H_orig, W_orig, 3) uint8

        # Preprocess each frame: RGB -> grayscale -> resize -> normalize
        processed = []
        for frame in raw_frames:
            # Convert RGB to grayscale — reduces 3 channels to 1, matching
            # the VAE's in_channels=1 training configuration
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            # Resize to square frame_size — INTER_AREA is best for downscaling
            # (anti-aliased averaging), avoids aliasing artifacts that would
            # add noise to the latent encoding
            resized = cv2.resize(gray, (frame_size, frame_size),
                                 interpolation=cv2.INTER_AREA)
            # Shape: (1, H, W) float32, normalized to [0,1] — unsqueeze adds
            # the channel dim, /255.0 maps uint8 [0,255] to float [0,1]
            # matching the sigmoid output range of the VAE decoder
            tensor = torch.from_numpy(resized).float().unsqueeze(0) / 255.0
            processed.append(tensor)

        # Infer policy/source label from parent directory name.
        # e.g., "/data/heuristic/episode_00042.npz" → policy="heuristic"
        # e.g., "/data/free-fall/episode_00001.npz" → policy="free-fall"
        # If the parent is a temp dir or root, label as "unknown"
        parent_name = Path(path).parent.name
        policy = parent_name if parent_name not in ("", ".") else "unknown"

        episodes.append({
            "frames": torch.stack(processed),  # (T+1, 1, H, W)
            "states": torch.from_numpy(data["states"].astype(np.float32)),
            "actions": torch.from_numpy(data["actions"].astype(np.float32)),
            "policy": policy,
        })
    return episodes


@torch.no_grad()
def evaluate_state_head_fidelity(
    vae,
    episodes: list[dict],
    device: str = "cpu",
) -> dict:
    """Layer 1a: State head fidelity — how well does z encode kinematics?

    Encodes every GT frame through the VAE encoder, then predicts kinematics
    via the auxiliary state head. Compares predictions to GT states. No
    dynamics model involved — this is purely measuring perception quality:
    can the encoder extract physical state from pixels?

    This is the most basic check: if the state head can't recover kinematics
    from z, then the latent space doesn't encode physics at all, and higher
    layers (dynamics accuracy, rollout consistency) are meaningless.

    Returns:
        per_dim_mse: list of 6 floats (x, y, vx, vy, angle, ang_vel)
        per_dim_r2: list of 6 floats (R² per dim)
        dim_names: ["x", "y", "vx", "vy", "angle", "ang_vel"]
        n_frames: total number of frames evaluated
    """
    # Guard: if the VAE was trained without a state head, we can't evaluate
    # state fidelity — this is a configuration error, not a runtime error
    if getattr(vae, "state_dim", 0) == 0:
        raise ValueError(
            "VAE has state_dim=0 (no state head). "
            "Pixel physics eval requires state_dim=6."
        )

    vae = vae.to(device)
    vae.eval()

    all_pred = []
    all_gt = []

    for ep in episodes:
        frames = ep["frames"].to(device)  # (T+1, 1, H, W)
        # Ground truth: first 6 dims are the kinematic state
        # (x, y, vx, vy, angle, angular_vel). Remaining dims (leg contacts)
        # are not kinematic and not predicted by the state head.
        gt_kin = ep["states"][:, :6].numpy()  # (T+1, 6)

        # Encode all GT frames to latent space — at eval, encode() returns
        # the mean (mu) deterministically (no sampling noise)
        z = vae.encode(frames)  # (T+1, latent_dim)
        # Predict kinematics from z via the auxiliary state head
        pred_kin = vae.predict_state(z).cpu().numpy()  # (T+1, 6)

        all_pred.append(pred_kin)
        all_gt.append(gt_kin)

    # Concatenate across episodes for aggregate metrics — treating all frames
    # as independent samples (they're not truly independent due to temporal
    # correlation, but for MSE/R² this is standard practice)
    pred = np.concatenate(all_pred, axis=0)  # (N, 6)
    gt = np.concatenate(all_gt, axis=0)      # (N, 6)

    # Per-dimension MSE: measures average squared prediction error for each
    # kinematic variable independently
    per_dim_mse = ((pred - gt) ** 2).mean(axis=0).tolist()

    # Per-dimension R²: coefficient of determination. R² = 1 means perfect
    # prediction; R² = 0 means no better than predicting the mean; R² < 0
    # means worse than the mean predictor (a strong signal of failure).
    # Using max(ss_tot, 1e-8) to avoid division by zero when a dimension
    # has constant ground truth (which shouldn't happen in practice but
    # could in degenerate test episodes).
    per_dim_r2 = []
    for d in range(6):
        ss_res = ((pred[:, d] - gt[:, d]) ** 2).sum()
        ss_tot = ((gt[:, d] - gt[:, d].mean()) ** 2).sum()
        r2 = 1.0 - ss_res / max(ss_tot, 1e-8)
        per_dim_r2.append(float(r2))

    return {
        "per_dim_mse": per_dim_mse,
        "per_dim_r2": per_dim_r2,
        "dim_names": _KIN_DIMS,
        "n_frames": len(pred),
    }


def _compute_ssim_map(pred: torch.Tensor, target: torch.Tensor,
                      window_size: int = 11) -> torch.Tensor:
    """Compute per-pixel SSIM map without averaging.

    Same computation as wm-ladder's _ssim_single but returns the full
    (B, C, H, W) map instead of a scalar. This is needed for fg-only SSIM:
    we compute the full SSIM map, then average only over foreground pixels
    rather than all pixels.

    Uses a box filter (uniform kernel) for local statistics — simpler than
    the Gaussian window in Wang et al. 2004, but consistent with the
    wm-ladder implementation and fast for eval purposes.

    Args:
        pred: (B, C, H, W) predicted frames in [0, 1]
        target: (B, C, H, W) target frames in [0, 1]
        window_size: side length of the local averaging window
    Returns:
        (B, C, H, W) per-pixel SSIM values in [-1, 1]
    """
    # SSIM constants from Wang et al. 2004 — stabilize division
    # when local means or variances are near zero
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    channels = pred.size(1)
    # Box filter: uniform kernel, applied per-channel via groups=channels.
    # Each channel gets its own 1x1xWxW kernel of value 1/W^2.
    kernel = torch.ones(channels, 1, window_size, window_size,
                        device=pred.device) / (window_size * window_size)
    pad = window_size // 2

    # Local means via convolution with the uniform kernel
    mu_p = F.conv2d(pred, kernel, padding=pad, groups=channels)
    mu_t = F.conv2d(target, kernel, padding=pad, groups=channels)
    mu_p_sq = mu_p * mu_p
    mu_t_sq = mu_t * mu_t
    mu_pt = mu_p * mu_t

    # Local variances: E[X^2] - E[X]^2 (biased, matches wm-ladder)
    sigma_p_sq = F.conv2d(pred * pred, kernel, padding=pad, groups=channels) - mu_p_sq
    sigma_t_sq = F.conv2d(target * target, kernel, padding=pad, groups=channels) - mu_t_sq
    # Cross-covariance: E[XY] - E[X]E[Y]
    sigma_pt = F.conv2d(pred * target, kernel, padding=pad, groups=channels) - mu_pt

    # SSIM formula: luminance * contrast terms
    ssim_map = ((2 * mu_pt + C1) * (2 * sigma_pt + C2)) / \
               ((mu_p_sq + mu_t_sq + C1) * (sigma_p_sq + sigma_t_sq + C2))
    return ssim_map


@torch.no_grad()
def evaluate_vae_reconstruction(
    vae,
    episodes: list[dict],
    device: str = "cpu",
) -> dict:
    """Layer 1b: VAE reconstruction quality.

    Encode-decode every GT frame, compute pixel MSE and SSIM. This measures
    how much visual information survives the bottleneck — a prerequisite for
    the dynamics model to have useful inputs. If reconstruction is poor, the
    latent space is losing critical visual features.

    Uses wm-ladder's pixel_metrics module for consistent metric computation
    across the project.

    Returns:
        pixel_mse: mean pixel MSE across all frames (lower = better)
        ssim: mean SSIM across all frames (higher = better, max 1.0)
        n_frames: total number of frames evaluated
    """
    # Import from wm-ladder evaluation module — these are the canonical
    # pixel metric implementations shared across all visual WM experiments
    from lwp.evaluation.pixel_metrics import pixel_mse, compute_ssim
    # Foreground weight mask from wm-ladder training losses — same mask used
    # during VAE training to upweight lander pixels, now reused for eval
    from lwp.training.pixel_losses import _foreground_weight_mask

    vae = vae.to(device)
    vae.eval()

    total_mse = 0.0
    total_ssim = 0.0
    total_fg_mse = 0.0
    total_fg_ssim = 0.0
    n_frames = 0

    for ep in episodes:
        frames = ep["frames"].to(device)  # (T+1, 1, H, W)
        # Encode-decode round trip: frame -> z -> reconstruction
        # At eval, encode returns deterministic mu (no sampling noise)
        z = vae.encode(frames)
        recon = vae.decode(z)
        # Accumulate weighted metrics — multiply by frame count so we can
        # compute a proper weighted average across variable-length episodes
        total_mse += pixel_mse(recon, frames).item() * len(frames)
        total_ssim += compute_ssim(recon, frames) * len(frames)

        # ---- Fg-weighted MSE ----
        # Standard pixel MSE is dominated by sky (74%) and terrain (23%).
        # The lander is only ~3% of pixels. By upweighting foreground pixels
        # by 50x (matching VAE training), we make the metric sensitive to the
        # visually important lander/flames/legs region.
        fg_weight = 50.0
        weights = _foreground_weight_mask(frames, fg_weight)  # (T+1, 1, H, W)
        # Weighted MSE: weight tensor multiplies per-pixel squared error,
        # so foreground pixel errors count 50x more than background
        fg_mse = (weights * (recon - frames).pow(2)).mean().item()
        total_fg_mse += fg_mse * len(frames)

        # ---- Foreground-only SSIM ----
        # Compute the full SSIM map, then average only over foreground pixels.
        # DO NOT zero out background first — that creates artificial agreement
        # (both zero) that inflates the score. Instead, compute SSIM everywhere
        # but only average where foreground pixels exist.
        ssim_map = _compute_ssim_map(recon, frames)  # (T+1, 1, H, W)
        # Same intensity band as _foreground_weight_mask: sky < 0.04, terrain > 0.78
        fg_mask = (frames > 0.04) & (frames < 0.78)
        fg_pixels = fg_mask.float()
        # Average SSIM only where foreground pixels exist — if none exist
        # in a frame, that frame contributes nothing (safe with max denominator)
        fg_ssim_val = (ssim_map * fg_pixels).sum() / max(fg_pixels.sum().item(), 1.0)
        total_fg_ssim += fg_ssim_val.item() * len(frames)

        n_frames += len(frames)

    return {
        "pixel_mse": total_mse / max(n_frames, 1),
        "ssim": total_ssim / max(n_frames, 1),
        "fg_pixel_mse": total_fg_mse / max(n_frames, 1),
        "fg_ssim": total_fg_ssim / max(n_frames, 1),
        "n_frames": n_frames,
    }


@torch.no_grad()
def evaluate_latent_dynamics_accuracy(
    vae,
    dynamics,
    episodes: list[dict],
    device: str = "cpu",
) -> dict:
    """Layer 2 preamble: latent dynamics accuracy WITHOUT state head.

    For each transition: encode GT frame_t → z_t, encode GT frame_{t+1} → z_{t+1}_gt,
    predict z_{t+1}_pred = dynamics(z_t, action_t, hidden), MSE(z_pred, z_gt).

    Teacher-forced: hidden state maintained across episode, fed encoded GT z each step.
    Handles both GRU (Tensor hidden) and RSSM (RSSMState) via duck typing.
    """
    vae = vae.to(device)
    dynamics = dynamics.to(device)
    vae.eval()
    dynamics.eval()

    total_mse = 0.0
    n_transitions = 0

    for ep in episodes:
        frames = ep["frames"].to(device)
        actions = ep["actions"].to(device)
        T = actions.shape[0]

        # Encode ALL GT frames upfront — batch is more efficient than per-step
        z_all = vae.encode(frames)  # (T+1, latent_dim)

        # Teacher-forced rollout: maintain hidden state, feed encoded GT z each step
        state = None
        for t in range(T):
            z_pred, state = dynamics.forward(z_all[t].unsqueeze(0),
                                              actions[t].unsqueeze(0), state)
            # Compare to encoded GT frame at t+1
            z_gt = z_all[t + 1].unsqueeze(0)
            total_mse += F.mse_loss(z_pred, z_gt).item()
            n_transitions += 1

    # Compute z variance from all encoded GT frames — measures how much
    # the latent space "spreads out" for this VAE. Normalizing MSE by
    # this variance makes the metric comparable across VAEs with different
    # latent distributions (e.g., different KL weights, different data).
    all_z_gt = []
    for ep in episodes:
        frames = ep["frames"].to(device)
        z_all = vae.encode(frames)
        all_z_gt.append(z_all.cpu())
    z_cat = torch.cat(all_z_gt, dim=0)  # (N, latent_dim)
    z_variance = float(z_cat.var().item())

    raw_mse = total_mse / max(n_transitions, 1)

    return {
        "latent_mse": raw_mse,
        "latent_mse_normalized": raw_mse / max(z_variance, 1e-8),
        "z_variance": z_variance,
        "n_transitions": n_transitions,
    }


@torch.no_grad()
def _predict_pixel_oracle_deltas(
    vae, dynamics, episode: dict, device: str,
) -> list[dict]:
    """Run teacher-forced oracle for one episode, return per-transition data.

    At each step:
    1. Encode GT frame_t → z_t (already in z_all from batch encode)
    2. Predict z_{t+1} via dynamics with running hidden state
    3. predict_state on both z_t and z_{t+1}_pred → delta_est
    4. Compute GT delta from episode states

    Returns list of dicts, one per transition, with:
        delta_est: (6,) numpy — predicted kinematic delta from state head
        gt_delta: (6,) numpy — actual kinematic delta from GT states
        gt_state: (6,) numpy — GT state at time t (for filtering)
        action: (2,) numpy — action at time t (for filtering)
        timestep: int
    """
    frames = episode["frames"].to(device)
    actions = episode["actions"].to(device)
    gt_states = episode["states"][:, :6].numpy()  # (T+1, 6)
    T = actions.shape[0]

    # Batch-encode all GT frames — more efficient than per-step encoding
    z_all = vae.encode(frames)  # (T+1, latent_dim)

    transitions = []
    state = None

    for t in range(T):
        # One-step dynamics prediction with teacher-forced hidden state
        # "Teacher-forced" = input is encoded GT z_t, not predicted z from prev step
        z_pred, state = dynamics.forward(
            z_all[t].unsqueeze(0), actions[t].unsqueeze(0), state)

        # State head predictions on current GT-encoded and dynamics-predicted latents
        # These give us the kinematic interpretation of each latent vector
        state_t_est = vae.predict_state(z_all[t].unsqueeze(0)).cpu().numpy()[0]  # (6,)
        state_next_est = vae.predict_state(z_pred).cpu().numpy()[0]              # (6,)
        # Delta = change in estimated kinematics from dynamics prediction
        delta_est = state_next_est - state_t_est

        # GT delta from recorded states — the "oracle" ground truth
        gt_delta = gt_states[t + 1] - gt_states[t]

        transitions.append({
            "delta_est": delta_est,
            "gt_delta": gt_delta,
            "gt_state": gt_states[t].copy(),
            "action": actions[t].cpu().numpy(),
            "timestep": t,
        })

    return transitions


def _extract_constants_from_transitions(all_transitions: list[dict]) -> dict:
    """Shared extraction logic for both oracle and rollout modes.

    Takes a list of transition dicts (each with delta_est, gt_delta, gt_state,
    action, timestep) and extracts all 6 physics constants with dependency
    ordering.

    This is factored out so oracle (GT-state filtering) and rollout
    (dreamed-state filtering) can share the same extraction pipeline —
    the only difference is how transitions are collected and what "gt_state"
    represents in each mode.
    """
    # --- Helper to extract one constant ---
    def _extract(filter_fn, measurement_fn, transitions):
        """Apply filter, compute measurement, return ConstantResult as dict."""
        model_vals, gt_vals, assoc_states, timesteps = [], [], [], []
        for tr in transitions:
            if not passes_general_filter(tr["gt_state"], tr["timestep"]):
                continue
            if not filter_fn(tr):
                continue
            m, g = measurement_fn(tr)
            if m is not None and g is not None:
                model_vals.append(m)
                gt_vals.append(g)
                assoc_states.append(tr["gt_state"].copy())
                timesteps.append(tr["timestep"])
        result = ConstantResult(
            model_values=np.array(model_vals, dtype=np.float32),
            gt_values=np.array(gt_vals, dtype=np.float32),
            associated_states=np.array(assoc_states, dtype=np.float32).reshape(-1, 6),
            associated_timesteps=np.array(timesteps, dtype=np.int32),
        )
        result_dict = result.to_dict() | {
            "model_values": result.model_values,
            "gt_values": result.gt_values,
            "associated_states": result.associated_states,
        }

        # Sign correctness: does the model's mean have the same sign as GT?
        # A model with 244% relative error but correct sign knows the DIRECTION
        # of the physical effect. Wrong sign means fundamentally wrong physics.
        sign_correct = False
        if result_dict["n_samples"] > 0:
            m = result_dict["model_mean"]
            g = result_dict["gt_mean"]
            # Both non-zero and same sign
            sign_correct = bool((m * g > 0)) if (abs(m) > 1e-8 and abs(g) > 1e-8) else False
        result_dict["sign_correct"] = sign_correct

        # Low-sample warning: flag constants with fewer than 100 qualifying
        # transitions. These are unreliable — the filter conditions rarely
        # match, so the extracted constant has high variance.
        result_dict["low_sample_warning"] = result_dict["n_samples"] < 100

        return result_dict

    # Step 1: Gravity — measured from free-fall (no engines, upright)
    # Gravity is the baseline vertical acceleration, independent of other constants
    gravity = _extract(
        filter_fn=lambda tr: passes_gravity_filter(tr["gt_state"], tr["action"]),
        measurement_fn=lambda tr: (float(tr["delta_est"][VY]), float(tr["gt_delta"][VY])),
        transitions=all_transitions,
    )
    # Store gravity estimate for use in dependent constants
    gravity_model = gravity["model_mean"] if gravity["n_samples"] > 0 else 0.0
    gravity_gt = gravity["gt_mean"] if gravity["n_samples"] > 0 else 0.0

    # Step 2: Main thrust — subtract gravity to isolate thrust contribution
    # dvy_thrust = dvy_total - gravity, measured when main engine is firing
    main_thrust = _extract(
        filter_fn=lambda tr: passes_main_thrust_filter(tr["gt_state"], tr["action"]),
        measurement_fn=lambda tr: (
            float(tr["delta_est"][VY]) - gravity_model,
            float(tr["gt_delta"][VY]) - gravity_gt,
        ),
        transitions=all_transitions,
    )

    # Step 2b: Angle-thrust coupling — ratio of dvx to (dvy - gravity)
    # Measures whether thrust direction follows angle: dvx/dvy_thrust ≈ -tan(angle)
    def _angle_thrust_measurement(tr):
        dvy_thrust = tr["delta_est"][VY] - gravity_model
        if abs(dvy_thrust) < 0.01:
            return None, None
        model_ratio = float(tr["delta_est"][VX] / dvy_thrust)
        gt_ratio = float(-np.tan(tr["gt_state"][ANGLE]))
        return model_ratio, gt_ratio

    angle_thrust = _extract(
        filter_fn=lambda tr: passes_angle_thrust_filter(
            tr["gt_state"], tr["action"], gravity_model),
        measurement_fn=_angle_thrust_measurement,
        transitions=all_transitions,
    )

    # Step 3: Independent constants (no dependencies on gravity or each other)

    # Side thrust — horizontal acceleration from side engines
    side_thrust = _extract(
        filter_fn=lambda tr: passes_side_thrust_filter(tr["gt_state"], tr["action"]),
        measurement_fn=lambda tr: (float(tr["delta_est"][VX]), float(tr["gt_delta"][VX])),
        transitions=all_transitions,
    )

    # Kinematics — position/velocity consistency: dx/vx should be constant (= dt)
    kinematics = _extract(
        filter_fn=lambda tr: passes_kinematic_filter(tr["gt_state"]),
        measurement_fn=lambda tr: (
            float(tr["delta_est"][X] / tr["gt_state"][VX]),
            float(tr["gt_delta"][X] / tr["gt_state"][VX]),
        ),
        transitions=all_transitions,
    )

    # Angular damping — rate of angular velocity decay without engines
    # Measured as 1 - (next_avel / current_avel), should be constant
    def _damping_measurement(tr):
        current_avel = tr["gt_state"][ANGULAR_VEL]
        if abs(current_avel) < 1e-6:
            return None, None
        model_next_avel = current_avel + tr["delta_est"][ANGULAR_VEL]
        gt_next_avel = current_avel + tr["gt_delta"][ANGULAR_VEL]
        return float(1.0 - model_next_avel / current_avel), \
               float(1.0 - gt_next_avel / current_avel)

    angular_damping = _extract(
        filter_fn=lambda tr: passes_angular_damping_filter(tr["gt_state"], tr["action"]),
        measurement_fn=_damping_measurement,
        transitions=all_transitions,
    )

    return {
        "gravity": gravity,
        "main_thrust": main_thrust,
        "side_thrust": side_thrust,
        "kinematics": kinematics,
        "angular_damping": angular_damping,
        "angle_thrust": angle_thrust,
    }


@torch.no_grad()
def extract_pixel_oracle_constants(
    vae, dynamics, episodes: list[dict], device: str = "cpu",
) -> dict:
    """Layer 2b: Extract 6 physics constants from 1-step oracle predictions.

    For each qualifying transition, uses the state head to decode kinematics
    from both the current z and the dynamics-predicted next z. Filters on
    GT state (not estimated) for fair comparison with state-space eval.

    Extraction order matters — some constants depend on others:
    1. Gravity (no dependencies — measured from free-fall transitions)
    2. Main thrust (subtract gravity_model from vertical acceleration)
    3. Angle-thrust coupling (needs gravity to isolate thrust component)
    4. Side thrust (independent)
    5. Kinematics (independent — position/velocity consistency)
    6. Angular damping (independent — angular velocity decay)
    """
    vae = vae.to(device)
    dynamics = dynamics.to(device)
    vae.eval()
    dynamics.eval()

    # Collect all transitions across episodes via teacher-forced oracle
    all_transitions = []
    for ep in episodes:
        all_transitions.extend(_predict_pixel_oracle_deltas(vae, dynamics, ep, device))

    # Delegate to shared extraction pipeline — oracle uses GT state for filtering
    return _extract_constants_from_transitions(all_transitions)


@torch.no_grad()
def extract_pixel_rollout_constants(
    vae, dynamics, episodes: list[dict],
    rollout_k: int = 10, device: str = "cpu",
) -> dict:
    """Layer 2c: Extract physics constants from k-step autoregressive rollouts.

    Unlike oracle mode which re-grounds to GT each step, this rolls out
    autoregressively — errors compound. Filtering uses STATE-HEAD PREDICTIONS
    of dreamed z (not GT state), matching state-space rollout convention.
    """
    vae = vae.to(device)
    dynamics = dynamics.to(device)
    vae.eval()
    dynamics.eval()

    # Collect transitions from dreamed trajectories across episodes
    all_transitions = []

    for ep in episodes:
        frames = ep["frames"].to(device)
        actions = ep["actions"].to(device)
        gt_states = ep["states"][:, :6].numpy()
        T = min(actions.shape[0], rollout_k)
        if T < 2:
            continue

        # Encode seed frame, rollout k steps autoregressively
        z_0 = vae.encode(frames[0:1])  # (1, latent_dim)
        z_seq, _ = dynamics.rollout(z_0, actions[:T].unsqueeze(0))  # (1, T+1, D)
        z_seq = z_seq.squeeze(0)  # (T+1, latent_dim)

        # Extract kinematics from dreamed z at each step via state head
        dreamed_kin = vae.predict_state(z_seq).cpu().numpy()  # (T+1, 6)

        for t in range(T):
            delta_est = dreamed_kin[t + 1] - dreamed_kin[t]
            gt_delta = gt_states[t + 1] - gt_states[t]
            # Use DREAMED state for filtering (not GT) — rollout convention
            dreamed_state = dreamed_kin[t]

            all_transitions.append({
                "delta_est": delta_est,
                "gt_delta": gt_delta,
                "gt_state": dreamed_state,  # NOTE: dreamed, not GT
                "action": actions[t].cpu().numpy(),
                "timestep": t,
            })

    # Same extraction pipeline as oracle — factored into shared helper
    return _extract_constants_from_transitions(all_transitions)


@torch.no_grad()
def _dream_episodes(
    vae, dynamics, episodes: list[dict], device: str,
    start_stride: int = 0, max_horizon: int = 0,
) -> list[dict]:
    """Shared helper: dream each episode, return z + frame trajectories.

    Both rollout kinematics and pixel metrics need the same dreams —
    compute once here, reuse in both eval functions.

    Args:
        start_stride: if > 0, dream from multiple start points per episode
            spaced this many steps apart. 0 = single start from frame 0 (default).
        max_horizon: if > 0 and start_stride > 0, limit each dream segment
            to this many steps. 0 = dream to end of episode (default).

    When start_stride > 0, each start offset beyond 0 gets warm-up:
    teacher-force steps 0..offset through dynamics to build hidden state,
    then switch to autoregressive dreaming from offset onward. This gives
    the GRU/RSSM temporal context from the real episode before dreaming.

    Default behavior (start_stride=0): single start from frame 0, dream to
    end of episode using dynamics.rollout(). This matches the old behavior
    exactly — no warm-up, no multi-start.

    Returns list of dicts with:
        z_dream: (seg_len+1, latent_dim) dreamed z trajectory
        dream_frames: (seg_len+1, 1, H, W) decoded dreamed frames
        gt_frames: (seg_len+1, 1, H, W) original GT frames for this segment
        gt_states: (seg_len+1, 6) GT kinematic states for this segment
        actions: (seg_len, action_dim) actions for this segment
        T: number of action steps in this segment
        start_offset: which GT frame this dream started from
    """
    vae = vae.to(device)
    dynamics = dynamics.to(device)
    vae.eval()
    dynamics.eval()

    dreams = []
    for ep in episodes:
        frames = ep["frames"].to(device)       # (T+1, 1, H, W)
        actions = ep["actions"].to(device)      # (T, 2)
        gt_states = ep["states"][:, :6]         # (T+1, 6) — stays on CPU
        T = actions.shape[0]

        # Compute start offsets for this episode
        if start_stride > 0 and max_horizon > 0:
            # Generate start points spaced by start_stride, ensuring each
            # has at least 1 step to dream (not max_horizon — shorter
            # segments at the end are still valuable for diversity)
            starts = list(range(0, T, start_stride))
            # Filter out starts that would have 0 steps to dream
            starts = [s for s in starts if s < T]
            if not starts:
                starts = [0]
        else:
            # Default: single start from frame 0 (backward-compatible)
            starts = [0]

        # Encode all GT frames once (shared across start points).
        # This is needed for warm-up (teacher-forcing) and for seeding
        # each dream segment with the correct initial z.
        z_all = vae.encode(frames)  # (T+1, latent_dim)

        for start in starts:
            if start == 0 and (start_stride <= 0 or max_horizon <= 0):
                # Default path: use dynamics.rollout() for backward compat.
                # This matches the old behavior exactly (same numerical output)
                # — important because test_default_matches_old_behavior validates
                # that default dreams are bit-identical to a direct rollout call.
                z_seq, _ = dynamics.rollout(
                    z_all[0:1], actions.unsqueeze(0))  # (1, T+1, latent_dim)
                z_seq = z_seq.squeeze(0)               # (T+1, latent_dim)
                dream_len = T
            else:
                # --- Warm-up phase ---
                # Teacher-force dynamics through steps 0..start to build
                # hidden state with temporal context from the episode.
                # This matters for GRU/RSSM models whose hidden state encodes
                # recent history — without warm-up, mid-episode dreams start
                # from a cold hidden state that doesn't match the episode context.
                state = None
                if start > 0:
                    for t in range(start):
                        _, state = dynamics.forward(
                            z_all[t:t+1], actions[t:t+1], state)

                # --- Dream phase ---
                # Autoregressive rollout from the warmed-up hidden state.
                # Can't use dynamics.rollout() because we need to pass the
                # warmed-up hidden state from the teacher-forcing loop.
                dream_len = min(max_horizon, T - start) if max_horizon > 0 else T - start

                # Manual rollout: seed with encoded GT frame at start offset,
                # then autoregressively predict forward
                z_seq_parts = [z_all[start:start+1].squeeze(0)]
                z = z_all[start:start+1]
                for t in range(dream_len):
                    z, state = dynamics.forward(
                        z, actions[start + t:start + t + 1], state)
                    z_seq_parts.append(z.squeeze(0))
                z_seq = torch.stack(z_seq_parts)  # (dream_len+1, latent_dim)

            # Decode dreamed z to pixel frames (for pixel metrics)
            dream_frames = vae.decode(z_seq)  # (dream_len+1, 1, H, W)

            dreams.append({
                "z_dream": z_seq.cpu(),
                "dream_frames": dream_frames.cpu(),
                "gt_frames": frames[start:start + dream_len + 1].cpu(),
                "gt_states": gt_states[start:start + dream_len + 1],
                "actions": actions[start:start + dream_len].cpu(),
                "T": dream_len,
                "start_offset": start,
            })
    return dreams


@torch.no_grad()
def evaluate_pixel_rollout_kinematics(
    vae, dynamics, episodes: list[dict],
    horizons: list[int] = None, device: str = "cpu",
    dreams: list[dict] | None = None,
    start_stride: int = 0, max_horizon: int = 0,
) -> dict:
    """Layer 3a: Kinematics error at each dream horizon.

    Dreams each episode autoregressively, extracts kinematics via state
    head at every step, compares to GT kinematics. Reports per-dim MSE
    at each requested horizon and fits a power law compounding exponent.

    Note: MSE includes both dynamics error AND state head noise. The
    perception tax (compute_perception_tax) provides the decomposition.
    """
    horizons = horizons or [1, 5, 10, 20, 50]
    vae = vae.to(device)
    vae.eval()

    # Dream all episodes. Caller (run_full_eval) can pass precomputed dreams
    # to avoid re-dreaming for each Layer 3 function. When called directly
    # (not via run_full_eval), start_stride/max_horizon control multi-start.
    if dreams is None:
        dreams = _dream_episodes(vae, dynamics, episodes, device,
                                  start_stride=start_stride, max_horizon=max_horizon)

    # Collect per-dim squared errors at each horizon across episodes
    # horizon → list of (6,) arrays
    errors_by_horizon: dict[int, list[np.ndarray]] = {h: [] for h in horizons}

    for dream in dreams:
        z_dream = dream["z_dream"].to(device)
        # gt_states may be a Tensor or numpy array depending on caller —
        # normalise to numpy for consistent indexing and arithmetic
        gt_states_raw = dream["gt_states"]
        if isinstance(gt_states_raw, torch.Tensor):
            gt_kin = gt_states_raw[:, :6].numpy()
        else:
            gt_kin = np.asarray(gt_states_raw)[:, :6]
        T = dream["T"]

        # Extract kinematics from dreamed z at every step via state head.
        # predict_state applies the auxiliary linear head trained alongside
        # the VAE encoder to decode (x, y, vx, vy, angle, ang_vel) from z.
        pred_kin = vae.predict_state(z_dream).cpu().numpy()  # (T+1, 6)

        for h in horizons:
            if h <= T:
                # Per-dim squared error at horizon h — compare step h of
                # the dream trajectory to GT step h. This isolates how
                # much error has compounded after h autoregressive steps.
                err = (pred_kin[h] - gt_kin[h]) ** 2  # (6,)
                errors_by_horizon[h].append(err)

    # Aggregate: mean per-dim MSE at each horizon
    horizon_mse = {}
    for h in horizons:
        if errors_by_horizon[h]:
            horizon_mse[h] = np.stack(errors_by_horizon[h]).mean(axis=0).tolist()
        else:
            horizon_mse[h] = [float("nan")] * 6

    # Fit power law: MSE(h) ~ h^b per dimension.
    # We expect b > 0 (error grows with horizon). b ≈ 2 is typical for
    # linear models (error grows quadratically). b >> 2 signals fast
    # compounding / chaotic dynamics.
    compounding_b = []
    for d in range(6):
        hs = []
        mses = []
        for h in sorted(horizons):
            if h in horizon_mse and np.isfinite(horizon_mse[h][d]) and horizon_mse[h][d] > 0:
                hs.append(h)
                mses.append(horizon_mse[h][d])
        if len(hs) >= 2:
            try:
                # Fit log(MSE) = b * log(h) + c  →  MSE ~ h^b
                log_h = np.log(np.array(hs, dtype=float))
                log_mse = np.log(np.array(mses, dtype=float))
                b, _ = np.polyfit(log_h, log_mse, 1)
                compounding_b.append(float(b))
            except (ValueError, np.linalg.LinAlgError):
                compounding_b.append(float("nan"))
        else:
            compounding_b.append(float("nan"))

    return {
        "horizon_mse": horizon_mse,
        "compounding_b": compounding_b,
        "dim_names": _KIN_DIMS,
        "n_dream_segments": len(dreams),
    }


@torch.no_grad()
def evaluate_pixel_rollout_frames(
    vae, dynamics, episodes: list[dict],
    horizons: list[int] = None, device: str = "cpu",
    dreams: list[dict] | None = None,
) -> dict:
    """Layer 3b: Pixel-level dream quality at each horizon.

    Dreams each episode, decodes to frames, computes pixel MSE and SSIM
    vs GT frames at each horizon. Also reports recognizable horizon
    (steps until SSIM drops below 0.5).

    Why pixel metrics on top of kinematics (Layer 3a)?
    - Kinematics error (via state head) mixes perception noise with dynamics
      error. Pixel MSE/SSIM are a direct, head-free measure of visual fidelity.
    - SSIM correlates with human-perceived quality — a low SSIM means the dream
      no longer "looks like" the game, regardless of what the state head says.
    - The recognizable horizon gives a single interpretable threshold: after how
      many steps does the dream degrade past a perceptual threshold (SSIM < 0.5)?
    """
    from lwp.evaluation.pixel_metrics import pixel_mse, compute_ssim, recognizable_horizon
    from lwp.training.pixel_losses import _foreground_weight_mask

    horizons = horizons or [1, 5, 10, 20, 50]
    vae = vae.to(device)

    # Use precomputed dreams if provided, otherwise dream fresh.
    # Callers that run multiple Layer 3 functions should pass dreams to avoid
    # redundant rollout+decode passes (each is expensive at scale).
    if dreams is None:
        dreams = _dream_episodes(vae, dynamics, episodes, device)

    # Collect per-horizon pixel metrics
    mse_by_horizon: dict[int, list[float]] = {h: [] for h in horizons}
    ssim_by_horizon: dict[int, list[float]] = {h: [] for h in horizons}
    # Fg-weighted variants: same structure, separate accumulators
    fg_mse_by_horizon: dict[int, list[float]] = {h: [] for h in horizons}
    fg_ssim_by_horizon: dict[int, list[float]] = {h: [] for h in horizons}
    recog_horizons: list[int] = []
    fg_recog_horizons: list[int] = []

    for dream in dreams:
        dream_fr = dream["dream_frames"]    # (T+1, 1, H, W)
        gt_fr = dream["gt_frames"]          # (T+1, 1, H, W)
        T = dream["T"]

        for h in horizons:
            if h <= T:
                # Single-frame pixel metrics at horizon h.
                # We compare one specific future frame to isolate how much
                # visual quality has degraded at exactly step h.
                pred_h = dream_fr[h:h+1]    # (1, 1, H, W)
                gt_h = gt_fr[h:h+1]
                mse_by_horizon[h].append(pixel_mse(pred_h, gt_h).item())
                ssim_by_horizon[h].append(compute_ssim(pred_h, gt_h))

                # ---- Fg-weighted MSE at this horizon ----
                # Mask from GT frame (NOT dream) — if the model loses the
                # lander, the dreamed frame has no foreground pixels, and
                # computing the mask from it would produce zero fg-MSE
                fg_weight = 50.0
                weights = _foreground_weight_mask(gt_h, fg_weight)
                fg_mse_h = (weights * (pred_h - gt_h).pow(2)).mean().item()
                fg_mse_by_horizon[h].append(fg_mse_h)

                # ---- Fg-only SSIM at this horizon ----
                ssim_map = _compute_ssim_map(pred_h, gt_h)
                fg_mask = (gt_h > 0.04) & (gt_h < 0.78)
                fg_pixels = fg_mask.float()
                fg_ssim_h = (ssim_map * fg_pixels).sum() / max(fg_pixels.sum().item(), 1.0)
                fg_ssim_by_horizon[h].append(fg_ssim_h.item())

        # Recognizable horizon: steps until SSIM < 0.5.
        # Threshold 0.5 is a common perceptual boundary — below it the image
        # is visibly degraded and hard to match to the original by eye.
        max_t = min(T + 1, dream_fr.shape[0])
        rh = recognizable_horizon(dream_fr[:max_t], gt_fr[:max_t], threshold=0.5)
        recog_horizons.append(rh)

        # ---- Fg recognizable horizon ----
        # Same concept but using fg-only SSIM: find the first step where
        # foreground SSIM drops below 0.5. This will typically be much lower
        # than the background-dominated recognizable horizon because the
        # lander (3% of pixels) degrades faster than the trivial sky/terrain.
        fg_rh = 0
        for t in range(1, max_t):
            ssim_map_t = _compute_ssim_map(dream_fr[t:t+1], gt_fr[t:t+1])
            fg_mask_t = (gt_fr[t:t+1] > 0.04) & (gt_fr[t:t+1] < 0.78)
            fg_pix = fg_mask_t.float()
            if fg_pix.sum().item() > 0:
                fg_ssim_t = (ssim_map_t * fg_pix).sum() / fg_pix.sum()
                if fg_ssim_t.item() < 0.5:
                    fg_rh = t
                    break
            else:
                # No foreground pixels in GT at this step — can't judge quality
                continue
        else:
            fg_rh = max_t  # Never dropped below threshold
        fg_recog_horizons.append(fg_rh)

    # Aggregate across episodes: simple mean per horizon
    horizon_pixel_mse = {}
    horizon_ssim = {}
    horizon_fg_pixel_mse = {}
    horizon_fg_ssim = {}
    for h in horizons:
        if mse_by_horizon[h]:
            horizon_pixel_mse[h] = float(np.mean(mse_by_horizon[h]))
            horizon_ssim[h] = float(np.mean(ssim_by_horizon[h]))
            horizon_fg_pixel_mse[h] = float(np.mean(fg_mse_by_horizon[h]))
            horizon_fg_ssim[h] = float(np.mean(fg_ssim_by_horizon[h]))
        else:
            # No episodes had length >= h — report NaN rather than silently
            # returning 0 which would be misleading
            horizon_pixel_mse[h] = float("nan")
            horizon_ssim[h] = float("nan")
            horizon_fg_pixel_mse[h] = float("nan")
            horizon_fg_ssim[h] = float("nan")

    return {
        "horizon_pixel_mse": horizon_pixel_mse,
        "horizon_ssim": horizon_ssim,
        "recognizable_horizon": float(np.mean(recog_horizons)) if recog_horizons else 0,
        "horizon_fg_pixel_mse": horizon_fg_pixel_mse,
        "horizon_fg_ssim": horizon_fg_ssim,
        "fg_recognizable_horizon": float(np.mean(fg_recog_horizons)) if fg_recog_horizons else 0,
    }


@torch.no_grad()
def evaluate_state_head_ood(
    vae, dynamics, episodes: list[dict],
    horizons: list[int] = None, device: str = "cpu",
    dreams: list[dict] | None = None,
) -> dict:
    """Layer 3c: State head OOD detection on dreamed z's.

    At each horizon, compare:
    - predict_state(z_dreamed)                     — state head reading raw dreamed z
    - predict_state(encode(decode(z_dreamed)))      — state head reading re-encoded z

    If these diverge, the dreamed z is outside the VAE encoder's output
    distribution and state head predictions are unreliable.

    Why does this matter?
    The state head was trained on z's produced by the encoder. Autoregressive
    dreaming can push z outside the encoder's typical output manifold — the
    dynamics model may produce z's that are "physically plausible" in latent
    space but don't correspond to any image that the encoder would produce for
    a real game frame. When that happens, state head predictions are extrapolating
    outside their training distribution and may be meaningless.

    The re-encoding trick: decode z → pixel frame → re-encode. The re-encoded z
    is guaranteed to be on the encoder manifold. If predict_state(z_direct) and
    predict_state(z_reencoded) disagree significantly, z_direct is OOD.

    Returns per-horizon L2 divergence between the two state predictions.
    """
    horizons = horizons or [1, 5, 10, 20, 50]
    vae = vae.to(device)

    if dreams is None:
        dreams = _dream_episodes(vae, dynamics, episodes, device)

    divergence_by_horizon: dict[int, list[float]] = {h: [] for h in horizons}

    for dream in dreams:
        z_dream = dream["z_dream"].to(device)   # (T+1, latent_dim)
        T = dream["T"]

        for h in horizons:
            if h <= T:
                z_h = z_dream[h:h+1]                          # (1, D)
                # Direct state head reading from raw dreamed z
                state_direct = vae.predict_state(z_h)          # (1, 6)
                # Re-encode: decode to pixels, then encode back to z.
                # This projects z_h onto the encoder manifold — the set of
                # z's the encoder would produce for real-looking frames.
                frame_decoded = vae.decode(z_h)                # (1, 1, H, W)
                z_reencoded = vae.encode(frame_decoded)        # (1, D)
                state_reencoded = vae.predict_state(z_reencoded)  # (1, 6)
                # L2 divergence between the two readings — large means OOD
                div = (state_direct - state_reencoded).pow(2).sum().sqrt().item()
                divergence_by_horizon[h].append(div)

    horizon_ood_divergence = {}
    for h in horizons:
        if divergence_by_horizon[h]:
            horizon_ood_divergence[h] = float(np.mean(divergence_by_horizon[h]))
        else:
            horizon_ood_divergence[h] = float("nan")

    return {"horizon_ood_divergence": horizon_ood_divergence}


def compute_perception_tax(
    layer1_results: dict,
    layer2_oracle_results: dict,
    layer2_latent_results: dict,
) -> dict:
    """Perception tax: decompose combined error into perception vs dynamics.

    Three complementary measures (from spec):
    1. perception_floor: Layer 1 per-dim MSE — best the state head can do
       even on GT frames. This is irreducible error from the encoder alone.
    2. latent_dynamics_mse: Layer 2 latent accuracy — pure dynamics error
       in z-space, independent of the state head entirely.
    3. dynamics_context: per-constant, reports oracle relative error and the
       perception floor for the relevant dimension side-by-side. These are
       in DIFFERENT units (relative error vs MSE) so we do NOT subtract —
       just present both for the reader to judge which source dominates.

    Note on decomposition limits: perception and dynamics errors interact
    nonlinearly in the full rollout (Layer 3). This function provides
    complementary upper-bound estimates, not an exact additive decomposition.
    A proper decomposition would require controlled ablation experiments.
    """
    perception_floor = layer1_results["per_dim_mse"]  # list of 6 floats
    latent_dynamics_mse = layer2_latent_results.get("latent_mse", float("nan"))

    # Map each physics constant to its primary kinematic dimension index.
    # This tells us which perception_floor entry is most relevant to each
    # constant's measurement (e.g., gravity is measured via vy → index VY=3).
    _CONST_TO_DIM = {
        "gravity": VY,              # free-fall measured as dvy → vy dimension
        "main_thrust": VY,          # thrust contribution also in vy
        "side_thrust": VX,          # horizontal thrust → vx dimension
        "kinematics": X,            # position/velocity consistency → x dimension
        "angular_damping": ANGULAR_VEL,  # angular velocity decay → ang_vel dim
        "angle_thrust": VX,         # angle-to-thrust coupling → vx dimension
    }

    # Per-constant context: report oracle error alongside perception floor
    # for the relevant dimension. Presenting them together lets the reader
    # compare scale: if perception_floor_mse >> oracle_relative_error, the
    # state head noise is likely dominating the oracle measurement error.
    dynamics_context = {}
    for const_name, dim_idx in _CONST_TO_DIM.items():
        if const_name in layer2_oracle_results:
            oracle_err = layer2_oracle_results[const_name].get("relative_error", float("nan"))
            floor = perception_floor[dim_idx] if dim_idx < len(perception_floor) else 0
            dynamics_context[const_name] = {
                "oracle_relative_error": oracle_err,
                "perception_floor_dim": _KIN_DIMS[dim_idx],
                "perception_floor_mse": floor,
                "n_samples": layer2_oracle_results[const_name].get("n_samples", 0),
            }

    return {
        "perception_floor": perception_floor,
        "latent_dynamics_mse": latent_dynamics_mse,
        "dynamics_context": dynamics_context,
    }


@torch.no_grad()
def compute_baselines(
    vae, episodes: list[dict],
    horizons: list[int] = None, device: str = "cpu",
) -> dict:
    """Compute trivial baseline kinematics MSE at each horizon.

    Two baselines, neither requiring a dynamics model:
    1. zero_predictor: predict state stays at initial state forever.
       MSE = (gt_state[h] - gt_state[0])^2. Measures how much the
       state changes — if the model can't beat this, it's worse than
       predicting "nothing moves."
    2. copy_previous: predict state[h] = state[h-1] (persist last known).
       MSE = (gt_state[h] - gt_state[h-1])^2. Measures the per-step
       delta magnitude — if the model can't beat this, it's adding noise
       rather than predicting dynamics.

    Both computed from GT states (no model involved, no state-head noise).
    These are LOWER BOUNDS — the model's kinematics MSE will always be
    higher even with perfect dynamics because it includes state-head
    perception noise that baselines don't have. They provide "is the model
    better than doing nothing?" context, but the comparison is not strict
    apples-to-apples. The report should note this caveat.
    """
    horizons = horizons or [1, 5, 10, 20, 50]

    # Collect per-horizon squared errors across episodes
    # Each entry is a (6,) array — one squared error per kinematic dim
    zero_errors: dict[int, list[np.ndarray]] = {h: [] for h in horizons}
    copy_errors: dict[int, list[np.ndarray]] = {h: [] for h in horizons}

    for ep in episodes:
        # GT kinematics: first 6 state dims (x, y, vx, vy, angle, ang_vel)
        gt_kin = ep["states"][:, :6].numpy()  # (T+1, 6)
        T = ep["actions"].shape[0]

        for h in horizons:
            if h <= T:
                # Zero predictor: predict state stays at step 0
                # High MSE = state changed a lot → easy to beat
                zero_err = (gt_kin[h] - gt_kin[0]) ** 2  # (6,)
                zero_errors[h].append(zero_err)

                # Copy previous: predict state[h] = state[h-1]
                # High MSE = large per-step jumps → harder to beat
                copy_err = (gt_kin[h] - gt_kin[h - 1]) ** 2  # (6,)
                copy_errors[h].append(copy_err)

    # Average across episodes, keeping per-dim breakdown
    zero_predictor = {}
    copy_previous = {}
    for h in horizons:
        if zero_errors[h]:
            # Stack (N, 6) → mean over episodes → (6,) per-dim MSE
            zero_predictor[h] = np.stack(zero_errors[h]).mean(axis=0).tolist()
            copy_previous[h] = np.stack(copy_errors[h]).mean(axis=0).tolist()
        else:
            # No episodes long enough for this horizon
            zero_predictor[h] = [float("nan")] * 6
            copy_previous[h] = [float("nan")] * 6

    return {
        "zero_predictor": zero_predictor,
        "copy_previous": copy_previous,
        "dim_names": _KIN_DIMS,
    }


# Contrasting test actions for sensitivity analysis
_TEST_ACTIONS = {
    "zero": np.array([0.0, 0.0], dtype=np.float32),
    "main_thrust": np.array([1.0, 0.0], dtype=np.float32),
    "side_left": np.array([0.0, -1.0], dtype=np.float32),
    "side_right": np.array([0.0, 1.0], dtype=np.float32),
}


@torch.no_grad()
def evaluate_action_sensitivity(
    vae, dynamics, episodes: list[dict], device: str = "cpu",
) -> dict:
    """Layer 4a: Action sensitivity — does the model respond to different actions?

    For sampled z's from GT frames, predict z_{t+1} under 4 contrasting
    actions from the SAME state. Compute pairwise L2 in both z-space and
    kinematics-space.

    A model that ignores actions produces identical z_next for all actions
    → L2 ≈ 0 → sensitivity ratio ≈ 0.

    RSSM limitation: forward() delegates to imagine_step() which uses
    prior only — the z_t input is ignored. Sensitivity measures how the
    prior responds to different actions from zero state, not from the
    actual encoded state. This tests the prior's action-conditioning
    quality, which is what matters for dreaming. For state-conditioned
    sensitivity, use step() (posterior) — but that's training-time only.
    """
    vae = vae.to(device)
    dynamics = dynamics.to(device)
    vae.eval()
    dynamics.eval()

    action_names = list(_TEST_ACTIONS.keys())
    n_actions = len(action_names)

    # Collect z_next predictions under each test action across sampled transitions
    # Key: action_name → list of z_next tensors
    z_nexts_by_action: dict[str, list[np.ndarray]] = {a: [] for a in action_names}
    kin_nexts_by_action: dict[str, list[np.ndarray]] = {a: [] for a in action_names}

    for ep in episodes:
        frames = ep["frames"].to(device)
        T = ep["actions"].shape[0]
        z_all = vae.encode(frames)  # (T+1, latent_dim)

        # Sample up to 20 transitions per episode (evenly spaced)
        step = max(1, T // 20)
        sample_indices = list(range(0, T, step))

        for t in sample_indices:
            z_t = z_all[t:t+1]  # (1, latent_dim)

            for action_name, action_val in _TEST_ACTIONS.items():
                action_tensor = torch.from_numpy(action_val).unsqueeze(0).to(device)
                # Single-step prediction with fresh hidden state
                # (stateless — we want to measure action sensitivity, not hidden state effects)
                z_next, _ = dynamics.forward(z_t, action_tensor, None)
                z_nexts_by_action[action_name].append(z_next.cpu().numpy()[0])

                # Also extract kinematics for per-dim breakdown
                if vae.state_dim > 0:
                    kin = vae.predict_state(z_next).cpu().numpy()[0]
                    kin_nexts_by_action[action_name].append(kin)

    # Compute pairwise L2 distances
    latent_l2_matrix = {}
    kinematics_l2_matrix = {}
    for i in range(n_actions):
        for j in range(i + 1, n_actions):
            a1, a2 = action_names[i], action_names[j]
            pair_key = f"{a1}_vs_{a2}"

            z1 = np.array(z_nexts_by_action[a1])  # (N, latent_dim)
            z2 = np.array(z_nexts_by_action[a2])
            # Mean L2 distance across sampled transitions
            latent_l2 = float(np.sqrt(((z1 - z2) ** 2).sum(axis=1)).mean())
            latent_l2_matrix[pair_key] = latent_l2

            # Per-dim kinematics L2
            if kin_nexts_by_action[a1]:
                k1 = np.array(kin_nexts_by_action[a1])  # (N, 6)
                k2 = np.array(kin_nexts_by_action[a2])
                per_dim_l2 = {}
                for d, dim_name in enumerate(_KIN_DIMS):
                    per_dim_l2[dim_name] = float(np.abs(k1[:, d] - k2[:, d]).mean())
                kinematics_l2_matrix[pair_key] = per_dim_l2

    # Sensitivity ratio: model L2 / expected L2 from GT
    # For GT, we'd need episodes with exactly those contrasting actions from
    # the same state — not available in recorded data. Use the model's own
    # L2 as a standalone metric. ratio is left as the raw L2 values.
    sensitivity_ratio = {k: v for k, v in latent_l2_matrix.items()}

    return {
        "latent_l2_matrix": latent_l2_matrix,
        "kinematics_l2_matrix": kinematics_l2_matrix,
        "sensitivity_ratio": sensitivity_ratio,
    }


@torch.no_grad()
def evaluate_action_ablation(
    vae, dynamics, episodes: list[dict],
    horizons: list[int] = None, device: str = "cpu",
) -> dict:
    """Layer 4b: Action ablation — dream same episode with real/zero/random actions.

    For each episode, dream three times from the same z_0:
    (a) real actions from the episode
    (b) zero actions throughout
    (c) random actions (uniform [-1, 1])

    Compare z trajectories. If MSE(real vs zero) ≈ 0, the model ignores actions.
    """
    horizons = horizons or [1, 5, 10, 20, 50]
    vae = vae.to(device)
    dynamics = dynamics.to(device)
    vae.eval()
    dynamics.eval()

    mse_real_zero: dict[int, list[float]] = {h: [] for h in horizons}
    mse_real_random: dict[int, list[float]] = {h: [] for h in horizons}

    # Per-dim kinematics collectors — tracks which kinematic dims
    # diverge when actions change (e.g., vy should differ for main thrust)
    kin_real_zero: dict[int, list[dict]] = {h: [] for h in horizons}
    kin_real_random: dict[int, list[dict]] = {h: [] for h in horizons}

    for ep in episodes:
        frames = ep["frames"].to(device)
        actions_real = ep["actions"].to(device)  # (T, 2)
        T = actions_real.shape[0]

        # Encode seed
        z_0 = vae.encode(frames[0:1])  # (1, latent_dim)

        # Dream with real actions
        z_real, _ = dynamics.rollout(z_0, actions_real.unsqueeze(0))  # (1, T+1, D)
        z_real = z_real.squeeze(0)  # (T+1, D)

        # Dream with zero actions
        actions_zero = torch.zeros_like(actions_real)
        z_zero, _ = dynamics.rollout(z_0, actions_zero.unsqueeze(0))
        z_zero = z_zero.squeeze(0)

        # Dream with random actions (uniform [-1, 1])
        actions_random = torch.rand_like(actions_real) * 2 - 1
        z_random, _ = dynamics.rollout(z_0, actions_random.unsqueeze(0))
        z_random = z_random.squeeze(0)

        # Extract kinematics from all three dream trajectories via state head
        # for per-dim breakdown of action effects
        if vae.state_dim > 0:
            kin_real = vae.predict_state(z_real.to(device)).cpu().numpy()    # (T+1, 6)
            kin_zero = vae.predict_state(z_zero.to(device)).cpu().numpy()
            kin_random = vae.predict_state(z_random.to(device)).cpu().numpy()

        # Compare at each horizon
        for h in horizons:
            if h <= T:
                mse_rz = F.mse_loss(z_real[h], z_zero[h]).item()
                mse_rr = F.mse_loss(z_real[h], z_random[h]).item()
                mse_real_zero[h].append(mse_rz)
                mse_real_random[h].append(mse_rr)

                # Per-dim kinematics difference: shows WHICH dimensions
                # respond to actions. vy should differ for main thrust,
                # vx for side thrust, angle for angular control.
                if vae.state_dim > 0:
                    rz_kin = {dim: float((kin_real[h, d] - kin_zero[h, d]) ** 2)
                              for d, dim in enumerate(_KIN_DIMS)}
                    rr_kin = {dim: float((kin_real[h, d] - kin_random[h, d]) ** 2)
                              for d, dim in enumerate(_KIN_DIMS)}
                    kin_real_zero[h].append(rz_kin)
                    kin_real_random[h].append(rr_kin)

    # Aggregate: mean MSE at each horizon
    result_rz = {}
    result_rr = {}
    for h in horizons:
        result_rz[h] = float(np.mean(mse_real_zero[h])) if mse_real_zero[h] else float("nan")
        result_rr[h] = float(np.mean(mse_real_random[h])) if mse_real_random[h] else float("nan")

    # Aggregate per-dim kinematics: mean across episodes at each horizon
    result_kin_rz = {}
    result_kin_rr = {}
    for h in horizons:
        if kin_real_zero[h]:
            # Average each dim's MSE across episodes
            result_kin_rz[h] = {
                dim: float(np.mean([d[dim] for d in kin_real_zero[h]]))
                for dim in _KIN_DIMS
            }
            result_kin_rr[h] = {
                dim: float(np.mean([d[dim] for d in kin_real_random[h]]))
                for dim in _KIN_DIMS
            }
        else:
            result_kin_rz[h] = {dim: float("nan") for dim in _KIN_DIMS}
            result_kin_rr[h] = {dim: float("nan") for dim in _KIN_DIMS}

    return {
        "mse_real_vs_zero": result_rz,
        "mse_real_vs_random": result_rr,
        "kin_real_vs_zero": result_kin_rz,
        "kin_real_vs_random": result_kin_rr,
    }


def run_full_eval(
    vae, dynamics, episodes: list[dict],
    horizons: list[int] = None, rollout_k: int = 10,
    device: str = "cpu",
    start_stride: int = 0, max_horizon: int = 0,
) -> dict:
    """Run all 4 evaluation layers and return combined results.

    Main entry point — called by CLI script and integration tests.
    Each layer is independent; errors in one don't block others.

    Layer execution order is intentional:
    - Layer 1 first (fastest, pure encoder — no dynamics needed)
    - Layer 2 second (oracle extraction feeds into Layer 3 tax)
    - Layer 3 third (needs dreams, which are expensive — compute once)
    - Layer 4 last (action conditioning, stateless per-episode)
    """
    horizons = horizons or [1, 5, 10, 20, 50]
    results = {}

    # Layer 1: perception baseline
    print("  Layer 1: perception baseline...")
    results["layer1_fidelity"] = evaluate_state_head_fidelity(vae, episodes, device)
    results["layer1_recon"] = evaluate_vae_reconstruction(vae, episodes, device)

    # Layer 2: oracle physics
    print("  Layer 2: oracle physics...")
    results["layer2_latent"] = evaluate_latent_dynamics_accuracy(vae, dynamics, episodes, device)
    results["layer2_oracle"] = extract_pixel_oracle_constants(vae, dynamics, episodes, device)
    results["layer2_rollout"] = extract_pixel_rollout_constants(
        vae, dynamics, episodes, rollout_k, device)
    results["layer2_consistency"] = compute_consistency_r2(results["layer2_oracle"])

    # Layer 3: rollout metrics — dream once, reuse for all three functions.
    # _dream_episodes is expensive (full-episode rollout for every episode),
    # so we share the computed dreams rather than each function re-dreaming.
    # Multi-start params (start_stride, max_horizon) control whether we dream
    # from multiple starting points per episode with GRU warm-up.
    print("  Layer 3: rollout metrics...")
    dreams = _dream_episodes(vae, dynamics, episodes, device,
                              start_stride=start_stride, max_horizon=max_horizon)
    results["layer3_kinematics"] = evaluate_pixel_rollout_kinematics(
        vae, dynamics, episodes, horizons, device, dreams=dreams)
    results["layer3_frames"] = evaluate_pixel_rollout_frames(
        vae, dynamics, episodes, horizons, device, dreams=dreams)
    results["layer3_ood"] = evaluate_state_head_ood(
        vae, dynamics, episodes, horizons, device, dreams=dreams)
    results["layer3_tax"] = compute_perception_tax(
        results["layer1_fidelity"], results["layer2_oracle"], results["layer2_latent"])

    # Baselines: trivial predictors for context — computed from GT states only,
    # so no dynamics model needed. Placed here (not in Layer 1) because the
    # baselines are compared to Layer 3 horizon metrics in the report.
    results["baselines"] = compute_baselines(vae, episodes, horizons, device)

    # Layer 4: action conditioning
    print("  Layer 4: action conditioning...")
    results["layer4_sensitivity"] = evaluate_action_sensitivity(vae, dynamics, episodes, device)
    results["layer4_ablation"] = evaluate_action_ablation(
        vae, dynamics, episodes, horizons, device)

    return results


def format_report(results: dict, model_name: str = "pixel_wm") -> str:
    """Format all eval results into a human-readable markdown report.

    Iterates through all four layers in order, pulling values from the
    results dict. Missing keys produce empty sections rather than crashes
    — partial results (e.g. from a failed layer) still render cleanly.
    """
    lines = [f"# Pixel WM Physics Evaluation: {model_name}\n"]

    # Layer 1
    lines.append("## Layer 1: Perception Baseline\n")
    fid = results.get("layer1_fidelity", {})
    if "per_dim_mse" in fid:
        lines.append("### State Head Fidelity\n")
        lines.append("| Dim | MSE | R² |")
        lines.append("|-----|-----|-----|")
        for i, dim in enumerate(_KIN_DIMS):
            mse = fid["per_dim_mse"][i]
            r2 = fid["per_dim_r2"][i]
            lines.append(f"| {dim} | {mse:.6f} | {r2:.4f} |")
    recon = results.get("layer1_recon", {})
    if recon:
        lines.append(f"\n### VAE Reconstruction\n")
        lines.append(f"- Pixel MSE: {recon.get('pixel_mse', float('nan')):.6f}")
        lines.append(f"- SSIM: {recon.get('ssim', float('nan')):.4f}")
        lines.append(f"- Fg-weighted MSE: {recon.get('fg_pixel_mse', float('nan')):.6f}")
        lines.append(f"- Fg-only SSIM: {recon.get('fg_ssim', float('nan')):.4f}")

    # Layer 2
    lines.append("\n## Layer 2: Oracle Physics\n")
    latent = results.get("layer2_latent", {})
    lines.append(f"Latent dynamics MSE (state-head-free): {latent.get('latent_mse', 'n/a'):.6f}\n")
    oracle = results.get("layer2_oracle", {})
    if oracle:
        lines.append("### Oracle Constants (1-step)\n")
        lines.append("| Constant | Model mean | GT mean | Rel. error | Sign | n_samples | Warning |")
        lines.append("|----------|-----------|---------|------------|------|-----------|---------|")
        for const in ["gravity", "main_thrust", "side_thrust",
                      "kinematics", "angular_damping", "angle_thrust"]:
            c = oracle.get(const, {})
            sign = "Y" if c.get("sign_correct", False) else "N"
            warning = "LOW" if c.get("low_sample_warning", False) else ""
            lines.append(
                f"| {const} | {c.get('model_mean', 'n/a'):.4f} "
                f"| {c.get('gt_mean', 'n/a'):.4f} "
                f"| {c.get('relative_error', 'n/a'):.2%} "
                f"| {sign} "
                f"| {c.get('n_samples', 0)} "
                f"| {warning} |")

    # Normalized latent MSE — shows how large the prediction error is relative
    # to the natural variance of the latent space (MSE / z_variance).
    if "latent_mse_normalized" in latent:
        lines.append(f"\nNormalized latent MSE (MSE/z_var): {latent['latent_mse_normalized']:.6f}")
        lines.append(f"z_variance: {latent.get('z_variance', 'n/a'):.6f}")

    # Layer 3
    lines.append("\n## Layer 3: Rollout Metrics\n")
    kin = results.get("layer3_kinematics", {})
    if "horizon_mse" in kin:
        lines.append("### Kinematics MSE by horizon\n")
        horizons = sorted(kin["horizon_mse"].keys())
        header = "| Dim | " + " | ".join(f"h={h}" for h in horizons) + " |"
        sep = "|-----|" + "|".join("---" for _ in horizons) + "|"
        lines.append(header)
        lines.append(sep)
        for i, dim in enumerate(_KIN_DIMS):
            vals = " | ".join(f"{kin['horizon_mse'][h][i]:.6f}" for h in horizons)
            lines.append(f"| {dim} | {vals} |")
    frames = results.get("layer3_frames", {})
    if frames:
        lines.append(f"\nRecognizable horizon: {frames.get('recognizable_horizon', 'n/a'):.1f} steps")
        # Fg recognizable horizon — same threshold but using fg-weighted SSIM,
        # which is more sensitive to small-object degradation.
        if "fg_recognizable_horizon" in frames:
            lines.append(f"Fg recognizable horizon: {frames['fg_recognizable_horizon']:.1f} steps")

    # Baselines: trivial predictors (zero-predictor, copy-previous) give context
    # for how much the model actually learns beyond naive strategies.
    baselines = results.get("baselines", {})
    if baselines:
        lines.append("\n### Baselines (GT-only, no model)\n")
        lines.append("*Caveat: Model kinematics MSE includes state-head perception noise; "
                      "baselines are computed from GT states directly. "
                      "Comparison is not apples-to-apples.*\n")
        zero_pred = baselines.get("zero_predictor", {})
        copy_prev = baselines.get("copy_previous", {})
        if zero_pred:
            b_horizons = sorted(zero_pred.keys())
            header = "| Baseline | " + " | ".join(f"h={h}" for h in b_horizons) + " |"
            sep = "|----------|" + "|".join("---" for _ in b_horizons) + "|"
            lines.append(header)
            lines.append(sep)
            # Report mean across 6 dims for each horizon — gives a single
            # summary number comparable to mean kinematics MSE from Layer 3.
            zero_vals = " | ".join(
                f"{np.mean(zero_pred[h]):.6f}" for h in b_horizons)
            copy_vals = " | ".join(
                f"{np.mean(copy_prev[h]):.6f}" for h in b_horizons)
            lines.append(f"| Zero predictor | {zero_vals} |")
            lines.append(f"| Copy previous | {copy_vals} |")

    # Layer 4 — action conditioning.
    # IMPORTANT: per-dim kinematics ablation is the primary metric.
    # z-space aggregate MSE is misleading because it includes appearance
    # dims (6-63) that the state head doesn't read. A model can show
    # large z-space action sensitivity while having identical kinematic
    # response — the "improvement" is in how things look, not how
    # physics works. (See Finding 08 correction.)
    lines.append("\n## Layer 4: Action Conditioning\n")

    abl = results.get("layer4_ablation", {})

    # Per-dim kinematics ablation FIRST — this is the metric that matters
    # for physics evaluation. Shows which kinematic dimensions respond to
    # action changes (vy should respond to thrust, vx to side thrust).
    if "kin_real_vs_zero" in abl:
        lines.append("### Per-dim Kinematics: Real vs Zero Actions (primary metric)\n")
        kin_horizons = sorted(abl["kin_real_vs_zero"].keys())
        header = "| Dim | " + " | ".join(f"h={h}" for h in kin_horizons) + " |"
        sep = "|-----|" + "|".join("---" for _ in kin_horizons) + "|"
        lines.append(header)
        lines.append(sep)
        for dim in _KIN_DIMS:
            vals = " | ".join(
                f"{abl['kin_real_vs_zero'][h].get(dim, float('nan')):.6f}" for h in kin_horizons)
            lines.append(f"| {dim} | {vals} |")

    if "kin_real_vs_random" in abl:
        lines.append("\n### Per-dim Kinematics: Real vs Random Actions\n")
        kin_horizons = sorted(abl["kin_real_vs_random"].keys())
        header = "| Dim | " + " | ".join(f"h={h}" for h in kin_horizons) + " |"
        sep = "|-----|" + "|".join("---" for _ in kin_horizons) + "|"
        lines.append(header)
        lines.append(sep)
        for dim in _KIN_DIMS:
            vals = " | ".join(
                f"{abl['kin_real_vs_random'][h].get(dim, float('nan')):.6f}" for h in kin_horizons)
            lines.append(f"| {dim} | {vals} |")

    # z-space ablation — secondary, with caveat
    if "mse_real_vs_zero" in abl:
        lines.append("\n### Action Ablation (z-space — includes appearance dims, interpret with caution)\n")
        lines.append("*Caveat: z-space MSE includes appearance dims (6-63) not read by state head. "
                      "A model can show large z-space sensitivity with identical kinematic response. "
                      "Use per-dim kinematics above for physics claims.*\n")
        lines.append("| Horizon | MSE(real vs zero) | MSE(real vs random) |")
        lines.append("|---------|-------------------|---------------------|")
        for h in sorted(abl["mse_real_vs_zero"].keys()):
            rz = abl["mse_real_vs_zero"][h]
            rr = abl["mse_real_vs_random"].get(h, float("nan"))
            lines.append(f"| {h} | {rz:.6f} | {rr:.6f} |")

    # Action sensitivity — also z-space, secondary
    sens = results.get("layer4_sensitivity", {})
    if "latent_l2_matrix" in sens:
        lines.append("\n### Action Sensitivity (L2 in z-space — same caveat as above)\n")
        for pair, l2 in sens["latent_l2_matrix"].items():
            lines.append(f"- {pair}: {l2:.6f}")

    return "\n".join(lines)


def compute_consistency_r2(oracle_results: dict) -> dict:
    """Consistency check: R² of extracted constants vs irrelevant state variables.

    For each constant with n_samples >= 10, fits R² of model_values against
    each of the 6 kinematic state variables from associated_states. High R²
    means the "constant" varies with state — spurious dependency.

    Returns: dict of constant_name -> dict of state_var_name -> R² float.
    """
    result = {}
    for const_name, const_data in oracle_results.items():
        if const_data.get("n_samples", 0) < 10:
            continue  # too few samples for meaningful R²
        model_vals = const_data.get("model_values")
        states = const_data.get("associated_states")
        if model_vals is None or states is None:
            continue
        if isinstance(model_vals, list):
            model_vals = np.array(model_vals)
        if isinstance(states, list):
            states = np.array(states)
        r2_per_var = {}
        for d, dim_name in enumerate(_KIN_DIMS):
            # Total variance of the extracted constant values
            ss_tot = ((model_vals - model_vals.mean()) ** 2).sum()
            if ss_tot < 1e-12:
                continue  # constant has zero variance — R² undefined
            # Simple linear R²: how much of model_vals variance is
            # explained by linear correlation with state dimension d?
            x = states[:, d]
            slope = np.cov(model_vals, x)[0, 1] / max(np.var(x), 1e-12)
            pred = slope * (x - x.mean()) + model_vals.mean()
            ss_res = ((model_vals - pred) ** 2).sum()
            r2_per_var[dim_name] = float(1.0 - ss_res / ss_tot)
        result[const_name] = r2_per_var
    return result
