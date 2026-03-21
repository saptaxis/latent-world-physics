#!/usr/bin/env python
"""Generate dream comparison videos: GT vs model predictions.

Produces MP4 videos with GT (left) and dream (right) side-by-side.
Supports full autoregressive dreaming and periodic re-grounding
(re-encode real frame every K steps).

Usage:
    # Full episodes + re-grounded at K=5,10,20
    python scripts/dream_compare.py \
        --vae-checkpoint /path/to/vae/best.pt \
        --dynamics-checkpoint /path/to/dynamics/best.pt \
        --data-path /path/to/episodes \
        --output-dir ~/vsr-tmp/dreams \
        --reground 5 10 20 0 \
        --n-episodes 5 --policies heuristic random

    # Just full dreams, no re-grounding
    python scripts/dream_compare.py \
        --vae-checkpoint /path/to/vae/best.pt \
        --dynamics-checkpoint /path/to/dynamics/best.pt \
        --data-path /path/to/episodes \
        --output-dir ~/vsr-tmp/dreams \
        --n-episodes 3
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import imageio
import numpy as np
import torch

# Add the project root to sys.path so this script can be run directly
# (e.g., `python scripts/dream_compare.py`) without requiring a package install.

from lwp.models.pixel_vae import PixelVAE
from lwp.models.pixel_dynamics import LatentDynamicsModel
from lwp.models.pixel_world_model import PixelWorldModel


def load_pixel_world_model(vae_path: str, dyn_path: str,
                           device: str) -> PixelWorldModel:
    """Load trained PixelWorldModel from VAE + dynamics checkpoints.

    Dispatches on model_type in both VAE and dynamics configs to support
    standard PixelVAE, FactoredPixelVAE, and all dynamics variants.
    """
    # --- VAE ---
    vae_ckpt = torch.load(vae_path, map_location=device, weights_only=False)
    cfg = vae_ckpt["config"]
    vae_model_type = cfg.get("model_type", "standard")

    if vae_model_type == "factored":
        from lwp.models.factored_pixel_vae import FactoredPixelVAE
        vae = FactoredPixelVAE(
            in_channels=cfg["in_channels"],
            latent_dim=cfg["latent_dim"],
            frame_size=cfg["frame_size"],
            channels=cfg.get("channels", [32, 64, 128, 256]),
            kin_targets=cfg.get("kin_targets", [0, 1, 2, 3, 4, 5]),
            decoder_type=cfg.get("decoder_type", "concat"),
            coord_conv=cfg.get("coord_conv", False),
        )
    else:
        vae = PixelVAE(
            in_channels=cfg["in_channels"],
            latent_dim=cfg["latent_dim"],
            frame_size=cfg["frame_size"],
            channels=cfg.get("channels", [32, 64, 128, 256]),
            state_dim=cfg.get("state_dim", 0),
            coord_conv=cfg.get("coord_conv", False),
        )
    vae.load_state_dict(vae_ckpt["model_state_dict"])

    # --- Dynamics ---
    dyn_ckpt = torch.load(dyn_path, map_location=device, weights_only=False)
    dyn_cfg = dyn_ckpt["config"]
    model_type = dyn_cfg.get("model_type", "gru")
    if model_type == "rssm":
        from lwp.models.pixel_rssm import LatentRSSM
        dynamics = LatentRSSM(
            latent_dim=cfg["latent_dim"],
            action_dim=dyn_cfg.get("action_dim", 2),
            deter_dim=dyn_cfg.get("deter_dim", 200),
            stoch_dim=dyn_cfg.get("stoch_dim", 30),
            hidden_dim=dyn_cfg.get("hidden_size", 200),
        )
    elif model_type == "film":
        from lwp.models.pixel_dynamics import FiLMDynamicsModel
        dynamics = FiLMDynamicsModel(
            latent_dim=cfg["latent_dim"],
            action_dim=dyn_cfg.get("action_dim", 2),
            hidden_size=dyn_cfg.get("hidden_size", 256),
        )
    elif model_type == "factored-dyn":
        from lwp.models.factored_dynamics import FactoredDynamicsModel
        dynamics = FactoredDynamicsModel(
            latent_dim=cfg["latent_dim"],
            action_dim=dyn_cfg.get("action_dim", 2),
            hidden_size=dyn_cfg.get("hidden_size", 256),
            kin_dims=dyn_cfg.get("kin_dims", 6),
        )
    else:
        dynamics = LatentDynamicsModel(
            latent_dim=cfg["latent_dim"],
            action_dim=dyn_cfg.get("action_dim", 2),
            hidden_size=dyn_cfg.get("hidden_size", 256),
        )
    dynamics.load_state_dict(dyn_ckpt["model_state_dict"])

    model = PixelWorldModel(vae, dynamics)
    model.to(device)
    model.eval()
    return model, cfg


def preprocess_episode(raw_frames: np.ndarray, frame_size: int,
                       grayscale: bool = True) -> np.ndarray:
    """Preprocess all frames from an episode. Returns (T+1, H, W) uint8.

    Duplicates the preprocessing logic from data/pixel_dataset.py so this
    script can run standalone without importing the dataset module. Must
    stay in sync with _preprocess_frame() there.
    """
    processed = []
    for i in range(len(raw_frames)):
        frame = raw_frames[i]
        if grayscale and frame.ndim == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # INTER_AREA for downscaling to match training preprocessing
        frame = cv2.resize(frame, (frame_size, frame_size),
                           interpolation=cv2.INTER_AREA)
        processed.append(frame)
    return np.stack(processed)


def dream_with_regrounding(model: PixelWorldModel, gt_frames: np.ndarray,
                           actions: np.ndarray, reground_every: int
                           ) -> np.ndarray:
    """Dream with optional periodic re-encoding of real frames.

    Args:
        model: trained PixelWorldModel
        gt_frames: (T+1, H, W) preprocessed uint8 frames
        actions: (T, action_dim) float32 actions
        reground_every: re-encode real frame every K steps (0 = no re-grounding)

    Returns:
        dreamed: (T+1, H, W) uint8 frames
    """
    n = len(actions)
    dreamed = []

    # Encode the first real frame into latent space as the dream seed.
    # Shape: (1, latent_dim) — unsqueeze adds batch and channel dims for
    # the VAE encoder which expects (B, C, H, W).
    # Move input tensors to same device as model — model.to(device) only
    # moves parameters, not the data we feed in.
    device = next(model.parameters()).device
    seed_tensor = torch.from_numpy(gt_frames[0]).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0
    z = model.vae.encode(seed_tensor)
    hidden = None  # GRU hidden state starts fresh for each dream

    # Decode the seed latent to get frame 0 of the dream — this shows how
    # much information the VAE retains (useful baseline for the comparison).
    decoded = model.vae.decode(z).squeeze().detach().cpu().numpy()
    dreamed.append((decoded * 255).clip(0, 255).astype(np.uint8))

    for t in range(n):
        action = torch.from_numpy(actions[t]).float().unsqueeze(0).to(device)

        # Re-grounding: periodically replace the predicted latent with the
        # real frame's encoding. This bounds error accumulation and lets us
        # measure how fast the dream diverges as a function of K.
        # hidden=None resets the GRU state since the re-grounded latent may
        # be inconsistent with the accumulated hidden state.
        if reground_every > 0 and t > 0 and t % reground_every == 0:
            real_tensor = torch.from_numpy(gt_frames[t]).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0
            z = model.vae.encode(real_tensor)
            hidden = None

        with torch.no_grad():
            # Step the dynamics model one timestep forward in latent space
            z_next, hidden = model.dynamics(z, action, hidden)
            # Decode predicted latent to pixel space for the output video
            frame = model.vae.decode(z_next).squeeze().detach().cpu().numpy()
            # Feed the predicted latent forward (autoregressive rollout)
            z = z_next

        dreamed.append((frame * 255).clip(0, 255).astype(np.uint8))

    return np.stack(dreamed)


def save_comparison_mp4(gt: np.ndarray, dreamed: np.ndarray,
                        path: str, fps: int = 10):
    """Save GT | Dream side-by-side MP4."""
    T = min(len(gt), len(dreamed))
    # Concatenate along width (axis=2) to produce [GT | Dream] layout.
    # The viewer sees real and predicted frames at the same timestamp
    # side-by-side, making divergence easy to spot visually.
    combined = np.concatenate([gt[:T], dreamed[:T]], axis=2)
    # Replicate grayscale across RGB channels — H.264 requires 3-channel input
    frames_rgb = np.stack([combined] * 3, axis=-1)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # macro_block_size=1 avoids the default 16-pixel padding that libx264
    # adds when frame dimensions aren't multiples of 16 (84*2=168 is not
    # a multiple of 16, so without this the video would have black bars).
    writer = imageio.get_writer(path, fps=fps, codec="libx264",
                                pixelformat="yuv420p",
                                macro_block_size=1)
    for f in frames_rgb:
        writer.append_data(f)
    writer.close()


def discover_policies(data_path: str) -> list[str]:
    """Auto-discover policy/trajectory subdirs that contain .npz episodes.

    Scans immediate subdirs of data_path for any that contain episode .npz
    files. Skips non-episode dirs (cache/, __pycache__/, etc.) by checking
    for actual .npz files inside.
    """
    policies = []
    root = Path(data_path)
    for subdir in sorted(root.iterdir()):
        if not subdir.is_dir():
            continue
        # Skip known non-episode dirs
        if subdir.name in ("cache", "__pycache__", "prepared-npy"):
            continue
        # Check if this dir actually contains episode npz files
        npz_files = list(subdir.glob("episode_*.npz"))
        if npz_files:
            policies.append(subdir.name)
    return policies


def find_episodes(data_path: str, policy: str, n_episodes: int) -> list[Path]:
    """Find episode .npz files for a given policy subdirectory."""
    policy_dir = Path(data_path) / policy
    if not policy_dir.exists():
        return []
    files = sorted(policy_dir.glob("episode_*.npz"))
    # Sample evenly across the episode range rather than taking the first N.
    # This gives a more representative cross-section of episode diversity
    # (early episodes may be short crashes, later ones longer flights).
    if len(files) <= n_episodes:
        return files
    step = len(files) // n_episodes
    return [files[i * step] for i in range(n_episodes)]


def main():
    parser = argparse.ArgumentParser(description="Generate dream comparison videos")
    parser.add_argument("--vae-checkpoint", type=str, required=True)
    parser.add_argument("--dynamics-checkpoint", type=str, required=True)
    parser.add_argument("--data-path", type=str, required=True,
                        help="Directory with policy subdirs (heuristic/, random/)")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--policies", type=str, nargs="+",
                        default=None,
                        help="Policy subdirectories to use. If not specified, "
                             "auto-discovers all subdirs containing .npz episodes.")
    parser.add_argument("--n-episodes", type=int, default=5,
                        help="Number of episodes per policy")
    parser.add_argument("--reground", type=int, nargs="+", default=[0],
                        help="Re-grounding intervals. 0 = no re-grounding (full dream). "
                             "Example: --reground 5 10 20 0")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    print(f"Loading model...")
    model, vae_cfg = load_pixel_world_model(
        args.vae_checkpoint, args.dynamics_checkpoint, args.device)
    # Pull preprocessing config from the VAE so dream frames match training
    frame_size = vae_cfg["frame_size"]
    grayscale = vae_cfg.get("in_channels", 1) == 1
    print(f"  frame_size={frame_size}, grayscale={grayscale}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Auto-discover policies if none specified — finds all subdirs with .npz files
    policies = args.policies or discover_policies(args.data_path)
    print(f"  Policies: {policies} ({args.n_episodes} episodes each)")

    total_videos = 0
    for policy in policies:
        episodes = find_episodes(args.data_path, policy, args.n_episodes)
        if not episodes:
            print(f"No episodes found for policy '{policy}' in {args.data_path}")
            continue

        print(f"\n{policy}: {len(episodes)} episodes")
        for ep_path in episodes:
            ep_name = ep_path.stem  # e.g., episode_00010
            data = np.load(str(ep_path), allow_pickle=True)
            raw_frames = data["rgb_frames"]
            actions = data["actions"]
            n_steps = len(actions)

            # Preprocess GT frames the same way the VAE saw them during training
            gt = preprocess_episode(raw_frames[:n_steps + 1], frame_size, grayscale)

            # Generate one video per re-grounding interval. K=0 means a fully
            # autoregressive dream (no re-grounding), which shows worst-case
            # error accumulation. K>0 re-encodes the real frame every K steps.
            for K in args.reground:
                tag = f"k{K}" if K > 0 else "full_dream"
                dreamed = dream_with_regrounding(model, gt, actions, reground_every=K)
                fname = f"{policy}_{ep_name}_{n_steps}steps_{tag}.mp4"
                save_comparison_mp4(gt, dreamed, str(out_dir / fname), fps=args.fps)
                total_videos += 1

            print(f"  {ep_name}: {n_steps} steps, {len(args.reground)} videos")

    print(f"\nDone. {total_videos} videos saved to {out_dir}")


if __name__ == "__main__":
    main()
