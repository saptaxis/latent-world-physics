#!/usr/bin/env python3
"""Interactive play with a pixel world model.

Two modes:
  teacher: Real env runs alongside. Model predicts 1 step from real frame.
           Shows side-by-side: Real | Predicted.
  dream:   No real env after seed. Model feeds own predictions.
           Shows only the model's imagination.

Controls:
  Arrow keys / WASD for thrust. ESC to quit. R to reset.

Usage:
    python lunar_lander/scripts/interactive_play.py \
        --vae-checkpoint path/to/vae/best.pt \
        --dynamics-checkpoint path/to/dynamics/best.pt \
        --mode teacher
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np
import torch


from lwp.models.pixel_vae import PixelVAE
from lwp.models.pixel_dynamics import LatentDynamicsModel
from lwp.models.pixel_world_model import PixelWorldModel


def load_pixel_world_model(vae_ckpt_path: str, dyn_ckpt_path: str,
                           device: str) -> PixelWorldModel:
    """Load a trained PixelWorldModel from VAE + dynamics checkpoints."""
    vae_ckpt = torch.load(vae_ckpt_path, map_location=device, weights_only=False)
    vae_cfg = vae_ckpt["config"]
    vae = PixelVAE(
        in_channels=vae_cfg["in_channels"],
        latent_dim=vae_cfg["latent_dim"],
        frame_size=vae_cfg["frame_size"],
        channels=vae_cfg.get("channels", [32, 64, 128, 256]),
    )
    vae.load_state_dict(vae_ckpt["model_state_dict"])

    dyn_ckpt = torch.load(dyn_ckpt_path, map_location=device, weights_only=False)
    dyn_cfg = dyn_ckpt["config"]
    dynamics = LatentDynamicsModel(
        latent_dim=vae_cfg["latent_dim"],
        action_dim=dyn_cfg.get("action_dim", 2),
        hidden_size=dyn_cfg.get("hidden_size", 256),
    )
    dynamics.load_state_dict(dyn_ckpt["model_state_dict"])

    model = PixelWorldModel(vae, dynamics)
    model.to(device)
    model.eval()
    return model


def preprocess_frame(frame: np.ndarray, frame_size: int,
                     grayscale: bool = True) -> torch.Tensor:
    """Convert env render (H, W, 3) to model input (1, C, H, W)."""
    if grayscale:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (frame_size, frame_size),
                       interpolation=cv2.INTER_AREA)
    if grayscale:
        t = torch.from_numpy(frame).float().unsqueeze(0).unsqueeze(0) / 255.0
    else:
        t = torch.from_numpy(frame).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    return t


def frame_to_display(tensor_frame: torch.Tensor, display_size: int = 256) -> np.ndarray:
    """Convert (1, C, H, W) tensor to displayable (display_size, display_size) uint8."""
    if tensor_frame.dim() == 4:
        tensor_frame = tensor_frame.squeeze(0)
    if tensor_frame.size(0) == 1:
        img = (tensor_frame.squeeze(0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    else:
        img = (tensor_frame[-1].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return cv2.resize(img, (display_size, display_size),
                      interpolation=cv2.INTER_NEAREST)


def get_action_from_keys(key: int) -> np.ndarray:
    """Map keyboard input to continuous (main_thrust, side_thrust) action."""
    main = 0.0
    side = 0.0
    if key == ord('w') or key == 82:
        main = 1.0
    if key == ord('a') or key == 81:
        side = -1.0
    if key == ord('d') or key == 83:
        side = 1.0
    return np.array([main, side], dtype=np.float32)


def run_teacher_forced(model: PixelWorldModel, device: str,
                       frame_size: int, display_size: int = 256):
    """Teacher-forced interactive: real env + model side-by-side."""
    env = gym.make("LunarLander-v3", continuous=True, render_mode="rgb_array")
    obs, info = env.reset()

    hidden = None
    print("Teacher-forced mode. WASD/arrows to control. ESC to quit. R to reset.")

    while True:
        real_frame = env.render()
        frame_tensor = preprocess_frame(real_frame, frame_size).to(device)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key == ord('r'):
            obs, info = env.reset()
            hidden = None
            continue

        action = get_action_from_keys(key)
        action_tensor = torch.from_numpy(action).unsqueeze(0).to(device)

        with torch.no_grad():
            pred_frame, _, hidden = model.predict_next(
                frame_tensor, action_tensor, hidden)

        real_disp = frame_to_display(frame_tensor, display_size)
        pred_disp = frame_to_display(pred_frame, display_size)
        combined = np.concatenate([real_disp, pred_disp], axis=1)

        cv2.putText(combined, "Real", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
        cv2.putText(combined, "Predicted", (display_size + 10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)

        cv2.imshow("Pixel World Model - Teacher Forced", combined)

        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
            hidden = None

        time.sleep(0.02)

    env.close()
    cv2.destroyAllWindows()


def run_dream(model: PixelWorldModel, device: str,
              frame_size: int, display_size: int = 256):
    """Fully dreamed interactive: model feeds own predictions."""
    env = gym.make("LunarLander-v3", continuous=True, render_mode="rgb_array")
    obs, info = env.reset()

    seed_frame = env.render()
    frame_tensor = preprocess_frame(seed_frame, frame_size).to(device)

    with torch.no_grad():
        z = model.vae.encode(frame_tensor)
    hidden = None

    print("Dream mode. WASD/arrows to control. ESC to quit. R to reset.")

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key == ord('r'):
            obs, info = env.reset()
            seed_frame = env.render()
            frame_tensor = preprocess_frame(seed_frame, frame_size).to(device)
            with torch.no_grad():
                z = model.vae.encode(frame_tensor)
            hidden = None
            continue

        action = get_action_from_keys(key)
        action_tensor = torch.from_numpy(action).unsqueeze(0).to(device)

        with torch.no_grad():
            z_next, hidden = model.dynamics(z, action_tensor, hidden)
            pred_frame = model.vae.decode(z_next)
            z = z_next

        disp = frame_to_display(pred_frame, display_size)
        cv2.putText(disp, "Dream", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
        cv2.imshow("Pixel World Model - Dream", disp)

        time.sleep(0.02)

    env.close()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Interactive pixel world model play")
    parser.add_argument("--vae-checkpoint", type=str, required=True)
    parser.add_argument("--dynamics-checkpoint", type=str, required=True)
    parser.add_argument("--mode", type=str, default="teacher",
                        choices=["teacher", "dream"])
    parser.add_argument("--frame-size", type=int, default=84)
    parser.add_argument("--display-size", type=int, default=256)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model = load_pixel_world_model(
        args.vae_checkpoint, args.dynamics_checkpoint, args.device)
    print(f"Model loaded on {args.device}")

    if args.mode == "teacher":
        run_teacher_forced(model, args.device, args.frame_size, args.display_size)
    else:
        run_dream(model, args.device, args.frame_size, args.display_size)


if __name__ == "__main__":
    main()
