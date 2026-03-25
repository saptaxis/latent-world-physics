#!/usr/bin/env python
"""Generate VAE reconstruction gallery images.

Produces grid images showing VAE reconstruction quality. For compositional
STN VAE, also shows decomposition: canonical lander patch, alpha mask,
warped lander, and background.

Each row is one sample: GT | Recon | O_hat | A_hat | O_warp | A_warp | B_hat
For standard/factored VAE: GT | Recon (2 columns only).

Usage:
    # Compositional VAE — full decomposition gallery
    python scripts/world_models/vae_gallery.py \
        --checkpoint /path/to/vae/best.pt \
        --data-path /path/to/episodes \
        --output-dir ~/vsr-tmp/vae-gallery \
        --n-samples 16

    # Standard VAE — reconstruction only
    python scripts/world_models/vae_gallery.py \
        --checkpoint /path/to/vae/best.pt \
        --data-path /path/to/episodes \
        --output-dir ~/vsr-tmp/vae-gallery
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image

# Add project root for script imports
_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))


def load_vae(checkpoint_path: str, device: str = "cpu"):
    """Load VAE from checkpoint, dispatching on model_type in config."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    model_type = config.get("model_type", "standard")

    if model_type == "compositional":
        from lwp.models.compositional_vae import CompositionalPixelVAE
        vae = CompositionalPixelVAE(
            in_channels=config.get("in_channels", 1),
            latent_dim=config["latent_dim"],
            frame_size=config["frame_size"],
            latent_mode=config.get("latent_mode", "flat"),
            bg_dim=config.get("bg_dim", 8),
            obj_dim=config.get("obj_dim", 32),
            canonical_size=config.get("canonical_size", 16),
            beta=config.get("beta", 0.0001),
        )
    elif model_type == "factored":
        from lwp.models.factored_pixel_vae import FactoredPixelVAE
        kin_targets = config.get("kin_targets", [0, 1, 2, 3, 4, 5])
        vae = FactoredPixelVAE(
            in_channels=config.get("in_channels", 1),
            latent_dim=config["latent_dim"],
            frame_size=config["frame_size"],
            kin_targets=kin_targets,
            decoder_type=config.get("decoder_type", "concat"),
            beta=config.get("beta", 0.0001),
        )
    else:
        from lwp.models.pixel_vae import PixelVAE
        vae = PixelVAE(
            in_channels=config.get("in_channels", 1),
            latent_dim=config["latent_dim"],
            frame_size=config["frame_size"],
            beta=config.get("beta", 0.0001),
            state_dim=config.get("state_dim", 0),
            coord_conv=config.get("coord_conv", False),
        )

    vae.load_state_dict(ckpt["model_state_dict"])
    vae.to(device)
    vae.eval()
    return vae, config


def load_frames(data_path: str, n_samples: int, frame_size: int = 84,
                grayscale: bool = True, device: str = "cpu"):
    """Load random frames from episode npz files."""
    import glob
    npz_files = sorted(glob.glob(f"{data_path}/**/*.npz", recursive=True))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {data_path}")

    frames = []
    states = []
    rng = np.random.default_rng(42)

    while len(frames) < n_samples:
        # Pick random episode, random frame
        npz_path = rng.choice(npz_files)
        data = np.load(npz_path)
        rgb = data["rgb_frames"]  # (T+1, H, W, 3) uint8
        n_frames = rgb.shape[0]
        idx = rng.integers(0, n_frames)

        frame = rgb[idx]  # (H, W, 3) uint8
        if grayscale:
            frame = np.mean(frame, axis=-1, keepdims=True)  # (H, W, 1)

        # Resize if needed
        frame = frame.astype(np.float32) / 255.0
        frame_t = torch.from_numpy(frame).permute(2, 0, 1)  # (C, H, W)
        if frame_t.shape[-1] != frame_size:
            frame_t = F.interpolate(
                frame_t.unsqueeze(0), size=frame_size, mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        frames.append(frame_t)

        # Load state if available
        if "states" in data:
            states.append(data["states"][idx])
        else:
            states.append(None)

    frames = torch.stack(frames).to(device)  # (N, C, H, W)
    return frames, states


def upscale_patch(patch: torch.Tensor, target_size: int) -> torch.Tensor:
    """Upscale a small patch to target_size for grid alignment.

    Uses nearest-neighbor to keep the patch crisp (no interpolation blur).
    """
    return F.interpolate(patch, size=target_size, mode="nearest")


def generate_compositional_gallery(vae, frames, output_dir: Path, tag: str = "gallery",
                                    rows_per_page: int = 5):
    """Generate decomposition gallery for compositional VAE.

    Each row: GT | Recon | O_hat | A_hat | O_warp | A_warp | B_hat
    Max rows_per_page rows per image — multiple pages generated if n_samples > rows_per_page.
    """
    fs = vae.frame_size
    n = frames.shape[0]

    with torch.no_grad():
        z = vae.encode(frames)
        decomp = vae.decode_decomposed(z)

    # Upscale canonical patches to frame size for grid alignment
    O_hat_up = upscale_patch(decomp["O_hat"], fs)
    A_hat_up = upscale_patch(decomp["A_hat"], fs)

    # All columns for the decomposition grid
    columns = [
        frames,              # GT
        decomp["x_hat"],     # composited reconstruction
        O_hat_up,            # canonical lander (upscaled)
        A_hat_up,            # alpha mask (upscaled)
        decomp["O_warp"],    # warped lander in frame
        decomp["A_warp"],    # warped mask in frame
        decomp["B_hat"],     # background
    ]
    col_labels = ["GT", "Recon", "O_hat", "A_hat", "O_warp", "A_warp", "B_hat"]

    # Generate pages of rows_per_page rows each
    n_pages = (n + rows_per_page - 1) // rows_per_page
    for page in range(n_pages):
        start = page * rows_per_page
        end = min(start + rows_per_page, n)

        all_images = []
        for i in range(start, end):
            for col in columns:
                all_images.append(col[i])

        grid = make_grid(all_images, nrow=len(columns), padding=2, pad_value=0.5)
        suffix = f"_p{page + 1}" if n_pages > 1 else ""
        save_image(grid, output_dir / f"{tag}_decomposition{suffix}.png")
        print(f"  Saved: {tag}_decomposition{suffix}.png  "
              f"(rows {start + 1}-{end}, columns: {', '.join(col_labels)})")

    # Canonical patches at native resolution — paginated too
    for page in range(n_pages):
        start = page * rows_per_page
        end = min(start + rows_per_page, n)
        patch_grid = make_grid(decomp["O_hat"][start:end], nrow=end - start, padding=1, pad_value=0.5)
        suffix = f"_p{page + 1}" if n_pages > 1 else ""
        save_image(patch_grid, output_dir / f"{tag}_canonical_patches{suffix}.png")

    # Masks at native resolution — paginated
    for page in range(n_pages):
        start = page * rows_per_page
        end = min(start + rows_per_page, n)
        mask_grid = make_grid(decomp["A_hat"][start:end], nrow=end - start, padding=1, pad_value=0.5)
        suffix = f"_p{page + 1}" if n_pages > 1 else ""
        save_image(mask_grid, output_dir / f"{tag}_masks{suffix}.png")

    # Print pose stats
    pose = decomp["pose_params"]  # (N, 5)
    print(f"\n  Pose stats (N={n}):")
    labels = ["tx", "ty", "sin", "cos", "scale"]
    for j, label in enumerate(labels):
        vals = pose[:, j]
        print(f"    {label:5s}: mean={vals.mean():.3f}  std={vals.std():.3f}  "
              f"min={vals.min():.3f}  max={vals.max():.3f}")

    # Mask area stats
    mask_areas = decomp["A_hat"].mean(dim=[1, 2, 3])
    print(f"\n  Mask area: mean={mask_areas.mean():.3f}  std={mask_areas.std():.3f}")


def generate_standard_gallery(vae, frames, output_dir: Path, tag: str = "gallery",
                               rows_per_page: int = 5):
    """Generate reconstruction gallery for standard/factored VAE.

    Each row: GT | Recon. Max rows_per_page rows per image.
    """
    with torch.no_grad():
        z = vae.encode(frames)
        recon = vae.decode(z)

    n = frames.shape[0]
    n_pages = (n + rows_per_page - 1) // rows_per_page
    for page in range(n_pages):
        start = page * rows_per_page
        end = min(start + rows_per_page, n)

        all_images = []
        for i in range(start, end):
            all_images.append(frames[i])
            all_images.append(recon[i])

        grid = make_grid(all_images, nrow=2, padding=2, pad_value=0.5)
        suffix = f"_p{page + 1}" if n_pages > 1 else ""
        save_image(grid, output_dir / f"{tag}_recon{suffix}.png")
        print(f"  Saved: {tag}_recon{suffix}.png (rows {start + 1}-{end}, GT | Recon)")


def parse_args():
    p = argparse.ArgumentParser(description="VAE reconstruction/decomposition gallery")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to VAE checkpoint (.pt)")
    p.add_argument("--data-path", type=str, required=True,
                   help="Directory with .npz episode files")
    p.add_argument("--output-dir", type=str, required=True,
                   help="Directory to save gallery images")
    p.add_argument("--n-samples", type=int, default=16,
                   help="Number of samples to include in gallery")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--tag", type=str, default="gallery",
                   help="Prefix for output filenames")
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading VAE from {args.checkpoint}")
    vae, config = load_vae(args.checkpoint, args.device)
    model_type = config.get("model_type", "standard")
    print(f"  Model type: {model_type}")
    if model_type == "compositional":
        print(f"  Latent mode: {config.get('latent_mode', 'flat')}")
        print(f"  Canonical size: {config.get('canonical_size', 16)}")
        print(f"  Latent dim: {config['latent_dim']}")

    frame_size = config.get("frame_size", 84)
    print(f"\nLoading {args.n_samples} frames from {args.data_path}")
    frames, states = load_frames(
        args.data_path, args.n_samples,
        frame_size=frame_size, device=args.device,
    )
    print(f"  Loaded {frames.shape[0]} frames, shape {frames.shape}")

    print(f"\nGenerating gallery...")
    if model_type == "compositional":
        generate_compositional_gallery(vae, frames, output_dir, args.tag)
    else:
        generate_standard_gallery(vae, frames, output_dir, args.tag)

    print(f"\nDone. Output in {output_dir}")


if __name__ == "__main__":
    main()
