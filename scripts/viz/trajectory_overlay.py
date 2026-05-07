"""Static trajectory overlay viz: paths from N episodes drawn over a lunar
lander backdrop (last frame of one rgb episode), with color encoding time.

Prototype V1: E1 full-variation easy, blind vs labeled.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

# Env constants (parametric_lunar_lander/env.py)
VIEWPORT_W, VIEWPORT_H = 600, 400
SCALE = 30.0
W = VIEWPORT_W / SCALE          # 20.0
H = VIEWPORT_H / SCALE          # 13.333
HELIPAD_Y = H / 4               # 3.333
LEG_OFFSET = 18 / SCALE         # 0.6


def state_to_pixels(states: np.ndarray) -> np.ndarray:
    """state[:,0:2] (normalized) -> pixel (px, py) in image coords."""
    sx, sy = states[:, 0], states[:, 1]
    pos_x = (sx + 1.0) * (W / 2)                   # state_x = (pos.x - W/2)/(W/2)
    pos_y = sy * (H / 2) + (HELIPAD_Y + LEG_OFFSET)
    px = pos_x * SCALE
    py = VIEWPORT_H - pos_y * SCALE
    return np.stack([px, py], axis=1)


def load_episode(path: Path) -> dict:
    d = np.load(path, allow_pickle=True)
    meta = json.loads(d["metadata_json"].item())
    return {
        "states": d["states"],
        "actions": d["actions"],
        "outcome": meta["outcome"],
        "length": meta["episode_length"],
        "rgb_frames": d["rgb_frames"] if "rgb_frames" in d.files else None,
    }


def pick_episodes(traj_dir: Path, n: int, outcome: str | None = "landed"):
    """Return first n episode paths matching outcome filter."""
    eps = sorted(traj_dir.glob("episode_*.npz"))
    selected = []
    for p in eps:
        ep = load_episode(p)
        if outcome is None or ep["outcome"] == outcome:
            selected.append((p, ep))
            if len(selected) >= n:
                break
    return selected


def plot_paths(ax, episodes, cmap_name: str, label: str, t_max: int):
    """Plot each episode's trajectory as a time-colored line.

    Color is normalized by absolute time across all conditions (0..t_max),
    so longer episodes traverse a larger range of the colormap.
    """
    cmap = plt.get_cmap(cmap_name)
    for idx, (_, ep) in enumerate(episodes):
        pts = state_to_pixels(ep["states"])
        T = len(pts)
        if T < 2:
            continue
        segs = np.concatenate([pts[:-1, None, :], pts[1:, None, :]], axis=1)
        # Absolute time fraction within [0, t_max]
        abs_t = np.arange(T - 1) / max(t_max - 1, 1)
        colors = cmap(0.15 + 0.85 * abs_t)
        lw = 0.7 if len(episodes) > 20 else 1.6
        a = 0.35 if len(episodes) > 20 else 0.9
        lc = LineCollection(segs, colors=colors, linewidths=lw, alpha=a,
                            label=label if idx == 0 else None)
        ax.add_collection(lc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blind-dir", required=True, type=Path)
    ap.add_argument("--labeled-dir", required=True, type=Path)
    ap.add_argument("--rgb-dir", required=True, type=Path,
                    help="Dir with rgb_frames in npz; uses one for backdrop")
    ap.add_argument("--rgb-episode", default="episode_0000.npz")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--outcome", default="landed",
                    help="Filter episodes by outcome; 'any' for no filter")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--blind-cmap", default="Oranges")
    ap.add_argument("--labeled-cmap", default="Blues")
    args = ap.parse_args()

    outcome = None if args.outcome == "any" else args.outcome

    backdrop_ep = load_episode(args.rgb_dir / args.rgb_episode)
    if backdrop_ep["rgb_frames"] is None:
        raise SystemExit(f"No rgb_frames in {args.rgb_dir / args.rgb_episode}")
    last_frame = backdrop_ep["rgb_frames"][-1]

    blind_eps = pick_episodes(args.blind_dir, args.n, outcome)
    labeled_eps = pick_episodes(args.labeled_dir, args.n, outcome)
    print(f"blind: {len(blind_eps)} eps, lengths={[ep['length'] for _, ep in blind_eps]}")
    print(f"labeled: {len(labeled_eps)} eps, lengths={[ep['length'] for _, ep in labeled_eps]}")

    t_max = max(
        max((ep["length"] for _, ep in blind_eps), default=1),
        max((ep["length"] for _, ep in labeled_eps), default=1),
    )
    print(f"absolute t_max = {t_max} steps (used to normalize time-color)")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    for ax, eps, cmap, title in [
        (axes[0], blind_eps, args.blind_cmap, f"Blind (n={len(blind_eps)})"),
        (axes[1], labeled_eps, args.labeled_cmap, f"Labeled (n={len(labeled_eps)})"),
    ]:
        ax.imshow(last_frame, extent=[0, VIEWPORT_W, VIEWPORT_H, 0])
        plot_paths(ax, eps, cmap, title, t_max=t_max)
        ax.set_xlim(0, VIEWPORT_W)
        ax.set_ylim(VIEWPORT_H, 0)
        ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(
        f"E1 full-variation easy — trajectory overlay "
        f"(color = absolute time, dark = end, max={t_max} steps)"
    )
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
