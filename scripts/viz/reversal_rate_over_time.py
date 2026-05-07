"""When along the trajectory does control diverge between blind and labeled?

For each episode, compute reversal events (main on/off and lateral L/R/none
changes), bin them into N normalized-time buckets (0=start, 1=end), and
average the per-bucket rate across episodes per condition. Plot blind vs
labeled curves to see if the gap is concentrated early, late, or uniform.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_actions(traj_dir: Path, n=100, outcome=None):
    eps = []
    for p in sorted(traj_dir.glob("episode_*.npz")):
        d = np.load(p, allow_pickle=True)
        meta = json.loads(d["metadata_json"].item())
        if outcome is None or meta["outcome"] == outcome:
            eps.append((d["actions"], meta["outcome"], meta["episode_length"]))
            if len(eps) >= n:
                break
    return eps


def reversal_mask(actions, kind="any"):
    main_on = (actions[:, 0] > 0).astype(int)
    lat = np.zeros(len(actions), dtype=int)
    lat[actions[:, 1] < -0.5] = -1
    lat[actions[:, 1] > 0.5] = 1
    main_change = np.diff(main_on) != 0
    lat_change = np.diff(lat) != 0
    rev = np.zeros(len(actions), dtype=bool)
    if kind == "main":
        rev[1:] = main_change
    elif kind == "lateral":
        rev[1:] = lat_change
    else:
        rev[1:] = main_change | lat_change
    return rev


def binned_rate(eps, n_bins=20, kind="any"):
    """Returns (mean_rate, sem) per bin across eps. Uses normalized time."""
    rates = np.zeros((len(eps), n_bins))
    for i, (a, _, _) in enumerate(eps):
        T = len(a)
        if T < n_bins:
            continue
        rev = reversal_mask(a, kind=kind)
        # Map step index -> bin
        bin_ids = (np.arange(T) * n_bins // T).clip(0, n_bins - 1)
        for b in range(n_bins):
            in_bin = bin_ids == b
            if in_bin.any():
                rates[i, b] = rev[in_bin].mean()
    mean = rates.mean(axis=0)
    sem = rates.std(axis=0) / np.sqrt(len(eps))
    return mean, sem


def main():
    base = Path("/media/hdd1/physics-priors-latent-space/lunar-lander-networks/rl_agents/full-variation")
    out_dir = Path.home() / "vsr-tmp/e1-overlay"
    scenarios = [
        ("easy-ID",
         base / "blind-ppo-easy-128-lowent-Feb122026/trajectories",
         base / "labeled-ppo-easy-128-lowent-Feb122026/trajectories"),
        ("easy-OOD-medium",
         base / "blind-ppo-easy-128-lowent-Feb122026/trajectories-ood-medium",
         base / "labeled-ppo-easy-128-lowent-Feb122026/trajectories-ood-medium"),
        ("medium-ID",
         base / "blind-ppo-medium-128-lowent-Feb122026/trajectories",
         base / "labeled-ppo-medium-128-lowent-Feb122026/trajectories"),
    ]

    n_bins = 20
    fig, axes = plt.subplots(len(scenarios), 3, figsize=(15, 3.3 * len(scenarios)),
                             dpi=150, constrained_layout=True, sharex=True)
    x = (np.arange(n_bins) + 0.5) / n_bins
    for row, (tag, bd, ld) in enumerate(scenarios):
        blind = load_actions(bd, n=100)
        labeled = load_actions(ld, n=100)
        for col, kind in enumerate(["any", "main", "lateral"]):
            ax = axes[row, col]
            for eps, name, color in [(blind, "Blind", "#ff7f0e"),
                                      (labeled, "Labeled", "#1f77b4")]:
                m, s = binned_rate(eps, n_bins=n_bins, kind=kind)
                ax.plot(x, m, color=color, linewidth=2, label=name)
                ax.fill_between(x, m - s, m + s, color=color, alpha=0.2)
            if row == 0:
                ax.set_title(f"{kind} reversals", fontsize=11)
            if col == 0:
                ax.set_ylabel(f"{tag}\nreversals / step", fontsize=10)
            if row == len(scenarios) - 1:
                ax.set_xlabel("normalized time within episode  (0 = start, 1 = end)")
            ax.grid(alpha=0.25)
            if row == 0 and col == 0:
                ax.legend(frameon=False, fontsize=9)
    fig.suptitle("Where in the trajectory do blind and labeled diverge?",
                 fontsize=13, fontweight="bold")
    out = out_dir / "reversal_rate_over_time.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
