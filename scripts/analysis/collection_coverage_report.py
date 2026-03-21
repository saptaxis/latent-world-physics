#!/usr/bin/env python3
"""Report coverage statistics for a primitive collection run.

Reads all .npz episodes in a collection directory (one subdir per config)
and reports:
  - Per-config: episode count, step length stats, initial state distributions
  - Cross-config: combined coverage of each state dimension
  - Plots: histograms, 2D scatter, and trajectory spread

Usage:
    # Report on a collection directory
    python lunar_lander/scripts/collection_coverage_report.py \
        ~/vsr-tmp/primitive-trial-v2/

    # Only specific subdirs
    python lunar_lander/scripts/collection_coverage_report.py \
        ~/vsr-tmp/primitive-trial-v2/ --configs free-fall impulse-main

    # Save plots to a specific directory
    python lunar_lander/scripts/collection_coverage_report.py \
        ~/vsr-tmp/primitive-trial-v2/ --plot-dir ~/vsr-tmp/coverage-plots/

    # Text-only (no plots)
    python lunar_lander/scripts/collection_coverage_report.py \
        ~/vsr-tmp/primitive-trial-v2/ --no-plots
"""

import argparse
import sys
from pathlib import Path

import numpy as np

DIM_NAMES = ["x", "y", "vx", "vy", "angle", "ang_vel", "left_leg", "right_leg"]
# First 6 are kinematic — the ones we care about for coverage.
KINEMATIC_DIMS = 6

# Meaningful state space ranges for Lunar Lander (normalized coords).
# Derived from the env's normalization math (env.py line 635-648):
#   x = (world_x - W/2) / (W/2)     → [-1, 1] is viewport, OOB terminates
#   y = (world_y - (helipad_y + LEG_DOWN/SCALE)) / (H/2)
#       ground ≈ -0.1, spawn ≈ 1.4, can't go much below -0.4
#   vx, vy scaled by viewport/FPS    → practical range from empirical data
#   angle in radians                  → rarely exceeds ±π in normal play
#   ang_vel scaled by 20/FPS         → practical range from empirical data
STATE_FULL_RANGE = {
    "x": (-1.0, 1.0),        # OOB terminates at ±1.0
    "y": (-0.4, 1.6),        # ground ≈ -0.1, spawn ≈ 1.4
    "vx": (-2.0, 2.0),       # agents/heuristic stay within ~±1.8
    "vy": (-2.0, 0.6),       # mostly downward; random can hit -2.5
    "angle": (-3.14, 3.14),  # ±π; rarely exceeds in normal trajectories
    "ang_vel": (-4.0, 4.0),  # agents up to ~±3.7
}


def load_collection(collection_dir: Path, config_names: list[str] | None = None):
    """Load initial states and metadata from a collection directory.

    Args:
        collection_dir: Root directory containing one subdir per config.
        config_names: If provided, only load these subdirs.

    Returns:
        Dict mapping config_name -> dict with keys:
            init_states: (N, state_dim) array of first-step states
            lengths: (N,) array of episode lengths
            n_episodes: int
    """
    results = {}
    subdirs = sorted(collection_dir.iterdir())

    for subdir in subdirs:
        if not subdir.is_dir():
            continue
        if config_names and subdir.name not in config_names:
            continue

        eps = sorted(subdir.glob("*.npz"))
        if not eps:
            # Skip dirs with no episodes (e.g. coverage/, logs/).
            continue

        init_states = []
        terminal_states = []
        all_states = []
        episodes = []  # list of per-episode state arrays (for line plots)
        lengths = []
        outcomes = []
        for ep_path in eps:
            data = np.load(str(ep_path), allow_pickle=False)
            states = data["states"]
            init_states.append(states[0])
            terminal_states.append(states[-1])
            all_states.append(states)
            episodes.append(states)
            lengths.append(len(states))
            # Infer outcome from terminal state.
            final = states[-1]
            if abs(final[0]) >= 0.95:  # OOB (x near ±1.0)
                outcomes.append("oob")
            elif abs(final[1]) < 0.15 and abs(final[3]) < 0.1:  # on ground, low vy
                outcomes.append("ground")
            elif len(states) >= data["actions"].shape[0]:  # max steps reached
                outcomes.append("timeout")
            else:
                outcomes.append("crash")

        results[subdir.name] = {
            "init_states": np.array(init_states),
            "terminal_states": np.array(terminal_states),
            "all_states": np.concatenate(all_states, axis=0),
            "episodes": episodes,
            "lengths": np.array(lengths),
            "outcomes": outcomes,
            "n_episodes": len(eps),
        }

    return results


def print_report(data: dict):
    """Print text coverage report to stdout."""
    total_episodes = 0
    total_steps = 0

    for config_name, info in sorted(data.items()):
        init = info["init_states"]
        traj = info["all_states"]
        lengths = info["lengths"]
        n = info["n_episodes"]
        total_episodes += n
        total_steps += lengths.sum()

        print(f"\n{'='*70}")
        print(f"{config_name}: {n} episodes, {len(traj):,} total steps, "
              f"length {lengths.mean():.0f} ± {lengths.std():.0f} "
              f"[{lengths.min()}-{lengths.max()}]")
        print(f"{'='*70}")

        print(f"\n  Initial states (where episodes start):")
        print(f"  {'dim':>8s}  {'min':>8s}  {'max':>8s}  "
              f"{'mean':>8s}  {'std':>8s}  {'p5':>8s}  {'p95':>8s}")
        print(f"  {'-'*8}  {'-'*8}  {'-'*8}  "
              f"{'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")
        for i in range(min(KINEMATIC_DIMS, init.shape[1])):
            vals = init[:, i]
            print(f"  {DIM_NAMES[i]:>8s}  {vals.min():+8.3f}  {vals.max():+8.3f}  "
                  f"{vals.mean():+8.3f}  {vals.std():8.3f}  "
                  f"{np.percentile(vals, 5):+8.3f}  {np.percentile(vals, 95):+8.3f}")

        print(f"\n  Trajectory coverage (all states visited):")
        print(f"  {'dim':>8s}  {'min':>8s}  {'max':>8s}  "
              f"{'mean':>8s}  {'std':>8s}  {'p5':>8s}  {'p95':>8s}")
        print(f"  {'-'*8}  {'-'*8}  {'-'*8}  "
              f"{'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")
        for i in range(min(KINEMATIC_DIMS, traj.shape[1])):
            vals = traj[:, i]
            print(f"  {DIM_NAMES[i]:>8s}  {vals.min():+8.3f}  {vals.max():+8.3f}  "
                  f"{vals.mean():+8.3f}  {vals.std():8.3f}  "
                  f"{np.percentile(vals, 5):+8.3f}  {np.percentile(vals, 95):+8.3f}")

        # Terminal state summary.
        outcomes = info["outcomes"]
        from collections import Counter
        counts = Counter(outcomes)
        parts = [f"{k}: {v} ({100*v/n:.0f}%)" for k, v in sorted(counts.items())]
        print(f"\n  Termination: {', '.join(parts)}")

        # Mean length by outcome.
        for outcome in sorted(counts.keys()):
            idxs = [j for j, o in enumerate(outcomes) if o == outcome]
            mean_len = float(lengths[idxs].mean())
            print(f"    {outcome:>8s}: mean length {mean_len:.0f} steps")

    print(f"\n{'='*70}")
    print(f"TOTAL: {total_episodes} episodes, {total_steps:,} steps "
          f"({total_steps / 50 / 60:.1f} min at 50 FPS)")
    print(f"{'='*70}")


def plot_coverage(data: dict, plot_dir: Path):
    """Generate coverage plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir.mkdir(parents=True, exist_ok=True)

    # Separate fresh_reset vs ground configs by checking if initial vy is
    # always zero (ground configs start from landed episodes).
    fresh_configs = []
    ground_configs = []
    for name, info in sorted(data.items()):
        vy_vals = info["init_states"][:, 3]
        vy_range = float(vy_vals.max() - vy_vals.min())
        if vy_range < 0.001:
            ground_configs.append(name)
        else:
            fresh_configs.append(name)

    # --- Histograms: fresh_reset configs ---
    if fresh_configs:
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        axes = axes.flatten()
        for i in range(KINEMATIC_DIMS):
            ax = axes[i]
            dim = DIM_NAMES[i]
            full_lo, full_hi = STATE_FULL_RANGE[dim]
            for cfg in fresh_configs:
                vals = data[cfg]["init_states"][:, i]
                ax.hist(vals, bins=30, alpha=0.5, label=cfg, density=True,
                        range=(full_lo, full_hi))
            ax.set_xlim(full_lo, full_hi)
            # Shade the full range lightly, highlight collected region.
            ax.axvspan(full_lo, full_hi, alpha=0.03, color="gray")
            ax.set_title(dim)
            ax.set_xlabel("value")
            if i == 0:
                ax.legend(fontsize=7)
        fig.suptitle("Fresh-reset: initial state distributions "
                     "(x-axis = full state space range)", fontsize=13)
        fig.tight_layout()
        fig.savefig(str(plot_dir / "coverage-fresh-histograms.png"), dpi=120)
        plt.close(fig)

    # --- Histograms: ground configs ---
    if ground_configs:
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        axes = axes.flatten()
        for i in range(KINEMATIC_DIMS):
            ax = axes[i]
            dim = DIM_NAMES[i]
            full_lo, full_hi = STATE_FULL_RANGE[dim]
            for cfg in ground_configs:
                vals = data[cfg]["init_states"][:, i]
                ax.hist(vals, bins=30, alpha=0.5, label=cfg, density=True,
                        range=(full_lo, full_hi))
            ax.set_xlim(full_lo, full_hi)
            ax.axvspan(full_lo, full_hi, alpha=0.03, color="gray")
            ax.set_title(dim)
            ax.set_xlabel("value")
            if i == 0:
                ax.legend(fontsize=7)
        fig.suptitle("Ground: initial state distributions "
                     "(x-axis = full state space range)", fontsize=13)
        fig.tight_layout()
        fig.savefig(str(plot_dir / "coverage-ground-histograms.png"), dpi=120)
        plt.close(fig)

    # --- 2D scatter: x vs y (fixed to full range) ---
    n_panels = bool(fresh_configs) + bool(ground_configs)
    if n_panels:
        fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 6))
        if n_panels == 1:
            axes = [axes]
        x_lo, x_hi = STATE_FULL_RANGE["x"]
        y_lo, y_hi = STATE_FULL_RANGE["y"]
        idx = 0
        if fresh_configs:
            ax = axes[idx]; idx += 1
            for cfg in fresh_configs:
                s = data[cfg]["init_states"]
                ax.scatter(s[:, 0], s[:, 1], alpha=0.3, s=10, label=cfg)
            ax.set_xlim(x_lo, x_hi); ax.set_ylim(y_lo, y_hi)
            ax.set_xlabel("x"); ax.set_ylabel("y")
            ax.set_title("Fresh-reset: x vs y")
            ax.legend(fontsize=8)
            ax.set_aspect("equal")
        if ground_configs:
            ax = axes[idx]
            for cfg in ground_configs:
                s = data[cfg]["init_states"]
                ax.scatter(s[:, 0], s[:, 1], alpha=0.3, s=10, label=cfg)
            ax.set_xlim(x_lo, x_hi); ax.set_ylim(y_lo, y_hi)
            ax.set_xlabel("x"); ax.set_ylabel("y")
            ax.set_title("Ground: x vs y")
            ax.legend(fontsize=8)
            ax.set_aspect("equal")
        fig.tight_layout()
        fig.savefig(str(plot_dir / "coverage-scatter-xy.png"), dpi=120)
        plt.close(fig)

    # --- Trajectory coverage histograms (all states visited) ---
    def _plot_traj_histograms(config_list, label, filename):
        if not config_list:
            return
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        axes = axes.flatten()
        for i in range(KINEMATIC_DIMS):
            ax = axes[i]
            dim = DIM_NAMES[i]
            full_lo, full_hi = STATE_FULL_RANGE[dim]
            for cfg in config_list:
                vals = data[cfg]["all_states"][:, i]
                ax.hist(vals, bins=60, alpha=0.4, label=cfg, density=True,
                        range=(full_lo, full_hi))
            ax.set_xlim(full_lo, full_hi)
            ax.set_title(dim)
            ax.set_xlabel("value")
            if i == 0:
                ax.legend(fontsize=7)
        fig.suptitle(f"{label}: trajectory coverage — all states visited "
                     "(x-axis = full state space)", fontsize=13)
        fig.tight_layout()
        fig.savefig(str(plot_dir / filename), dpi=120)
        plt.close(fig)

    _plot_traj_histograms(fresh_configs, "Fresh-reset",
                          "coverage-fresh-trajectory.png")
    _plot_traj_histograms(ground_configs, "Ground",
                          "coverage-ground-trajectory.png")

    # --- Trajectory lines: x vs y (sample N episodes per config) ---
    n_sample_eps = 500  # episodes per config to draw
    rng = np.random.default_rng(42)
    # Color cycle for configs.
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    def _plot_trajectory_lines(config_list, label, filename):
        if not config_list:
            return
        fig, ax = plt.subplots(figsize=(10, 8))
        x_lo, x_hi = STATE_FULL_RANGE["x"]
        y_lo, y_hi = STATE_FULL_RANGE["y"]
        for ci, cfg in enumerate(config_list):
            episodes = data[cfg]["episodes"]
            n = len(episodes)
            idxs = rng.choice(n, size=min(n_sample_eps, n), replace=False)
            color = colors[ci % len(colors)]
            for j, idx in enumerate(idxs):
                ep = episodes[idx]
                lbl = cfg if j == 0 else None
                ax.plot(ep[:, 0], ep[:, 1], color=color, alpha=0.3,
                        linewidth=0.8, label=lbl)
                # Mark start and end.
                ax.plot(ep[0, 0], ep[0, 1], "o", color=color,
                        markersize=3, alpha=0.6)
                ax.plot(ep[-1, 0], ep[-1, 1], "x", color=color,
                        markersize=4, alpha=0.6)
        ax.set_xlim(x_lo, x_hi); ax.set_ylim(y_lo, y_hi)
        ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.set_title(f"{label}: sample trajectories "
                     f"({n_sample_eps}/config, o=start, x=end)")
        ax.legend(fontsize=8)
        ax.set_aspect("equal")
        fig.tight_layout()
        fig.savefig(str(plot_dir / filename), dpi=150)
        plt.close(fig)

    _plot_trajectory_lines(fresh_configs, "Fresh-reset",
                           "coverage-trajectories-fresh.png")
    _plot_trajectory_lines(ground_configs, "Ground",
                           "coverage-trajectories-ground.png")

    # --- Episode length distributions ---
    fig, ax = plt.subplots(figsize=(10, 5))
    all_names = sorted(data.keys())
    positions = range(len(all_names))
    lengths_list = [data[n]["lengths"] for n in all_names]
    ax.boxplot(lengths_list, positions=positions, vert=True, patch_artist=True)
    ax.set_xticks(positions)
    ax.set_xticklabels(all_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Episode length (steps)")
    ax.set_title("Episode length distribution per config")
    fig.tight_layout()
    fig.savefig(str(plot_dir / "coverage-lengths.png"), dpi=120)
    plt.close(fig)

    print(f"\nPlots saved to {plot_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Report coverage statistics for primitive collections.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("collection_dir", type=str,
                        help="Root directory with one subdir per config.")
    parser.add_argument("--configs", nargs="+", default=None,
                        help="Only report on these config subdirs.")
    parser.add_argument("--plot-dir", type=str, default=None,
                        help="Directory for plots (default: <collection_dir>/coverage/).")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip plot generation (text report only).")

    args = parser.parse_args()
    collection_dir = Path(args.collection_dir)

    if not collection_dir.is_dir():
        print(f"Error: {collection_dir} is not a directory")
        sys.exit(1)

    data = load_collection(collection_dir, args.configs)
    if not data:
        print(f"Error: no episode subdirs found in {collection_dir}")
        sys.exit(1)

    print_report(data)

    if not args.no_plots:
        plot_dir = Path(args.plot_dir) if args.plot_dir else collection_dir / "coverage"
        plot_coverage(data, plot_dir)


if __name__ == "__main__":
    main()
