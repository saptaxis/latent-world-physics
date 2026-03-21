#!/usr/bin/env python
"""Generate a visual report describing a world model training dataset.

Reads metrics.csv files from collection directories (or a combined CSV)
and produces summary stats + diagnostic plots. Use this to validate
that a dataset has good physics coverage, behavioral diversity, and
sensible outcome distributions before training.

Supports multiple input paths for comparing physics profiles (e.g.,
full-variation + easy + medium) in the same plots. Each path gets a
"profile" label derived from its directory name.

Usage:
    # Single directory of collections
    python lunar_lander/scripts/dataset_report.py \
        /media/hdd1/.../world_model_data/full-variation \
        --output ~/vsr-tmp/wm-dataset-report

    # Multiple profiles together (main use case for training data selection)
    python lunar_lander/scripts/dataset_report.py \
        /media/hdd1/.../world_model_data/full-variation \
        /media/hdd1/.../world_model_data/easy \
        /media/hdd1/.../world_model_data/medium \
        --output ~/vsr-tmp/wm-dataset-report-combined

    # From a pre-combined CSV
    python lunar_lander/scripts/dataset_report.py \
        /media/hdd1/.../all_metrics.csv \
        --output ~/vsr-tmp/wm-dataset-report

    # Filter to specific sources
    python lunar_lander/scripts/dataset_report.py \
        /media/hdd1/.../world_model_data/full-variation \
        --sources blind-s42 heuristic random \
        --output ~/vsr-tmp/wm-dataset-report
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd


# --- Physics parameters and their display names ---
PHYSICS_PARAMS = [
    ("gravity", "Gravity"),
    ("main_engine_power", "Main Engine"),
    ("side_engine_power", "Side Engine"),
    ("lander_density", "Density"),
    ("angular_damping", "Angular Damping"),
    ("wind_power", "Wind"),
    ("turbulence_power", "Turbulence"),
]

# Key behavioral metrics to summarize
BEHAVIORAL_METRICS = [
    ("thrust_duty_cycle", "Thrust Duty Cycle"),
    ("thrust_autocorr_lag1", "Thrust Autocorrelation"),
    ("total_fuel", "Total Fuel"),
    ("mean_main_thrust", "Mean Main Thrust"),
    ("mean_abs_angular_vel", "Mean |Angular Vel|"),
]

# Source ordering and colors for consistent plots
SOURCE_COLORS = {
    "blind": "#2196F3",
    "labeled": "#FF9800",
    "heuristic": "#4CAF50",
    "random": "#9E9E9E",
    "noisy-expert": "#9C27B0",
}

# Profile colors for multi-profile comparisons
PROFILE_COLORS = {
    "full-variation": "#2196F3",
    "easy": "#4CAF50",
    "medium": "#FF9800",
}


def _profile_color(profile_name):
    """Map a profile name to a color, with fallback."""
    return PROFILE_COLORS.get(profile_name, "#607D8B")


def _source_color(source_name):
    """Map a source name like 'blind-s42' to its category color."""
    for prefix, color in SOURCE_COLORS.items():
        if source_name.startswith(prefix):
            return color
    return "#607D8B"


def _source_category(source_name):
    """Map a source name like 'blind-s42' to its category."""
    for prefix in SOURCE_COLORS:
        if source_name.startswith(prefix):
            return prefix
    return source_name


def load_data(paths, sources=None):
    """Load metrics from CSV files or directories of collections.

    Supports multiple paths for combining profiles. Each directory path
    gets a 'profile' label from its directory name (e.g., "full-variation",
    "easy"). CSV files get profile="csv".

    Args:
        paths: List of paths. Each can be a CSV file or a directory
               containing subdirectories with metrics.csv files.
        sources: Optional list of source names to include.

    Returns:
        DataFrame with all metrics plus 'source', 'profile', and
        'source_category' columns.
    """
    all_dfs = []

    for p in paths:
        path = Path(p)

        if path.suffix == ".csv":
            df = pd.read_csv(path)
            if "profile" not in df.columns:
                df["profile"] = "csv"
            all_dfs.append(df)

        elif path.is_dir():
            profile_name = path.name
            for sub in sorted(path.iterdir()):
                csv = sub / "metrics.csv"
                if csv.exists():
                    sub_df = pd.read_csv(csv)
                    sub_df["source"] = sub.name
                    sub_df["profile"] = profile_name
                    all_dfs.append(sub_df)

        else:
            print(f"WARNING: {path} is not a CSV file or directory, skipping")

    if not all_dfs:
        print(f"ERROR: No metrics.csv files found in any of the provided paths")
        sys.exit(1)

    df = pd.concat(all_dfs, ignore_index=True)

    if sources:
        df = df[df["source"].isin(sources)]
        if df.empty:
            print(f"ERROR: No data after filtering to sources: {sources}")
            sys.exit(1)

    # Add derived columns
    df["source_category"] = df["source"].apply(_source_category)

    return df


def write_summary(df, output_dir):
    """Write a text summary of the dataset."""
    lines = []
    lines.append("=" * 70)
    lines.append("WORLD MODEL DATASET REPORT")
    lines.append("=" * 70)

    profiles = sorted(df["profile"].unique())
    lines.append(f"\nTotal episodes: {len(df):,}")
    lines.append(f"Profiles: {profiles}")
    for prof in profiles:
        pdf = df[df["profile"] == prof]
        lines.append(f"  {prof}: {len(pdf):,} episodes ({pdf['source'].nunique()} sources)")
    lines.append(f"Sources: {sorted(df['source'].unique())}")

    # Outcome distribution
    lines.append(f"\n--- Outcomes ---")
    for outcome, count in df["outcome"].value_counts().items():
        lines.append(f"  {outcome:15s} {count:6,d} ({100*count/len(df):.1f}%)")

    # Per-source outcomes
    lines.append(f"\n--- Outcomes by source ---")
    pivot = df.groupby("source")["outcome"].value_counts().unstack(fill_value=0)
    pivot["total"] = pivot.sum(axis=1)
    if "landed" in pivot.columns:
        pivot["land_rate"] = (100 * pivot["landed"] / pivot["total"]).round(1)
    lines.append(pivot.to_string())

    # Physics distributions
    lines.append(f"\n--- Physics parameter distributions ---")
    for param, label in PHYSICS_PARAMS:
        if param in df.columns:
            s = df[param]
            lines.append(
                f"  {label:20s}  "
                f"mean={s.mean():.2f}  std={s.std():.2f}  "
                f"min={s.min():.2f}  max={s.max():.2f}  "
                f"median={s.median():.2f}"
            )

    # TWR distribution
    if "twr" in df.columns:
        lines.append(f"\n--- TWR (Thrust-to-Weight Ratio) ---")
        twr = df["twr"]
        lines.append(f"  mean={twr.mean():.2f}  std={twr.std():.2f}  "
                      f"min={twr.min():.2f}  max={twr.max():.2f}")
        lines.append(f"  TWR < 1.0 (can't hover): {(twr < 1.0).sum():,d} "
                      f"({100*(twr < 1.0).mean():.1f}%)")
        lines.append(f"  TWR 1.0-2.0 (moderate):  {((twr >= 1.0) & (twr < 2.0)).sum():,d} "
                      f"({100*((twr >= 1.0) & (twr < 2.0)).mean():.1f}%)")
        lines.append(f"  TWR > 2.0 (easy):        {(twr >= 2.0).sum():,d} "
                      f"({100*(twr >= 2.0).mean():.1f}%)")

    # Behavioral metrics by source category
    lines.append(f"\n--- Behavioral metrics by source category ---")
    for col, label in BEHAVIORAL_METRICS:
        if col in df.columns:
            lines.append(f"\n  {label}:")
            grouped = df.groupby("source_category")[col].agg(["mean", "std", "median"])
            lines.append("    " + grouped.round(3).to_string().replace("\n", "\n    "))

    # Episode length stats
    lines.append(f"\n--- Episode length ---")
    lines.append(f"  mean={df['episode_steps'].mean():.0f}  "
                  f"std={df['episode_steps'].std():.0f}  "
                  f"min={df['episode_steps'].min()}  "
                  f"max={df['episode_steps'].max()}")
    lines.append(f"  Timeouts (1000 steps): {(df['episode_steps'] >= 1000).sum():,d}")

    text = "\n".join(lines) + "\n"

    summary_path = output_dir / "summary.txt"
    summary_path.write_text(text)
    print(text)
    print(f"Summary saved to {summary_path}")

    return text


def plot_physics_distributions(df, output_dir):
    """Histograms of all 7 physics parameters + TWR.

    When multiple profiles are present, overlays them with different colors
    so you can see how the distributions differ (e.g., easy has no wind,
    medium is narrower gravity range, etc.).
    """
    n_params = len(PHYSICS_PARAMS) + (1 if "twr" in df.columns else 0)
    cols = 4
    rows = (n_params + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    axes = axes.flatten()

    profiles = sorted(df["profile"].unique())
    multi_profile = len(profiles) > 1

    for i, (param, label) in enumerate(PHYSICS_PARAMS):
        if param not in df.columns:
            continue
        if multi_profile:
            for prof in profiles:
                subset = df[df["profile"] == prof][param]
                axes[i].hist(subset, bins=50, alpha=0.5, edgecolor="none",
                             color=_profile_color(prof), label=prof)
            axes[i].legend(fontsize=8)
        else:
            axes[i].hist(df[param], bins=50, color="#2196F3", alpha=0.7, edgecolor="none")
        axes[i].set_title(label, fontsize=12)
        axes[i].set_xlabel(param)
        axes[i].set_ylabel("Count")

    # TWR as the last panel
    if "twr" in df.columns:
        idx = len(PHYSICS_PARAMS)
        if multi_profile:
            for prof in profiles:
                subset = df[df["profile"] == prof]["twr"]
                axes[idx].hist(subset, bins=50, alpha=0.5, edgecolor="none",
                               color=_profile_color(prof), label=prof)
            axes[idx].legend(fontsize=8)
        else:
            axes[idx].hist(df["twr"], bins=50, color="#FF5722", alpha=0.7, edgecolor="none")
        axes[idx].axvline(1.0, color="red", linestyle="--", alpha=0.7, label="TWR=1")
        axes[idx].set_title("TWR (derived)", fontsize=12)
        axes[idx].set_xlabel("twr")

    # Hide unused axes
    for j in range(n_params, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Physics Parameter Distributions", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "physics_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_physics_2d(df, output_dir):
    """2D scatter plots of key physics parameter pairs.

    When multiple profiles are present, colors by profile instead of outcome
    so you can see how the physics spaces overlap or separate.
    """
    pairs = [
        ("gravity", "main_engine_power", "Gravity vs Engine Power"),
        ("gravity", "lander_density", "Gravity vs Density"),
        ("twr", "wind_power", "TWR vs Wind"),
        ("twr", "turbulence_power", "TWR vs Turbulence"),
    ]
    pairs = [(x, y, t) for x, y, t in pairs if x in df.columns and y in df.columns]

    if not pairs:
        return

    profiles = sorted(df["profile"].unique())
    multi_profile = len(profiles) > 1

    fig, axes = plt.subplots(1, len(pairs), figsize=(5 * len(pairs), 4.5))
    if len(pairs) == 1:
        axes = [axes]

    from matplotlib.lines import Line2D

    # Subsample per profile to keep point counts balanced in the plot
    per_profile = min(3000, len(df) // max(len(profiles), 1))
    sample = pd.concat([
        g.sample(min(per_profile, len(g)), random_state=42)
        for _, g in df.groupby("profile")
    ], ignore_index=True)

    for ax, (x, y, title) in zip(axes, pairs):
        if multi_profile:
            for prof in profiles:
                s = sample[sample["profile"] == prof]
                ax.scatter(s[x], s[y], c=_profile_color(prof), alpha=0.3,
                           s=4, edgecolors="none", label=prof)
        else:
            colors = sample["outcome"].map({
                "landed": "#4CAF50",
                "crashed": "#F44336",
                "out_of_bounds": "#FF9800",
                "timeout": "#9E9E9E",
            })
            ax.scatter(sample[x], sample[y], c=colors, alpha=0.3, s=4, edgecolors="none")
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(title, fontsize=11)

    if multi_profile:
        legend_elements = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor=_profile_color(p),
                   markersize=6, label=p)
            for p in profiles
        ]
    else:
        legend_elements = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#4CAF50", markersize=6, label="Landed"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#F44336", markersize=6, label="Crashed"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#FF9800", markersize=6, label="OOB"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#9E9E9E", markersize=6, label="Timeout"),
        ]
    axes[-1].legend(handles=legend_elements, loc="upper right", fontsize=8)

    color_label = "profile" if multi_profile else "outcome"
    fig.suptitle(f"Physics Space Coverage (colored by {color_label})", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "physics_2d_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_outcomes_by_source(df, output_dir):
    """Stacked bar chart of outcomes per source."""
    pivot = df.groupby("source")["outcome"].value_counts().unstack(fill_value=0)

    outcome_colors = {
        "landed": "#4CAF50",
        "crashed": "#F44336",
        "out_of_bounds": "#FF9800",
        "timeout": "#9E9E9E",
    }

    fig, ax = plt.subplots(figsize=(14, 5))
    bottom = np.zeros(len(pivot))
    for outcome in ["landed", "crashed", "out_of_bounds", "timeout"]:
        if outcome in pivot.columns:
            vals = pivot[outcome].values
            ax.bar(range(len(pivot)), vals, bottom=bottom,
                   color=outcome_colors.get(outcome, "#607D8B"),
                   label=outcome, edgecolor="none")
            bottom += vals

    ax.set_xticks(range(len(pivot)))
    ax.set_xticklabels(pivot.index, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Episodes")
    ax.set_title("Outcomes by Source", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_dir / "outcomes_by_source.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_episode_lengths(df, output_dir):
    """Episode length distributions by source category."""
    categories = sorted(df["source_category"].unique())

    fig, ax = plt.subplots(figsize=(10, 5))
    for cat in categories:
        subset = df[df["source_category"] == cat]["episode_steps"]
        ax.hist(subset, bins=50, alpha=0.5, label=f"{cat} (n={len(subset):,})",
                color=SOURCE_COLORS.get(cat, "#607D8B"), edgecolor="none")

    ax.set_xlabel("Episode Length (steps)")
    ax.set_ylabel("Count")
    ax.set_title("Episode Length Distribution by Source Category", fontsize=13, fontweight="bold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "episode_lengths.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_reward_distributions(df, output_dir):
    """Reward distributions by source category."""
    categories = sorted(df["source_category"].unique())

    fig, ax = plt.subplots(figsize=(10, 5))
    for cat in categories:
        subset = df[df["source_category"] == cat]["total_reward"]
        ax.hist(subset, bins=80, alpha=0.5, label=f"{cat} (n={len(subset):,})",
                color=SOURCE_COLORS.get(cat, "#607D8B"), edgecolor="none",
                range=(-1000, 500))

    ax.set_xlabel("Total Reward")
    ax.set_ylabel("Count")
    ax.set_title("Reward Distribution by Source Category", fontsize=13, fontweight="bold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "reward_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_behavioral_metrics(df, output_dir):
    """Box plots of behavioral metrics by source category."""
    metrics = [(col, label) for col, label in BEHAVIORAL_METRICS if col in df.columns]
    if not metrics:
        return

    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]

    categories = sorted(df["source_category"].unique())

    for ax, (col, label) in zip(axes, metrics):
        data = [df[df["source_category"] == cat][col].dropna() for cat in categories]
        bp = ax.boxplot(data, tick_labels=categories, patch_artist=True, showfliers=False)
        for patch, cat in zip(bp["boxes"], categories):
            patch.set_facecolor(SOURCE_COLORS.get(cat, "#607D8B"))
            patch.set_alpha(0.6)
        ax.set_title(label, fontsize=11)
        ax.tick_params(axis="x", rotation=45)

    fig.suptitle("Behavioral Metrics by Source Category", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "behavioral_metrics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_landing_rate_vs_twr(df, output_dir):
    """Landing rate as a function of TWR, by source category."""
    if "twr" not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    categories = sorted(df["source_category"].unique())

    twr_bins = np.linspace(df["twr"].min(), min(df["twr"].max(), 8.0), 20)

    for cat in categories:
        subset = df[df["source_category"] == cat]
        if len(subset) < 100:
            continue
        subset = subset.copy()
        subset["twr_bin"] = pd.cut(subset["twr"], bins=twr_bins)
        rates = subset.groupby("twr_bin", observed=True).apply(
            lambda g: (g["outcome"] == "landed").mean() if len(g) > 10 else np.nan
        )
        bin_centers = [(b.left + b.right) / 2 for b in rates.index]
        ax.plot(bin_centers, rates.values, "o-", label=cat,
                color=SOURCE_COLORS.get(cat, "#607D8B"), alpha=0.8, markersize=4)

    ax.axvline(1.0, color="red", linestyle="--", alpha=0.5, label="TWR=1")
    ax.set_xlabel("TWR (Thrust-to-Weight Ratio)")
    ax.set_ylabel("Landing Rate")
    ax.set_title("Landing Rate vs TWR by Source Category", fontsize=13, fontweight="bold")
    ax.legend()
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    fig.savefig(output_dir / "landing_rate_vs_twr.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_per_profile_summary(df, output_dir):
    """Per-profile comparison plots: TWR distribution, outcome rates, episode counts.

    Only generated when multiple profiles are present. Gives a quick
    side-by-side comparison of how the profiles differ.
    """
    profiles = sorted(df["profile"].unique())
    if len(profiles) < 2:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: TWR distributions per profile
    ax = axes[0]
    if "twr" in df.columns:
        for prof in profiles:
            subset = df[df["profile"] == prof]["twr"]
            ax.hist(subset, bins=40, alpha=0.5, label=f"{prof} (n={len(subset):,})",
                    color=_profile_color(prof), edgecolor="none", density=True)
        ax.axvline(1.0, color="red", linestyle="--", alpha=0.5)
        ax.set_xlabel("TWR")
        ax.set_ylabel("Density")
        ax.set_title("TWR Distribution by Profile")
        ax.legend(fontsize=9)

    # Panel 2: Outcome proportions per profile (stacked bars)
    ax = axes[1]
    outcome_colors = {
        "landed": "#4CAF50",
        "crashed": "#F44336",
        "out_of_bounds": "#FF9800",
        "timeout": "#9E9E9E",
    }
    x_pos = np.arange(len(profiles))
    bottom = np.zeros(len(profiles))
    for outcome in ["landed", "crashed", "out_of_bounds", "timeout"]:
        vals = []
        for prof in profiles:
            pdf = df[df["profile"] == prof]
            vals.append((pdf["outcome"] == outcome).mean())
        vals = np.array(vals)
        ax.bar(x_pos, vals, bottom=bottom, color=outcome_colors.get(outcome, "#607D8B"),
               label=outcome, edgecolor="none", width=0.6)
        bottom += vals
    ax.set_xticks(x_pos)
    ax.set_xticklabels(profiles, fontsize=10)
    ax.set_ylabel("Proportion")
    ax.set_title("Outcome Proportions by Profile")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(0, 1.05)

    # Panel 3: Episode counts per source category, grouped by profile
    ax = axes[2]
    categories = sorted(df["source_category"].unique())
    n_profiles = len(profiles)
    bar_width = 0.8 / n_profiles
    x_pos = np.arange(len(categories))
    for i, prof in enumerate(profiles):
        counts = [len(df[(df["profile"] == prof) & (df["source_category"] == cat)])
                  for cat in categories]
        offset = (i - n_profiles / 2 + 0.5) * bar_width
        ax.bar(x_pos + offset, counts, bar_width, label=prof,
               color=_profile_color(prof), alpha=0.8, edgecolor="none")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Episodes")
    ax.set_title("Episodes by Source Category & Profile")
    ax.legend(fontsize=9)

    fig.suptitle("Per-Profile Comparison", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "per_profile_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Generate a visual report describing a world model training dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("paths", type=str, nargs="+",
                        help="One or more directories of collections (with metrics.csv each) "
                             "or combined CSV files. Multiple paths are combined with a "
                             "'profile' label from the directory name.")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory for report (summary.txt + plots)")
    parser.add_argument("--sources", nargs="+", default=None,
                        help="Filter to specific source names")

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {len(args.paths)} path(s)...")
    for p in args.paths:
        print(f"  {p}")
    df = load_data(args.paths, sources=args.sources)
    profiles = sorted(df["profile"].unique())
    print(f"Loaded {len(df):,} episodes from {df['source'].nunique()} sources, "
          f"{len(profiles)} profile(s): {profiles}\n")

    # Text summary
    write_summary(df, output_dir)

    # Plots
    print("\nGenerating plots...")
    plot_physics_distributions(df, output_dir)
    print("  physics_distributions.png")

    plot_physics_2d(df, output_dir)
    print("  physics_2d_scatter.png")

    plot_outcomes_by_source(df, output_dir)
    print("  outcomes_by_source.png")

    plot_episode_lengths(df, output_dir)
    print("  episode_lengths.png")

    plot_reward_distributions(df, output_dir)
    print("  reward_distributions.png")

    plot_behavioral_metrics(df, output_dir)
    print("  behavioral_metrics.png")

    plot_landing_rate_vs_twr(df, output_dir)
    print("  landing_rate_vs_twr.png")

    if len(profiles) > 1:
        plot_per_profile_summary(df, output_dir)
        print("  per_profile_summary.png")

    n_plots = 7 + (1 if len(profiles) > 1 else 0)
    print(f"\nReport complete: {output_dir}/")
    print(f"  summary.txt + {n_plots} plots")


if __name__ == "__main__":
    main()
