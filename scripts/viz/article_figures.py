"""Article figure generator for "How Information Channels Shape Learned Controllers".

Builds the 5 main metric-panel figures (Fig 1-5), a compact Result 4 table,
and re-wraps two existing artifacts (state V_D overlay, normalized
reversal-rate-over-time) into the publication figures directory with
companion numbers.csv / numbers.md / metadata.json files.

Style constants (colors, sizes, labels) are defined at the top of the file
so future sessions can tweak them without touching plotting code.

Sources (all in dated cross-config dirs from the 2026-05-13 recompute):
  Fig 1: parametric-vs-behavioral-v2-2026-05-13/full-variation-easy
  Fig 2: parametric-vs-behavioral-2026-05-13/{physics-only-easy,vehicle-only-easy,full-variation-easy}
  Fig 3: body-vs-world-corruption-2026-05-13/{full-variation-easy,physics-only-easy,vehicle-only-easy}
  Fig 4: parametric-vs-behavioral-v2-2026-05-13/full-variation-easy
         + ood-generalization-{medium,hard}-2026-05-13
  Fig 5: e3-n3-frozen-only-final + e3-n3-all-6-final-ood-medium
  R4:    e1-blind-vs-labeled-gym-default-128 + matched-enc-all-6-conditions

Usage:
  python scripts/viz/article_figures.py --figure all
  python scripts/viz/article_figures.py --figure fig1
  python scripts/viz/article_figures.py --figure fig1 --out-dir /tmp/preview
"""
from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend, avoid X11 crash on headless runs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ====================================================================
# Style constants — tweak here, not in the plotting functions.
# ====================================================================

# Color palette
COLORS = {
    # State-vector pair
    "blind": "#ff7f0e",       # orange
    "labeled": "#1f77b4",     # blue
    # Visual fusion gradient (Fig 5)
    "blind-visual": "#ff7f0e",
    "raw": "#9467bd",         # purple
    "branch": "#2ca02c",      # green
    # Corruption ladder (Fig 3)
    "baseline": "#444444",
    "world-zero": "#d62728",  # red
    "body-zero": "#e377c2",   # pink
    "all-zero": "#7f7f7f",    # mid-grey
    # Result 4 (state vs visual fixed-physics)
    "state-blind": "#ff7f0e",
    "state-labeled": "#1f77b4",
    "visual-frozen": "#9467bd",
    "visual-ft": "#2ca02c",
}

# Figure sizes (inches)
FIG_SIZE_2x2 = (8.0, 7.0)        # Fig 1
FIG_SIZE_SINGLE = (6.5, 4.2)     # Fig 2
FIG_SIZE_LADDER = (9.0, 4.8)     # Fig 3
FIG_SIZE_OOD = (7.0, 4.5)        # Fig 4
FIG_SIZE_FUSION = (9.0, 4.8)     # Fig 5

# Typography
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

DPI = 300

# Human-readable metric labels (used in panel titles, axes)
METRIC_LABELS = {
    "landed_pct": "Landing rate (%)",
    "crashed_pct": "Crash rate (%)",
    "timeout_pct": "Timeout rate (%)",
    "thrust_autocorr_lag1": "Thrust autocorrelation (lag-1)",
    "side_thrust_autocorr_lag1": "Side-thrust autocorrelation (lag-1)",
    "side_thrust_reversals": "Side-thrust reversals (per episode)",
    "side_thrust_reversals_rate": "Side-thrust reversals (per step)",
    "side_thrust_reversals_per_step": "Side-thrust reversals (per step)",
    "thrust_duty_cycle": "Thrust duty cycle",
    "total_fuel": "Total fuel (per episode)",
    "frac_upright": "Fraction upright",
    "time_to_first_contact": "Steps to first contact",
    "time_to_descend": "Steps to descend",
    "descent_steadiness": "Descent steadiness",
    "mean_main_thrust": "Mean main thrust",
    "frac_correcting": "Fraction correcting",
    "fuel_efficiency": "Fuel efficiency",
}

# Default repo-relative paths
DEFAULT_NETS = Path("/media/hdd1/physics-priors-latent-space/lunar-lander-networks")
DEFAULT_OUT = Path(
    "/media/hdd1/physics-priors-latent-space/lunar-lander-networks/"
    "articles/01-what-do-agents-learn/figures"
)
DATE_TAG = "2026-05-14"

# Existing overlay artifacts to wrap
OVERLAY_VD_SRC = Path("/home/vsr/vsr-tmp/e1-overlay/easy-ID_vD.png")
REVERSAL_TIME_SRC = Path("/home/vsr/vsr-tmp/e1-overlay/reversal_rate_over_time.png")

# Backdrop frame used for trajectory overlays (V_BD). Any episode_*.npz with
# rgb_frames will do — the last frame becomes the static background.
BACKDROP_RGB = Path(
    "/media/hdd1/physics-priors-latent-space/lunar-lander-networks/rl_agents/"
    "full-variation/labeled-ppo-easy-128-lowent-Feb122026/"
    "trajectories-rgb/episode_0000.npz"
)

# Cross-config metrics CSVs feeding the V_BD seed picker.
#   easy-ID            -> v2 (the main full-variation easy comparison)
#   medium-ID          -> v1, full-variation-medium subdir
#   easy-OOD-medium    -> ood-generalization-medium
BD_BODYWORLD_SCENARIOS = {
    "physics-only_easy": Path(
        "/media/hdd1/physics-priors-latent-space/lunar-lander-networks/"
        f"analysis/cross-config/parametric-vs-behavioral-{DATE_TAG}/"
        "physics-only-easy/metrics_by_variant.csv"
    ),
    "physics-only_medium": Path(
        "/media/hdd1/physics-priors-latent-space/lunar-lander-networks/"
        f"analysis/cross-config/parametric-vs-behavioral-{DATE_TAG}/"
        "physics-only-medium/metrics_by_variant.csv"
    ),
    "vehicle-only_easy": Path(
        "/media/hdd1/physics-priors-latent-space/lunar-lander-networks/"
        f"analysis/cross-config/parametric-vs-behavioral-{DATE_TAG}/"
        "vehicle-only-easy/metrics_by_variant.csv"
    ),
    "vehicle-only_hard": Path(
        "/media/hdd1/physics-priors-latent-space/lunar-lander-networks/"
        f"analysis/cross-config/parametric-vs-behavioral-{DATE_TAG}/"
        "vehicle-only-hard/metrics_by_variant.csv"
    ),
}

BD_SCENARIOS = {
    "easy-ID": Path(
        "/media/hdd1/physics-priors-latent-space/lunar-lander-networks/"
        f"analysis/cross-config/parametric-vs-behavioral-v2-{DATE_TAG}/"
        "full-variation-easy/metrics_by_variant.csv"
    ),
    "medium-ID": Path(
        "/media/hdd1/physics-priors-latent-space/lunar-lander-networks/"
        f"analysis/cross-config/parametric-vs-behavioral-{DATE_TAG}/"
        "full-variation-medium/metrics_by_variant.csv"
    ),
    "easy-OOD-medium": Path(
        "/media/hdd1/physics-priors-latent-space/lunar-lander-networks/"
        f"analysis/cross-config/ood-generalization-medium-{DATE_TAG}/"
        "full-variation-easy-trained-on-medium/metrics_by_variant.csv"
    ),
    "easy-OOD-hard": Path(
        "/media/hdd1/physics-priors-latent-space/lunar-lander-networks/"
        f"analysis/cross-config/ood-generalization-hard-{DATE_TAG}/"
        "full-variation-easy-trained-on-hard/metrics_by_variant.csv"
    ),
}

# ====================================================================
# Data loading + stats utilities
# ====================================================================


def cross_config_path(name: str) -> Path:
    return DEFAULT_NETS / "analysis" / "cross-config" / name


def load_metrics_csv(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p)
    return df


def per_seed_means(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Per-(variant, seed) mean of a metric across episodes.

    For percentage outcomes (landed_pct, crashed_pct, timeout_pct), the
    underlying CSV has `outcome` as a categorical string per episode.
    We compute the percentage per (variant, seed).
    """
    if metric.endswith("_pct"):
        outcome_label = metric.replace("_pct", "")
        df = df.assign(_match=(df["outcome"] == outcome_label).astype(float) * 100.0)
        return (
            df.groupby(["variant", "seed"])["_match"]
              .mean().reset_index().rename(columns={"_match": metric})
        )
    return (
        df.groupby(["variant", "seed"])[metric]
          .mean().reset_index()
    )


def variant_summary(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Compute mean, std, SE, N across seeds for each variant."""
    per_seed = per_seed_means(df, metric)
    g = per_seed.groupby("variant")[metric]
    return pd.DataFrame({
        "variant": g.mean().index,
        "mean": g.mean().values,
        "std": g.std().values,
        "se": (g.std() / np.sqrt(g.count())).values,
        "n": g.count().values,
    })


def load_stat_tests(name: str, comparison: str) -> dict:
    p = cross_config_path(name) / "stat_tests.json"
    if not p.exists():
        return {}
    return json.load(open(p)).get(comparison, {})


def lookup_pvalue_d(stats: dict, metric: str) -> tuple[float | None, float | None]:
    entry = stats.get(metric)
    if not isinstance(entry, dict):
        return None, None
    return entry.get("p_value"), entry.get("effect_size_cohens_d")


# ====================================================================
# Plot primitives
# ====================================================================


def bar_with_seed_dots(
    ax,
    summary: pd.DataFrame,
    per_seed: pd.DataFrame,
    metric: str,
    color_map: dict,
    title: str | None = None,
    ylabel: str | None = None,
    show_xticks: bool = True,
):
    """Bar plot with per-seed dots overlaid. summary has columns
    [variant, mean, se, n]; per_seed has [variant, seed, <metric>]."""
    variants = list(summary["variant"])
    # Pack bars closer together by compressing x spacing.
    x = np.arange(len(variants)) * 0.6
    means = summary["mean"].values
    ses = summary["se"].values
    colors = [color_map.get(v, "#888") for v in variants]

    bars = ax.bar(x, means, 0.28, yerr=ses, capsize=4, color=colors,
                  alpha=0.85, edgecolor="black", linewidth=0.5)
    # seed dots
    for i, v in enumerate(variants):
        vals = per_seed[per_seed["variant"] == v][metric].values
        jitter = (np.random.RandomState(42 + i).uniform(-0.08, 0.08, size=len(vals)))
        ax.scatter(np.full_like(vals, x[i], dtype=float) + jitter, vals,
                   s=18, color="white", edgecolor=colors[i],
                   linewidth=0.8, zorder=3, alpha=0.95)
    ax.set_xlim(x[0] - 0.4, x[-1] + 0.4)
    if show_xticks:
        ax.set_xticks(x)
        ax.set_xticklabels(variants, fontsize=9)
    else:
        ax.set_xticks([])
    if title:
        ax.set_title(title, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.yaxis.grid(True, alpha=0.25, linestyle="-", linewidth=0.5)
    ax.set_axisbelow(True)


def point_with_seed_dots(
    ax, summary: pd.DataFrame, per_seed: pd.DataFrame, metric: str,
    color_map: dict, title: str | None = None, ylabel: str | None = None,
    show_xticks: bool = True,
):
    """Point + SE + jittered per-seed scatter (no bars). For bounded style
    metrics on a zoomed axis: filled bars are misleading there.
    See feedback_bounded_style_metric_plotting.
    """
    variants = list(summary["variant"])
    x = np.arange(len(variants))
    for i, v in enumerate(variants):
        row = summary[summary["variant"] == v].iloc[0]
        seed_vals = per_seed[per_seed["variant"] == v][metric].values
        rng = np.random.RandomState(hash((metric, v)) & 0xFFFF)
        jitter = rng.uniform(-0.10, 0.10, size=len(seed_vals))
        ax.scatter(np.full_like(seed_vals, i, dtype=float) + jitter, seed_vals,
                   s=22, color=color_map.get(v, "#888"),
                   edgecolor="white", linewidth=0.4,
                   zorder=3, alpha=0.85)
        ax.errorbar(i, row["mean"], yerr=row["se"],
                    fmt="D", color="black",
                    markerfacecolor=color_map.get(v, "#888"),
                    markersize=5, capsize=3,
                    markeredgecolor="black", markeredgewidth=0.6,
                    elinewidth=1.0, zorder=5)
    if show_xticks:
        ax.set_xticks(x)
        ax.set_xticklabels(variants, fontsize=9)
    else:
        ax.set_xticks([])
    if title:
        ax.set_title(title, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.yaxis.grid(True, alpha=0.25, linestyle="-", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_xlim(-0.5, len(variants) - 0.5)


def grouped_points(
    ax, df_summary: pd.DataFrame, group_col: str, sub_col: str,
    value_col: str, se_col: str, color_map: dict,
    per_seed: pd.DataFrame | None = None, metric: str | None = None,
    title: str | None = None, ylabel: str | None = None,
    group_order: list | None = None, sub_order: list | None = None,
):
    """Grouped point + SE + jittered seed scatter. For bounded style metrics
    plotted across env groups on a zoomed axis. Mirrors grouped_bars layout
    so swapping is straightforward.
    """
    groups = group_order or list(df_summary[group_col].unique())
    subs = sub_order or list(df_summary[sub_col].unique())
    n_sub = len(subs)
    width = 0.8 / n_sub
    x_outer = np.arange(len(groups))
    for j, sub in enumerate(subs):
        offset = (j - (n_sub - 1) / 2) * width
        for i, g in enumerate(groups):
            row = df_summary[(df_summary[group_col] == g) & (df_summary[sub_col] == sub)]
            if len(row) == 0:
                continue
            mean = float(row[value_col].iloc[0])
            se = float(row[se_col].iloc[0])
            xc = x_outer[i] + offset
            if per_seed is not None and metric is not None:
                vals = per_seed[(per_seed[group_col] == g)
                                & (per_seed[sub_col] == sub)][metric].values
                rng = np.random.RandomState(hash((metric, sub, g)) & 0xFFFF)
                jitter = rng.uniform(-width * 0.25, width * 0.25, size=len(vals))
                ax.scatter(np.full_like(vals, xc, dtype=float) + jitter, vals,
                           s=22, color=color_map.get(sub, "#888"),
                           edgecolor="white", linewidth=0.4,
                           zorder=3, alpha=0.85)
            ax.errorbar(xc, mean, yerr=se, fmt="D", color="black",
                        markerfacecolor=color_map.get(sub, "#888"),
                        markersize=5, capsize=3,
                        markeredgecolor="black", markeredgewidth=0.6,
                        elinewidth=1.0, zorder=5,
                        label=sub if i == 0 else None)
    ax.set_xticks(x_outer)
    ax.set_xticklabels(groups, fontsize=9)
    if title:
        ax.set_title(title, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.yaxis.grid(True, alpha=0.25, linestyle="-", linewidth=0.5)
    ax.set_axisbelow(True)


def grouped_bars(
    ax,
    df_summary: pd.DataFrame,
    group_col: str,
    sub_col: str,
    value_col: str,
    se_col: str,
    color_map: dict,
    per_seed: pd.DataFrame | None = None,
    metric: str | None = None,
    title: str | None = None,
    ylabel: str | None = None,
    group_order: list | None = None,
    sub_order: list | None = None,
):
    """Grouped bars: outer x = group_col (e.g. env), inner = sub_col (variant)."""
    groups = group_order or list(df_summary[group_col].unique())
    subs = sub_order or list(df_summary[sub_col].unique())
    n_sub = len(subs)
    width = 0.8 / n_sub
    x_outer = np.arange(len(groups))
    for j, sub in enumerate(subs):
        offset = (j - (n_sub - 1) / 2) * width
        means = []
        ses = []
        for g in groups:
            row = df_summary[(df_summary[group_col] == g) & (df_summary[sub_col] == sub)]
            means.append(float(row[value_col].iloc[0]) if len(row) else np.nan)
            ses.append(float(row[se_col].iloc[0]) if len(row) else 0.0)
        ax.bar(x_outer + offset, means, width, yerr=ses, capsize=3,
               color=color_map.get(sub, "#888"), alpha=0.85,
               edgecolor="black", linewidth=0.5, label=sub)
        # Seed dots if provided
        if per_seed is not None and metric is not None:
            for i, g in enumerate(groups):
                vals = per_seed[
                    (per_seed[group_col] == g) & (per_seed[sub_col] == sub)
                ][metric].values
                rng = np.random.RandomState(hash((sub, g)) & 0xFFFF)
                jitter = rng.uniform(-width * 0.3, width * 0.3, size=len(vals))
                ax.scatter(np.full_like(vals, x_outer[i] + offset, dtype=float) + jitter,
                           vals, s=14, color="white",
                           edgecolor=color_map.get(sub, "#888"),
                           linewidth=0.7, zorder=3, alpha=0.95)
    ax.set_xticks(x_outer)
    ax.set_xticklabels(groups, fontsize=9)
    if title:
        ax.set_title(title, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.yaxis.grid(True, alpha=0.25, linestyle="-", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9)


# ====================================================================
# Number-slice writers
# ====================================================================


def save_numbers(
    fig_dir: Path,
    long_df: pd.DataFrame,
    table_md: str,
    metadata: dict,
):
    """Write numbers.csv (long form), numbers.md (paste-ready table),
    metadata.json (provenance)."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(fig_dir / "numbers.csv", index=False)
    (fig_dir / "numbers.md").write_text(table_md)
    metadata["generated_at"] = datetime.now(timezone.utc).isoformat()
    (fig_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))


def md_table(
    headers: list[str],
    rows: list[list],
    align: list[str] | None = None,
) -> str:
    """Build a markdown table string."""
    if align is None:
        align = ["right" if i > 0 else "left" for i in range(len(headers))]
    sep_map = {"left": ":---", "right": "---:", "center": ":---:"}
    sep_row = [sep_map[a] for a in align]
    out = ["| " + " | ".join(headers) + " |"]
    out.append("| " + " | ".join(sep_row) + " |")
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out) + "\n"


def fmt(v, prec=2):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    if isinstance(v, float):
        return f"{v:.{prec}f}"
    return str(v)


def save_figure(fig, fig_dir: Path, slug: str):
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / f"{slug}.png", dpi=DPI, bbox_inches="tight")
    fig.savefig(fig_dir / f"{slug}.svg", bbox_inches="tight")
    plt.close(fig)


# ====================================================================
# Fig 1 — E1 strategy split (Result 1)
# ====================================================================


FIG1_METRICS = [
    "landed_pct",
    "thrust_autocorr_lag1",
    "side_thrust_reversals_per_step",
    "time_to_first_contact",
]


def _ttest_cohens_d(per_seed: pd.DataFrame, metric: str,
                    group_a: str = "labeled", group_b: str = "blind"
                    ) -> tuple[float | None, float | None]:
    """Welch's t-test + Cohen's d on per-seed means. Used for derived metrics
    not present in the precomputed stat_tests.json."""
    a = per_seed[per_seed["variant"] == group_a][metric].values
    b = per_seed[per_seed["variant"] == group_b][metric].values
    if len(a) < 2 or len(b) < 2:
        return None, None
    from scipy import stats as sp
    t, p = sp.ttest_ind(a, b, equal_var=False)
    sa, sb = a.std(ddof=1), b.std(ddof=1)
    sp_pool = np.sqrt(((len(a)-1)*sa**2 + (len(b)-1)*sb**2) / (len(a) + len(b) - 2))
    d = (a.mean() - b.mean()) / sp_pool if sp_pool > 0 else None
    return float(p), float(d) if d is not None else None


def build_fig1(out_root: Path):
    name = f"parametric-vs-behavioral-v2-{DATE_TAG}"
    comp = "full-variation-easy"
    csv = cross_config_path(name) / comp / "metrics_by_variant.csv"
    df = load_metrics_csv(csv)
    stats = load_stat_tests(name, comp)

    fig_dir = out_root / "fig1-strategy-split"
    fig, axes = plt.subplots(2, 2, figsize=FIG_SIZE_2x2, dpi=DPI)

    long_rows = []
    md_rows = []
    color_map = {"blind": COLORS["blind"], "labeled": COLORS["labeled"]}

    for ax, metric in zip(axes.flat, FIG1_METRICS):
        summ = variant_summary(df, metric)
        per_seed = per_seed_means(df, metric)
        # Order labeled first then blind for consistent reading L→B
        summ = summ.set_index("variant").loc[["labeled", "blind"]].reset_index()
        bar_with_seed_dots(
            ax, summ, per_seed, metric, color_map,
            title=METRIC_LABELS.get(metric, metric),
            ylabel=None, show_xticks=True,
        )
        # Force full natural range for bounded metrics so the panel does not
        # imply landing rates above the plotted top are impossible / off-scale.
        if metric.endswith("_pct"):
            ax.set_ylim(0, 100)
        elif metric == "thrust_autocorr_lag1" or metric.endswith("_autocorr_lag1"):
            ax.set_ylim(0, 1.0)
        elif metric == "frac_upright" or metric.startswith("frac_"):
            ax.set_ylim(0, 1.0)
        # Unbounded count/step metrics (reversals, time_to_first_contact) keep autoscale.
        p, d = lookup_pvalue_d(stats, metric)
        if p is None:
            # Derived metric not in precomputed stat_tests.json; compute inline
            p, d = _ttest_cohens_d(per_seed, metric)
        if p is not None:
            ax.text(0.98, 0.96,
                    f"p={p:.4f}, d={d:.2f}" if d is not None else f"p={p:.4f}",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=8.5, color="#444")
        for v in ["labeled", "blind"]:
            row = summ[summ["variant"] == v].iloc[0]
            long_rows.append({
                "figure": "fig1", "metric": metric, "variant": v,
                "mean": row["mean"], "std": row["std"],
                "se": row["se"], "n_seeds": int(row["n"]),
                "p_value": p, "cohens_d": d,
            })
        # markdown row per metric
        lab_row = summ[summ["variant"] == "labeled"].iloc[0]
        bli_row = summ[summ["variant"] == "blind"].iloc[0]
        md_rows.append([
            METRIC_LABELS.get(metric, metric),
            f"{fmt(lab_row['mean'], 3)} ± {fmt(lab_row['se'], 3)}",
            f"{fmt(bli_row['mean'], 3)} ± {fmt(bli_row['se'], 3)}",
            fmt(p, 4), fmt(d, 2),
            "blind" if bli_row["mean"] > lab_row["mean"] else "labeled",
        ])

    fig.suptitle(
        r"$\bf{Same\ Task,\ Different\ Controllers}$" "\n"
        "\n"
        "Strategy split: full-variation easy",
        fontsize=14, y=1.0, linespacing=1.3,
    )
    fig.text(0.5, -0.01,
             "mean ± SE across N=10 seeds, dots = per-seed means",
             ha="center", va="top", fontsize=9, color="#555")
    fig.tight_layout()
    save_figure(fig, fig_dir, "fig1-strategy-split")

    md = (
        "# Fig 1. State-vector strategy split (full-variation easy)\n\n"
        f"Source: `parametric-vs-behavioral-v2-{DATE_TAG}/full-variation-easy`.\n"
        "N=10 seeds per variant. Mean ± SE across seed-level means.\n\n"
    )
    md += md_table(
        ["Metric", "Labeled (mean ± SE)", "Blind (mean ± SE)", "p", "Cohen's d", "winner"],
        md_rows,
    )
    save_numbers(
        fig_dir, pd.DataFrame(long_rows), md,
        {
            "figure": "fig1",
            "title": "State-vector strategy split (full-variation easy)",
            "source_csv": str(csv),
            "source_stats": str(cross_config_path(name) / "stat_tests.json"),
            "manifest": "parametric-vs-behavioral-v2.yaml",
            "comparison": comp,
            "n_seeds_per_variant": 10,
            "metrics": FIG1_METRICS,
            "variants": ["labeled", "blind"],
        },
    )
    print(f"  wrote: {fig_dir}/")


# ====================================================================
# Fig 2 — Body-vs-world diagnostic (Result 1 / Result 2 bridge)
# ====================================================================


FIG2_METRICS = ["thrust_autocorr_lag1", "side_thrust_reversals_per_step"]


def build_fig2(out_root: Path):
    """Body-vs-world diagnostic. Two panels (1x2), one per metric.
    Each panel: grouped bars across {world-only, body-only, full-variation}
    envs × {labeled, blind} with per-seed dots and per-group p/d annotation.
    """
    fig_dir = out_root / "fig2-body-vs-world"
    name = f"parametric-vs-behavioral-{DATE_TAG}"
    env_to_comp = [
        ("world-only", "physics-only-easy"),
        ("body-only", "vehicle-only-easy"),
        ("full-variation", "full-variation-easy"),
    ]
    group_order = [e for e, _ in env_to_comp]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), dpi=DPI)
    long_rows = []
    md_rows_by_metric = {m: [] for m in FIG2_METRICS}

    for ax, metric in zip(axes, FIG2_METRICS):
        rows_summary = []
        rows_per_seed = []
        pd_per_group = {}
        for env_label, comp in env_to_comp:
            df = load_metrics_csv(
                cross_config_path(name) / comp / "metrics_by_variant.csv")
            stats = load_stat_tests(name, comp)
            summ = variant_summary(df, metric)
            ps = per_seed_means(df, metric)
            for v in ["labeled", "blind"]:
                r = summ[summ["variant"] == v].iloc[0]
                rows_summary.append({
                    "env": env_label, "variant": v,
                    "mean": r["mean"], "se": r["se"],
                    "std": r["std"], "n": int(r["n"]),
                })
            for _, r in ps.iterrows():
                rows_per_seed.append({"env": env_label, "variant": r["variant"],
                                      metric: r[metric]})
            p, d = lookup_pvalue_d(stats, metric)
            pd_per_group[env_label] = (p, d)
            lab = summ[summ["variant"] == "labeled"].iloc[0]
            bli = summ[summ["variant"] == "blind"].iloc[0]
            md_rows_by_metric[metric].append([
                env_label,
                f"{fmt(lab['mean'], 3)} ± {fmt(lab['se'], 3)}",
                f"{fmt(bli['mean'], 3)} ± {fmt(bli['se'], 3)}",
                fmt(p, 4), fmt(d, 2),
            ])
            long_rows.append({
                "figure": "fig2", "env": env_label, "metric": metric,
                "labeled_mean": lab["mean"], "labeled_se": lab["se"],
                "blind_mean": bli["mean"], "blind_se": bli["se"],
                "p_value": p, "cohens_d": d, "n_seeds": int(lab["n"]),
            })

        df_summary = pd.DataFrame(rows_summary)
        df_per_seed = pd.DataFrame(rows_per_seed)
        grouped_bars(
            ax, df_summary, "env", "variant", "mean", "se",
            {"labeled": COLORS["labeled"], "blind": COLORS["blind"]},
            per_seed=df_per_seed, metric=metric,
            title=METRIC_LABELS.get(metric, metric),
            ylabel=METRIC_LABELS.get(metric, metric),
            group_order=group_order, sub_order=["labeled", "blind"],
        )
        if metric == "thrust_autocorr_lag1":
            ax.set_ylim(0, 1.0)
        elif metric == "side_thrust_reversals_per_step":
            ymax = df_summary["mean"].max() + df_summary["se"].max() * 2
            ax.set_ylim(0, ymax * 1.18)

        # Per-group p/d annotation, placed just above the taller bar in the pair.
        y_lo, y_hi = ax.get_ylim()
        offset = 0.025 * (y_hi - y_lo)
        for i, env_label in enumerate(group_order):
            p, d = pd_per_group[env_label]
            if p is None:
                continue
            top = df_summary[df_summary["env"] == env_label]
            bar_top = float((top["mean"] + top["se"]).max())
            txt = f"p={p:.4f}\nd={d:.2f}" if d is not None else f"p={p:.4f}"
            ax.text(i, bar_top + offset, txt,
                    ha="center", va="bottom", fontsize=8, color="#444")

        if metric == FIG2_METRICS[1]:
            ax.legend(loc="upper left", frameon=False, fontsize=9)
        else:
            leg = ax.get_legend()
            if leg is not None:
                leg.remove()

    fig.suptitle(
        r"$\bf{Body\ Uncertainty\ Changes\ the\ Controller}$" "\n"
        "\n"
        "Body-vs-world behavioral diagnostic: blind vs labeled (easy profile)",
        fontsize=14, y=1.0, linespacing=1.3,
    )
    fig.text(0.5, -0.01,
             "mean ± SE across N=10 seeds, dots = per-seed means",
             ha="center", va="top", fontsize=9, color="#555")
    fig.tight_layout()
    save_figure(fig, fig_dir, "fig2-body-vs-world")

    md = (
        "# Fig 2. Body Uncertainty Opens the Reactive Gap\n\n"
        f"Sources: `parametric-vs-behavioral-{DATE_TAG}/"
        "{physics-only-easy,vehicle-only-easy,full-variation-easy}`.\n"
        "World-only = physics-only env (only gravity/wind/turbulence vary). "
        "Body-only = vehicle-only env (only engine/density/damping vary). "
        "Full-variation = both vary.\n\n"
    )
    for metric in FIG2_METRICS:
        md += f"## {METRIC_LABELS.get(metric, metric)}\n\n"
        md += md_table(
            ["Env",
             f"Labeled (mean ± SE)",
             f"Blind (mean ± SE)",
             "p", "Cohen's d"],
            md_rows_by_metric[metric],
        )
        md += "\n"
    save_numbers(
        fig_dir, pd.DataFrame(long_rows), md,
        {
            "figure": "fig2",
            "title": "Body Uncertainty Opens the Reactive Gap",
            "source_csvs": [
                str(cross_config_path(name) / c / "metrics_by_variant.csv")
                for _, c in env_to_comp
            ],
            "source_stats": str(cross_config_path(name) / "stat_tests.json"),
            "manifest": "parametric-vs-behavioral.yaml",
            "metrics": FIG2_METRICS,
            "envs": group_order,
            "n_seeds_per_variant": 10,
        },
    )
    print(f"  wrote: {fig_dir}/")


# ====================================================================
# Fig 3 — Corruption ladder (Result 2)
# ====================================================================


CORRUPT_VARIANTS = {
    "labeled": "baseline",
    "labeled-world-zeroed": "world-zero",
    "labeled-body-zeroed": "body-zero",
    "labeled-zeroed": "all-zero",
}
CORRUPT_ORDER = ["baseline", "world-zero", "body-zero", "all-zero"]


FIG3_ENV_LABELS = {
    "full-variation-easy": "full-variation",
    "physics-only-easy":   "world-only",
    "vehicle-only-easy":   "body-only",
}
FIG3_MODE_COLORS = {
    "baseline":   "#2c7fb8",
    "world-zero": "#7fcdbb",
    "body-zero":  "#f2a154",
    "all-zero":   "#c0392b",
}
# Cross-config CSVs to read blind-reference autocorr from for each env column.
# Keyed by env-column label used in build_fig3.
FIG3_BLIND_REF_CSV = {
    "full-variation": lambda: cross_config_path(
        f"parametric-vs-behavioral-v2-{DATE_TAG}") / "full-variation-easy" /
        "metrics_by_variant.csv",
    "world-only":     lambda: cross_config_path(
        f"parametric-vs-behavioral-{DATE_TAG}") / "physics-only-easy" /
        "metrics_by_variant.csv",
    "body-only":      lambda: cross_config_path(
        f"parametric-vs-behavioral-{DATE_TAG}") / "vehicle-only-easy" /
        "metrics_by_variant.csv",
}


def _blind_autocorr_ref(env_label: str):
    """Per-seed mean + SE of thrust_autocorr_lag1 for the blind variant."""
    csv = FIG3_BLIND_REF_CSV[env_label]()
    if not csv.exists():
        return None, None
    df = pd.read_csv(csv)
    df = df[df["variant"] == "blind"]
    ps = df.groupby("seed")["thrust_autocorr_lag1"].mean()
    if len(ps) < 2:
        return float(ps.mean()), 0.0
    return float(ps.mean()), float(ps.std(ddof=1) / np.sqrt(len(ps)))


def build_fig3(out_root: Path):
    """Wrong Labels, Same Controller Style. 2 rows × 3 cols.
    Autocorr row uses filled bars on full [0,1] range (per user preference).
    """
    _build_fig3_variant(out_root, autocorr_style="bars-full",
                        out_slug="fig3-corruption-style")


def _build_fig3_variant(out_root: Path, autocorr_style: str, out_slug: str):
    fig_dir = out_root / "fig3-corruption-style"
    name = f"body-vs-world-corruption-{DATE_TAG}"
    env_comps = [
        ("full-variation", "full-variation-easy"),
        ("world-only",     "physics-only-easy"),
        ("body-only",      "vehicle-only-easy"),
    ]
    metrics = ["landed_pct", "thrust_autocorr_lag1"]

    fig, axes = plt.subplots(
        2, 3, figsize=(11.5, 6.2), dpi=DPI,
        gridspec_kw={"height_ratios": [2.4, 1.4]},
        sharex="col",
    )

    long_rows = []
    md_rows_by_metric = {m: [] for m in metrics}

    for col, (env_label, comp) in enumerate(env_comps):
        df = load_metrics_csv(cross_config_path(name) / comp / "metrics_by_variant.csv")
        df = df[df["variant"].isin(CORRUPT_VARIANTS.keys())].copy()
        df["mode"] = df["variant"].map(CORRUPT_VARIANTS)
        # per_seed_means groups by ("variant", "seed") and handles _pct outcomes.
        df_tmp = df.rename(columns={"variant": "_orig_variant", "mode": "variant"})

        for row_idx, metric in enumerate(metrics):
            ax = axes[row_idx, col]
            ps = per_seed_means(df_tmp, metric)
            per_mode = {}
            for mode in CORRUPT_ORDER:
                vals = ps[ps["variant"] == mode][metric].values
                per_mode[mode] = vals

            baseline_vals = per_mode["baseline"]
            # Squeeze x spacing now that bars are narrow so the cell isn't
            # mostly whitespace.
            x = np.arange(len(CORRUPT_ORDER)) * 0.6
            means = np.array([
                per_mode[m].mean() if len(per_mode[m]) else np.nan
                for m in CORRUPT_ORDER
            ])
            ses = np.array([
                per_mode[m].std(ddof=1) / np.sqrt(len(per_mode[m]))
                if len(per_mode[m]) > 1 else 0.0
                for m in CORRUPT_ORDER
            ])
            colors = [FIG3_MODE_COLORS[m] for m in CORRUPT_ORDER]
            use_bars = (metric == "landed_pct"
                        or (metric == "thrust_autocorr_lag1"
                            and autocorr_style == "bars-full"))
            if use_bars:
                # Bars + SE + per-seed dots for the headline collapse story.
                ax.bar(x, means, 0.28, yerr=ses, capsize=3,
                       color=colors, alpha=0.88,
                       edgecolor="black", linewidth=0.5)
                for i, mode in enumerate(CORRUPT_ORDER):
                    v = per_mode[mode]
                    if len(v) == 0:
                        continue
                    rng = np.random.RandomState(hash((metric, env_label, mode)) & 0xFFFF)
                    jitter = rng.uniform(-0.10, 0.10, size=len(v))
                    ax.scatter(np.full_like(v, x[i], dtype=float) + jitter, v,
                               s=14, color="white",
                               edgecolor=FIG3_MODE_COLORS[mode],
                               linewidth=0.7, zorder=3, alpha=0.95)
            else:
                # Dot + SE error bars + jittered seed points, no bars.
                # Bars on a zoomed [0.6,1.0] axis would imply the floor is
                # meaningful and visually exaggerate the differences.
                for i, mode in enumerate(CORRUPT_ORDER):
                    v = per_mode[mode]
                    if len(v) == 0:
                        continue
                    rng = np.random.RandomState(hash((metric, env_label, mode)) & 0xFFFF)
                    jitter = rng.uniform(-0.10, 0.10, size=len(v))
                    ax.scatter(np.full_like(v, x[i], dtype=float) + jitter, v,
                               s=22, color=FIG3_MODE_COLORS[mode],
                               edgecolor="white", linewidth=0.4,
                               zorder=3, alpha=0.85)
                    # Mean marker: smaller diamond + thin error bar so seed
                    # points stay readable alongside it.
                    ax.errorbar(x[i], means[i], yerr=ses[i],
                                fmt="D", color="black",
                                markerfacecolor=FIG3_MODE_COLORS[mode],
                                markersize=5, capsize=3,
                                markeredgecolor="black", markeredgewidth=0.6,
                                elinewidth=1.0, zorder=5)

            ax.set_xticks(x)
            ax.set_xticklabels(CORRUPT_ORDER, fontsize=8, rotation=20)
            ax.set_xlim(x[0] - 0.4, x[-1] + 0.4)
            ax.yaxis.grid(True, alpha=0.25, linestyle="-", linewidth=0.5)
            ax.set_axisbelow(True)
            if metric == "landed_pct":
                ax.set_ylim(0, 100)
            elif metric == "thrust_autocorr_lag1":
                ax.set_ylim(0, 1.0) if autocorr_style == "bars-full" else ax.set_ylim(0.6, 1.0)
                # Blind reference: horizontal dashed line + ±SE band, so the
                # reader can judge whether corrupted labeled drops toward blind.
                mu, se = _blind_autocorr_ref(env_label)
                if mu is not None:
                    ax.axhline(mu, color="#444", lw=1.0, linestyle="--",
                               zorder=1)
                    # Label sits below the dashed line at the left edge with
                    # enough offset to not touch the line itself.
                    y_lo, y_hi = ax.get_ylim()
                    label_y = mu - 0.03 * (y_hi - y_lo)
                    # Sit just inside the left edge (a smidge right of the
                    # y-axis spine) so it doesn't crash into the axis line.
                    ax.text(x[0] - 0.30, label_y, f"blind ref = {mu:.2f}",
                            ha="left", va="top", fontsize=7.5,
                            color="#444", fontstyle="italic")
            if col == 0:
                ax.set_ylabel(METRIC_LABELS[metric], fontsize=10)
            if row_idx == 0:
                ax.set_title(env_label, fontsize=11, fontweight="bold")

            # Per-bar p/d vs clean baseline (skip baseline itself).
            y_lo, y_hi = ax.get_ylim()
            offset = 0.025 * (y_hi - y_lo)
            for i, mode in enumerate(CORRUPT_ORDER):
                if mode == "baseline" or len(baseline_vals) < 2 or len(per_mode[mode]) < 2:
                    continue
                p, d = _ttest_cohens_d_arrays(baseline_vals, per_mode[mode])
                if p is None:
                    continue
                bar_top = means[i] + ses[i]
                txt = f"p={p:.3f}\nd={d:.2f}" if d is not None else f"p={p:.3f}"
                ax.text(x[i], bar_top + offset, txt,
                        ha="center", va="bottom", fontsize=7, color="#444")

            # Record numbers.
            for i, mode in enumerate(CORRUPT_ORDER):
                v = per_mode[mode]
                if len(v) == 0:
                    continue
                p, d = (None, None)
                if mode != "baseline" and len(baseline_vals) >= 2:
                    p, d = _ttest_cohens_d_arrays(baseline_vals, v)
                long_rows.append({
                    "figure": "fig3", "env": env_label, "mode": mode,
                    "metric": metric, "mean": float(v.mean()),
                    "se": float(ses[i]), "std": float(v.std(ddof=1) if len(v) > 1 else 0.0),
                    "n_seeds": int(len(v)),
                    "p_vs_clean": p, "cohens_d_vs_clean": d,
                })
                md_rows_by_metric[metric].append([
                    env_label, mode,
                    f"{fmt(v.mean(), 3)} ± {fmt(ses[i], 3)}",
                    fmt(p, 4) if p is not None else "—",
                    fmt(d, 2) if d is not None else "—",
                ])

    fig.suptitle(
        r"$\bf{Wrong\ Labels,\ Same\ Controller\ Style}$" "\n"
        "\n"
        "Labeled agent under label corruption (easy profile)",
        fontsize=14, y=1.0, linespacing=1.3,
    )
    fig.text(0.5, -0.02,
             "mean ± SE across N=10 seeds, dots = per-seed means",
             ha="center", va="top", fontsize=9, color="#555")
    fig.tight_layout()
    save_figure(fig, fig_dir, out_slug)

    md = (
        "# Fig 3. Wrong Labels, Same Controller Style\n\n"
        f"Source: `body-vs-world-corruption-{DATE_TAG}/"
        "{full-variation-easy,physics-only-easy,vehicle-only-easy}`. "
        "Labeled agent only. N=10 seeds per (env, mode).\n\n"
        "Reading: rows are metrics; columns are envs. Each cell shows the "
        "labeled agent under clean labels vs three corruption modes. "
        "p/d above each non-clean bar is Welch's t-test vs the clean baseline "
        "for that env+metric, per-seed means.\n\n"
    )
    for metric in metrics:
        md += f"## {METRIC_LABELS.get(metric, metric)}\n\n"
        md += md_table(
            ["Env", "Mode", f"{metric} (mean ± SE)", "p (vs clean)", "d (vs clean)"],
            md_rows_by_metric[metric],
        )
        md += "\n"
    save_numbers(
        fig_dir, pd.DataFrame(long_rows), md,
        {
            "figure": "fig3",
            "title": "Wrong Labels, Same Controller Style",
            "source_csvs": [str(cross_config_path(name) / c / "metrics_by_variant.csv")
                            for _, c in env_comps],
            "manifest": "body-vs-world-corruption.yaml",
            "metrics": metrics,
            "envs": [e for e, _ in env_comps],
            "modes": CORRUPT_ORDER,
            "n_seeds_per_variant": 10,
        },
    )
    print(f"  wrote: {fig_dir}/")


def _ttest_cohens_d_arrays(a, b):
    """Welch's t-test + Cohen's d on two 1-D numpy arrays (per-seed means)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return None, None
    from scipy import stats as sp
    _, p = sp.ttest_ind(a, b, equal_var=False)
    sa, sb = a.std(ddof=1), b.std(ddof=1)
    sp_pool = np.sqrt(((len(a)-1)*sa**2 + (len(b)-1)*sb**2) / (len(a) + len(b) - 2))
    d = (a.mean() - b.mean()) / sp_pool if sp_pool > 0 else None
    return float(p), float(d) if d is not None else None


# ====================================================================
# Fig 4 — E1 OOD divergence (Result 3)
# ====================================================================


def build_fig4(out_root: Path):
    fig_dir = out_root / "fig4-ood-divergence"
    sources = [
        ("easy", "parametric-vs-behavioral-v2-2026-05-13", "full-variation-easy"),
        ("medium", "ood-generalization-medium-2026-05-13",
         "full-variation-easy-trained-on-medium"),
        ("hard", "ood-generalization-hard-2026-05-13",
         "full-variation-easy-trained-on-hard"),
    ]

    rows_summary = []
    rows_per_seed = []
    long_rows = []
    md_rows = []
    for eval_profile, name, comp in sources:
        df = load_metrics_csv(cross_config_path(name) / comp / "metrics_by_variant.csv")
        stats = load_stat_tests(name, comp)
        for metric in ["landed_pct", "crashed_pct"]:
            summ = variant_summary(df, metric)
            ps = per_seed_means(df, metric)
            for v in ["labeled", "blind"]:
                r = summ[summ["variant"] == v].iloc[0]
                rows_summary.append({
                    "eval": eval_profile, "variant": v, "metric": metric,
                    "mean": r["mean"], "se": r["se"], "std": r["std"], "n": int(r["n"]),
                })
            for _, r in ps.iterrows():
                rows_per_seed.append({"eval": eval_profile, "variant": r["variant"],
                                      "metric": metric, "value": r[metric]})
            p, d = lookup_pvalue_d(stats, metric)
            lab = summ[summ["variant"] == "labeled"].iloc[0]
            bli = summ[summ["variant"] == "blind"].iloc[0]
            md_rows.append([
                eval_profile, METRIC_LABELS[metric],
                f"{fmt(lab['mean'], 1)} ± {fmt(lab['se'], 1)}",
                f"{fmt(bli['mean'], 1)} ± {fmt(bli['se'], 1)}",
                fmt(p, 4), fmt(d, 2),
            ])
            long_rows.append({
                "figure": "fig4", "eval": eval_profile, "metric": metric,
                "labeled_mean": lab["mean"], "labeled_se": lab["se"],
                "blind_mean": bli["mean"], "blind_se": bli["se"],
                "p_value": p, "cohens_d": d,
            })

    df_summary = pd.DataFrame(rows_summary)
    df_per_seed = pd.DataFrame(rows_per_seed)

    fig, (ax_land, ax_crash) = plt.subplots(1, 2, figsize=FIG_SIZE_OOD, dpi=DPI,
                                            sharex=True)
    profiles = ["easy", "medium", "hard"]
    x = np.arange(len(profiles))
    for ax, metric, ylabel in [
        (ax_land, "landed_pct", "Landing rate (%)"),
        (ax_crash, "crashed_pct", "Crash rate (%)"),
    ]:
        for variant in ["labeled", "blind"]:
            sub = df_summary[
                (df_summary["variant"] == variant) & (df_summary["metric"] == metric)
            ].set_index("eval").loc[profiles]
            ax.errorbar(
                x, sub["mean"], yerr=sub["se"],
                marker="o", capsize=4, color=COLORS[variant],
                linewidth=2.2, markersize=7, label=variant,
            )
            # seed dots
            for i, prof in enumerate(profiles):
                vals = df_per_seed[
                    (df_per_seed["eval"] == prof) & (df_per_seed["variant"] == variant)
                    & (df_per_seed["metric"] == metric)
                ]["value"].values
                rng = np.random.RandomState(hash((variant, metric, prof)) & 0xFFFF)
                jitter = rng.uniform(-0.08, 0.08, size=len(vals))
                ax.scatter(np.full_like(vals, x[i], dtype=float) + jitter, vals,
                           s=14, color="white", edgecolor=COLORS[variant],
                           linewidth=0.7, zorder=3, alpha=0.95)
        ax.set_xticks(x)
        ax.set_xticklabels(profiles)
        ax.set_xlabel("Evaluation profile")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 100)
        ax.yaxis.grid(True, alpha=0.25, linestyle="-", linewidth=0.5)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=9)

    fig.suptitle("Out-of-distribution divergence: easy-trained state-vector "
                 "agents evaluated across profiles",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_figure(fig, fig_dir, "fig4-ood-divergence")

    md = (
        "# Fig 4. State-vector out-of-distribution divergence\n\n"
        "Easy-trained labeled and blind agents evaluated on easy / medium / hard. "
        "Sources: easy column from `parametric-vs-behavioral-v2-2026-05-13`, "
        "medium and hard from `ood-generalization-{medium,hard}-2026-05-13`. "
        "N=10 seeds per variant.\n\n"
    )
    md += md_table(
        ["Eval profile", "Metric", "Labeled (mean ± SE)", "Blind (mean ± SE)",
         "p", "Cohen's d"],
        md_rows,
    )
    save_numbers(
        fig_dir, pd.DataFrame(long_rows), md,
        {
            "figure": "fig4",
            "title": "State-vector out-of-distribution divergence",
            "sources": [
                {"eval": e, "csv": str(cross_config_path(n) / c / "metrics_by_variant.csv")}
                for e, n, c in sources
            ],
            "metrics": ["landed_pct", "crashed_pct"],
            "n_seeds_per_variant": 10,
        },
    )
    print(f"  wrote: {fig_dir}/")


# ====================================================================
# Fig 5 — E3 fusion-strength gradient (Result 5)
# ====================================================================


def build_fig5(out_root: Path):
    """Two panels: ID landed% (frozen only — blind/raw/branch),
    Medium OOD landed% (same 3 conditions), with secondary strip of reversals."""
    fig_dir = out_root / "fig5-fusion-gradient"
    NETS = DEFAULT_NETS
    # ID source: e3-n3-frozen-only-final (only the 3 frozen conditions)
    id_csv = NETS / "analysis/cross-config/e3-n3-frozen-only-final/frozen-only-final/metrics_by_variant.csv"
    id_stats = NETS / "analysis/cross-config/e3-n3-frozen-only-final/stat_tests.json"
    # OOD source: e3-n3-all-6-final-ood-medium (filtered to frozen rows)
    ood_csv = NETS / "analysis/cross-config/e3-n3-all-6-final-ood-medium/all-6-final-ood-medium/metrics_by_variant.csv"
    ood_stats = NETS / "analysis/cross-config/e3-n3-all-6-final-ood-medium/stat_tests.json"

    df_id = load_metrics_csv(id_csv)
    df_ood = load_metrics_csv(ood_csv)
    df_ood = df_ood[df_ood["variant"].str.endswith("-frozen")].copy()

    conditions = ["blind-frozen", "raw-frozen", "branch-frozen"]
    cond_color = {
        "blind-frozen": COLORS["blind-visual"],
        "raw-frozen": COLORS["raw"],
        "branch-frozen": COLORS["branch"],
    }

    long_rows = []
    md_rows_landed = []
    md_rows_rev = []

    fig = plt.figure(figsize=FIG_SIZE_FUSION, dpi=DPI)
    gs = fig.add_gridspec(2, 2, height_ratios=[2.5, 1], hspace=0.45, wspace=0.2)

    # --- Top row: landed% ID and OOD ---
    for col, (ax, df, eval_label, src) in enumerate([
        (fig.add_subplot(gs[0, 0]), df_id, "easy (ID)", "id"),
        (fig.add_subplot(gs[0, 1]), df_ood, "medium (OOD)", "ood"),
    ]):
        summ = variant_summary(df, "landed_pct")
        ps = per_seed_means(df, "landed_pct")
        summ = summ.set_index("variant").loc[conditions].reset_index()
        bar_with_seed_dots(
            ax, summ, ps, "landed_pct", cond_color,
            title=f"{eval_label}: landing rate",
            ylabel="Landing rate (%)",
            show_xticks=True,
        )
        ax.set_xticklabels(["blind", "raw", "branch"])
        ax.set_ylim(0, 100)
        for v in conditions:
            r = summ[summ["variant"] == v].iloc[0]
            long_rows.append({
                "figure": "fig5", "panel": "landed_pct", "eval": src,
                "variant": v, "mean": r["mean"], "se": r["se"],
                "std": r["std"], "n_seeds": int(r["n"]),
            })
            md_rows_landed.append([
                eval_label, v.replace("-frozen", ""),
                f"{fmt(r['mean'], 1)} ± {fmt(r['se'], 1)}", int(r["n"]),
            ])

    # --- Bottom row: side_thrust_reversals strip (ID and OOD) ---
    # Both strips share a y-axis so the reader can read OOD as a vertical
    # drop from the ID values directly above. Without a shared range,
    # matplotlib autoscales each strip independently and the visual
    # "ID → OOD" comparison breaks.
    metric_strip = "side_thrust_reversals"
    id_summ = variant_summary(df_id, metric_strip).set_index("variant").loc[conditions]
    ood_summ = variant_summary(df_ood, metric_strip).set_index("variant").loc[conditions]
    rev_top = float(max((id_summ["mean"] + id_summ["se"]).max(),
                        (ood_summ["mean"] + ood_summ["se"]).max())) * 1.18
    ax_rev_id = fig.add_subplot(gs[1, 0])
    ax_rev_ood = fig.add_subplot(gs[1, 1], sharey=ax_rev_id)
    for ax, df, eval_label, src in [
        (ax_rev_id, df_id, "easy (ID)", "id"),
        (ax_rev_ood, df_ood, "medium (OOD)", "ood"),
    ]:
        summ = variant_summary(df, metric_strip)
        ps = per_seed_means(df, metric_strip)
        summ = summ.set_index("variant").loc[conditions].reset_index()
        bar_with_seed_dots(
            ax, summ, ps, metric_strip, cond_color,
            title=None,
            ylabel="Reversals/ep" if ax is ax_rev_id else None,
            show_xticks=True,
        )
        ax.set_xticklabels(["blind", "raw", "branch"])
        ax.set_ylim(0, rev_top)
        for v in conditions:
            r = summ[summ["variant"] == v].iloc[0]
            long_rows.append({
                "figure": "fig5", "panel": "side_thrust_reversals", "eval": src,
                "variant": v, "mean": r["mean"], "se": r["se"],
                "std": r["std"], "n_seeds": int(r["n"]),
            })
            md_rows_rev.append([
                eval_label, v.replace("-frozen", ""),
                f"{fmt(r['mean'], 1)} ± {fmt(r['se'], 1)}", int(r["n"]),
            ])

    fig.suptitle(
        "Visual fusion-strength gradient: blind, raw, branch (frozen encoder, N=3)",
        fontsize=12, fontweight="bold", y=1.0,
    )
    save_figure(fig, fig_dir, "fig5-fusion-gradient")

    md = (
        "# Fig 5. Visual fusion-strength gradient\n\n"
        "Visual frozen agents (blind, raw-fusion, branch-fusion) on easy (ID) and "
        "medium (OOD). Sources: ID from `e3-n3-frozen-only-final`, OOD from "
        "`e3-n3-all-6-final-ood-medium` (frozen rows). N=3 seeds.\n\n"
        "## Landing rate\n\n"
    )
    md += md_table(["Eval", "Condition", "Landed % (mean ± SE)", "N"], md_rows_landed)
    md += "\n## Side-thrust reversals\n\n"
    md += md_table(["Eval", "Condition", "Reversals/ep (mean ± SE)", "N"], md_rows_rev)

    save_numbers(
        fig_dir, pd.DataFrame(long_rows), md,
        {
            "figure": "fig5",
            "title": "Visual fusion-strength gradient",
            "sources": {
                "id_csv": str(id_csv),
                "ood_csv": str(ood_csv),
            },
            "conditions": conditions,
            "n_seeds_per_variant": 3,
            "metrics": ["landed_pct", "side_thrust_reversals"],
        },
    )
    print(f"  wrote: {fig_dir}/")


# ====================================================================
# Result 4 compact table (state vs visual gym-default)
# ====================================================================


def build_result4_table(out_root: Path):
    """Compact comparison: state blind/labeled vs visual frozen/finetuned.
    Source: e1-blind-vs-labeled-gym-default-128 (state) +
            matched-enc-all-6-conditions (visual).
    Metrics: landed_pct, crashed_pct, side_thrust_reversals,
             side_thrust_autocorr_lag1, thrust_duty_cycle, frac_upright.
    """
    fig_dir = out_root / "result4-state-vs-visual"
    NETS = DEFAULT_NETS
    state_csv = NETS / "analysis/cross-config/e1-blind-vs-labeled-gym-default-128"
    state_inner = next((p for p in state_csv.iterdir() if p.is_dir()), None)
    state_df = load_metrics_csv(state_inner / "metrics_by_variant.csv")

    visual_csv = NETS / "analysis/cross-config/matched-enc-all-6-conditions/all-6/metrics_by_variant.csv"
    visual_df = load_metrics_csv(visual_csv)
    # matched-enc has variants {frozen-baseline, finetune-baseline, ...swap variants...}
    visual_df = visual_df[visual_df["variant"].isin(["frozen-baseline", "finetune-baseline"])].copy()

    metrics_r4 = [
        ("landed_pct", "Landing rate (%)"),
        ("crashed_pct", "Crash rate (%)"),
        ("side_thrust_reversals", "Side-thrust reversals"),
        ("side_thrust_autocorr_lag1", "Side-thrust autocorr"),
        ("thrust_duty_cycle", "Thrust duty cycle"),
        ("frac_upright", "Fraction upright"),
    ]

    long_rows = []
    columns = ["State blind", "State labeled", "Visual frozen", "Visual fine-tuned"]
    rows = []
    for m, label in metrics_r4:
        state_summ = variant_summary(state_df, m).set_index("variant")
        vis_summ = variant_summary(visual_df, m).set_index("variant")
        # If frac_upright was not in the state CSV (old E1 set), might be missing here too — guard:
        def cell(summ, key):
            if key not in summ.index:
                return "—", None, None, None
            r = summ.loc[key]
            return f"{fmt(r['mean'], 3 if 'pct' not in m and 'reversals' not in m else 1)} ± {fmt(r['se'], 3 if 'pct' not in m and 'reversals' not in m else 1)}", r["mean"], r["se"], int(r["n"])
        sb, sb_m, sb_se, sb_n = cell(state_summ, "blind")
        sl, sl_m, sl_se, sl_n = cell(state_summ, "labeled")
        vf, vf_m, vf_se, vf_n = cell(vis_summ, "frozen-baseline")
        vt, vt_m, vt_se, vt_n = cell(vis_summ, "finetune-baseline")
        rows.append([label, sb, sl, vf, vt])
        for cond_label, mean, se, n in [
            ("state-blind", sb_m, sb_se, sb_n),
            ("state-labeled", sl_m, sl_se, sl_n),
            ("visual-frozen", vf_m, vf_se, vf_n),
            ("visual-finetuned", vt_m, vt_se, vt_n),
        ]:
            long_rows.append({
                "figure": "result4", "metric": m, "condition": cond_label,
                "mean": mean, "se": se, "n_seeds": n,
            })

    md = (
        "# Result 4. Perception pushes control toward the reactive regime (gym-default, fixed physics)\n\n"
        "State agents vs visual agents under fixed Lunar Lander physics. State sources: "
        "`e1-blind-vs-labeled-gym-default-128`. Visual sources: "
        "`matched-enc-all-6-conditions` (frozen-baseline = frozen encoder, "
        "finetune-baseline = fine-tuned encoder, N=5).\n\n"
    )
    md += md_table(["Metric"] + columns, rows)

    save_numbers(
        fig_dir, pd.DataFrame(long_rows), md,
        {
            "figure": "result4-table",
            "title": "State vs visual gym-default comparison",
            "state_source": str(state_inner / "metrics_by_variant.csv") if state_inner else "",
            "visual_source": str(visual_csv),
            "metrics": [m for m, _ in metrics_r4],
            "conditions": columns,
        },
    )
    print(f"  wrote: {fig_dir}/  (table only, no plot)")


# ====================================================================
# Wrap existing overlay artifacts
# ====================================================================


def wrap_overlay(out_root: Path):
    """Copy state V_D overlay and reversal-rate-over-time into figures/ with
    minimal companion files. Numbers are not regenerated since the overlay
    aggregates 50 trajectories per condition — the underlying data is the
    per-episode npz files; metadata.json documents that.
    """
    for src, slug, title, note in [
        (Path("/home/vsr/vsr-tmp/e1-overlay/easy-ID_vB.png"), "fig-overlay-state-vb-easy-ID",
         "V_B reversal-dot overlay: full-variation easy ID (blind vs labeled)",
         "Faint grey paths are sampled trajectories. Yellow dots: main-thrust on/off transitions. "
         "Magenta dots: lateral L/R/none changes. Dense speckled bands indicate repeated correction; "
         "sparse dots along a narrow corridor indicate smoother control. Underlying script: "
         "`scripts/viz/trajectory_overlay_variants.py` (variant_B)."),
        (Path("/home/vsr/vsr-tmp/e1-overlay/medium-ID_vB.png"), "fig-overlay-state-vb-medium-ID",
         "V_B reversal-dot overlay: full-variation medium ID (blind vs labeled)",
         "Same conventions as the easy-ID V_B overlay. Medium physics typically produces a wider, "
         "denser blind correction band, making the smooth-vs-reactive contrast more visually obvious."),
        (Path("/home/vsr/vsr-tmp/e1-overlay/easy-OOD-medium_vB.png"), "fig-overlay-state-vb-easy-OOD-medium",
         "V_B reversal-dot overlay: easy-trained-on-medium OOD (blind vs labeled)",
         "Easy-trained agents evaluated on medium OOD physics. Dot density reveals where corrective "
         "events concentrate spatially. Labeled OOD trajectories often end earlier (early impact)."),
        (OVERLAY_VD_SRC, "fig-overlay-state-vd",
         "State-vector trajectory overlay: full-variation easy (blind vs labeled)",
         "50 trajectories per condition, time-shaded color, with per-episode "
         "thrust-action staircases. Outcome triangles: green = landed, red = failed."),
        (REVERSAL_TIME_SRC, "fig-reversal-rate-over-time",
         "Reversal rate over normalized trajectory time (3 scenarios × main/lateral/any)",
         "Per-bin mean ± SE across N=100 episodes per condition. Rows: "
         "easy-ID / easy-OOD-medium / medium-ID. Columns: any / main / lateral reversals."),
    ]:
        if not src.exists():
            print(f"  SKIP: {src} not found")
            continue
        fig_dir = out_root / slug
        fig_dir.mkdir(parents=True, exist_ok=True)
        dst = fig_dir / f"{slug}.png"
        shutil.copy2(src, dst)
        (fig_dir / "numbers.md").write_text(
            f"# {title}\n\n{note}\n\n"
            f"Source image: `{src}`. "
            f"Underlying script: `scripts/viz/trajectory_overlay_variants.py` "
            f"(overlay V_D) or `scripts/viz/reversal_rate_over_time.py`.\n"
        )
        (fig_dir / "metadata.json").write_text(json.dumps({
            "figure": slug,
            "title": title,
            "source_image": str(src),
            "underlying_data": "per-episode npz under analysis/encoder runs (see overlay scripts)",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }, indent=2))
        print(f"  wrote: {fig_dir}/")


def _run_overlay_bd(out_root: Path, scenarios: dict, out_slug: str,
                    title: str, top_note: str, seed: int = 0,
                    suptitle_map: dict | None = None):
    """Shared V_BD render+wrap path. Iterates `scenarios` (dict of name -> CSV
    path) x {median, extreme} strategies, with optional RNG seed suffix.
    Writes PNGs, numbers.csv (merged across RNG seeds), numbers.md, metadata.json
    under `out_root/out_slug/`.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from trajectory_overlay_variants import pick_seed_dirs, render_scenario_bd

    fig_dir = out_root / out_slug
    fig_dir.mkdir(parents=True, exist_ok=True)

    if not BACKDROP_RGB.exists():
        print(f"  ABORT: backdrop {BACKDROP_RGB} missing")
        return

    records = []
    for scenario, csv_path in scenarios.items():
        if not csv_path.exists():
            print(f"  SKIP scenario={scenario}: {csv_path} missing")
            continue
        for strategy in ("median", "extreme"):
            try:
                picks = pick_seed_dirs(csv_path, strategy=strategy)
            except Exception as e:
                print(f"  SKIP {scenario}/{strategy}: {e}")
                continue
            seed_tag = "" if seed == 0 else f"_seed{seed}"
            slug = f"{scenario}_{strategy}{seed_tag}"
            out_png = fig_dir / f"{slug}.png"
            sup = (suptitle_map or {}).get((scenario, strategy))
            try:
                meta = render_scenario_bd(
                    slug,
                    picks["blind"]["traj_dir"],
                    picks["labeled"]["traj_dir"],
                    BACKDROP_RGB, out_png,
                    seed=seed,
                    suptitle=sup,
                )
            except Exception as e:
                print(f"  FAIL {scenario}/{strategy}: {e}")
                continue
            records.append({
                "scenario": scenario,
                "strategy": strategy,
                "rng_seed": seed,
                "blind_seed": picks["blind"]["seed"],
                "blind_metric": picks["blind"]["metric_value"],
                "blind_traj_dir": str(picks["blind"]["traj_dir"]),
                "labeled_seed": picks["labeled"]["seed"],
                "labeled_metric": picks["labeled"]["metric_value"],
                "labeled_traj_dir": str(picks["labeled"]["traj_dir"]),
                "n_blind_plotted": meta["n_blind"],
                "n_labeled_plotted": meta["n_labeled"],
                "image": str(out_png),
            })
            print(f"  wrote: {out_png}")

    if records:
        df = pd.DataFrame(records)
        csv_path = fig_dir / "numbers.csv"
        if csv_path.exists():
            old = pd.read_csv(csv_path)
            df = pd.concat([old, df], ignore_index=True).drop_duplicates(
                subset=["scenario", "strategy", "rng_seed"], keep="last")
        df.to_csv(csv_path, index=False)
        md = [f"# {title}",
              "",
              top_note,
              "",
              "Selection metric: `side_thrust_reversals` (per-episode mean).",
              "  - median:  seed whose mean is closest to the variant mean",
              "  - extreme: blind = most reactive seed; labeled = smoothest seed",
              "",
              "| scenario | strategy | rng_seed | blind seed | blind mean | labeled seed | labeled mean |",
              "|---|---|---|---|---|---|---|",
              ]
        for r in records:
            md.append(f"| {r['scenario']} | {r['strategy']} | {r['rng_seed']} | "
                      f"s{r['blind_seed']} | {r['blind_metric']:.1f} | "
                      f"s{r['labeled_seed']} | {r['labeled_metric']:.1f} |")
        (fig_dir / "numbers.md").write_text("\n".join(md) + "\n")
        (fig_dir / "metadata.json").write_text(json.dumps({
            "figure": out_slug,
            "title": title,
            "scenarios": list(scenarios.keys()),
            "strategies": ["median", "extreme"],
            "selection_metric": "side_thrust_reversals",
            "sources": {k: str(v) for k, v in scenarios.items()},
            "backdrop_rgb": str(BACKDROP_RGB),
            "underlying_script": "scripts/viz/trajectory_overlay_variants.py",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }, indent=2))


def build_overlay_bd(out_root: Path, seed: int = 0):
    """V_BD trajectory overlays (V_B reversal-dot top + V_D thrust-strip bottom)
    for full-variation state-vector blind vs labeled across easy-ID, medium-ID,
    easy-OOD-medium, easy-OOD-hard. Median + extreme seed-selection per scenario.
    """
    _run_overlay_bd(
        out_root, BD_SCENARIOS, "fig-overlay-state-vbd",
        title="State-vector trajectory overlay (V_BD)",
        suptitle_map={("easy-ID", "median"): "Same Task, Different Controllers"},
        top_note=(
            "Top panel: 50 trajectories per condition on lunar lander backdrop, "
            "with reversal dots (yellow = main on/off; magenta = lateral L/R/none change). "
            "Bottom panel: per-episode thrust action strip (sorted by episode length); "
            "▶ green = landed, ▶ red = failed."
        ),
        seed=seed,
    )


# Pairwise visual fusion-gradient overlays. Variant names come straight from
# the E3 cross-config CSVs. For each scenario we point at one CSV that
# contains all three (and optionally fine-tuned) variants.
E3_BD_SCENARIOS = {
    "easy-ID": Path(
        "/media/hdd1/physics-priors-latent-space/lunar-lander-networks/"
        "analysis/cross-config/e3-n3-frozen-only-final/frozen-only-final/"
        "metrics_by_variant.csv"
    ),
    "easy-OOD-medium": Path(
        "/media/hdd1/physics-priors-latent-space/lunar-lander-networks/"
        "analysis/cross-config/e3-n3-all-6-final-ood-medium/all-6-final-ood-medium/"
        "metrics_by_variant.csv"
    ),
}

# (col_A_variant, col_B_variant, col_A_label, col_B_label)
E3_BD_PAIRS = [
    ("blind-frozen", "raw-frozen",    "Visual blind",     "Visual raw fusion"),
    ("blind-frozen", "branch-frozen", "Visual blind",     "Visual branch fusion"),
    ("raw-frozen",   "branch-frozen", "Visual raw fusion", "Visual branch fusion"),
]


def build_e3_ood_behavioral_panel(out_root: Path):
    """4-panel behavioral panel for visual agents under medium-OOD (R5 supp).
    Mirrors Fig 1 layout but with three variants (visual blind / raw / branch,
    all frozen) and uses the e3-n3-all-6-final-ood-medium cross-config CSV.

    Metrics: landed_pct, thrust_autocorr_lag1, side_thrust_reversals,
    frac_upright (posture is the visual-specific 4th panel per writing-brief).
    """
    csv = Path(
        "/media/hdd1/physics-priors-latent-space/lunar-lander-networks/"
        "analysis/cross-config/e3-n3-all-6-final-ood-medium/"
        "all-6-final-ood-medium/metrics_by_variant.csv"
    )
    if not csv.exists():
        print(f"  ABORT: {csv} missing")
        return

    df = load_metrics_csv(csv)
    variants_order = ["blind-frozen", "raw-frozen", "branch-frozen"]
    color_map = {
        "blind-frozen":  COLORS["blind-visual"],
        "raw-frozen":    COLORS["raw"],
        "branch-frozen": COLORS["branch"],
    }
    display_labels = {
        "blind-frozen":  "Visual blind",
        "raw-frozen":    "Raw fusion",
        "branch-frozen": "Branch fusion",
    }
    metrics = ["landed_pct", "thrust_autocorr_lag1", "side_thrust_reversals",
               "frac_upright"]

    fig_dir = out_root / "fig5-ood-behavioral-panel"
    fig, axes = plt.subplots(2, 2, figsize=FIG_SIZE_2x2, dpi=DPI)

    long_rows = []
    for ax, metric in zip(axes.flat, metrics):
        summ = variant_summary(df, metric).set_index("variant")
        summ = summ.loc[[v for v in variants_order if v in summ.index]].reset_index()
        # Rename variant column to display labels for the bar primitive
        summ_display = summ.copy()
        summ_display["variant"] = summ_display["variant"].map(display_labels)
        per_seed = per_seed_means(df, metric)
        per_seed_display = per_seed.copy()
        per_seed_display["variant"] = per_seed_display["variant"].map(display_labels)
        color_map_display = {display_labels[k]: c for k, c in color_map.items()}
        bar_with_seed_dots(
            ax, summ_display, per_seed_display, metric, color_map_display,
            title=METRIC_LABELS.get(metric, metric),
            ylabel=None, show_xticks=True,
        )
        for v in variants_order:
            if v not in summ.set_index("variant").index:
                continue
            row = summ.set_index("variant").loc[v]
            long_rows.append({
                "figure": "fig5-ood-behavioral-panel",
                "metric": metric, "variant": v,
                "display_variant": display_labels[v],
                "mean": float(row["mean"]), "std": float(row["std"]),
                "se": float(row["se"]), "n_seeds": int(row["n"]),
            })

    fig.suptitle(
        "Visual fusion gradient: behavioral panel under medium-OOD"
        "   (mean ± SE across N=3 seeds, dots = per-seed means)",
        fontsize=12, fontweight="bold", y=1.0,
    )
    fig.tight_layout()
    save_figure(fig, fig_dir, "fig5-ood-behavioral-panel")

    long_df = pd.DataFrame(long_rows)
    md_rows = []
    for metric in metrics:
        cells = []
        for v in variants_order:
            sub = long_df[(long_df["metric"] == metric) & (long_df["variant"] == v)]
            if sub.empty:
                cells.append("—")
            else:
                r = sub.iloc[0]
                cells.append(f"{fmt(r['mean'], 3)} ± {fmt(r['se'], 3)}")
        md_rows.append([METRIC_LABELS.get(metric, metric), *cells])

    md = (
        "# Fig 5 supplementary. Visual fusion gradient under medium-OOD\n\n"
        "Behavioral panel for visual agents (frozen encoder) evaluated on "
        "medium-OOD physics. Mirrors the Fig 1 layout but with three variants "
        "and the posture metric (`frac_upright`) replacing time-to-first-contact.\n\n"
        "Source: `e3-n3-all-6-final-ood-medium/all-6-final-ood-medium/"
        "metrics_by_variant.csv`. N=3 seeds per variant.\n\n"
    )
    md += md_table(
        ["Metric"] + [display_labels[v] for v in variants_order],
        md_rows,
    )
    save_numbers(
        fig_dir, long_df, md,
        {
            "figure": "fig5-ood-behavioral-panel",
            "title": "Visual fusion gradient: behavioral panel under medium-OOD",
            "source_csv": str(csv),
            "comparison": "all-6-final-ood-medium",
            "n_seeds_per_variant": 3,
            "metrics": metrics,
            "variants": variants_order,
            "variant_display_labels": display_labels,
        },
    )
    print(f"  wrote: {fig_dir}/")


def build_overlay_bd_e3(out_root: Path, seed: int = 0):
    """V_BD trajectory overlays for E3 visual fusion gradient (Result 5).
    Renders three pairwise comparisons (blind-vs-raw, blind-vs-branch,
    raw-vs-branch) at easy-ID and easy-OOD-medium, median + extreme strategies.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from trajectory_overlay_variants import pick_seed_dirs, render_scenario_bd

    fig_dir = out_root / "fig-overlay-e3-vbd"
    fig_dir.mkdir(parents=True, exist_ok=True)

    if not BACKDROP_RGB.exists():
        print(f"  ABORT: backdrop {BACKDROP_RGB} missing")
        return

    records = []
    for scenario, csv_path in E3_BD_SCENARIOS.items():
        if not csv_path.exists():
            print(f"  SKIP scenario={scenario}: {csv_path} missing")
            continue
        for var_a, var_b, label_a, label_b in E3_BD_PAIRS:
            pair_name = f"{var_a.replace('-frozen', '')}-vs-{var_b.replace('-frozen', '')}"
            for strategy in ("median", "extreme"):
                try:
                    picks = pick_seed_dirs(csv_path, strategy=strategy,
                                           blind_variant=var_a,
                                           labeled_variant=var_b)
                except Exception as e:
                    print(f"  SKIP {pair_name}/{scenario}/{strategy}: {e}")
                    continue
                seed_tag = "" if seed == 0 else f"_seed{seed}"
                slug = f"{pair_name}_{scenario}_{strategy}{seed_tag}"
                out_png = fig_dir / f"{slug}.png"
                try:
                    meta = render_scenario_bd(
                        slug,
                        picks["blind"]["traj_dir"],
                        picks["labeled"]["traj_dir"],
                        BACKDROP_RGB, out_png,
                        seed=seed,
                        titles=[label_a, label_b],
                    )
                except Exception as e:
                    print(f"  FAIL {slug}: {e}")
                    continue
                records.append({
                    "pair": pair_name,
                    "scenario": scenario,
                    "strategy": strategy,
                    "rng_seed": seed,
                    "col_a_variant": var_a,
                    "col_a_seed": picks["blind"]["seed"],
                    "col_a_metric": picks["blind"]["metric_value"],
                    "col_a_traj_dir": str(picks["blind"]["traj_dir"]),
                    "col_b_variant": var_b,
                    "col_b_seed": picks["labeled"]["seed"],
                    "col_b_metric": picks["labeled"]["metric_value"],
                    "col_b_traj_dir": str(picks["labeled"]["traj_dir"]),
                    "n_a_plotted": meta["n_blind"],
                    "n_b_plotted": meta["n_labeled"],
                    "image": str(out_png),
                })
                print(f"  wrote: {out_png}")

    if records:
        df = pd.DataFrame(records)
        csv_path = fig_dir / "numbers.csv"
        if csv_path.exists():
            old = pd.read_csv(csv_path)
            df = pd.concat([old, df], ignore_index=True).drop_duplicates(
                subset=["pair", "scenario", "strategy", "rng_seed"], keep="last")
        df.to_csv(csv_path, index=False)
        md = ["# E3 visual fusion gradient trajectory overlay (V_BD)",
              "",
              "Pairwise visual-agent overlays for Result 5 (fusion strength). "
              "Variants compared: visual-blind, visual-raw (parameters concatenated), "
              "visual-branch (parameters through a learned linear branch). "
              "All three are frozen-encoder configurations.",
              "",
              "Selection metric: `side_thrust_reversals` (per-episode mean).",
              "  - median:  seed whose mean is closest to the variant mean",
              "  - extreme: col-A = highest reversals; col-B = lowest reversals",
              "",
              "| pair | scenario | strategy | rng_seed | col-A seed | col-A mean | col-B seed | col-B mean |",
              "|---|---|---|---|---|---|---|---|",
              ]
        for r in records:
            md.append(f"| {r['pair']} | {r['scenario']} | {r['strategy']} | {r['rng_seed']} | "
                      f"s{r['col_a_seed']} | {r['col_a_metric']:.1f} | "
                      f"s{r['col_b_seed']} | {r['col_b_metric']:.1f} |")
        (fig_dir / "numbers.md").write_text("\n".join(md) + "\n")
        (fig_dir / "metadata.json").write_text(json.dumps({
            "figure": "fig-overlay-e3-vbd",
            "title": "E3 visual fusion gradient trajectory overlay (V_BD)",
            "scenarios": list(E3_BD_SCENARIOS.keys()),
            "pairs": [{"col_a": a, "col_b": b, "label_a": la, "label_b": lb}
                      for a, b, la, lb in E3_BD_PAIRS],
            "strategies": ["median", "extreme"],
            "selection_metric": "side_thrust_reversals",
            "sources": {k: str(v) for k, v in E3_BD_SCENARIOS.items()},
            "backdrop_rgb": str(BACKDROP_RGB),
            "underlying_script": "scripts/viz/trajectory_overlay_variants.py",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }, indent=2))


def build_overlay_bd_bodyworld(out_root: Path, seed: int = 0):
    """V_BD trajectory overlays for body/world diagnostic: physics-only and
    vehicle-only profiles (blind vs labeled). Renders all available scenarios
    from `parametric-vs-behavioral-{DATE_TAG}`: physics-only easy+medium,
    vehicle-only easy+hard.
    """
    _run_overlay_bd(
        out_root, BD_BODYWORLD_SCENARIOS, "fig-overlay-bodyworld-vbd",
        title="Body/world trajectory overlay (V_BD)",
        top_note=(
            "Body/world diagnostic overlays. Physics-only varies world parameters "
            "(gravity/wind/turbulence). Vehicle-only varies body parameters "
            "(engine power, density, angular damping). Same V_BD format as the "
            "state-vector overlay: reversal dots on top, per-episode action strips below."
        ),
        seed=seed,
    )


def build_overlay_b_bodyworld(out_root: Path, seed: int = 0):
    """Body/world trajectory companion (5b). Produces under
    `fig-overlay-body-world/`:
      - vb-grid.{png,svg}: V_B-only 3-row × 2-col grid (full-variation,
        world-only, body-only × blind, labeled), median seed per cell.
      - world-only_easy_{median,extreme}.{png,svg}: V_BD per-condition.
      - body-only_easy_{median,extreme}.{png,svg}: V_BD per-condition.
    All seeds picked from `parametric-vs-behavioral-{DATE_TAG}` cross-config.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from trajectory_overlay_variants import (
        variant_B_grid, pick_seed_dirs, pick, pick_by_reversal_rate,
        load_episode, render_scenario_bd,
    )

    fig_dir = out_root / "fig-overlay-body-world"
    fig_dir.mkdir(parents=True, exist_ok=True)
    if not BACKDROP_RGB.exists():
        print(f"  ABORT: backdrop {BACKDROP_RGB} missing")
        return

    name = f"parametric-vs-behavioral-{DATE_TAG}"
    row_specs = [
        ("full-variation", "full-variation-easy"),
        ("world-only",     "physics-only-easy"),
        ("body-only",      "vehicle-only-easy"),
    ]
    backdrop = load_episode(BACKDROP_RGB)["rgb_frames"][-1]

    # --- (1) V_B 3-row grid: produce median + extreme variants.
    grid_records = []
    for strategy in ("median", "extreme"):
        rows = []
        for row_label, comp in row_specs:
            csv_path = cross_config_path(name) / comp / "metrics_by_variant.csv"
            if not csv_path.exists():
                print(f"  SKIP grid {row_label}/{strategy}: {csv_path} missing")
                continue
            picks = pick_seed_dirs(csv_path, strategy=strategy)
            if strategy == "extreme":
                # Within the cherry-picked seed, cherry-pick the 20 most-extreme
                # episodes by per-step lateral reversal rate.
                blind_eps   = pick_by_reversal_rate(picks["blind"]["traj_dir"],
                                                    n=20, mode="most")
                labeled_eps = pick_by_reversal_rate(picks["labeled"]["traj_dir"],
                                                    n=20, mode="least")
            else:
                rng_b = np.random.default_rng(seed)
                rng_l = np.random.default_rng(seed + 1)
                blind_eps   = pick(picks["blind"]["traj_dir"],   n=50, outcome=None, rng=rng_b)
                labeled_eps = pick(picks["labeled"]["traj_dir"], n=50, outcome=None, rng=rng_l)
            rows.append({"label": row_label,
                         "blind_eps": blind_eps,
                         "labeled_eps": labeled_eps})
            grid_records.append({
                "artifact": f"vb-grid_{strategy}",
                "env": row_label, "strategy": strategy,
                "comp": comp, "rng_seed": seed,
                "blind_seed":  picks["blind"]["seed"],
                "blind_metric": picks["blind"]["metric_value"],
                "blind_traj_dir": str(picks["blind"]["traj_dir"]),
                "n_blind_plotted": len(blind_eps),
                "labeled_seed": picks["labeled"]["seed"],
                "labeled_metric": picks["labeled"]["metric_value"],
                "labeled_traj_dir": str(picks["labeled"]["traj_dir"]),
                "n_labeled_plotted": len(labeled_eps),
            })

        if rows:
            out_png = fig_dir / f"vb-grid_{strategy}.png"
            qualifier = (
                "representative seeds (median), 50 random episodes per cell"
                if strategy == "median"
                else "contrast seeds + cherry-picked episodes "
                     "(blind = top-20 most reactive, labeled = bottom-20 smoothest)"
            )
            variant_B_grid(
                rows, backdrop, out_png,
                suptitle=(
                    r"$\bf{Body\ Uncertainty\ Changes\ the\ Controller}$" "\n"
                    "\n"
                    "State-vector blind vs labeled across body/world envs (easy profile)\n"
                    f"{qualifier}"
                ),
                col_titles=("Blind agent", "Labeled agent"),
            )
            print(f"  wrote: {out_png}")

    # --- (2) V_BD per-condition: world-only / body-only × median / extreme.
    bd_specs = [
        ("full-variation_easy", "full-variation-easy"),
        ("world-only_easy",     "physics-only-easy"),
        ("body-only_easy",      "vehicle-only-easy"),
    ]
    bd_records = []
    for slug, comp in bd_specs:
        csv_path = cross_config_path(name) / comp / "metrics_by_variant.csv"
        if not csv_path.exists():
            print(f"  SKIP V_BD {slug}: {csv_path} missing")
            continue
        for strategy in ("median", "extreme"):
            picks = pick_seed_dirs(csv_path, strategy=strategy)
            out_png = fig_dir / f"{slug}_{strategy}.png"
            try:
                meta = render_scenario_bd(
                    f"{slug}_{strategy}",
                    picks["blind"]["traj_dir"],
                    picks["labeled"]["traj_dir"],
                    BACKDROP_RGB, out_png, seed=seed,
                )
            except Exception as e:
                print(f"  FAIL V_BD {slug}/{strategy}: {e}")
                continue
            bd_records.append({
                "artifact": "vbd", "env": slug, "strategy": strategy,
                "comp": comp, "rng_seed": seed,
                "blind_seed":  picks["blind"]["seed"],
                "blind_metric": picks["blind"]["metric_value"],
                "blind_traj_dir": str(picks["blind"]["traj_dir"]),
                "labeled_seed": picks["labeled"]["seed"],
                "labeled_metric": picks["labeled"]["metric_value"],
                "labeled_traj_dir": str(picks["labeled"]["traj_dir"]),
                "n_blind_plotted": meta["n_blind"],
                "n_labeled_plotted": meta["n_labeled"],
                "image": str(out_png),
            })
            print(f"  wrote: {out_png}")

    records = grid_records + bd_records
    if records:
        df = pd.DataFrame(records)
        df.to_csv(fig_dir / "numbers.csv", index=False)
        md = [
            "# Fig 5b. Body Uncertainty Changes the Controller",
            "",
            f"Source: `{name}/{{full-variation-easy,physics-only-easy,vehicle-only-easy}}`.",
            "",
            "Artifacts:",
            "- `vb-grid_median.png` — V_B only, 3 rows × 2 cols. Median (representative) seed per cell.",
            "- `vb-grid_extreme.png` — V_B only, same grid, contrast seeds (blind = most reactive, labeled = smoothest).",
            "- `full-variation_easy_{median,extreme}.png` — V_BD per-condition.",
            "- `world-only_easy_{median,extreme}.png` — V_BD per-condition.",
            "- `body-only_easy_{median,extreme}.png` — V_BD per-condition.",
            "",
            "Selection metric: `side_thrust_reversals`.",
            "  - median: seed mean closest to within-variant mean (representative).",
            "  - extreme: blind = most reactive seed; labeled = smoothest seed.",
            "",
            "| artifact | env | strategy | blind seed | blind sel.metric | labeled seed | labeled sel.metric |",
            "|---|---|---|---|---|---|---|",
        ]
        for r in records:
            md.append(f"| {r['artifact']} | {r['env']} | {r['strategy']} | "
                      f"s{r['blind_seed']} | {r['blind_metric']:.1f} | "
                      f"s{r['labeled_seed']} | {r['labeled_metric']:.1f} |")
        (fig_dir / "numbers.md").write_text("\n".join(md) + "\n")
        (fig_dir / "metadata.json").write_text(json.dumps({
            "figure": "fig-overlay-body-world",
            "title": "Body Uncertainty Changes the Controller",
            "cross_config_dir": str(cross_config_path(name)),
            "selection_metric": "side_thrust_reversals",
            "backdrop_rgb": str(BACKDROP_RGB),
            "underlying_script": "scripts/viz/trajectory_overlay_variants.py",
            "rng_seed": seed,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }, indent=2))


# ====================================================================
# CLI
# ====================================================================


BUILDERS = {
    "fig1": build_fig1,
    "fig2": build_fig2,
    "fig3": build_fig3,
    "fig4": build_fig4,
    "fig5": build_fig5,
    "result4": build_result4_table,
    "overlays": wrap_overlay,
    "overlay_bd": build_overlay_bd,
    "overlay_bd_bodyworld": build_overlay_bd_bodyworld,
    "overlay_b_bodyworld": build_overlay_b_bodyworld,
    "overlay_bd_e3": build_overlay_bd_e3,
    "fig5_ood_panel": build_e3_ood_behavioral_panel,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--figure", choices=list(BUILDERS) + ["all"], default="all")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--seed", type=int, default=0,
                   help="random seed for episode sampling in V_BD overlays")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    targets = list(BUILDERS) if args.figure == "all" else [args.figure]
    for t in targets:
        print(f"\n=== building {t} ===")
        if t in ("overlay_bd", "overlay_bd_bodyworld", "overlay_bd_e3"):
            BUILDERS[t](args.out_dir, seed=args.seed)
        else:
            BUILDERS[t](args.out_dir)
    print(f"\nAll figures written under: {args.out_dir}")


if __name__ == "__main__":
    main()
