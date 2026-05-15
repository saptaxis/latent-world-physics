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
FIG_SIZE_2x2 = (11.0, 8.0)       # Fig 1 (widened from 8x7 to match family)
FIG_SIZE_SINGLE = (6.5, 4.2)     # Fig 2 (unused; build_fig2 hard-codes 11x4.6)
FIG_SIZE_LADDER = (11.5, 6.2)    # Fig 3 (matches existing hard-coded value)
FIG_SIZE_OOD = (13.5, 8.0)       # Fig 4 OOD (matches restyled value)
FIG_SIZE_FUSION = (11.0, 5.5)    # Fig 5 E3 fusion (widened to family range)

# Typography — canonical sizes referenced by every figure. Per-call
# fontsize= overrides should be avoided so changes here propagate.
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Suptitle canonical: applied via explicit `fontsize=SUPTITLE_FONTSIZE,
# linespacing=SUPTITLE_LINESPACING` since matplotlib's figure.titlesize
# rcParam doesn't cover `fig.suptitle`.
SUPTITLE_FONTSIZE = 15
SUPTITLE_LINESPACING = 1.4

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
        ax.set_title(title)
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
        ax.set_title(title)
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
        ax.set_title(title)
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
        ax.set_title(title)
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
        fontsize=SUPTITLE_FONTSIZE, linespacing=SUPTITLE_LINESPACING,
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
        fontsize=SUPTITLE_FONTSIZE, linespacing=SUPTITLE_LINESPACING,
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
        fontsize=SUPTITLE_FONTSIZE, linespacing=SUPTITLE_LINESPACING,
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
    """Fig 5 (slug fig4-ood-divergence): "Labeled Agents Fail Harder
    Under Shift". Easy-trained state-vector agents evaluated on easy /
    medium / hard. Two rows:
      - Top (wider): landed_pct | crashed_pct  (shared y 0-100%)
      - Bottom (3): thrust_autocorr_lag1 | side_thrust_reversals_per_step
                    | thrust_duty_cycle (independent y; bounded ones use 0-1)
    Each panel: connected lines + markers + SE errorbars + per-seed dots.
    """
    fig_dir = out_root / "fig4-ood-divergence"
    sources = [
        ("easy",   f"parametric-vs-behavioral-v2-{DATE_TAG}",
                   "full-variation-easy"),
        ("medium", f"ood-generalization-medium-{DATE_TAG}",
                   "full-variation-easy-trained-on-medium"),
        ("hard",   f"ood-generalization-hard-{DATE_TAG}",
                   "full-variation-easy-trained-on-hard"),
    ]
    metrics = [
        "landed_pct", "crashed_pct",
        "thrust_autocorr_lag1",
        "side_thrust_reversals_per_step",
        "thrust_duty_cycle",
    ]

    rows_summary = []
    rows_per_seed = []
    long_rows = []
    md_rows = []
    for eval_profile, name, comp in sources:
        df = load_metrics_csv(cross_config_path(name) / comp / "metrics_by_variant.csv")
        stats = load_stat_tests(name, comp)
        for metric in metrics:
            summ = variant_summary(df, metric)
            ps = per_seed_means(df, metric)
            for v in ["labeled", "blind"]:
                r = summ[summ["variant"] == v].iloc[0]
                rows_summary.append({
                    "eval": eval_profile, "variant": v, "metric": metric,
                    "mean": r["mean"], "se": r["se"], "std": r["std"],
                    "n": int(r["n"]),
                })
            for _, r in ps.iterrows():
                rows_per_seed.append({"eval": eval_profile,
                                      "variant": r["variant"],
                                      "metric": metric, "value": r[metric]})
            p, d = lookup_pvalue_d(stats, metric)
            lab = summ[summ["variant"] == "labeled"].iloc[0]
            bli = summ[summ["variant"] == "blind"].iloc[0]
            md_rows.append([
                eval_profile, METRIC_LABELS.get(metric, metric),
                f"{fmt(lab['mean'], 3)} ± {fmt(lab['se'], 3)}",
                f"{fmt(bli['mean'], 3)} ± {fmt(bli['se'], 3)}",
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

    # Layout: 2 rows. Top row: ONE wide outcome panel (landed + crashed,
    # 4 lines). Bottom row: 3 behavioral panels.
    fig = plt.figure(figsize=(13.5, 8.0), dpi=DPI, constrained_layout=True)
    gs = fig.add_gridspec(2, 6, hspace=0.15)
    ax_land    = fig.add_subplot(gs[0, 0:3])
    ax_crash   = fig.add_subplot(gs[0, 3:6])
    ax_ac      = fig.add_subplot(gs[1, 0:2])
    ax_rev     = fig.add_subplot(gs[1, 2:4])
    ax_duty    = fig.add_subplot(gs[1, 4:6])

    profiles = ["easy", "medium", "hard"]
    x = np.arange(len(profiles))

    # --- Top: two outcome panels (landed, crashed), shared y 0-100% ---
    outcome_panel_specs = [
        (ax_land,  "landed_pct",  "Landing rate"),
        (ax_crash, "crashed_pct", "Crash rate"),
    ]
    for ax, metric, title in outcome_panel_specs:
        for variant in ["labeled", "blind"]:
            sub = df_summary[
                (df_summary["variant"] == variant)
                & (df_summary["metric"] == metric)
            ].set_index("eval").loc[profiles]
            ax.errorbar(
                x, sub["mean"], yerr=sub["se"],
                marker="o", capsize=4, color=COLORS[variant],
                linewidth=2.2, markersize=7, label=variant.capitalize(),
            )
            for i, prof in enumerate(profiles):
                vals = df_per_seed[
                    (df_per_seed["eval"] == prof)
                    & (df_per_seed["variant"] == variant)
                    & (df_per_seed["metric"] == metric)
                ]["value"].values
                rng = np.random.RandomState(hash((variant, metric, prof)) & 0xFFFF)
                jitter = rng.uniform(-0.08, 0.08, size=len(vals))
                ax.scatter(
                    np.full_like(vals, x[i], dtype=float) + jitter, vals,
                    s=14, color="white", edgecolor=COLORS[variant],
                    linewidth=0.7, zorder=3, alpha=0.95)
        ax.set_xticks(x)
        ax.set_xticklabels(profiles)
        ax.set_ylabel("% of episodes")
        ax.set_ylim(0, 100)
        ax.set_title(title, pad=10)
        ax.yaxis.grid(True, alpha=0.25, linestyle="-", linewidth=0.5)
        ax.set_axisbelow(True)

    # Single shared figure-level legend (Blind / Labeled), top-right.
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color=COLORS["blind"], marker="o",
               linewidth=2.2, markersize=7, label="Blind"),
        Line2D([0], [0], color=COLORS["labeled"], marker="o",
               linewidth=2.2, markersize=7, label="Labeled"),
    ]
    fig.legend(handles=legend_handles, loc="upper right", frameon=False,
               fontsize=10, bbox_to_anchor=(0.99, 0.97))

    # --- Bottom: 3 behavioral panels ---
    panel_specs = [
        (ax_ac,   "thrust_autocorr_lag1",
         "Thrust autocorrelation",
         "lag-1 autocorrelation",      (0, 1)),
        (ax_rev,  "side_thrust_reversals_per_step",
         "Side-thrust reversals",
         "events per step",            None),
        (ax_duty, "thrust_duty_cycle",
         "Thrust duty cycle",
         "fraction of steps on",       (0, 1)),
    ]
    for ax, metric, title, ylabel, ylim in panel_specs:
        for variant in ["labeled", "blind"]:
            sub = df_summary[
                (df_summary["variant"] == variant)
                & (df_summary["metric"] == metric)
            ].set_index("eval").loc[profiles]
            ax.errorbar(
                x, sub["mean"], yerr=sub["se"],
                marker="o", capsize=4, color=COLORS[variant],
                linewidth=2.2, markersize=7, label=variant,
            )
            for i, prof in enumerate(profiles):
                vals = df_per_seed[
                    (df_per_seed["eval"] == prof)
                    & (df_per_seed["variant"] == variant)
                    & (df_per_seed["metric"] == metric)
                ]["value"].values
                rng = np.random.RandomState(hash((variant, metric, prof)) & 0xFFFF)
                jitter = rng.uniform(-0.08, 0.08, size=len(vals))
                ax.scatter(np.full_like(vals, x[i], dtype=float) + jitter, vals,
                           s=14, color="white", edgecolor=COLORS[variant],
                           linewidth=0.7, zorder=3, alpha=0.95)
        ax.set_xticks(x)
        ax.set_xticklabels(profiles)
        ax.set_ylabel(ylabel)
        ax.set_title(title, pad=10)
        if ylim is not None:
            ax.set_ylim(*ylim)
        else:
            ax.set_ylim(bottom=0)
        ax.yaxis.grid(True, alpha=0.25, linestyle="-", linewidth=0.5)
        ax.set_axisbelow(True)

    fig.suptitle(
        r"$\bf{Labeled\ Agents\ Fail\ Harder\ Under\ Shift}$" "\n"
        "\n"
        "Easy-trained agents evaluated on easy / medium / hard"
        "\n ",
        fontsize=SUPTITLE_FONTSIZE, linespacing=SUPTITLE_LINESPACING,
    )
    fig.text(0.5, -0.01,
             "lines = mean across N=10 seeds; bars = ±SE across seeds; "
             "open circles = per-seed means",
             ha="center", va="top", fontsize=9, color="#555")

    save_figure(fig, fig_dir, "fig4-ood-divergence")

    md = (
        "# Fig 5. State-vector out-of-distribution divergence "
        "(slug: fig4-ood-divergence)\n\n"
        "Easy-trained labeled and blind agents evaluated on easy / medium "
        "/ hard. Sources: easy from "
        f"`parametric-vs-behavioral-v2-{DATE_TAG}`, medium and hard from "
        f"`ood-generalization-{{medium,hard}}-{DATE_TAG}`. "
        "N=10 seeds per variant.\n\n"
    )
    md += md_table(
        ["Eval profile", "Metric", "Labeled (mean ± SE)",
         "Blind (mean ± SE)", "p", "Cohen's d"],
        md_rows,
    )
    save_numbers(
        fig_dir, pd.DataFrame(long_rows), md,
        {
            "figure": "fig4",
            "title": "Labeled Agents Fail Harder Under Shift",
            "sources": [
                {"eval": e,
                 "csv": str(cross_config_path(n) / c / "metrics_by_variant.csv")}
                for e, n, c in sources
            ],
            "metrics": metrics,
            "n_seeds_per_variant": 10,
        },
    )
    print(f"  wrote: {fig_dir}/")


# ====================================================================
# Fig 5 — E3 fusion-strength gradient (Result 5)
# ====================================================================


def build_fig5(out_root: Path):
    """Fig 6 (slug fig5-fusion-gradient): "Stronger Fusion, Sharper Collapse".

    Visual frozen agents (blind, raw, branch) on ID and medium-OOD. Three
    panels:
      Top: landed_pct (lines, 3 variants × {ID, OOD-medium}) | crashed_pct
      Bot: side_thrust_reversals_per_step (ID-only, 3 bars)

    Brief calls out: don't use OOD reversals -- in OOD, branch can collapse
    toward zero because the agent crashes before acting, confounding
    controller style with episode truncation.
    """
    fig_dir = out_root / "fig5-fusion-gradient"
    NETS = DEFAULT_NETS
    id_csv = (NETS / "analysis/cross-config"
              / f"e3-n3-all-6-final-{DATE_TAG}" / "all-6-final"
              / "metrics_by_variant.csv")
    ood_csv = (NETS / "analysis/cross-config"
               / f"e3-n3-all-6-final-ood-medium-{DATE_TAG}"
               / "all-6-final-ood-medium" / "metrics_by_variant.csv")

    df_id_all  = load_metrics_csv(id_csv)
    df_ood_all = load_metrics_csv(ood_csv)
    df_id  = df_id_all[df_id_all["variant"].str.endswith("-frozen")].copy()
    df_ood = df_ood_all[df_ood_all["variant"].str.endswith("-frozen")].copy()

    conditions = ["blind-frozen", "raw-frozen", "branch-frozen"]
    public_label = {
        "blind-frozen":  "Blind",
        "raw-frozen":    "Labeled, raw fusion",
        "branch-frozen": "Labeled, branch fusion",
    }
    cond_color = {
        "blind-frozen":  COLORS["blind-visual"],
        "raw-frozen":    COLORS["raw"],
        "branch-frozen": COLORS["branch"],
    }

    # ----- Figure layout -----
    fig = plt.figure(figsize=(13.0, 10.0), dpi=DPI, constrained_layout=True)
    gs = fig.add_gridspec(2, 6, height_ratios=[1.4, 1.0], hspace=0.10)
    ax_land  = fig.add_subplot(gs[0, 0:3])
    ax_crash = fig.add_subplot(gs[0, 3:6])
    ax_rev   = fig.add_subplot(gs[1, 0:3])
    ax_ac    = fig.add_subplot(gs[1, 3:6])

    # ----- Top row: outcome lines across [ID, OOD-medium] -----
    eval_labels = ["Easy physics\n(ID)", "Medium physics\n(OOD)"]
    x = np.arange(len(eval_labels))
    panel_specs = [
        (ax_land,  "landed_pct",  "Landing rate", "% of episodes"),
        (ax_crash, "crashed_pct", "Crash rate",   "% of episodes"),
    ]
    long_rows = []
    md_rows = []

    for ax, metric, title, ylabel in panel_specs:
        for variant in conditions:
            means, ses, per_seed_vals = [], [], []
            for df in (df_id, df_ood):
                summ = variant_summary(df, metric).set_index("variant")
                ps   = per_seed_means(df, metric)
                means.append(float(summ.loc[variant, "mean"]))
                ses.append(float(summ.loc[variant, "se"]))
                per_seed_vals.append(
                    ps[ps["variant"] == variant][metric].values)
            ax.errorbar(x, means, yerr=ses, marker="o", capsize=4,
                        color=cond_color[variant], linewidth=2.4,
                        markersize=8, label=public_label[variant])
            for i, vals in enumerate(per_seed_vals):
                jitter = (np.random.RandomState(
                    hash((variant, metric, i)) & 0xFFFF)
                          .uniform(-0.04, 0.04, size=len(vals)))
                ax.scatter(np.full_like(vals, x[i], dtype=float) + jitter,
                           vals, s=18, color="white",
                           edgecolor=cond_color[variant], linewidth=0.8,
                           zorder=3, alpha=0.95)
                long_rows.append({
                    "figure": "fig5", "panel": metric,
                    "eval": "id" if i == 0 else "ood",
                    "variant": variant,
                    "mean": means[i], "se": ses[i],
                })
                md_rows.append([
                    "ID" if i == 0 else "OOD-medium",
                    title, public_label[variant],
                    f"{fmt(means[i], 1)} ± {fmt(ses[i], 1)}",
                ])
        ax.set_xticks(x)
        ax.set_xticklabels(eval_labels)
        ax.set_ylabel(ylabel)
        # Small headroom above 100 so endpoint markers/SE caps aren't
        # pressed against the top spine.
        ax.set_ylim(-3, 105)
        ax.set_title(title, pad=10)
        ax.yaxis.grid(True, alpha=0.25, linestyle="-", linewidth=0.5)
        ax.set_axisbelow(True)
        if metric == "landed_pct":
            ax.legend(frameon=False, loc="upper right")

    # ----- Bottom: ID-only controller-style strips -----
    color_pub = {public_label[k]: v for k, v in cond_color.items()}
    strip_specs = [
        (ax_rev, "side_thrust_reversals_per_step",
         "Side-thrust reversals (ID only)", "events per step", None),
        (ax_ac,  "thrust_autocorr_lag1",
         "Thrust autocorrelation (ID only)",
         "lag-1 autocorrelation", (0, 1)),
    ]
    for ax, metric, title, ylabel, ylim in strip_specs:
        summ = variant_summary(df_id, metric).set_index("variant")
        summ_ordered = summ.loc[conditions].reset_index()
        summ_ordered["variant"] = summ_ordered["variant"].map(public_label)
        ps = per_seed_means(df_id, metric)
        ps_renamed = ps.copy()
        ps_renamed["variant"] = ps_renamed["variant"].map(public_label)
        bar_with_seed_dots(
            ax, summ_ordered, ps_renamed, metric, color_pub,
            title=title, ylabel=ylabel,
        )
        if ylim is not None:
            lo, hi = ylim
            ax.set_ylim(lo, hi + 0.18 * (hi - lo))
        else:
            top = float((summ_ordered["mean"] + summ_ordered["se"]).max())
            ax.set_ylim(0, top * 1.30)
        # p/d annotation vs blind-frozen baseline (leftmost bar).
        baseline_vals = ps[ps["variant"] == "blind-frozen"][metric].values
        bar_x = np.arange(len(conditions)) * 0.6
        y_lo, y_hi = ax.get_ylim()
        offset = 0.025 * (y_hi - y_lo)
        for i, cond in enumerate(conditions):
            row = summ_ordered.iloc[i]
            bar_top = float(row["mean"] + row["se"])
            if cond == "blind-frozen":
                ax.text(bar_x[i], bar_top + offset, "(baseline)",
                        ha="center", va="bottom", fontsize=8,
                        color="#888", style="italic")
                continue
            cmp_vals = ps[ps["variant"] == cond][metric].values
            p, d = _ttest_cohens_d_arrays(baseline_vals, cmp_vals)
            if p is None:
                continue
            txt = (f"p={p:.3f}\nd={d:.2f}"
                   if d is not None else f"p={p:.3f}")
            ax.text(bar_x[i], bar_top + offset, txt,
                    ha="center", va="bottom", fontsize=8, color="#444")
        for v in conditions:
            r = summ.loc[v]
            long_rows.append({
                "figure": "fig5", "panel": metric, "eval": "id",
                "variant": v, "mean": float(r["mean"]),
                "se": float(r["se"]),
            })
            md_rows.append([
                "ID", title.replace(" (ID only)", ""),
                public_label[v],
                f"{fmt(r['mean'], 3)} ± {fmt(r['se'], 3)}",
            ])

    fig.suptitle(
        r"$\bf{Stronger\ Fusion,\ Sharper\ Collapse}$" "\n"
        "\n"
        "Visual frozen agents trained on easy-variation physics, evaluated "
        "on easy (ID) and medium (OOD)\n"
        "Blind / raw fusion / branch fusion (N=3 seeds per condition)"
        "\n ",
        fontsize=SUPTITLE_FONTSIZE, linespacing=SUPTITLE_LINESPACING,
    )
    save_figure(fig, fig_dir, "fig5-fusion-gradient")

    md = (
        "# Fig 6. Stronger Fusion, Sharper Collapse "
        "(slug: fig5-fusion-gradient)\n\n"
        "Visual frozen agents (blind, raw-fusion, branch-fusion) on easy "
        "(ID) and medium-OOD. N=3 seeds per condition.\n\n"
        f"Sources: ID `e3-n3-all-6-final-{DATE_TAG}/all-6-final`; "
        f"OOD `e3-n3-all-6-final-ood-medium-{DATE_TAG}/all-6-final-ood-medium`. "
        "Filtered to `-frozen` rows. Fine-tuned and peak-checkpoint variants "
        "excluded per writing brief.\n\n"
    )
    md += md_table(["Eval", "Metric", "Variant", "Mean ± SE"], md_rows)

    save_numbers(
        fig_dir, pd.DataFrame(long_rows), md,
        {
            "figure": "fig5-fusion-gradient",
            "title": "Stronger Fusion, Sharper Collapse",
            "sources": {"id_csv": str(id_csv), "ood_csv": str(ood_csv)},
            "conditions": conditions,
            "n_seeds_per_variant": 3,
            "metrics": ["landed_pct", "crashed_pct",
                        "side_thrust_reversals_per_step",
                        "thrust_autocorr_lag1"],
        },
    )
    print(f"  wrote: {fig_dir}/")


# ====================================================================
# Result 4 compact table (state vs visual gym-default)
# ====================================================================


def build_result4_table(out_root: Path):
    """Result 4: "With Physics Fixed, Perception Changes Control". State
    agents (gym-default, fixed physics) vs visual matched-encoder agents.

    The framing is a deliberate null control: under fixed physics, state
    labels carry no signal, so state-blind and state-labeled converge to
    near-identical smooth controllers. Visual agents, given the same fixed
    physics, sit much further toward reactive correction — the shift is
    caused by perception, not parameter variation.

    Produces both a table (numbers.md) AND a plot
    (result4-state-vs-visual.{png,svg}).

    Layout (2 rows × 6-col gridspec):
      Top:    landed_pct | crashed_pct (2 wide panels, shared y 0-100%)
      Bottom: thrust_autocorr_lag1 | side_thrust_reversals_per_step
              | frac_upright

    Each panel: 4 grouped bars (state blind / state labeled / visual frozen /
    visual fine-tuned) with per-seed dots. N=5 seeds per condition.
    """
    fig_dir = out_root / "result4-state-vs-visual"
    NETS = DEFAULT_NETS
    state_csv = (NETS / "analysis/cross-config"
                 / f"e1-blind-vs-labeled-gym-default-128-{DATE_TAG}"
                 / "e1-blind-vs-labeled-gym-default-128"
                 / "metrics_by_variant.csv")
    visual_csv = (NETS / "analysis/cross-config"
                  / f"matched-enc-all-6-conditions-{DATE_TAG}"
                  / "all-6" / "metrics_by_variant.csv")
    state_df = load_metrics_csv(state_csv)
    visual_df = load_metrics_csv(visual_csv)
    # matched-enc has variants {frozen-baseline, finetune-baseline, ...swap variants...}
    visual_df = visual_df[visual_df["variant"].isin(
        ["frozen-baseline", "finetune-baseline"])].copy()

    # Public condition labels (used in plot x-ticks AND table columns).
    COND_STATE_BLIND  = "State,\nblind"
    COND_STATE_LAB    = "State,\nlabeled"
    COND_VIS_FROZEN   = "Visual blind,\nfrozen"
    COND_VIS_FT       = "Visual blind,\nfine-tuned"
    cond_order = [COND_STATE_BLIND, COND_STATE_LAB,
                  COND_VIS_FROZEN,  COND_VIS_FT]
    cond_color = {
        COND_STATE_BLIND: COLORS["blind"],
        COND_STATE_LAB:   COLORS["labeled"],
        COND_VIS_FROZEN:  COLORS["visual-frozen"],
        COND_VIS_FT:      COLORS["visual-ft"],
    }
    source_map = [
        (state_df,  "blind",             COND_STATE_BLIND),
        (state_df,  "labeled",           COND_STATE_LAB),
        (visual_df, "frozen-baseline",   COND_VIS_FROZEN),
        (visual_df, "finetune-baseline", COND_VIS_FT),
    ]

    metrics_r4 = [
        ("landed_pct",                     "Landing rate",          "% of episodes",         (0, 100)),
        # Crash rates here are ~0.8-5.7% — keep zoomed (not 0-100) for legibility.
        ("crashed_pct",                    "Crash rate",            "% of episodes",         (0, 15)),
        ("thrust_autocorr_lag1",           "Thrust autocorrelation",
                                                                    "lag-1 autocorrelation", (0, 1)),
        ("side_thrust_reversals_per_step", "Side-thrust reversals",
                                                                    "events per step",       None),
        ("frac_upright",                   "Fraction upright",      "fraction of steps",     (0, 1)),
    ]

    # Build a combined long-form per-seed dataframe with public condition
    # names, then a summary frame, so we can reuse `bar_with_seed_dots`.
    combined_per_seed = []
    combined_summary  = []
    for metric, *_ in metrics_r4:
        rows_ps, rows_summ = [], []
        for src_df, src_variant, cond_name in source_map:
            ps = per_seed_means(src_df, metric)
            ps = ps[ps["variant"] == src_variant].copy()
            ps["condition"] = cond_name
            rows_ps.append(ps[["condition", "seed", metric]])
            summ = variant_summary(src_df, metric)
            summ = summ[summ["variant"] == src_variant].copy()
            summ["condition"] = cond_name
            rows_summ.append(summ[["condition", "mean", "std", "se", "n"]])
        ps_df = pd.concat(rows_ps, ignore_index=True)
        summ_df = pd.concat(rows_summ, ignore_index=True)
        ps_df["metric"] = metric
        summ_df["metric"] = metric
        combined_per_seed.append(ps_df)
        combined_summary.append(summ_df)
    per_seed_all = pd.concat(combined_per_seed, ignore_index=True)
    summary_all  = pd.concat(combined_summary, ignore_index=True)

    # ---------------- Plot ----------------
    # 2-row layout per the writing brief:
    #   Top:    landed_pct | crashed_pct (shared y 0-100%)
    #   Bottom: side-thrust reversals | side-thrust autocorr | frac upright
    fig = plt.figure(figsize=(13.0, 8.8), dpi=DPI, constrained_layout=True)
    gs = fig.add_gridspec(2, 6, hspace=0.15)
    axes_map = {
        "landed_pct":                     fig.add_subplot(gs[0, 0:3]),
        "crashed_pct":                    fig.add_subplot(gs[0, 3:6]),
        "thrust_autocorr_lag1":           fig.add_subplot(gs[1, 0:2]),
        "side_thrust_reversals_per_step": fig.add_subplot(gs[1, 2:4]),
        "frac_upright":                   fig.add_subplot(gs[1, 4:6]),
    }

    for metric, title, ylabel, ylim in metrics_r4:
        ax = axes_map[metric]
        # Order summary + per-seed rows by canonical cond order so bars
        # appear in the expected sequence.
        ms = summary_all[summary_all["metric"] == metric].copy()
        ms = ms.set_index("condition").loc[cond_order].reset_index()
        ms = ms.rename(columns={"condition": "variant"})
        ps = per_seed_all[per_seed_all["metric"] == metric].rename(
            columns={"condition": "variant"}).copy()
        bar_with_seed_dots(ax, ms, ps, metric, cond_color,
                           title=title, ylabel=ylabel)
        # Give headroom for the p/d annotation labels above each bar.
        if ylim is not None:
            lo, hi = ylim
            ax.set_ylim(lo, hi + 0.18 * (hi - lo))
        else:
            top = float((ms["mean"] + ms["se"]).max())
            ax.set_ylim(0, top * 1.30)

        # Per-bar p/d vs state-blind (leftmost). Baseline gets a "(baseline)"
        # label so the convention is explicit.
        baseline_vals = ps[ps["variant"] == COND_STATE_BLIND][metric].values
        bar_x = np.arange(len(cond_order)) * 0.6
        y_lo, y_hi = ax.get_ylim()
        offset = 0.025 * (y_hi - y_lo)
        for i, cond in enumerate(cond_order):
            bar_top = float(
                ms[ms["variant"] == cond]["mean"].iloc[0]
                + ms[ms["variant"] == cond]["se"].iloc[0])
            if cond == COND_STATE_BLIND:
                ax.text(bar_x[i], bar_top + offset, "(baseline)",
                        ha="center", va="bottom", fontsize=8,
                        color="#888", style="italic")
                continue
            cmp_vals = ps[ps["variant"] == cond][metric].values
            p, d = _ttest_cohens_d_arrays(baseline_vals, cmp_vals)
            if p is None:
                continue
            txt = (f"p={p:.3f}\nd={d:.2f}"
                   if d is not None else f"p={p:.3f}")
            ax.text(bar_x[i], bar_top + offset, txt,
                    ha="center", va="bottom", fontsize=8, color="#444")

    fig.suptitle(
        r"$\bf{With\ Physics\ Fixed,\ Perception\ Changes\ Control}$" "\n"
        "\n"
        "Under gym-default (fixed) physics, state labels carry no signal "
        "and state-blind ≈ state-labeled\n"
        "Visual-blind agents (no physics labels) shift toward reactive "
        "correction anyway (N=5 seeds)"
        "\n ",
        fontsize=SUPTITLE_FONTSIZE, linespacing=SUPTITLE_LINESPACING,
    )
    fig.text(0.5, -0.01,
             "bars = mean ± SE across seeds; dots = per-seed means",
             ha="center", va="top", fontsize=9, color="#555")
    save_figure(fig, fig_dir, "result4-state-vs-visual")

    # ---------------- Table / numbers ----------------
    long_rows = []
    md_rows = []
    for metric, title, *_ in metrics_r4:
        ms = summary_all[summary_all["metric"] == metric].set_index("condition")
        cells = []
        for cond in cond_order:
            if cond not in ms.index:
                cells.append("—")
                long_rows.append({
                    "figure": "result4", "metric": metric,
                    "condition": cond.replace("\n", " "),
                    "mean": None, "se": None, "n_seeds": None,
                })
                continue
            r = ms.loc[cond]
            digits = 1 if "pct" in metric else 3
            cells.append(f"{fmt(r['mean'], digits)} ± {fmt(r['se'], digits)}")
            long_rows.append({
                "figure": "result4", "metric": metric,
                "condition": cond.replace("\n", " "),
                "mean": float(r["mean"]), "se": float(r["se"]),
                "n_seeds": int(r["n"]),
            })
        md_rows.append([title] + cells)

    md = (
        "# Result 4. With Physics Fixed, Perception Changes Control "
        "(gym-default)\n\n"
        "State agents vs visual matched-encoder agents under fixed Lunar "
        "Lander physics. N=5 seeds per condition.\n\n"
        f"State source: `e1-blind-vs-labeled-gym-default-128-{DATE_TAG}`.\n"
        f"Visual source: `matched-enc-all-6-conditions-{DATE_TAG}/all-6/` "
        "(only `frozen-baseline` and `finetune-baseline` — swap variants "
        "Fr+Fr / Ft+Ft / Fr+Ft / Ft+Fr excluded).\n\n"
    )
    md += md_table(["Metric"] + [c.replace("\n", " ") for c in cond_order],
                   md_rows)
    save_numbers(
        fig_dir, pd.DataFrame(long_rows), md,
        {
            "figure": "result4-state-vs-visual",
            "title": "With Physics Fixed, Perception Changes Control",
            "state_source": str(state_csv),
            "visual_source": str(visual_csv),
            "metrics": [m for m, *_ in metrics_r4],
            "conditions": [c.replace("\n", " ") for c in cond_order],
            "n_seeds_per_condition": 5,
        },
    )
    print(f"  wrote: {fig_dir}/")


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
        # fig-reversal-rate-over-time is now produced by build_fig_reversal_time
        # from cross-config dirs with per-seed aggregation.
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
        suptitle_map={
            ("easy-ID", "median"): "Same Task, Different Controllers",
            ("easy-OOD-medium", "median"): "Medium-OOD Trajectories: Labeled vs Blind",
            ("easy-OOD-medium", "extreme"): "Medium-OOD Trajectories: Labeled vs Blind",
        },
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
        fontsize=SUPTITLE_FONTSIZE, fontweight="bold", y=1.0,
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


def _per_episode_binned_rates(eps, n_bins: int, kind: str) -> np.ndarray:
    """Per-episode per-bin reversal rate (shape: (n_eps, n_bins)). Episodes
    shorter than n_bins are dropped (not zero-filled, to avoid downward bias
    on the per-bin mean)."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from reversal_rate_over_time import reversal_mask
    out = []
    for a, _, _ in eps:
        T = len(a)
        if T < n_bins:
            continue
        rev = reversal_mask(a, kind=kind)
        bin_ids = (np.arange(T) * n_bins // T).clip(0, n_bins - 1)
        row = np.zeros(n_bins)
        for b in range(n_bins):
            mask = bin_ids == b
            if mask.any():
                row[b] = rev[mask].mean()
        out.append(row)
    if not out:
        return np.zeros((0, n_bins))
    return np.stack(out, axis=0)


def _load_per_seed_reversal_curves(csv_path: Path, n_bins: int,
                                   n_eps_per_seed: int) -> dict:
    """For a cross-config CSV, return
        {variant: {"any": (n_seeds, n_bins),
                   "main": ..., "lateral": ...,
                   "n_seeds": int}}.
    Each row of the (n_seeds, n_bins) matrix is one seed's per-bin mean
    rate across that seed's ~n_eps_per_seed episodes. The plot then takes
    mean ± SE across these per-seed means (SE = std/sqrt(n_seeds)).
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from reversal_rate_over_time import load_actions
    df = pd.read_csv(csv_path)
    out: dict = {}
    for variant, g in df.groupby("variant"):
        per_seed_rows = {k: [] for k in ("any", "main", "lateral")}
        seeds = []
        for seed, gs in g.groupby("seed"):
            traj_dir = Path(gs["npz_path"].iloc[0]).parent
            eps = load_actions(traj_dir, n=n_eps_per_seed)
            if not eps:
                continue
            seeds.append(int(seed))
            for kind in per_seed_rows:
                mat = _per_episode_binned_rates(eps, n_bins, kind)
                per_seed_rows[kind].append(mat.mean(axis=0))
        entry = {"n_seeds": len(seeds)}
        for kind in per_seed_rows:
            if per_seed_rows[kind]:
                entry[kind] = np.stack(per_seed_rows[kind], axis=0)
            else:
                entry[kind] = np.zeros((0, n_bins))
        out[str(variant)] = entry
    return out


def _plot_reversal_time_row(axes_row, per_seed_curves: dict, n_bins: int,
                            show_xlabel: bool = False,
                            ylabel_tag: str | None = None,
                            show_legend: bool = False,
                            ylim: tuple = (0.0, 0.6)):
    """One row (3 cols: any/main/lateral). Bold aggregate mean line + SE
    band pooled across all episodes (pooled across all seeds × episodes;
    each episode contributes one per-bin sample). SE = std / sqrt(N_eps).
    """
    colors = {"blind": "#ff7f0e", "labeled": "#1f77b4"}
    x = (np.arange(n_bins) + 0.5) / n_bins
    for col, kind in enumerate(("any", "main", "lateral")):
        ax = axes_row[col]
        for variant in ("blind", "labeled"):
            entry = per_seed_curves.get(variant)
            if entry is None or entry[kind].shape[0] == 0:
                continue
            color = colors[variant]
            mat = entry[kind]  # (n_seeds, n_bins)
            mean = mat.mean(axis=0)
            se = mat.std(axis=0, ddof=1) / np.sqrt(mat.shape[0])
            ax.fill_between(x, mean - se, mean + se, color=color, alpha=0.22,
                            linewidth=0, zorder=2)
            label = variant.capitalize() if (col == 0 and show_legend) else None
            ax.plot(x, mean, color=color, linewidth=2.4,
                    label=label, zorder=5)
        ax.set_xlim(0, 1)
        ax.set_ylim(*ylim)
        ax.grid(alpha=0.25)
        ax.set_axisbelow(True)
        if show_xlabel:
            ax.set_xlabel("normalized trajectory time  (0 = start, 1 = end)",
                          fontsize=9)
    label_text = (f"{ylabel_tag}\nreversal rate (per step)"
                  if ylabel_tag else "reversal rate (per step)")
    axes_row[0].set_ylabel(label_text, fontsize=10)
    if show_legend:
        axes_row[0].legend(frameon=False, fontsize=9, loc="upper right")


def build_fig_reversal_time(out_root: Path, seed: int = 0):
    """Fig 4: Reversal Rate Over Normalized Trajectory Time.

    Produces both:
      - Option A (main): 1x3 grid, full-variation easy-ID only.
        "Main-Thrust Correction Persists Through Descent"
      - Option B (supplementary): 4x3 grid, easy-ID / medium-ID /
        easy-OOD-medium / easy-OOD-hard. "Controller Timing Changes Under
        Shift"

    Aggregation: per-(variant, seed) curve = per-bin mean reversal rate
    across that seed's episodes. The figure shows thin pale per-seed
    curves plus a bold aggregate mean line across N=10 seeds. No SE band
    -- the per-seed curves are the error display.
    """
    fig_dir = out_root / "fig-reversal-rate-over-time"
    fig_dir.mkdir(parents=True, exist_ok=True)

    n_bins = 20
    n_eps_per_seed = 200
    kind_titles = ("Any reversal", "Main-thrust transition",
                   "Lateral transition")

    # ---- Option A: 1x3 easy-ID only ----
    csv_a = BD_SCENARIOS["easy-ID"]
    if not csv_a.exists():
        print(f"  ABORT: {csv_a} missing")
        return
    print(f"  loading easy-ID curves from {csv_a.parent.name}/{csv_a.name}")
    curves_a = _load_per_seed_reversal_curves(csv_a, n_bins, n_eps_per_seed)

    fig, axes = plt.subplots(1, 3, figsize=(13, 5.8), dpi=150,
                             constrained_layout=True, sharex=True)
    _plot_reversal_time_row(axes, curves_a, n_bins,
                            show_xlabel=True, show_legend=True,
                            ylabel_tag="full-variation easy-ID")
    for col, t in enumerate(kind_titles):
        axes[col].set_title(t, pad=10)
    fig.suptitle(
        r"$\bf{Main\text{-}Thrust\ Correction\ Persists\ Through\ Descent}$" "\n"
        "\n"
        "Reversal rate over normalized trajectory time "
        "(full-variation easy-ID)"
        "\n ",
        fontsize=SUPTITLE_FONTSIZE, linespacing=SUPTITLE_LINESPACING,
    )
    fig.text(0.5, -0.02,
             "line = mean across N=10 seeds; "
             "band = ±SE across seeds",
             ha="center", va="top", fontsize=9, color="#555")
    out_a = fig_dir / "fig-reversal-rate-over-time-easy-id.png"
    fig.savefig(out_a, bbox_inches="tight")
    fig.savefig(out_a.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote: {out_a}")

    # ---- Option B: 4x3 ID/OOD grid ----
    rows_b = [
        ("easy-ID",         "full-variation easy-ID"),
        ("medium-ID",       "full-variation medium-ID"),
        ("easy-OOD-medium", "easy-trained, OOD medium"),
        ("easy-OOD-hard",   "easy-trained, OOD hard"),
    ]
    all_curves = {"easy-ID": curves_a}
    for tag, _ in rows_b:
        if tag in all_curves:
            continue
        csv = BD_SCENARIOS.get(tag)
        if csv is None or not csv.exists():
            print(f"  SKIP {tag}: csv missing")
            all_curves[tag] = {}
            continue
        print(f"  loading {tag} curves from {csv.parent.name}/{csv.name}")
        all_curves[tag] = _load_per_seed_reversal_curves(
            csv, n_bins, n_eps_per_seed)

    fig, axes = plt.subplots(len(rows_b), 3, figsize=(13, 3.0 * len(rows_b)),
                             dpi=150, constrained_layout=True, sharex=True)
    for r, (tag, ylabel) in enumerate(rows_b):
        _plot_reversal_time_row(
            axes[r], all_curves[tag], n_bins,
            show_xlabel=(r == len(rows_b) - 1),
            ylabel_tag=ylabel,
            show_legend=(r == 0),
        )
        if r == 0:
            for col, t in enumerate(kind_titles):
                axes[r, col].set_title(t)
    fig.suptitle(
        r"$\bf{Controller\ Timing\ Changes\ Under\ Shift}$" "\n"
        "\n"
        "Reversal rate over normalized trajectory time, ID vs OOD"
        "\n ",
        fontsize=SUPTITLE_FONTSIZE, linespacing=SUPTITLE_LINESPACING,
    )
    fig.text(0.5, -0.01,
             "line = mean across N=10 seeds; "
             "band = ±SE across seeds",
             ha="center", va="top", fontsize=9, color="#555")
    out_b = fig_dir / "fig-reversal-rate-over-time-id-ood-grid.png"
    fig.savefig(out_b, bbox_inches="tight")
    fig.savefig(out_b.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote: {out_b}")

    (fig_dir / "metadata.json").write_text(json.dumps({
        "figure": "fig-reversal-rate-over-time",
        "option_A": {
            "title": "Main-Thrust Correction Persists Through Descent",
            "scenarios": ["easy-ID"],
            "source_csv": str(BD_SCENARIOS["easy-ID"]),
            "image": str(out_a),
        },
        "option_B": {
            "title": "Controller Timing Changes Under Shift",
            "scenarios": [t for t, _ in rows_b],
            "sources": {t: str(BD_SCENARIOS[t]) for t, _ in rows_b
                        if t in BD_SCENARIOS},
            "image": str(out_b),
        },
        "n_bins": n_bins,
        "max_episodes_per_seed": n_eps_per_seed,
        "underlying_script": "scripts/viz/reversal_rate_over_time.py",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }, indent=2))


def build_overlay_corrupt(out_root: Path, seed: int = 0):
    """6b. Label-corruption trajectory companion.

    For full-variation-easy, produces three V_BD overlays: clean labeled vs
    each of {world-zeroed, body-zeroed, all-zeroed} labels. The clean column
    is the same seed's `trajectories/` sibling of the corrupted run, so the
    contrast is honestly within-seed.

    Median strategy only (5b lesson: extreme/cherry-picked variants
    over-emphasize degenerate "did very little" episodes when both columns
    are smooth controllers, which they likely are here — see Fig 3 autocorr).
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from trajectory_overlay_variants import pick_seed_dirs, render_scenario_bd

    fig_dir = out_root / "fig-overlay-corrupt"
    fig_dir.mkdir(parents=True, exist_ok=True)
    if not BACKDROP_RGB.exists():
        print(f"  ABORT: backdrop {BACKDROP_RGB} missing")
        return

    name = f"body-vs-world-corruption-{DATE_TAG}"
    comp = "full-variation-easy"
    csv_path = cross_config_path(name) / comp / "metrics_by_variant.csv"
    if not csv_path.exists():
        print(f"  ABORT: {csv_path} missing")
        return

    modes = [
        ("world-zeroed", "labeled-world-zeroed", r"World\text{-}zeroed\ labels"),
        ("body-zeroed",  "labeled-body-zeroed",  r"Body\text{-}zeroed\ labels"),
        ("all-zeroed",   "labeled-zeroed",       r"All\text{-}zeroed\ labels"),
    ]
    records = []
    for slug, variant, right_title_tex in modes:
        # pick_seed_dirs returns {"blind": ..., "labeled": ...}. We repurpose:
        # "blind" role = clean labeled; "labeled" role = corrupted variant.
        # Median strategy picks the seed whose per-seed mean is closest to
        # the variant mean (representative seed).
        picks = pick_seed_dirs(
            csv_path, strategy="median",
            blind_variant="labeled",
            labeled_variant=variant,
        )
        corrupt_dir = picks["labeled"]["traj_dir"]
        # Force clean baseline to be the same seed's `trajectories/` sibling
        # of the corrupted dir (honest within-seed contrast).
        clean_dir = corrupt_dir.parent / "trajectories"
        if not clean_dir.is_dir():
            print(f"  SKIP {slug}: clean sibling {clean_dir} missing")
            continue

        out_png = fig_dir / f"{slug}_median.png"
        titles = (r"Clean\ labeled", right_title_tex)
        suptitle = (
            r"$\bf{Wrong\ Labels,\ Same\ Controller\ Style}$" "\n"
            "\n"
            f"Labeled agent: clean labels vs {slug} labels "
            "(full-variation easy)\n"
            f"representative seed (s{picks['labeled']['seed']}), "
            "50 random episodes per column"
        )
        try:
            meta = render_scenario_bd(
                f"corrupt-{slug}-median",
                clean_dir, corrupt_dir,
                BACKDROP_RGB, out_png, seed=seed,
                titles=titles, suptitle=suptitle,
            )
        except Exception as e:
            print(f"  FAIL {slug}: {e}")
            continue
        records.append({
            "artifact": "vbd_corrupt",
            "mode": slug, "strategy": "median",
            "comp": comp,
            "corrupt_variant": variant,
            "corrupt_seed": picks["labeled"]["seed"],
            "corrupt_metric": picks["labeled"]["metric_value"],
            "clean_traj_dir": str(clean_dir),
            "corrupt_traj_dir": str(corrupt_dir),
            "n_clean_plotted": meta["n_blind"],
            "n_corrupt_plotted": meta["n_labeled"],
            "image": str(out_png),
        })
        print(f"  wrote: {out_png}")

    if records:
        df = pd.DataFrame(records)
        df.to_csv(fig_dir / "numbers.csv", index=False)
        md = [
            "# Fig 6b. Label-corruption trajectory companion",
            "",
            f"Source: `{name}/{comp}`.",
            "",
            "Artifacts (median strategy, within-seed contrast):",
            "- `world-zeroed_median.{png,svg}` — clean labels vs world-zeroed labels",
            "- `body-zeroed_median.{png,svg}`  — clean labels vs body-zeroed labels",
            "- `all-zeroed_median.{png,svg}`   — clean labels vs all-zeroed labels",
            "",
            "Each row picks the representative seed within the corrupted",
            "variant (per-seed mean closest to variant mean on",
            "`side_thrust_reversals`); the clean column is that same seed's",
            "`trajectories/` sibling.",
            "",
            "Per the 5b lesson: extreme/cherry-picked variants are not",
            "produced — when both columns are smooth controllers, the",
            "extreme picks degenerate into 'did very little' episodes.",
            "",
            "| mode | corrupt seed | corrupt sel.metric |",
            "|---|---|---|",
        ]
        for r in records:
            md.append(f"| {r['mode']} | s{r['corrupt_seed']} | "
                      f"{r['corrupt_metric']:.1f} |")
        (fig_dir / "numbers.md").write_text("\n".join(md) + "\n")
        (fig_dir / "metadata.json").write_text(json.dumps({
            "figure": "fig-overlay-corrupt",
            "title": "Wrong Labels, Same Controller Style (trajectory)",
            "cross_config_dir": str(cross_config_path(name)),
            "comp": comp,
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
    "overlay_corrupt": build_overlay_corrupt,
    "fig_reversal_time": build_fig_reversal_time,
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
