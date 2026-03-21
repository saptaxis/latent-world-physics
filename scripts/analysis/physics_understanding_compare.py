"""Compare physics understanding results across multiple world models.

Reads physics_understanding.json files from multiple model runs and produces:
  1. Summary table (console + markdown) — all models x all constants
  2. Consistency heatmap — R² of spurious dependencies
  3. Compounding overlay — MSE vs horizon curves
  4. Per-constant error bar chart

Usage:
  # Auto-discover all reports under a directory:
  python lunar_lander/scripts/physics_understanding_compare.py \
    --search-dir /media/hdd1/.../world-model-ladder-runs \
    --output-dir ~/vsr-tmp/physics-understanding-comparison

  # Or specify individual JSON files:
  python lunar_lander/scripts/physics_understanding_compare.py \
    --json-files path/to/model1/physics_understanding.json \
                 path/to/model2/physics_understanding.json \
    --output-dir ~/vsr-tmp/physics-understanding-comparison
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.use("Agg")

# Constants to compare (order for display).
SCALAR_CONSTANTS = ["gravity", "main_thrust", "side_thrust", "kinematics", "angular_damping"]
CONSTANT_LABELS = {
    "gravity": "Gravity",
    "main_thrust": "Main Thrust",
    "side_thrust": "Side Thrust",
    "kinematics": "Kinematics",
    "angular_damping": "Ang. Damping",
}
STATE_DIMS = ["x", "y", "vx", "vy", "angle", "angular_vel"]


def load_reports(json_paths: list[Path]) -> dict[str, dict]:
    """Load JSON reports, keyed by model name (parent directory name)."""
    reports = {}
    for p in sorted(json_paths):
        # Model name = grandparent dir (parent is physics_understanding/).
        model_name = p.parent.parent.name
        with open(p) as f:
            reports[model_name] = json.load(f)
    return reports


def find_reports(search_dir: Path) -> list[Path]:
    """Recursively find all physics_understanding.json files."""
    return sorted(search_dir.rglob("physics_understanding/physics_understanding.json"))


def shorten_name(name: str) -> str:
    """Shorten model run name for display.

    mlp-delta-single_step_k1-policy--v3 -> mlp-ss-v3
    gru-delta-multi_step_k10-primitives -> gru-prims
    """
    parts = name.split("-delta-")
    if len(parts) != 2:
        return name
    arch = parts[0]
    rest = parts[1]

    # Step type.
    if "single_step" in rest:
        step = "ss"
    elif "multi_step" in rest:
        step = "ms"
    elif "elbo" in rest:
        step = "elbo"
    else:
        step = ""

    # Data type.
    if "primitives" in rest:
        data = "prims"
    elif "v3" in rest:
        data = "v3"
    elif "v2" in rest:
        data = "v2"
    else:
        data = "pol"

    return f"{arch}-{step}-{data}"


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def make_summary_table(reports: dict[str, dict]) -> str:
    """Build a markdown summary table of all constants x all models."""
    models = list(reports.keys())
    short = [shorten_name(m) for m in models]

    lines = []
    lines.append("# Physics Understanding — Model Comparison")
    lines.append("")

    # --- Oracle errors table ---
    lines.append("## Oracle (1-step) Relative Error (%)")
    lines.append("")
    header = "| Model | " + " | ".join(CONSTANT_LABELS[c] for c in SCALAR_CONSTANTS) + " |"
    sep = "|-------|" + "|".join("--------:" for _ in SCALAR_CONSTANTS) + "|"
    lines.append(header)
    lines.append(sep)

    for model, sname in zip(models, short):
        row = f"| {sname} |"
        for const in SCALAR_CONSTANTS:
            oracle = reports[model]["constants"].get(const, {}).get("oracle", {})
            err = oracle.get("relative_error")
            n = oracle.get("n_samples", 0)
            if err is not None and n > 0:
                row += f" {err * 100:.1f}% (n={n}) |"
            else:
                row += " — |"
        lines.append(row)
    lines.append("")

    # --- Rollout errors table ---
    lines.append("## Rollout (10-step) Relative Error (%)")
    lines.append("")
    lines.append(header)
    lines.append(sep)

    for model, sname in zip(models, short):
        row = f"| {sname} |"
        for const in SCALAR_CONSTANTS:
            rollout = reports[model]["constants"].get(const, {}).get("rollout", {})
            err = rollout.get("relative_error")
            n = rollout.get("n_samples", 0)
            if err is not None and n > 0:
                row += f" {err * 100:.1f}% (n={n}) |"
            else:
                row += " — |"
        lines.append(row)
    lines.append("")

    # --- GT reference ---
    # Take GT from the first model (should be same for same data source).
    lines.append("## GT Reference Values")
    lines.append("")
    first = reports[models[0]]
    for const in SCALAR_CONSTANTS:
        gt_mean = first["constants"].get(const, {}).get("oracle", {}).get("gt_mean")
        if gt_mean is not None:
            lines.append(f"- **{CONSTANT_LABELS[const]}:** {gt_mean:.6f}")
    lines.append("")

    # --- Compounding ---
    lines.append("## Compounding (Error Growth)")
    lines.append("")
    header_c = "| Model | a | b (exponent) | Useful h (angular_vel) | Useful h (vy) |"
    sep_c = "|-------|----:|----:|----:|----:|"
    lines.append(header_c)
    lines.append(sep_c)

    for model, sname in zip(models, short):
        comp = reports[model].get("compounding", {})
        fit = comp.get("fit_params", {})
        a = fit.get("a")
        b = fit.get("b")
        uh = comp.get("useful_horizon", {})
        h_avel = uh.get("angular_vel", "—")
        h_vy = uh.get("vy", "—")
        if a is not None:
            lines.append(f"| {sname} | {a:.4f} | {b:.2f} | {h_avel} | {h_vy} |")
        else:
            lines.append(f"| {sname} | — | — | — | — |")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_error_bars(reports: dict[str, dict], output_dir: Path):
    """Bar chart of oracle relative error per constant, grouped by model."""
    models = list(reports.keys())
    short = [shorten_name(m) for m in models]
    n_models = len(models)
    n_consts = len(SCALAR_CONSTANTS)

    fig, ax = plt.subplots(figsize=(max(12, n_models * 1.5), 6))

    x = np.arange(n_consts)
    width = 0.8 / n_models

    for i, (model, sname) in enumerate(zip(models, short)):
        errors = []
        for const in SCALAR_CONSTANTS:
            oracle = reports[model]["constants"].get(const, {}).get("oracle", {})
            err = oracle.get("relative_error")
            if err is not None:
                errors.append(err * 100)
            else:
                errors.append(0)
        offset = (i - n_models / 2 + 0.5) * width
        ax.bar(x + offset, errors, width, label=sname)

    ax.set_ylabel("Relative Error (%)")
    ax.set_title("Oracle (1-step) Relative Error by Constant")
    ax.set_xticks(x)
    ax.set_xticklabels([CONSTANT_LABELS[c] for c in SCALAR_CONSTANTS])
    ax.legend(loc="upper right", fontsize=8)
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "error_bars_oracle.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_dir / 'error_bars_oracle.png'}")


def plot_consistency_heatmap(reports: dict[str, dict], output_dir: Path):
    """Heatmap of consistency R² values across models and constants."""
    models = list(reports.keys())
    short = [shorten_name(m) for m in models]

    # Build columns: (constant, state_dim) pairs.
    col_labels = []
    for const in SCALAR_CONSTANTS:
        for dim in STATE_DIMS:
            col_labels.append(f"{CONSTANT_LABELS[const]}\nvs {dim}")

    matrix = np.full((len(models), len(col_labels)), np.nan)

    for i, model in enumerate(models):
        col = 0
        for const in SCALAR_CONSTANTS:
            consistency = reports[model]["constants"].get(const, {}).get("consistency", {})
            for dim in STATE_DIMS:
                r2 = consistency.get(dim)
                if r2 is not None:
                    matrix[i, col] = r2
                col += 1

    fig, ax = plt.subplots(figsize=(max(20, len(col_labels) * 0.7), max(4, len(models) * 0.6)))

    # Custom colormap: green (ok) -> yellow (borderline) -> red (spurious).
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("consistency", [
        (0.0, "#2d8a4e"),   # green: R² ~ 0
        (0.05, "#2d8a4e"),  # still green at 0.05
        (0.1, "#f5c542"),   # yellow: borderline
        (0.3, "#e85d3a"),   # orange
        (1.0, "#b71c1c"),   # red: strong spurious
    ])

    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=0.6)

    # Annotate cells with R² values.
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if not np.isnan(val):
                color = "white" if val > 0.3 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=6, color=color)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=7, rotation=45, ha="right")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(short, fontsize=9)
    ax.set_title("Consistency R² (want < 0.05, red = spurious dependency)")

    # Add vertical separators between constants.
    for k in range(1, len(SCALAR_CONSTANTS)):
        ax.axvline(x=k * len(STATE_DIMS) - 0.5, color="white", linewidth=2)

    fig.colorbar(im, ax=ax, shrink=0.8, label="R²")
    fig.tight_layout()
    fig.savefig(output_dir / "consistency_heatmap.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_dir / 'consistency_heatmap.png'}")


def plot_compounding_overlay(reports: dict[str, dict], output_dir: Path):
    """Overlay MSE-vs-horizon curves for all models."""
    models = list(reports.keys())
    short = [shorten_name(m) for m in models]

    fig, ax = plt.subplots(figsize=(10, 6))

    for model, sname in zip(models, short):
        comp = reports[model].get("compounding", {})
        mse_by_h = comp.get("mse_by_horizon", {})
        if not mse_by_h:
            continue

        horizons = sorted([int(h) for h in mse_by_h.keys()])
        mses = [mse_by_h[str(h)] for h in horizons]

        ax.plot(horizons, mses, "o-", label=sname, markersize=4)

    ax.set_xlabel("Horizon (steps)")
    ax.set_ylabel("MSE")
    ax.set_title("Compounding: MSE vs Rollout Horizon")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "compounding_overlay.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_dir / 'compounding_overlay.png'}")


def plot_oracle_vs_rollout(reports: dict[str, dict], output_dir: Path):
    """Scatter: oracle error vs rollout error per model per constant.

    Shows how much each constant degrades under autoregression.
    Points on the diagonal mean no compounding cost for that constant.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    models = list(reports.keys())
    short = [shorten_name(m) for m in models]
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "<"]

    for i, (model, sname) in enumerate(zip(models, short)):
        oracle_errs = []
        rollout_errs = []
        labels = []
        for const in SCALAR_CONSTANTS:
            cdata = reports[model]["constants"].get(const, {})
            o_err = cdata.get("oracle", {}).get("relative_error")
            r_err = cdata.get("rollout", {}).get("relative_error")
            if o_err is not None and r_err is not None:
                oracle_errs.append(o_err * 100)
                rollout_errs.append(r_err * 100)
                labels.append(CONSTANT_LABELS[const])

        if oracle_errs:
            marker = markers[i % len(markers)]
            ax.scatter(oracle_errs, rollout_errs, marker=marker, s=60,
                       label=sname, zorder=3)

    # Diagonal line (no compounding cost).
    lims = ax.get_xlim()
    ax.plot([0.1, 10000], [0.1, 10000], "k--", alpha=0.3, linewidth=1)
    ax.set_xlabel("Oracle Error (%)")
    ax.set_ylabel("Rollout Error (%)")
    ax.set_title("Oracle vs Rollout Error (diagonal = no compounding cost)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    fig.tight_layout()
    fig.savefig(output_dir / "oracle_vs_rollout.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_dir / 'oracle_vs_rollout.png'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare physics understanding results across world models."
    )
    parser.add_argument(
        "--search-dir", type=str, default=None,
        help="Directory to recursively search for physics_understanding.json files.",
    )
    parser.add_argument(
        "--json-files", type=str, nargs="+", default=None,
        help="Explicit list of physics_understanding.json file paths.",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Where to save comparison outputs (plots + markdown).",
    )
    args = parser.parse_args()

    # Discover reports.
    if args.json_files:
        json_paths = [Path(p) for p in args.json_files]
    elif args.search_dir:
        json_paths = find_reports(Path(args.search_dir))
    else:
        parser.error("Provide either --search-dir or --json-files.")

    if not json_paths:
        print("No physics_understanding.json files found.")
        return

    print(f"Found {len(json_paths)} reports:")
    for p in json_paths:
        print(f"  {p.parent.parent.name}")

    # Load all reports.
    reports = load_reports(json_paths)
    print(f"\nLoaded {len(reports)} models.\n")

    # Output directory.
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Summary table (console + markdown).
    table = make_summary_table(reports)
    print(table)

    md_path = output_dir / "comparison.md"
    with open(md_path, "w") as f:
        f.write(table + "\n")
    print(f"\nMarkdown saved to: {md_path}")

    # Plots.
    print("\nGenerating plots...")
    plot_error_bars(reports, output_dir)
    plot_consistency_heatmap(reports, output_dir)
    plot_compounding_overlay(reports, output_dir)
    plot_oracle_vs_rollout(reports, output_dir)

    print(f"\nDone. All outputs in: {output_dir}")


if __name__ == "__main__":
    main()
