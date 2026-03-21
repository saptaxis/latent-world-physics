"""Plotting utilities for world-model evaluation."""
from __future__ import annotations

from pathlib import Path


def plot_horizon_curve(horizon_data: dict, output_path: str,
                       title: str = "Horizon Error Curve",
                       log_scale: bool = False) -> None:
    """Plot MSE vs horizon as a line chart."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    horizons = sorted(horizon_data.keys())
    errors = [horizon_data[h] for h in horizons]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(horizons, errors, "o-", linewidth=2, markersize=6)
    ax.set_xlabel("Horizon (steps)")
    ax.set_ylabel("Mean MSE")
    ax.set_title(title)
    if log_scale:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_per_dim_bars(per_dim_mse: dict, output_path: str,
                      title: str = "Per-Dimension MSE") -> None:
    """Plot per-dimension MSE as a bar chart."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    names = list(per_dim_mse.keys())
    values = list(per_dim_mse.values())

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(names)), values)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("MSE")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def export_plots(tb_dir: str, plot_dir: str) -> None:
    """Read TensorBoard event files and save each scalar tag as a PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        return

    tb_path = Path(tb_dir)
    if not tb_path.exists():
        return

    ea = EventAccumulator(str(tb_path))
    try:
        ea.Reload()
    except Exception:
        return

    tags = ea.Tags().get("scalars", [])
    if not tags:
        return

    out = Path(plot_dir)
    out.mkdir(parents=True, exist_ok=True)

    for tag in tags:
        events = ea.Scalars(tag)
        if len(events) < 2:
            continue

        steps = [e.step for e in events]
        values = [e.value for e in events]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(steps, values)
        ax.set_xlabel("Step")
        ax.set_ylabel(tag)
        ax.set_title(tag)
        ax.grid(True, alpha=0.3)

        filename = tag.replace("/", "_").replace(" ", "_") + ".png"
        fig.savefig(out / filename, dpi=100, bbox_inches="tight")
        plt.close(fig)
