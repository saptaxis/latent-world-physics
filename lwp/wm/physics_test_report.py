# lunar_lander/src/wm/physics_test_report.py
"""Report generation for physics unit tests.

Produces:
  1. JSON report — machine-readable, full detail
  2. Markdown report — human-readable summary with tables and key findings
  3. Console summary — formatted table for quick reading
  4. Per-maneuver plots — predicted vs GT state dims over time
  5. Per-maneuver videos — side-by-side lander animations

All five report types are implemented in this module.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def save_json_report(
    results: dict,
    output_path: str | Path,
    run_name: str = "",
    gt_mode: str = "replay",
    n_episodes: int = 0,
) -> Path:
    """Save physics test results to JSON.

    Args:
        results: Dict from run_physics_tests() — per-maneuver aggregate + per-episode.
        output_path: Where to save the JSON file.
        run_name: Name of the model run being evaluated.
        gt_mode: Ground truth mode used ("teleport" or "replay").
        n_episodes: Total episodes tested.

    Returns:
        Path to saved file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Compute overall pass rate.
    total_tested = sum(r["n_tested"] for r in results.values())
    total_passed = sum(r["n_passed"] for r in results.values())

    report = {
        "run_name": run_name,
        "gt_mode": gt_mode,
        "n_episodes": n_episodes,
        "overall_pass_rate": total_passed / total_tested if total_tested > 0 else 0.0,
        "total_tested": total_tested,
        "total_passed": total_passed,
        "maneuvers": results,
    }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    return output_path


def generate_markdown_report(
    results: dict,
    output_path: str | Path,
    run_name: str,
    gt_mode: str,
    n_episodes: int,
) -> Path:
    """Generate a human-readable markdown diagnostics report.

    Structure:
    1. Header with run info (model name, GT mode, episode count)
    2. Summary table (all maneuvers, pass rate, mean error)
    3. Key findings (which passed, which failed, worst offenders)
    4. Per-maneuver detail sections (not yet populated — placeholder
       for future expansion with per-episode breakdown)

    This report is intended for quick scanning — a researcher should
    be able to open the .md file and immediately see which physics
    properties the model has learned vs. which it struggles with.

    Args:
        results: Output from run_physics_tests() — per-maneuver aggregate
            dicts with keys n_tested, n_passed, pass_rate, mean_error,
            max_error, per_episode.
        output_path: Where to save the .md file.
        run_name: Run identifier (e.g. "ladder-gru" or a wandb run name).
        gt_mode: GT mode used ("teleport" or "replay").
        n_episodes: Number of episodes tested.

    Returns:
        output_path (as a Path object).
    """
    lines = []
    lines.append(f"# Physics Test Report: {run_name}")
    lines.append("")
    lines.append(f"- **GT mode:** {gt_mode}")
    lines.append(f"- **Episodes tested:** {n_episodes}")
    lines.append(f"- **Maneuvers:** {len(results)}")
    lines.append("")

    # ---- Summary table ----
    # One row per maneuver: tested count, passed count, pass rate,
    # mean relative/absolute error, max error. Sorted alphabetically
    # for stable ordering across runs.
    lines.append("## Summary")
    lines.append("")
    lines.append("| Maneuver | Tested | Passed | Pass Rate | Mean Error | Max Error |")
    lines.append("|----------|--------|--------|-----------|------------|-----------|")
    for name, agg in sorted(results.items()):
        lines.append(
            f"| {name} | {agg['n_tested']} | {agg['n_passed']} | "
            f"{agg['pass_rate']:.0%} | {agg['mean_error']:.4f} | {agg['max_error']:.4f} |"
        )
    lines.append("")

    # ---- Key findings ----
    # Quick triage: which maneuvers are fully passing, which have
    # failures, and which couldn't be tested (e.g., precondition unmet).
    passed_all = [n for n, a in results.items() if a["pass_rate"] == 1.0 and a["n_tested"] > 0]
    failed_some = [n for n, a in results.items() if a["pass_rate"] < 1.0 and a["n_tested"] > 0]
    not_tested = [n for n, a in results.items() if a["n_tested"] == 0]

    lines.append("## Key Findings")
    lines.append("")
    if passed_all:
        lines.append(f"- **All passed:** {', '.join(passed_all)}")
    if failed_some:
        lines.append(f"- **Some failures:** {', '.join(failed_some)}")
    if not_tested:
        lines.append(f"- **Not tested (precondition unmet):** {', '.join(not_tested)}")
    lines.append("")

    # Write the report to disk.
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
    return output_path


def print_console_summary(
    results: dict,
    run_name: str = "",
):
    """Print a formatted physics test summary to console.

    Shows a table with per-maneuver pass rate, mean error, and max error.
    """
    # Description labels for the summary table.
    labels = {
        "free_fall": "Free fall (gravity)",
        "full_thrust": "Full thrust (F=ma)",
        "side_thrust": "Side thrust (lateral)",
        "angular_decay": "Angular decay (damping)",
        "hover": "Hover (equilibrium)",
        "conservation": "Conservation (momentum)",
        "angle_thrust": "Angle-thrust (vectoring)",
    }

    total_tested = sum(r["n_tested"] for r in results.values())
    total_passed = sum(r["n_passed"] for r in results.values())

    print(f"\nPhysics Unit Tests: {run_name}")
    print("=" * 65)
    print(f"{'Maneuver':<28} {'Pass Rate':>10} {'Mean Err':>10} {'Max Err':>10}")
    print("-" * 65)

    for name, data in results.items():
        label = labels.get(name, name)
        n = data["n_tested"]
        if n == 0:
            print(f"{label:<28} {'(skipped)':>10}")
            continue

        rate = f"{data['pass_rate']:.0%}"
        mean = data["mean_error"]
        mx = data["max_error"]

        # Use % for relative-error maneuvers, absolute for near-zero targets.
        if name in ("hover", "conservation"):
            mean_str = f"{mean:.3f}"
            max_str = f"{mx:.3f}"
        else:
            mean_str = f"{mean:.1%}"
            max_str = f"{mx:.1%}"

        print(f"{label:<28} {rate:>10} {mean_str:>10} {max_str:>10}")

    print("-" * 65)
    overall = total_passed / total_tested if total_tested > 0 else 0.0
    print(f"{'Overall':<28} {overall:.0%} ({total_passed}/{total_tested})")
    print()


# --------------------------------------------------------------------------- #
# Per-maneuver visualization: state overlay plots + side-by-side videos       #
# --------------------------------------------------------------------------- #

# Kinematic dim names for plot labels.
KINEMATIC_DIM_NAMES = [
    "x", "y", "vx", "vy", "angle", "angular_vel", "left_leg", "right_leg",
]


def plot_maneuver_comparison(
    maneuver_name: str,
    model_states: np.ndarray,
    gt_states: np.ndarray,
    output_path: str | Path,
    fps: int = 50,
):
    """Plot per-dimension model vs GT trajectories for a maneuver.

    Similar to rollout_viz.plot_state_overlay but labeled for physics tests.
    Each kinematic dimension gets its own subplot: GT in blue, model in red,
    with a shaded error band showing the absolute difference. This makes it
    easy to see WHERE the model diverges (which dim, which timestep).

    Args:
        maneuver_name: Name of the maneuver (for title).
        model_states: (T+1, model_state_dim) predicted trajectory.
        gt_states: (T+1, 15) ground truth trajectory.
        output_path: Where to save the PNG.
        fps: Timesteps per second (for x-axis in seconds).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    T = len(model_states)
    # Plot the kinematic dims that both model and GT share.
    # Model may have 8 dims (blind) while GT has 15 — only compare the overlap.
    n_dims = min(model_states.shape[1], 8, gt_states.shape[1])
    time_axis = np.arange(T) / fps

    fig, axes = plt.subplots(n_dims, 1, figsize=(10, 2.2 * n_dims), sharex=True)
    if n_dims == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        if i >= n_dims:
            break
        # GT in blue, model in red — consistent color scheme across the report.
        ax.plot(time_axis, gt_states[:T, i], color="#2196F3", linewidth=1.2,
                label="Box2D GT", alpha=0.8)
        ax.plot(time_axis, model_states[:, i], color="#F44336", linewidth=1.2,
                label="Model", alpha=0.8)
        # Shaded error band: shows absolute error around the GT line.
        error = np.abs(model_states[:, i] - gt_states[:T, i])
        ax.fill_between(time_axis, gt_states[:T, i] - error, gt_states[:T, i] + error,
                        color="#F44336", alpha=0.1)
        dim_name = KINEMATIC_DIM_NAMES[i] if i < len(KINEMATIC_DIM_NAMES) else f"dim_{i}"
        ax.set_ylabel(dim_name, fontsize=10)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc="upper right", fontsize=9)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"Physics Test: {maneuver_name}", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=120, bbox_inches="tight")
    plt.close(fig)


def render_maneuver_video(
    maneuver_name: str,
    gt_frames: np.ndarray,
    model_frames: np.ndarray,
    output_path: str | Path,
    fps: int = 50,
    gap: int = 4,
):
    """Render side-by-side comparison video from RGB frame arrays.

    Composites GT frames (left, blue label) and model frames (right, red
    label) side-by-side with a gap between them. Same compositing pattern
    as platformer's render_comparison_frame — horizontal concatenation
    via PIL. Overlays timestep counter and labels.

    Both frame arrays come from Box2D rendering:
      - GT frames: captured during Box2D stepping (generate_gt_trajectory
        with save_frames=True)
      - Model frames: rendered by teleporting Box2D body to predicted states
        (render_states_to_frames in physics_test_gt.py)

    The final frame is held for 1.5 seconds so viewers can see the end state.

    Args:
        maneuver_name: Maneuver name for title overlay.
        gt_frames: (T, H, W, 3) uint8 — ground truth RGB frames.
        model_frames: (T, H, W, 3) uint8 — model prediction RGB frames.
        output_path: MP4 output path.
        fps: Frames per second.
        gap: Pixel gap between left and right panels.
    """
    from PIL import Image, ImageDraw, ImageFont
    import imageio.v3 as iio

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    T = min(len(gt_frames), len(model_frames))
    H, W = gt_frames.shape[1], gt_frames.shape[2]
    canvas_w = W * 2 + gap

    # Try to load a nice monospace font; fall back to PIL default if unavailable.
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 14,
        )
    except (OSError, IOError):
        font = ImageFont.load_default()

    frames = []
    for t in range(T):
        # Composite: GT left, model right, gap in between (dark background).
        canvas = Image.new("RGB", (canvas_w, H), (20, 20, 30))

        gt_img = Image.fromarray(gt_frames[t])
        model_img = Image.fromarray(model_frames[t])

        canvas.paste(gt_img, (0, 0))
        canvas.paste(model_img, (W + gap, 0))

        # Labels and timestep counter overlay.
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 10), f"t={t:03d}  {maneuver_name}", fill="white", font=font)
        draw.text((10, H - 22), "Box2D GT", fill="#2196F3", font=font)
        draw.text((W + gap + 10, H - 22), "Model", fill="#F44336", font=font)

        frames.append(np.array(canvas))

    # Hold final frame for 1.5s so viewers can examine the end state.
    for _ in range(int(fps * 1.5)):
        frames.append(frames[-1])

    iio.imwrite(str(output_path), frames, fps=fps,
                codec="libx264", macro_block_size=1)


def plot_per_dim_trajectory_error(
    maneuver_name: str,
    per_dim_error: np.ndarray,
    output_path: Path,
    dim_names: list[str] | None = None,
):
    """Heatmap + line plot of per-dim error over time within a maneuver.

    Two-panel figure showing how prediction error evolves across timesteps
    for each state dimension. The line plot (top) uses symlog scale to handle
    both near-zero and large errors gracefully. The heatmap (bottom) gives a
    dense overview of which dims diverge when.

    This is a diagnostic tool — it answers "which dimension breaks first?"
    and "does error grow linearly or exponentially?" at a glance.

    Args:
        maneuver_name: For title.
        per_dim_error: (T, n_dims) squared error array.
        output_path: Where to save the PNG.
        dim_names: Optional dim labels. Defaults to x, y, vx, vy, angle, ang_vel, ll, rl.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if dim_names is None:
        dim_names = ["x", "y", "vx", "vy", "angle", "ang_vel", "left_leg", "right_leg"]
    n_dims = min(per_dim_error.shape[1], len(dim_names))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[1, 2])

    # Top: line plot per dim (symlog handles zero values gracefully).
    for d in range(n_dims):
        ax1.plot(per_dim_error[:, d], label=dim_names[d], alpha=0.8)
    ax1.set_ylabel("Squared Error")
    ax1.set_title(f"{maneuver_name} — Per-Dim Error Over Time")
    ax1.legend(ncol=4, fontsize=8)
    ax1.set_yscale("symlog", linthresh=1e-8)

    # Bottom: heatmap — rows are dims, columns are timesteps.
    # "hot" colormap: black (zero) -> red -> yellow (high error).
    im = ax2.imshow(
        per_dim_error[:, :n_dims].T,
        aspect="auto", interpolation="nearest",
        cmap="hot",
    )
    ax2.set_yticks(range(n_dims))
    ax2.set_yticklabels(dim_names[:n_dims])
    ax2.set_xlabel("Timestep")
    ax2.set_ylabel("State Dimension")
    plt.colorbar(im, ax=ax2, label="Squared Error")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
