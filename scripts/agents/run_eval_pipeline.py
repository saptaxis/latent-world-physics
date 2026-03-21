#!/usr/bin/env python
"""Full evaluation pipeline: collect trajectories + metrics + behavioral analysis.

Compound script that runs the complete evaluation pipeline for one or
more trained agents. Chains:
  1. collect_trajectories -> .npz files
  2. compute_metrics -> metrics.csv
  3. behavioral_analysis -> plots + summary JSON

Each step can also be run independently via its own script:
  - collect_trajectories.py — just collect .npz files
  - compute_metrics.py — just compute metrics from existing .npz files
  - analyze_behavior.py — just behavioral analysis from existing .npz files
  - eval_agent.py — eval with reward stats, plots, and videos

Usage:
    # Single agent, full pipeline
    python lunar_lander/scripts/run_eval_pipeline.py \
        --checkpoint-dir /path/to/agent --episodes 100

    # Specific intermediate checkpoint
    python lunar_lander/scripts/run_eval_pipeline.py \
        --checkpoint-dir /path/to/agent \
        --model rl_model_2000000_steps.zip --episodes 100

    # Batch: all agents under a parent directory
    python lunar_lander/scripts/run_eval_pipeline.py \
        --agents-dir /path/to/rl_agents/ --episodes 100

    # Skip collection (reuse existing trajectories)
    python lunar_lander/scripts/run_eval_pipeline.py \
        --checkpoint-dir /path/to/agent --skip-collect

    # Skip analysis (just collect + metrics)
    python lunar_lander/scripts/run_eval_pipeline.py \
        --checkpoint-dir /path/to/agent --episodes 100 --skip-analysis

    # Custom output location
    python lunar_lander/scripts/run_eval_pipeline.py \
        --checkpoint-dir /path/to/agent --episodes 100 \
        --output-dir /tmp/eval-output
"""

import os
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
import argparse
from pathlib import Path


from lwp.agents.eval_utils import (
    resolve_model_path, resolve_vec_normalize_path,
    load_training_config, make_env_factory, build_eval_batches,
)
from scripts.collection.collect_trajectories import (
    _collect_episodes, _find_agent_dirs, _check_existing_npz,
)
from scripts.analysis.analyze_behavior import _plot_physics_distribution
from lwp.analysis.trajectory_metrics import compute_collection_metrics
from lwp.analysis.behavioral_metrics import compute_collection_histograms
from lwp.analysis.behavioral_comparison import (
    aggregate_model_distribution,
    compute_adaptation_score,
    compute_binned_performance,
)


def _format_metrics_summary(df) -> str:
    """Format a human-readable summary of metrics.csv for console + file output."""
    lines = []
    n = len(df)

    # Outcomes
    lines.append(f"\n  === Summary ({n} episodes) ===")
    outcomes = df["outcome"].value_counts()
    for outcome, count in outcomes.items():
        lines.append(f"    {outcome}: {count} ({100*count/n:.0f}%)")

    # Reward
    lines.append(f"  Reward: {df['total_reward'].mean():.1f} ± {df['total_reward'].std():.1f}"
                 f"  (median {df['total_reward'].median():.1f},"
                 f" range [{df['total_reward'].min():.1f}, {df['total_reward'].max():.1f}])")

    # Episode length
    lines.append(f"  Episode steps: {df['episode_steps'].mean():.0f} ± {df['episode_steps'].std():.0f}")

    # Thrust behavior
    lines.append(f"  --- Thrust ---")
    lines.append(f"    duty_cycle:       {df['thrust_duty_cycle'].mean():.3f} ± {df['thrust_duty_cycle'].std():.3f}")
    lines.append(f"    mean_main:        {df['mean_main_thrust'].mean():.3f} ± {df['mean_main_thrust'].std():.3f}")
    lines.append(f"    total_fuel:       {df['total_fuel'].mean():.1f} ± {df['total_fuel'].std():.1f}")
    lines.append(f"    frac_full:        {df['main_thrust_frac_full'].mean():.3f}")
    lines.append(f"    frac_zero:        {df['main_thrust_frac_zero'].mean():.3f}")
    lines.append(f"    autocorr_lag1:    {df['thrust_autocorr_lag1'].mean():.3f} ± {df['thrust_autocorr_lag1'].std():.3f}")

    # Side thrust
    lines.append(f"  --- Side Thrust ---")
    lines.append(f"    mean_abs:         {df['mean_abs_side_thrust'].mean():.3f} ± {df['mean_abs_side_thrust'].std():.3f}")
    lines.append(f"    reversals:        {df['side_thrust_reversals'].mean():.1f} ± {df['side_thrust_reversals'].std():.1f}")
    lines.append(f"    autocorr_lag1:    {df['side_thrust_autocorr_lag1'].mean():.3f} ± {df['side_thrust_autocorr_lag1'].std():.3f}")

    # Flight phases
    lines.append(f"  --- Flight Phases ---")
    lines.append(f"    descending:       {df['frac_descending'].mean():.3f}")
    lines.append(f"    hovering:         {df['frac_hovering'].mean():.3f}")
    lines.append(f"    approaching:      {df['frac_approaching'].mean():.3f}")
    lines.append(f"    correcting:       {df['frac_correcting'].mean():.3f}")

    # Orientation
    lines.append(f"  --- Orientation ---")
    lines.append(f"    mean_abs_ang_vel: {df['mean_abs_angular_vel'].mean():.3f} ± {df['mean_abs_angular_vel'].std():.3f}")
    if "frac_upright" in df.columns:
        lines.append(f"    frac_upright:     {df['frac_upright'].mean():.3f} ± {df['frac_upright'].std():.3f}")
        lines.append(f"    max_angle_exc:    {df['max_angle_excursion'].mean():.3f} ± {df['max_angle_excursion'].std():.3f}")
    if "angle_recovery_time" in df.columns:
        valid = df["angle_recovery_time"].dropna()
        if len(valid) > 0:
            lines.append(f"    recovery_time:    {valid.mean():.1f} ± {valid.std():.1f} steps")
    if "thrust_angle_correlation" in df.columns:
        lines.append(f"    thrust_angle_corr:{df['thrust_angle_correlation'].mean():.3f} ± {df['thrust_angle_correlation'].std():.3f}")
    if "correction_lag" in df.columns:
        lines.append(f"    correction_lag:   {df['correction_lag'].mean():.1f} ± {df['correction_lag'].std():.1f} steps")

    # Landing quality (only for landed episodes)
    landed = df[df["outcome"] == "landed"]
    if len(landed) > 0:
        lines.append(f"  --- Landing Quality ({len(landed)} episodes) ---")
        lines.append(f"    x_error:          {landed['landing_x_error'].mean():.4f} ± {landed['landing_x_error'].std():.4f}")
        lines.append(f"    angle:            {landed['angle_at_landing'].mean():.4f} ± {landed['angle_at_landing'].std():.4f}")
        lines.append(f"    vy:               {landed['landing_vy'].mean():.3f} ± {landed['landing_vy'].std():.3f}")

    # Trajectory shape
    lines.append(f"  --- Trajectory ---")
    lines.append(f"    max_altitude:     {df['max_altitude'].mean():.3f} ± {df['max_altitude'].std():.3f}")
    lines.append(f"    max_lat_drift:    {df['max_lateral_drift'].mean():.3f} ± {df['max_lateral_drift'].std():.3f}")
    if "time_to_descend" in df.columns:
        lines.append(f"    time_to_descend:  {df['time_to_descend'].mean():.0f} ± {df['time_to_descend'].std():.0f} steps")
    if "descent_steadiness" in df.columns:
        valid = df["descent_steadiness"].dropna()
        if len(valid) > 0:
            lines.append(f"    descent_steady:   {valid.mean():.4f} ± {valid.std():.4f}")
    if "frac_approaching_pad" in df.columns:
        lines.append(f"    frac_approach_pad:{df['frac_approaching_pad'].mean():.3f} ± {df['frac_approaching_pad'].std():.3f}")

    return "\n".join(lines)


def _run_behavioral_analysis(traj_dir, output_base, workers, n_bins, n_quartiles, twr_bins):
    """Stage 3: compute action histograms, adaptation score, plots, summary JSON.

    Reads .npz trajectories from traj_dir, produces analysis outputs in
    output_base/behavioral_analysis/. Reuses metrics.csv if it exists.
    """
    import json
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    traj_path = Path(traj_dir)
    analysis_dir = Path(output_base) / "behavioral_analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(traj_path.glob("*.npz"))
    print(f"\n  Behavioral analysis: {len(npz_files)} episodes in {traj_dir}")

    # Step 1: Action histograms
    histograms = compute_collection_histograms(
        str(traj_dir), workers=workers, n_bins=n_bins,
    )
    print(f"    Histograms computed for {len(histograms)} episodes")

    # Step 2: Load or compute metrics.csv
    metrics_path = traj_path / "metrics.csv"
    if metrics_path.exists():
        episodes_df = pd.read_csv(str(metrics_path))
    else:
        episodes_df = compute_collection_metrics(str(traj_dir), workers=workers)
        episodes_df.to_csv(str(metrics_path), index=False)
        print(f"    Saved metrics to {metrics_path}")

    n_landed = (episodes_df["outcome"] == "landed").sum()
    landed_pct = (episodes_df["outcome"] == "landed").mean() * 100
    print(f"    {len(episodes_df)} episodes, {n_landed} landed ({landed_pct:.1f}%)")

    # Step 3: Aggregate model-level distribution
    model_dist = aggregate_model_distribution(list(histograms.values()))

    # Step 4: Adaptation score
    adaptation_result = compute_adaptation_score(
        histograms, episodes_df, physics_col="twr", n_quartiles=n_quartiles,
    )
    print(f"    Adaptation score: {adaptation_result['adaptation_score']:.4f}")

    # Step 5: Binned performance
    binned_perf = compute_binned_performance(
        episodes_df, bin_col="twr", n_bins=twr_bins,
    )

    # Step 6: Plots
    # Plot 1: Action distributions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    edges = model_dist["main_thrust_edges"]
    centers = (edges[:-1] + edges[1:]) / 2
    ax1.bar(centers, model_dist["main_thrust_probs"],
            width=edges[1] - edges[0], alpha=0.8, color="steelblue",
            edgecolor="white", linewidth=0.3)
    ax1.set_xlabel("Main Thrust")
    ax1.set_ylabel("Probability")
    ax1.set_title("P(main_thrust)")
    ax1.set_xlim(0, 1)

    edges = model_dist["side_thrust_edges"]
    centers = (edges[:-1] + edges[1:]) / 2
    ax2.bar(centers, model_dist["side_thrust_probs"],
            width=edges[1] - edges[0], alpha=0.8, color="coral",
            edgecolor="white", linewidth=0.3)
    ax2.set_xlabel("Side Thrust")
    ax2.set_ylabel("Probability")
    ax2.set_title("P(side_thrust)")
    ax2.set_xlim(-1, 1)

    fig.suptitle("Model-Level Action Distributions (all episodes)", fontsize=13)
    fig.tight_layout()
    fig.savefig(str(analysis_dir / "action_distributions.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Plot 2: Adaptation by TWR quartile
    quartile_dists = adaptation_result["quartile_distributions"]
    boundaries = adaptation_result["quartile_boundaries"]
    score = adaptation_result["adaptation_score"]
    counts = adaptation_result["quartile_episode_counts"]

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(quartile_dists)))
    for i, (q_dist, (lo, hi), count, color) in enumerate(
        zip(quartile_dists, boundaries, counts, colors)
    ):
        q_edges = q_dist.get(
            "main_thrust_edges",
            np.linspace(0, 1, len(q_dist["main_thrust_probs"]) + 1),
        )
        q_centers = (q_edges[:-1] + q_edges[1:]) / 2
        label = f"Q{i+1}: TWR [{lo:.1f}, {hi:.1f}] (n={count})"
        ax.step(q_centers, q_dist["main_thrust_probs"],
                where="mid", alpha=0.8, color=color, label=label, linewidth=1.5)
    ax.set_xlabel("Main Thrust")
    ax.set_ylabel("Probability")
    ax.set_title(f"P(main_thrust) by TWR Quartile — Adaptation Score: {score:.4f}")
    ax.set_xlim(0, 1)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(str(analysis_dir / "adaptation_by_twr.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Plot 3: Landed% vs TWR
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    x = range(len(binned_perf))
    bars = ax.bar(x, binned_perf["landed_pct"], color="seagreen", alpha=0.8,
                  edgecolor="white", linewidth=0.5)
    for bar, row in zip(bars, binned_perf.itertuples()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"n={row.n_episodes}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(list(x))
    ax.set_xticklabels(binned_perf["bin_label"], rotation=30, ha="right", fontsize=8)
    ax.set_xlabel("TWR Bin")
    ax.set_ylabel("Landed %")
    ax.set_title("Landing Success Rate by TWR")
    ax.set_ylim(0, 110)
    fig.tight_layout()
    fig.savefig(str(analysis_dir / "landed_vs_twr.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Plot 4: Physics parameter distributions
    _plot_physics_distribution(episodes_df, analysis_dir / "physics_distribution.png")

    # Step 7: Summary JSON
    summary = {
        "n_episodes": len(episodes_df),
        "landed_pct": float(landed_pct),
        "adaptation_score": adaptation_result["adaptation_score"],
        "quartile_boundaries": adaptation_result["quartile_boundaries"],
        "quartile_episode_counts": adaptation_result["quartile_episode_counts"],
        "binned_performance": binned_perf.to_dict(orient="records"),
    }
    new_scalars = [
        "std_main_thrust", "std_side_thrust",
        "main_thrust_frac_full", "main_thrust_frac_zero",
        "frac_descending", "frac_hovering",
        "frac_approaching", "frac_correcting",
        "thrust_autocorr_lag1", "side_thrust_autocorr_lag1",
    ]
    for col in new_scalars:
        if col in episodes_df.columns:
            summary[f"mean_{col}"] = float(episodes_df[col].mean())

    summary_path = analysis_dir / "behavioral_summary.json"
    with open(str(summary_path), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"    Saved: {analysis_dir}/ (4 plots + summary JSON)")


def _find_npz_dirs(output_base):
    """Find all directories with .npz files under output_base.

    Returns list of directory path strings. Could be output_base itself
    and/or profile subdirectories underneath it.
    """
    base = Path(output_base)
    if not base.exists():
        return []

    dirs = []
    if list(base.glob("*.npz")):
        dirs.append(str(base))
    for subdir in sorted(base.iterdir()):
        if subdir.is_dir() and list(subdir.glob("*.npz")):
            dirs.append(str(subdir))
    return dirs


def _run_pipeline_for_agent(
    agent_dir,
    output_base,
    episodes,
    seed,
    model_name,
    profiles_str,
    deterministic,
    workers,
    skip_collect,
    skip_metrics,
    skip_analysis,
    force,
    n_bins,
    n_quartiles,
    twr_bins,
    corruption=None,
    corruption_sigma=0.1,
    corruption_means_dir=None,
    save_frames=False,
):
    """Run the full pipeline (collect + metrics + analysis) for a single agent.

    Orchestrates three stages:
    1. Collect .npz trajectory files (unless --skip-collect)
    2. Compute per-episode metrics CSV (unless --skip-metrics)
    3. Behavioral analysis: histograms, adaptation score, plots (unless --skip-analysis)

    Returns a summary dict for the master comparison table.
    """
    agent_name = os.path.basename(agent_dir)

    # --- Load agent config (only needed for collection) ---
    variant = None
    algo = None

    if not skip_collect:
        try:
            model_path = resolve_model_path(agent_dir, model_name)
        except FileNotFoundError as e:
            print(f"  SKIP (no model): {e}")
            return None

        vec_norm_path = resolve_vec_normalize_path(agent_dir, model_path)

        try:
            train_config = load_training_config(agent_dir)
        except FileNotFoundError as e:
            print(f"  SKIP (no config): {e}")
            return None

        variant = train_config["variant"]
        algo = train_config.get("algo", "ppo")
        history_k = train_config.get("history_k", 8)
        n_rays = train_config.get("n_rays", 7)
        is_visual = variant.startswith("visual")
        n_stack = train_config.get("n_stack", 4 if is_visual else 0)
        frame_size = train_config.get("frame_size", 84)

        print(f"  Config: {variant} / {algo.upper()}")
    else:
        # Try to load config for summary table, but don't fail if missing.
        try:
            train_config = load_training_config(agent_dir)
            variant = train_config["variant"]
            algo = train_config.get("algo", "ppo")
            print(f"  Config: {variant} / {algo.upper()}")
        except FileNotFoundError:
            print(f"  Config: unknown (no config.json, running analysis only)")

    # --- Stage 1: Collect trajectories ---
    if not skip_collect:
        from stable_baselines3 import PPO, SAC
        AlgoClass = PPO if algo == "ppo" else SAC
        model = AlgoClass.load(model_path, device="auto")

        # Auto-detect training profile from config.json unless --profiles overrides.
        effective_profiles = profiles_str
        if not effective_profiles:
            train_profile = train_config.get("profile")
            if train_profile:
                effective_profiles = train_profile
                print(f"  Auto-detected training profile: {train_profile}")

        render_mode = "rgb_array" if save_frames else None
        env_fn = make_env_factory(variant=variant, n_rays=n_rays, history_k=history_k,
                                  render_mode=render_mode, frame_size=frame_size)
        eval_batches = build_eval_batches(
            variant=variant, n_rays=n_rays, history_k=history_k,
            profiles_str=effective_profiles, default_env_fn=env_fn,
            render_mode=render_mode, frame_size=frame_size,
        )

        # Apply label corruption if requested (same logic as collect_trajectories.py).
        if corruption:
            training_means = None
            if corruption == "mean":
                from lwp.agents.label_corruption import compute_training_means
                means_dir = corruption_means_dir
                if means_dir is None:
                    means_dir = os.path.join(agent_dir, "trajectories")
                if not os.path.isdir(means_dir):
                    print(f"  SKIP: --corruption mean requires trajectory dir at {means_dir}")
                    return None
                print(f"  Computing training means from {means_dir}")
                training_means = compute_training_means(means_dir)

            from lwp.agents.eval_utils import wrap_env_fn_with_corruption
            eval_batches = [
                (prof_name, wrap_env_fn_with_corruption(
                    batch_env_fn,
                    corruption_type=corruption,
                    corruption_sigma=corruption_sigma,
                    training_means=training_means,
                ))
                for prof_name, batch_env_fn in eval_batches
            ]
            print(f"  Corruption: {corruption}" +
                  (f" (sigma={corruption_sigma})" if corruption == "noise" else ""))

        for prof_name, batch_env_fn in eval_batches:
            if len(eval_batches) > 1:
                collect_dir = os.path.join(output_base, prof_name)
            else:
                collect_dir = output_base

            # Check for existing trajectories before collecting.
            existing = _check_existing_npz(collect_dir)
            if existing and not force:
                print(f"\n  STOP: {existing} existing .npz files in {collect_dir}/")
                print(f"  Use --force to overwrite, or --skip-collect to reuse them.")
                return None
            if existing and force:
                import glob as globmod
                old_files = globmod.glob(os.path.join(collect_dir, "*.npz"))
                for f in old_files:
                    os.remove(f)
                print(f"\n  Removed {len(old_files)} old .npz files (--force)")

            print(f"\n  Collecting {episodes} episodes (profile: {prof_name})...")
            results = _collect_episodes(
                model=model,
                env_fn=batch_env_fn,
                output_dir=collect_dir,
                n_episodes=episodes,
                seed=seed,
                vec_normalize_path=vec_norm_path,
                deterministic=deterministic,
                profile=prof_name,
                save_frames=save_frames,
                n_stack=n_stack,
            )

            n_landed = sum(1 for r in results if r["outcome"] == "landed")
            print(f"  Collected {len(results)} episodes -> {collect_dir}/")
            print(f"  Landed: {n_landed}/{len(results)} ({100*n_landed/len(results):.0f}%)")
    else:
        print("  Skipping collection (--skip-collect)")

    # --- Stage 2: Compute metrics ---
    if not skip_metrics:
        # Find all directories with .npz files under the output base.
        # Could be output_base itself or profile subdirectories.
        metric_dirs = _find_npz_dirs(output_base)
        if not metric_dirs:
            print(f"  WARNING: No .npz files found in {output_base}")
            return None

        for mdir in metric_dirs:
            npz_count = len(list(Path(mdir).glob("*.npz")))
            print(f"\n  Computing metrics for {npz_count} episodes in {mdir}/")
            df = compute_collection_metrics(mdir, workers=workers)
            csv_path = os.path.join(mdir, "metrics.csv")
            df.to_csv(csv_path, index=False)
            print(f"  Saved metrics to {csv_path}")

            # Print and save summary stats for this directory
            summary_text = _format_metrics_summary(df)
            print(summary_text)
            summary_path = os.path.join(mdir, "metrics_summary.txt")
            with open(summary_path, "w") as f:
                f.write(summary_text + "\n")
            print(f"  Saved summary to {summary_path}")
    else:
        print("  Skipping metrics (--skip-metrics)")

    # --- Stage 3: Behavioral analysis ---
    if not skip_analysis:
        analysis_dirs = _find_npz_dirs(output_base)
        if not analysis_dirs:
            print(f"  WARNING: No .npz files for analysis in {output_base}")
        else:
            for adir in analysis_dirs:
                _run_behavioral_analysis(
                    traj_dir=adir,
                    output_base=output_base,
                    workers=workers,
                    n_bins=n_bins,
                    n_quartiles=n_quartiles,
                    twr_bins=twr_bins,
                )
    else:
        print("  Skipping behavioral analysis (--skip-analysis)")

    return {
        "agent": agent_name,
        "variant": variant or "unknown",
        "algo": algo or "unknown",
        "output_dir": output_base,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Full evaluation pipeline: collect trajectories + compute metrics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint-dir",
                       help="Single agent checkpoint directory")
    group.add_argument("--agents-dir",
                       help="Parent dir — run pipeline for all agents underneath")

    parser.add_argument("--model", default=None,
                        help="Specific model file (default: model.zip)")
    parser.add_argument("--episodes", type=int, default=50,
                        help="Episodes per agent (default: 50)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Base random seed (default: 0)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: {checkpoint-dir}/trajectories/)")
    parser.add_argument("--profiles", type=str, default=None,
                        help="Comma-separated profile names (e.g. 'easy,medium')")
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel workers for metrics (default: 8)")
    parser.add_argument("--deterministic", action="store_true", default=True,
                        help="Use deterministic policy (default: True)")
    parser.add_argument("--no-deterministic", dest="deterministic",
                        action="store_false",
                        help="Use stochastic policy")
    parser.add_argument("--skip-collect", action="store_true",
                        help="Skip collection, reuse existing .npz files")
    parser.add_argument("--skip-metrics", action="store_true",
                        help="Skip metrics computation")
    parser.add_argument("--skip-analysis", action="store_true",
                        help="Skip behavioral analysis (plots + adaptation score)")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing trajectories (deletes old .npz files first)")
    parser.add_argument("--corruption", type=str, default=None,
                        choices=["zero", "shuffle", "mean", "noise"],
                        help="Label corruption type for eval-time manipulation. "
                             "Only meaningful for labeled agents. Affects both "
                             "collection (wraps env) and output path (trajectories-{tag}/).")
    parser.add_argument("--corruption-sigma", type=float, default=0.1,
                        help="Noise std as fraction of param range (default: 0.1). "
                             "Only used with --corruption noise.")
    parser.add_argument("--corruption-means-dir", type=str, default=None,
                        help="Directory of .npz files to compute training means from. "
                             "Required for --corruption mean.")
    parser.add_argument("--save-frames", action="store_true",
                        help="Capture RGB frames at each timestep and include in .npz files. "
                             "Increases file size from ~20KB to ~1-2MB per episode.")
    parser.add_argument("--trajectory-subdir", type=str, default="trajectories",
                        help="Subdirectory name for trajectory output "
                             "(default: 'trajectories'). Use e.g. 'trajectories-v2' "
                             "for reproducibility verification runs.")
    parser.add_argument("--n-bins", type=int, default=50,
                        help="Histogram bins for action distributions (default: 50)")
    parser.add_argument("--n-quartiles", type=int, default=4,
                        help="Quartiles for adaptation score (default: 4)")
    parser.add_argument("--twr-bins", type=int, default=5,
                        help="Bins for landed%% vs TWR plot (default: 5)")

    args = parser.parse_args()

    if args.checkpoint_dir:
        agent_dirs = [args.checkpoint_dir]
    else:
        agent_dirs = _find_agent_dirs(args.agents_dir)
        if not agent_dirs:
            print(f"ERROR: No agent checkpoints found under {args.agents_dir}")
            sys.exit(1)
        print(f"Found {len(agent_dirs)} agents under {args.agents_dir}")

    # Run pipeline for each agent
    all_summaries = []
    for agent_dir in agent_dirs:
        agent_name = os.path.basename(agent_dir)
        print(f"\n{'='*60}")
        print(f"Agent: {agent_name}")
        print(f"{'='*60}")

        # In batch mode with --output-dir, use per-agent subdirs to avoid overwrites.
        # In single-agent mode, use --output-dir directly or default to {agent}/trajectories/.
        if args.output_dir and len(agent_dirs) > 1:
            output_base = os.path.join(args.output_dir, agent_name)
        else:
            output_base = args.output_dir or os.path.join(agent_dir, args.trajectory_subdir)

        # Corruption runs save to a separate subdir (e.g. trajectories-zero/).
        if args.corruption:
            corruption_tag = args.corruption
            if args.corruption == "noise":
                corruption_tag = f"noise-s{args.corruption_sigma}"
            if output_base.endswith(args.trajectory_subdir):
                output_base = output_base[:-len(args.trajectory_subdir)] + f"{args.trajectory_subdir}-{corruption_tag}"
            else:
                output_base = f"{output_base}-{corruption_tag}"

            # Skip non-labeled agents — corruption targets physics dims 8-14,
            # which are ray values (not physics) in blind/history agents.
            if not args.skip_collect:
                try:
                    tc = load_training_config(agent_dir)
                    if tc["variant"] != "labeled":
                        print(f"  SKIP: --corruption only applies to labeled agents "
                              f"(this is '{tc['variant']}')")
                        continue
                except FileNotFoundError:
                    pass  # Let _run_pipeline_for_agent handle missing config

        summary = _run_pipeline_for_agent(
            agent_dir=agent_dir,
            output_base=output_base,
            episodes=args.episodes,
            seed=args.seed,
            model_name=args.model,
            profiles_str=args.profiles,
            deterministic=args.deterministic,
            workers=args.workers,
            skip_collect=args.skip_collect,
            skip_metrics=args.skip_metrics,
            skip_analysis=args.skip_analysis,
            force=args.force,
            n_bins=args.n_bins,
            n_quartiles=args.n_quartiles,
            twr_bins=args.twr_bins,
            corruption=args.corruption,
            corruption_sigma=args.corruption_sigma,
            corruption_means_dir=args.corruption_means_dir,
            save_frames=args.save_frames,
        )

        if summary:
            all_summaries.append(summary)

    # Master comparison table for batch mode
    if len(all_summaries) > 1:
        print(f"\n{'='*70}")
        print("Pipeline complete for all agents:")
        print(f"{'='*70}")
        for s in all_summaries:
            print(f"  {s['agent']:<40s} {s['variant']}/{s['algo'].upper()}")

    print("\nDone.")


if __name__ == "__main__":
    main()
