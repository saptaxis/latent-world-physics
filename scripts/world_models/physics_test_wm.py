# lunar_lander/scripts/physics_test_wm.py
"""Run physics unit tests on a trained world model.

Controlled maneuvers testing specific physics relationships (gravity,
F=ma, damping, etc.). By default, starts from a clean reset state (lander
upright, centered, zero velocity) for pure physics isolation. Use
--from-episodes to branch from recorded episode states instead.

Runs 8 primary maneuvers by default. Use --maneuvers to select specific
maneuvers including secondary ones (hover, conservation, angular_decay).

Default duration is 300 steps (6 seconds at 50 FPS). Override with --duration.

Supports two model backends:
  - lwg models (--run-dir): ContextMLP, GRUWorldModel from this repo
  - wm-ladder models (--ladder-checkpoint): LinearModel, MLPModel, GRUModel, RSSMModel

Usage:
    # Reset mode (default) — clean initial state, 5 terrain seeds
    python lunar_lander/scripts/physics_test_wm.py \
        --ladder-checkpoint /path/to/best.pt --plot --video

    # More seeds for statistical confidence
    python lunar_lander/scripts/physics_test_wm.py \
        --ladder-checkpoint /path/to/best.pt --n-seeds 20 --plot

    # From recorded episodes (old behavior)
    python lunar_lander/scripts/physics_test_wm.py \
        --ladder-checkpoint /path/to/best.pt --from-episodes \
        --data-dir /path/to/episodes/ --episodes 20 --plot

Output: {run_dir}/physics_tests/ or {checkpoint_dir}/physics_tests/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


import torch

from lwp.wm.physics_tests import (
    MANEUVERS, ALL_MANEUVERS, run_physics_tests, run_maneuver_test,
    override_maneuver_duration,
)
from lwp.wm.physics_test_report import (
    save_json_report, print_console_summary,
    generate_markdown_report, plot_maneuver_comparison,
)
from lwp.wm.rollout_viz import (
    render_trajectory_video,  # triangle video renderer
)


def main():
    parser = argparse.ArgumentParser(
        description="Run physics unit tests on a trained world model.",
    )
    parser.add_argument(
        "--run-dir", required=False, default=None,
        help="Path to lwg trained model run directory. Required unless --ladder-checkpoint is used.",
    )
    parser.add_argument(
        "--ladder-checkpoint", type=str, default=None,
        help="Path to wm-ladder checkpoint (.pt). Uses wm-ladder's build_model + NormStats. "
             "Requires --data-dir for episode data.",
    )
    parser.add_argument(
        "--from-episodes", action="store_true",
        help="Branch from recorded episode states instead of env reset. "
             "Requires --data-dir (ladder) or uses split_index.json (lwg). "
             "Default: use clean reset state (upright, centered, zero velocity).",
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Directory containing episode .npz files. Required with "
             "--ladder-checkpoint --from-episodes.",
    )
    parser.add_argument(
        "--episodes", type=int, default=20,
        help="Number of episodes to test when using --from-episodes (default: 20). "
             "In reset mode (default), uses a single reset episode.",
    )
    parser.add_argument(
        "--n-seeds", type=int, default=5,
        help="Number of different terrain seeds to test in reset mode (default: 5). "
             "Each seed generates different terrain. Ignored with --from-episodes.",
    )
    parser.add_argument(
        "--maneuvers", nargs="+", default=None,
        help="Which maneuvers to run (default: primary only). "
             "Choices: " + ", ".join(ALL_MANEUVERS.keys()),
    )
    parser.add_argument(
        "--gt-mode", choices=["teleport", "replay", "both"], default="replay",
        help="Box2D ground truth mode. 'both' runs both and compares (default: replay).",
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Generate per-maneuver state overlay plots.",
    )
    parser.add_argument(
        "--video", action="store_true",
        help="Generate per-maneuver trajectory videos.",
    )
    parser.add_argument(
        "--duration", type=int, default=None,
        help="Override maneuver duration in steps (default: 300). "
             "At 50 FPS, 300 steps = 6 seconds.",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Where to save results (default: {run_dir}/physics_tests/).",
    )
    args = parser.parse_args()

    # -- Validate model source --
    if not args.run_dir and not args.ladder_checkpoint:
        parser.error("Either --run-dir or --ladder-checkpoint is required.")
    if args.from_episodes and args.ladder_checkpoint and not args.data_dir:
        parser.error("--from-episodes with --ladder-checkpoint requires --data-dir.")

    # -- Load model and build adapter (two code paths) --
    if args.ladder_checkpoint:
        # --- wm-ladder model path ---
        from lwp.utils.checkpoint import load_checkpoint as load_ladder_ckpt
        from lwp.models.factory import build_model as build_ladder_model
        from lwp.data.normalization import NormStats
        from lwp.wm.physics_tests import WMLadderAdapter

        print(f"\n  Loading ladder model from {args.ladder_checkpoint}...")
        ckpt = load_ladder_ckpt(args.ladder_checkpoint)
        model = build_ladder_model(ckpt["config"])
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        norm_stats = NormStats.from_dict(ckpt["norm_stats"])

        arch = ckpt["config"].arch
        recurrent = arch in ("gru", "rssm")
        state_dim = ckpt["config"].state_dim
        model_subsample = getattr(ckpt["config"], "subsample", 1)
        run_name = f"ladder-{arch}"
        print(f"  Model: {arch}, state_dim={state_dim}, recurrent={recurrent}")
        if model_subsample > 1:
            print(f"  Subsample: {model_subsample}x ({50 // model_subsample} FPS)")

        adapter = WMLadderAdapter(
            norm_stats=norm_stats,
            state_dim=state_dim,
            recurrent=recurrent,
            subsample=model_subsample,
        )

        output_dir = (
            Path(args.output_dir) if args.output_dir
            else Path(args.ladder_checkpoint).parent / "physics_tests"
        )

    else:
        parser.error("--ladder-checkpoint is required.")

    # -- Load or generate episodes --
    branch_offset: int | None = None  # None = use adapter's context_k

    if args.from_episodes:
        # Branch from recorded episode states (old behavior).
        if args.ladder_checkpoint:
            from lwp.wm.diagnostics import _load_episodes_from_dir
            episodes = _load_episodes_from_dir(Path(args.data_dir), args.episodes)
            print(f"  Episodes: {len(episodes)} from {args.data_dir}")
        else:
            from lwp.wm.diagnostics import _load_episodes
            split_index = run_dir / "split_index.json"
            episodes = _load_episodes(split_index, "val", args.episodes)
            print(f"  Episodes: {len(episodes)} validation")
        print(f"  Mode: from-episodes (branching from recorded states)")
    else:
        # Default: clean reset state. Generate one reset episode per seed.
        # Lander starts upright, centered, zero velocity — pure physics isolation.
        from lwp.wm.physics_test_gt import create_reset_episode

        # Padding must cover the model's context window.
        context_k = getattr(adapter, "context_k", 0) or 0
        n_padding = max(context_k + 5, 20)  # At least 20 for recurrent burn-in
        episodes = []
        for seed in range(args.n_seeds):
            episodes.append(create_reset_episode(
                n_padding=n_padding,
                seed=seed,
            ))
        # Branch at n_padding so adapter has full context of neutral state.
        # GT uses teleport mode: teleports lander to the reset state (already
        # the env's initial state — no joint artifacts). Replay mode would step
        # zero-action for n_padding steps, causing the lander to fall.
        branch_offset = n_padding
        print(f"  Episodes: {len(episodes)} reset episodes "
              f"(seeds 0-{args.n_seeds - 1}, padding={n_padding})")
        print(f"  Mode: reset (clean initial state — upright, centered, zero velocity)")

    # -- Override maneuver duration if specified --
    if args.duration is not None:
        override_maneuver_duration(args.duration)
        print(f"  Duration override: {args.duration} steps "
              f"({args.duration / 50:.1f}s)")

    # -- Run physics tests --
    if args.maneuvers:
        maneuver_names = args.maneuvers
        for m in maneuver_names:
            if m not in ALL_MANEUVERS:
                parser.error(f"Unknown maneuver: {m}. Choices: {list(ALL_MANEUVERS.keys())}")
    else:
        maneuver_names = list(MANEUVERS.keys())

    gt_modes = ["teleport", "replay"] if args.gt_mode == "both" else [args.gt_mode]

    # Force teleport GT for reset episodes — replay alters the initial state.
    if not args.from_episodes and "replay" in gt_modes:
        gt_modes = ["teleport"]
        print(f"  Note: using teleport GT for reset mode "
              f"(replay would alter initial state)")

    for gt_mode in gt_modes:
        print(f"\n  Running physics tests (GT mode: {gt_mode})...")
        results = run_physics_tests(
            model=model,
            episodes=episodes,
            adapter=adapter,
            maneuver_names=maneuver_names,
            gt_mode=gt_mode,
            branch_point_offset=branch_offset,
            subsample=model_subsample,
        )

        # -- Output --
        gt_output_dir = output_dir
        if len(gt_modes) > 1:
            gt_output_dir = output_dir / gt_mode
        gt_output_dir.mkdir(parents=True, exist_ok=True)

        # JSON report.
        json_path = save_json_report(
            results=results,
            output_path=gt_output_dir / "physics_tests.json",
            run_name=run_name,
            gt_mode=gt_mode,
            n_episodes=len(episodes),
        )
        print(f"  JSON report: {json_path}")

        # Console summary.
        print_console_summary(results, run_name=f"{run_name} ({gt_mode})")

        # Markdown report.
        md_path = generate_markdown_report(
            results=results,
            output_path=gt_output_dir / "physics_tests_report.md",
            run_name=run_name,
            gt_mode=gt_mode,
            n_episodes=len(episodes),
        )
        print(f"  Markdown report: {md_path}")

        # Plots and videos — render all tested episodes, one dir per maneuver.
        if args.plot or args.video:
            print(f"  Generating visualizations...")
            for m_name in maneuver_names:
                m_results = results[m_name]
                if m_results["n_tested"] == 0:
                    continue

                # Create per-maneuver directory.
                m_dir = gt_output_dir / m_name
                m_dir.mkdir(parents=True, exist_ok=True)

                maneuver = ALL_MANEUVERS[m_name]
                default_bp = branch_offset if branch_offset is not None else 0

                n_viz = 0
                n_ep = len(m_results["per_episode"])
                print(f"    {m_name}: rendering {n_ep} episodes...",
                      flush=True)

                for i, ep_result in enumerate(m_results["per_episode"]):
                    ep_idx = ep_result["episode_idx"]
                    episode = episodes[ep_idx]
                    bp = ep_result.get("branch_point", default_bp)

                    # Re-run to get full state arrays.
                    full_result = run_maneuver_test(
                        model=model,
                        episode=episode,
                        maneuver=maneuver,
                        adapter=adapter,
                        branch_point=bp,
                        gt_mode=gt_mode,
                        subsample=model_subsample,
                    )

                    # Skip if re-run was skipped (truncation).
                    if full_result.get("skipped"):
                        print(f"      ep{ep_idx:03d}: skipped", flush=True)
                        continue

                    tag = "pass" if full_result.get("passed") else "fail"
                    n_steps = full_result["model_states"].shape[0] - 1

                    if args.plot:
                        plot_maneuver_comparison(
                            maneuver_name=m_name,
                            model_states=full_result["model_states"],
                            gt_states=full_result["gt_states"],
                            output_path=m_dir / f"ep{ep_idx:03d}_{tag}.png",
                        )

                    if args.video:
                        rollout = {
                            "predicted_states": full_result["model_states"],
                            "actual_states": full_result["gt_states"],
                        }
                        render_trajectory_video(
                            rollout=rollout,
                            output_path=m_dir / f"ep{ep_idx:03d}_{tag}.mp4",
                            title=f"{m_name} ep{ep_idx} ({tag})",
                            actions=full_result["controlled_actions"],
                        )

                    n_viz += 1
                    print(f"      ep{ep_idx:03d}: {tag} ({n_steps} steps) "
                          f"[{i+1}/{n_ep}]", flush=True)

                print(f"    {m_name}: {n_viz} plots/videos → {m_dir}/",
                      flush=True)

            print(f"  Visualizations saved to {gt_output_dir}/")

    print(f"\n  Done.")


if __name__ == "__main__":
    main()
