#!/usr/bin/env python3
"""Grid-aware parallel trajectory collection across physics types and policies.

Systematically collects episodes across the physics_type x policy grid using
multiprocessing. Each cell gets N episodes with randomized physics PARAMETERS
(continuous variation within a fixed physics equation type).

Currently the only physics type is "newtonian" (Phase 1 — all 7 continuous params
with standard Newtonian mechanics). Phase 2 will add equation-family variations
(different gravity laws, thrust models, drag models, etc.).

Each episode:
  1. Randomizes a LunarLanderPhysicsConfig from the valid parameter ranges
  2. Runs calibration (4 analytical maneuvers + 1 hover test) to validate the config
  3. Runs the chosen policy for up to max_steps steps
  4. Classifies outcome: landed, crashed, out_of_bounds, or timeout
  5. Saves trajectory as .npz via episode_io

Usage:
    # Quick test: 4 episodes/cell, 2 workers
    python lunar_lander/scripts/collect_grid.py --run-name test --episodes-per-cell 4 --workers 2

    # Validation tier: 16 episodes/cell = 32 total (1 physics type x 2 policies)
    python lunar_lander/scripts/collect_grid.py --run-name validation-v1 --episodes-per-cell 16

    # Medium tier: 256 episodes/cell = 512 total
    python lunar_lander/scripts/collect_grid.py --run-name medium-v1 --episodes-per-cell 256 --workers 10

    # With RGB frames saved (much larger files)
    python lunar_lander/scripts/collect_grid.py --run-name with-frames --episodes-per-cell 16 --save-frames

    # Resume interrupted run
    python lunar_lander/scripts/collect_grid.py --manifest lunar_lander/data/collections/medium-v1/manifest.json

    # Subset of policies
    python lunar_lander/scripts/collect_grid.py --run-name heuristic-only --episodes-per-cell 16 --policies heuristic
"""

import sys
import json
import time
import argparse
import numpy as np
from multiprocessing import Pool
from pathlib import Path

# Add the repo root to sys.path so that `lunar_lander.src.*` imports work
# when running this script directly (e.g., `python lunar_lander/scripts/collect_grid.py`).
# The repo root is two levels up from this file: scripts/ -> lunar_lander/ -> repo_root/

# --- Grid axis definitions ---
# Phase 1: only "newtonian" physics type (standard Newtonian mechanics with
# 7 continuous parameters varied per episode). Phase 2 will add equation-family
# variations (e.g., "inverse_cube_gravity", "drag_thrust", etc.).
ALL_PHYSICS_TYPES = ["newtonian"]

# Two policies: random exploration and a PD heuristic controller.
# - random: uniform random actions, gives broad trajectory coverage
# - heuristic: PD controller tuned for default physics, degrades under
#   varied physics (this degradation IS the experimental signal)
ALL_POLICIES = ["random", "heuristic"]


def _collect_cell(cell_spec):
    """Worker function: collect episodes for one (physics_type, policy) cell.

    Runs in a subprocess via multiprocessing.Pool. All imports are inside
    the function body because each subprocess needs its own module state
    (Box2D world, etc. are not fork-safe).

    For each episode:
      1. Randomize physics config (continuous params) while keeping the
         physics type fixed (equation family).
      2. Run calibration to validate the config and measure behavioral
         properties (these go into the metadata for later analysis).
      3. Run the chosen policy to generate a trajectory.
      4. Classify the outcome and save as .npz.

    Args:
        cell_spec: dict with keys:
            physics_type, policy, n_episodes, start_idx, output_dir,
            seed, max_steps, save_frames

    Returns:
        dict with collection stats for this cell:
            physics_type, policy, episodes_collected, total_steps,
            outcomes, calibration_failures
    """
    # --- Imports inside worker (multiprocessing subprocess) ---
    # Each subprocess needs fresh module state. Box2D worlds are not
    # safe to share across fork boundaries.
    from parametric_lunar_lander.env import ParameterizedLunarLander
    from parametric_lunar_lander.physics_config import LunarLanderPhysicsConfig
    from parametric_lunar_lander.calibration import calibrate
    from parametric_lunar_lander.heuristic import heuristic_policy
    from parametric_lunar_lander.episode_io import save_episode

    physics_type = cell_spec["physics_type"]
    policy_name = cell_spec["policy"]
    n_episodes = cell_spec["n_episodes"]
    start_idx = cell_spec["start_idx"]
    output_dir = Path(cell_spec["output_dir"])
    base_seed = cell_spec["seed"]
    max_steps = cell_spec["max_steps"]
    save_frames = cell_spec["save_frames"]

    # Output directory: {collection_dir}/trajectories/{physics_type}/
    # Multiple policies share the same physics_type dir, distinguished by filename prefix.
    cell_dir = output_dir / "trajectories" / physics_type
    cell_dir.mkdir(parents=True, exist_ok=True)

    # Per-cell RNG derived from base seed + cell identity for reproducibility.
    # Using hash() ensures different cells get different seeds even with the
    # same base_seed. The modulo keeps it within numpy's valid seed range.
    cell_seed_key = hash((physics_type, policy_name, base_seed)) % (2**31)
    rng = np.random.default_rng(cell_seed_key)

    # Stats tracking
    outcomes = {"landed": 0, "crashed": 0, "out_of_bounds": 0, "timeout": 0}
    total_steps = 0
    collected = 0
    calibration_failures = 0

    for ep_idx in range(n_episodes):
        episode_num = start_idx + ep_idx

        # --- 1. Randomize physics config ---
        # Physics TYPE stays fixed (defines the equation family),
        # but continuous PARAMETERS are randomized within valid ranges.
        # Each episode explores a different point in the 7D physics space.
        config = LunarLanderPhysicsConfig.randomize(rng=rng)

        # --- 2. Run calibration ---
        # Validates the config produces physically coherent behavior.
        # We pass heuristic_policy to also run the hover test (maneuver 5),
        # which measures controllability under this config.
        # Calibration result goes into metadata for later analysis (e.g.,
        # correlating TWR with agent performance, probing for physics structure).
        calibration_result = calibrate(config, heuristic_fn=heuristic_policy)

        if not calibration_result.all_passed:
            calibration_failures += 1
            # Still collect the episode — failed calibration is informative data,
            # not a reason to skip. The failure is recorded in metadata.

        # --- 3. Create env and run episode ---
        # render_mode="rgb_array" only when saving frames (allows env.render()
        # to return pixel arrays). None = headless, much faster.
        render_mode = "rgb_array" if save_frames else None
        env = ParameterizedLunarLander(
            render_mode=render_mode,
            physics_config=config,
        )

        ep_seed = int(rng.integers(0, 2**31))
        obs, _info = env.reset(seed=ep_seed)

        # Collect trajectory arrays — states has T+1 entries (initial + one per step),
        # actions/rewards/dones have T entries (one per step).
        states_list = [obs.copy()]
        actions_list = []
        rewards_list = []
        dones_list = []
        frames_list = []

        if save_frames:
            frame = env.render()
            frames_list.append(frame)

        # Track cumulative reward for metadata.
        total_reward = 0.0
        outcome = "timeout"  # default if we hit max_steps without termination

        for _ in range(max_steps):
            # --- Select action from policy ---
            if policy_name == "random":
                # Uniform random actions in the continuous action space [-1, 1]^2.
                # (main_thrust, side_thrust) — gives broad trajectory coverage
                # including unusual maneuvers that heuristic would never attempt.
                action = np.array(
                    [rng.uniform(-1, 1), rng.uniform(-1, 1)],
                    dtype=np.float32,
                )
            elif policy_name == "heuristic":
                # PD controller tuned for default physics. Under varied physics
                # it degrades naturally — this degradation IS the experimental
                # signal showing why physics-aware agents are needed.
                action = heuristic_policy(obs)
            else:
                raise ValueError(f"Unknown policy: {policy_name}")

            obs, reward, terminated, _, _ = env.step(action)

            states_list.append(obs.copy())
            actions_list.append(action.copy())
            rewards_list.append(reward)
            dones_list.append(terminated)
            total_reward += reward

            if save_frames:
                frame = env.render()
                frames_list.append(frame)

            if terminated:
                # --- 4. Classify outcome ---
                # Outcome classification from the reward signal and state:
                #   +100 reward = landed (lander came to rest on pad)
                #   -100 reward with |x| >= 1.0 = out of bounds (drifted off screen)
                #   -100 reward with |x| < 1.0 = crashed (body contact with ground)
                if reward >= 100:
                    outcome = "landed"
                else:
                    # obs[0] is normalized x position; |x| >= 1 means OOB
                    if abs(obs[0]) >= 1.0:
                        outcome = "out_of_bounds"
                    else:
                        outcome = "crashed"
                break

        n_steps = len(actions_list)

        # --- 5. Build metadata ---
        # Everything needed to reconstruct the episode context and analyze
        # relationships between physics, calibration, and behavior.
        metadata = {
            "physics_type": physics_type,
            "physics_config": config.to_dict(),
            "calibration": calibration_result.to_dict(),
            "policy": policy_name,
            "seed": ep_seed,
            "outcome": outcome,
            "n_steps": n_steps,
            "total_reward": float(total_reward),
        }

        # --- 6. Save episode as .npz ---
        # Filename format: {policy}_{NNNN}.npz — matches platformer convention.
        # episode_num ensures unique names even when resuming a partial run.
        ep_filename = f"{policy_name}_{episode_num:04d}.npz"
        ep_path = cell_dir / ep_filename

        # Convert lists to numpy arrays for save_episode().
        states = np.array(states_list, dtype=np.float32)
        actions = np.array(actions_list, dtype=np.float32)
        rewards = np.array(rewards_list, dtype=np.float32)
        dones = np.array(dones_list, dtype=bool)
        rgb_frames = (
            np.array(frames_list, dtype=np.uint8) if save_frames else None
        )

        save_episode(
            path=ep_path,
            states=states,
            actions=actions,
            rewards=rewards,
            dones=dones,
            metadata=metadata,
            rgb_frames=rgb_frames,
        )

        total_steps += n_steps
        collected += 1
        outcomes[outcome] += 1

        env.close()

    return {
        "physics_type": physics_type,
        "policy": policy_name,
        "episodes_collected": collected,
        "total_steps": total_steps,
        "outcomes": outcomes,
        "calibration_failures": calibration_failures,
    }


def _count_existing_episodes(collection_dir, physics_type, policy_name):
    """Count existing episodes for a (physics_type, policy) cell.

    Used for resume support — if a run was interrupted, we count how many
    episodes already exist for each cell and only collect the remaining ones.

    Counts files matching the pattern {policy}_{NNNN}.npz in the cell directory.
    """
    cell_dir = Path(collection_dir) / "trajectories" / physics_type
    if not cell_dir.exists():
        return 0
    pattern = f"{policy_name}_*.npz"
    return len(list(cell_dir.glob(pattern)))


def _build_manifest(args):
    """Build manifest dict from CLI args.

    The manifest captures all settings needed to reproduce or resume a run.
    It's saved as manifest.json in the collection directory and read back
    on resume (--manifest flag).
    """
    return {
        "run_name": args.run_name,
        "seed": args.seed,
        "physics_types": args.physics_types,
        "policies": args.policies,
        "episodes_per_cell": args.episodes_per_cell,
        "max_steps": args.max_steps,
        "save_frames": args.save_frames,
        "n_workers": args.workers,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Grid-aware parallel trajectory collection for Lunar Lander.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--run-name", type=str, default=None,
        help="Name for this collection run (creates {data-root}/collections/{run_name}/)",
    )
    parser.add_argument(
        "--manifest", type=str, default=None,
        help="Path to existing manifest.json to resume a run",
    )
    parser.add_argument(
        "--episodes-per-cell", type=int, default=16,
        help="Episodes per (physics_type, policy) cell (default: 16)",
    )
    parser.add_argument(
        "--physics-types", type=str, nargs="+", default=ALL_PHYSICS_TYPES,
        choices=ALL_PHYSICS_TYPES,
        help="Physics equation types to collect (default: ['newtonian'])",
    )
    parser.add_argument(
        "--policies", type=str, nargs="+", default=ALL_POLICIES,
        choices=ALL_POLICIES,
        help="Policies to collect (default: ['random', 'heuristic'])",
    )
    parser.add_argument(
        "--max-steps", type=int, default=400,
        help="Max steps per episode (default: 400)",
    )
    parser.add_argument(
        "--save-frames", action="store_true",
        help="Save RGB frames (default: off, produces much larger files)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Base random seed (default: 42)",
    )
    parser.add_argument(
        "--workers", type=int, default=10,
        help="Number of parallel workers (default: 10)",
    )
    parser.add_argument(
        "--data-root", type=str, default="lunar_lander/data",
        help="Root data directory (default: lunar_lander/data)",
    )

    args = parser.parse_args()

    # --- Load or build manifest ---
    # The manifest is the single source of truth for run configuration.
    # On fresh runs we build it from CLI args; on resume we read it from disk.
    if args.manifest:
        manifest_path = Path(args.manifest)
        with open(manifest_path) as f:
            manifest = json.load(f)
        collection_dir = manifest_path.parent
        print(f"Resuming run '{manifest['run_name']}' from {collection_dir}")
    else:
        if not args.run_name:
            parser.error("--run-name is required (or use --manifest to resume)")

        # Validate physics types against known set.
        for pt in args.physics_types:
            if pt not in ALL_PHYSICS_TYPES:
                parser.error(f"Unknown physics type: {pt}. Valid: {ALL_PHYSICS_TYPES}")

        collection_dir = Path(args.data_root) / "collections" / args.run_name
        collection_dir.mkdir(parents=True, exist_ok=True)
        manifest = _build_manifest(args)

        # Save manifest immediately so the run can be resumed if interrupted.
        manifest_path = collection_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"Created manifest: {manifest_path}")

    # --- Extract config from manifest ---
    physics_types = manifest["physics_types"]
    policies = manifest["policies"]
    episodes_per_cell = manifest["episodes_per_cell"]
    max_steps = manifest["max_steps"]
    save_frames = manifest["save_frames"]
    base_seed = manifest["seed"]
    n_workers = manifest["n_workers"]

    # --- Build cell specs, checking for existing episodes (resume support) ---
    # For each (physics_type, policy) cell, count existing episodes and only
    # schedule collection for the remaining ones. This makes interrupted runs
    # resumable without re-collecting already-saved episodes.
    cell_specs = []
    total_new = 0
    total_existing = 0

    for physics_type in physics_types:
        for policy_name in policies:
            existing = _count_existing_episodes(collection_dir, physics_type, policy_name)
            needed = episodes_per_cell - existing

            if needed <= 0:
                total_existing += existing
                continue

            total_existing += existing
            total_new += needed

            cell_specs.append({
                "physics_type": physics_type,
                "policy": policy_name,
                "n_episodes": needed,
                "start_idx": existing,
                "output_dir": str(collection_dir),
                "seed": base_seed,
                "max_steps": max_steps,
                "save_frames": save_frames,
            })

    total_cells = len(physics_types) * len(policies)
    cells_to_run = len(cell_specs)
    cells_done = total_cells - cells_to_run

    print(f"\n=== Grid Collection: {manifest.get('run_name', 'unknown')} ===")
    print(f"Grid: {len(physics_types)} physics types x {len(policies)} policies = {total_cells} cells")
    print(f"Episodes per cell: {episodes_per_cell}")
    print(f"Total target: {total_cells * episodes_per_cell} episodes")
    print(f"Already collected: {total_existing} episodes ({cells_done} cells complete)")
    print(f"Remaining: {total_new} episodes across {cells_to_run} cells")
    print(f"Workers: {n_workers}")
    print(f"Save frames: {save_frames}")
    print(f"Output: {collection_dir}")
    print()

    if not cell_specs:
        print("Nothing to collect -- all cells complete!")
        return

    # --- Run collection with multiprocessing ---
    t_start = time.time()

    if n_workers <= 1:
        # Single process mode — easier to debug, useful for development.
        results = []
        for i, spec in enumerate(cell_specs):
            print(f"  Cell {i+1}/{cells_to_run}: {spec['physics_type']} / {spec['policy']} "
                  f"({spec['n_episodes']} episodes)...")
            result = _collect_cell(spec)
            results.append(result)
            elapsed = time.time() - t_start
            done = sum(r["episodes_collected"] for r in results)
            rate = done / elapsed if elapsed > 0 else 0
            print(f"    -> {result['episodes_collected']} collected, "
                  f"outcomes: {result['outcomes']}, "
                  f"cal_failures: {result['calibration_failures']}, "
                  f"{rate:.1f} ep/s overall")
    else:
        # Multiprocessing with progress reporting via imap_unordered.
        # imap_unordered returns results as they complete (not in submission order),
        # which gives the most responsive progress updates.
        results = []
        done_count = 0

        print(f"Dispatching {cells_to_run} cells to {n_workers} workers...")
        with Pool(n_workers) as pool:
            for result in pool.imap_unordered(_collect_cell, cell_specs):
                results.append(result)
                done_count += result["episodes_collected"]
                elapsed = time.time() - t_start
                rate = done_count / elapsed if elapsed > 0 else 0
                remaining = total_new - done_count
                eta = remaining / rate if rate > 0 else 0

                print(
                    f"  [{done_count:5d}/{total_new}] "
                    f"{result['physics_type']:15s} / {result['policy']:10s} | "
                    f"{result['episodes_collected']:3d} eps, "
                    f"{result['outcomes']} | "
                    f"cal_fail={result['calibration_failures']} | "
                    f"{rate:.1f} ep/s, ETA {eta/60:.0f}m"
                )

    elapsed = time.time() - t_start
    total_collected = sum(r["episodes_collected"] for r in results)
    total_steps = sum(r["total_steps"] for r in results)
    total_cal_failures = sum(r["calibration_failures"] for r in results)

    # Aggregate outcomes across all cells.
    total_outcomes = {"landed": 0, "crashed": 0, "out_of_bounds": 0, "timeout": 0}
    for r in results:
        for k in total_outcomes:
            total_outcomes[k] += r["outcomes"][k]

    # --- Save summary ---
    # Summary captures the final state of the collection run. Unlike the
    # manifest (which captures config), the summary captures results.
    summary = {
        "run_name": manifest.get("run_name", "unknown"),
        "episodes_collected": total_collected + total_existing,
        "episodes_new": total_collected,
        "episodes_resumed": total_existing,
        "total_steps": total_steps,
        "outcomes": total_outcomes,
        "calibration_failures": total_cal_failures,
        "wall_time_seconds": elapsed,
        "episodes_per_second": total_collected / elapsed if elapsed > 0 else 0,
        "completed": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    summary_path = collection_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== Done ===")
    print(f"Collected: {total_collected} new episodes ({total_collected + total_existing} total)")
    print(f"Steps: {total_steps:,}")
    print(f"Outcomes: {total_outcomes['landed']} landed, "
          f"{total_outcomes['crashed']} crashed, "
          f"{total_outcomes['out_of_bounds']} out_of_bounds, "
          f"{total_outcomes['timeout']} timeout")
    print(f"Calibration failures: {total_cal_failures}")
    print(f"Time: {elapsed:.1f}s ({total_collected / elapsed:.1f} ep/s)")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
