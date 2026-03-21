#!/usr/bin/env python3
"""Collection summary and validation tool for Lunar Lander trajectories.

Reads a collection directory produced by collect_grid.py, prints stats
(episodes per cell, outcomes, disk size), and validates completeness
against the manifest. Spot-checks metadata in random episodes.

Usage:
    python lunar_lander/scripts/collection_summary.py lunar_lander/data/collections/validation-v1/
    python lunar_lander/scripts/collection_summary.py lunar_lander/data/collections/test/ --spot-check 10
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict


def _dir_size_mb(path):
    """Total size of directory in MB."""
    total = 0
    for f in Path(path).rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total / (1024 * 1024)


def _load_metadata(npz_path):
    """Load and parse metadata_json from an .npz file."""
    try:
        data = np.load(npz_path, allow_pickle=True)
        if "metadata_json" not in data:
            return None
        return json.loads(str(data["metadata_json"]))
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Collection summary and validation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "collection_dir", type=str,
        help="Path to collection directory (contains manifest.json)",
    )
    parser.add_argument(
        "--spot-check", type=int, default=5,
        help="Number of random episodes to spot-check metadata (default: 5)",
    )

    args = parser.parse_args()

    collection_dir = Path(args.collection_dir)
    manifest_path = collection_dir / "manifest.json"
    trajectories_dir = collection_dir / "trajectories"

    # Load manifest
    if not manifest_path.exists():
        print(f"ERROR: No manifest.json in {collection_dir}")
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    print(f"=== Collection Summary: {manifest.get('run_name', 'unknown')} ===")
    print(f"Created: {manifest.get('created', '?')}")
    print(f"Seed: {manifest.get('seed', '?')}")
    print(f"Save frames: {manifest.get('save_frames', '?')}")
    print()

    physics_types = manifest["physics_types"]
    policies = manifest["policies"]
    episodes_per_cell = manifest["episodes_per_cell"]

    # Scan trajectories
    cell_counts = {}  # (physics_type, policy) -> count
    all_npz_files = []

    for physics_type in physics_types:
        type_dir = trajectories_dir / physics_type
        for policy_name in policies:
            pattern = f"{policy_name}_*.npz"
            files = sorted(type_dir.glob(pattern)) if type_dir.exists() else []
            cell_counts[(physics_type, policy_name)] = len(files)
            all_npz_files.extend(files)

    # Print grid matrix
    print(f"--- Episode Counts (target: {episodes_per_cell}/cell) ---")
    print()

    # Header
    col_width = max(len(p) for p in policies) + 2
    header = f"{'physics_type':25s}"
    for p in policies:
        header += f" {p:>{col_width}}"
    header += f" {'total':>{col_width}}"
    print(header)
    print("-" * len(header))

    total_by_policy = defaultdict(int)
    total_episodes = 0
    incomplete_cells = []

    for physics_type in physics_types:
        row = f"{physics_type:25s}"
        row_total = 0
        for policy_name in policies:
            count = cell_counts[(physics_type, policy_name)]
            marker = "" if count >= episodes_per_cell else " *"
            row += f" {count:>{col_width - len(marker)}}{marker}"
            total_by_policy[policy_name] += count
            row_total += count
            total_episodes += count

            if count < episodes_per_cell:
                incomplete_cells.append((physics_type, policy_name, count))

        row += f" {row_total:>{col_width}}"
        print(row)

    # Totals row
    print("-" * len(header))
    totals_row = f"{'TOTAL':25s}"
    for p in policies:
        totals_row += f" {total_by_policy[p]:>{col_width}}"
    totals_row += f" {total_episodes:>{col_width}}"
    print(totals_row)
    print()

    # Completeness validation
    expected_total = len(physics_types) * len(policies) * episodes_per_cell
    complete_cells = len(physics_types) * len(policies) - len(incomplete_cells)
    total_cells = len(physics_types) * len(policies)

    print(f"--- Completeness ---")
    print(f"Cells: {complete_cells}/{total_cells} complete")
    print(f"Episodes: {total_episodes}/{expected_total} ({100 * total_episodes / expected_total:.1f}%)")

    if incomplete_cells:
        print(f"\nIncomplete cells (* in table above):")
        for pt, pol, count in incomplete_cells:
            print(f"  {pt} / {pol}: {count}/{episodes_per_cell}")
    print()

    # Disk usage
    disk_mb = _dir_size_mb(collection_dir)
    print(f"--- Disk Usage ---")
    print(f"Total: {disk_mb:.1f} MB")
    if total_episodes > 0:
        print(f"Per episode: {disk_mb / total_episodes:.2f} MB")
    print()

    # Summary.json stats (if exists)
    summary_path = collection_dir / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        outcomes = summary.get("outcomes", {})
        print(f"--- Outcomes ---")
        total_outcome = sum(outcomes.values())
        for outcome, count in outcomes.items():
            pct = 100 * count / total_outcome if total_outcome > 0 else 0
            print(f"  {outcome}: {count} ({pct:.1f}%)")
        if "episodes_per_second" in summary:
            print(f"\nThroughput: {summary['episodes_per_second']:.1f} ep/s")
        if "wall_time_seconds" in summary:
            wt = summary["wall_time_seconds"]
            print(f"Wall time: {wt:.0f}s ({wt/60:.1f}m)")
        print()

    # Spot-check metadata
    if all_npz_files and args.spot_check > 0:
        n_check = min(args.spot_check, len(all_npz_files))
        rng = np.random.default_rng(42)
        check_files = rng.choice(all_npz_files, size=n_check, replace=False)

        print(f"--- Metadata Spot-Check ({n_check} episodes) ---")
        issues = []

        for npz_path in check_files:
            rel_path = npz_path.relative_to(collection_dir)
            meta = _load_metadata(npz_path)

            if meta is None:
                issues.append(f"  {rel_path}: NO metadata_json (file may be corrupted)")
                continue

            checks = []
            # Check required top-level fields
            if "physics_type" not in meta:
                checks.append("missing physics_type")
            if "physics_config" not in meta:
                checks.append("missing physics_config")
            if "calibration" not in meta:
                checks.append("missing calibration")
            if "policy" not in meta:
                checks.append("missing policy")
            if "outcome" not in meta:
                checks.append("missing outcome")

            # Validate outcome value if present
            valid_outcomes = {"landed", "crashed", "out_of_bounds", "timeout"}
            if "outcome" in meta and meta["outcome"] not in valid_outcomes:
                checks.append(f"invalid outcome '{meta['outcome']}'")

            if checks:
                issues.append(f"  {rel_path}: {', '.join(checks)}")
            else:
                print(f"  {rel_path}: OK "
                      f"(policy={meta['policy']}, "
                      f"physics_type={meta.get('physics_type', '?')}, "
                      f"outcome={meta.get('outcome', '?')})")

        if issues:
            print(f"\nISSUES FOUND:")
            for issue in issues:
                print(issue)
        else:
            print(f"\nAll {n_check} spot-checks passed.")
        print()

    # Loadability check — verify shapes on one file
    if all_npz_files:
        loaded = False
        for npz_path in all_npz_files:
            try:
                sample = np.load(npz_path, allow_pickle=True)
                print(f"--- Sample Episode ({npz_path.name}) ---")
                for key in sorted(sample.keys()):
                    arr = sample[key]
                    if arr.shape:
                        print(f"  {key}: {arr.dtype} {arr.shape}")
                    else:
                        print(f"  {key}: scalar ({type(arr.item()).__name__})")
                loaded = True
                break
            except Exception as e:
                print(f"WARNING: {npz_path.name} is corrupted ({e}), trying next...")
        if not loaded:
            print("WARNING: Could not load any episode file (all corrupted?)")


if __name__ == "__main__":
    main()
