#!/usr/bin/env python
"""Validate physics extraction pipeline against ground truth.

Runs the full physics understanding extraction (6 constants, oracle +
rollout) on a GroundTruthModel that returns actual Box2D deltas. If the
pipeline is correct, extracted constants should match known Box2D physics.

This is the sanity check for all Finding 05 results: if extraction gives
wrong numbers on GT, then the model evaluation numbers are meaningless.

Usage:
    python scripts/world_models/validate_physics_extraction.py \
        --data-path /media/hdd1/.../world_model_data/gym-default \
        --max-episodes 100
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

from lwp.wm.gt_model import GroundTruthModel, prepare_episodes_for_norm
from lwp.wm.physics_understanding import (
    generate_report, format_console_report,
    extract_gravity_oracle, extract_main_thrust_oracle,
    extract_side_thrust_oracle, extract_kinematics_oracle,
    extract_damping_oracle, extract_angle_thrust_oracle,
    VY,
)
from lwp.data.normalization import compute_norm_stats


def load_episodes(data_path: str, max_episodes: int = 100) -> list[dict]:
    npz_files = sorted(glob.glob(f"{data_path}/**/*.npz", recursive=True))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files in {data_path}")
    npz_files = npz_files[:max_episodes]
    episodes = []
    for f in npz_files:
        data = np.load(f)
        episodes.append({
            "states": data["states"].astype(np.float32),
            "actions": data["actions"].astype(np.float32),
        })
    print(f"Loaded {len(episodes)} episodes from {data_path}")
    return episodes


def main():
    p = argparse.ArgumentParser(description="Validate physics extraction on GT")
    p.add_argument("--data-path", type=str, required=True,
                   help="Directory with gym-default .npz episodes")
    p.add_argument("--max-episodes", type=int, default=100)
    args = p.parse_args()

    # Load data and compute normalization
    # compute_norm_stats needs 'deltas' key + torch Tensors
    episodes = load_episodes(args.data_path, args.max_episodes)
    prepared = prepare_episodes_for_norm(episodes)
    norm_stats = compute_norm_stats(prepared)

    # Create GT model
    gt_model = GroundTruthModel(episodes, norm_stats)
    print(f"GT model: {len(gt_model._states)} transitions indexed")

    # --- Run individual extractions with detailed output ---
    print("\n" + "=" * 70)
    print("PHYSICS EXTRACTION ON GROUND TRUTH")
    print("If these numbers are wrong, ALL model evaluations are suspect.")
    print("=" * 70)

    # Gravity
    grav = extract_gravity_oracle(gt_model, norm_stats, episodes)
    gravity_model = grav['model_mean']
    gravity_gt = grav['gt_mean']
    print(f"\n--- Gravity ---")
    print(f"  n_samples:    {grav['n_samples']}")
    print(f"  GT mean:      {gravity_gt:.6f} (delta_vy per step)")
    print(f"  Model mean:   {gravity_model:.6f}")
    print(f"  GT std:       {grav['gt_std']:.6f}")
    print(f"  Rel error:    {grav['relative_error']:.6f}")
    print(f"  Box2D gravity: -10.0 m/s^2")
    print(f"  Expected delta_vy/step depends on dt and coordinate scaling")

    # Main thrust
    thrust = extract_main_thrust_oracle(gt_model, norm_stats, episodes, gravity_model, gravity_gt)
    print(f"\n--- Main Thrust ---")
    print(f"  n_samples:    {thrust['n_samples']}")
    print(f"  GT mean:      {thrust['gt_mean']:.6f}")
    print(f"  Model mean:   {thrust['model_mean']:.6f}")
    print(f"  Rel error:    {thrust['relative_error']:.6f}")
    print(f"  Box2D main_engine_power: 13.0")

    # Side thrust
    side = extract_side_thrust_oracle(gt_model, norm_stats, episodes)
    print(f"\n--- Side Thrust ---")
    print(f"  n_samples:    {side['n_samples']}")
    print(f"  GT mean:      {side['gt_mean']:.6f}")
    print(f"  Model mean:   {side['model_mean']:.6f}")
    print(f"  Rel error:    {side['relative_error']:.6f}")
    print(f"  Box2D side_engine_power: 0.6")

    # Kinematics
    kin = extract_kinematics_oracle(gt_model, norm_stats, episodes)
    print(f"\n--- Kinematics (dx/dt consistency) ---")
    print(f"  n_samples:    {kin['n_samples']}")
    print(f"  GT mean:      {kin['gt_mean']:.6f}")
    print(f"  Model mean:   {kin['model_mean']:.6f}")
    print(f"  Rel error:    {kin['relative_error']:.6f}")
    print(f"  Expected: ~1.0 (velocity correctly predicts position change)")

    # Angular damping
    damp = extract_damping_oracle(gt_model, norm_stats, episodes)
    print(f"\n--- Angular Damping ---")
    print(f"  n_samples:    {damp['n_samples']}")
    print(f"  GT mean:      {damp['gt_mean']:.6f}")
    print(f"  Model mean:   {damp['model_mean']:.6f}")
    print(f"  Rel error:    {damp['relative_error']:.6f}")
    print(f"  Box2D angular_damping: 0.0 (no damping)")

    # Angle-thrust coupling
    angle = extract_angle_thrust_oracle(gt_model, norm_stats, episodes, gravity_model)
    print(f"\n--- Angle-Thrust Coupling ---")
    print(f"  n_samples:    {angle['n_samples']}")
    print(f"  GT mean:      {angle['gt_mean']:.6f}")
    print(f"  Model mean:   {angle['model_mean']:.6f}")
    print(f"  Rel error:    {angle['relative_error']:.6f}")

    # --- Summary table ---
    print(f"\n{'=' * 70}")
    print(f"SUMMARY: GT Model Extraction vs GT Values")
    print(f"{'=' * 70}")
    print(f"{'Constant':<20} {'GT Mean':>12} {'Model Mean':>12} {'Rel Err':>10} {'N':>6}")
    print(f"{'-' * 60}")
    for name, result in [
        ("Gravity", grav),
        ("Main Thrust", thrust),
        ("Side Thrust", side),
        ("Kinematics", kin),
        ("Ang. Damping", damp),
        ("Angle-Thrust", angle),
    ]:
        print(f"{name:<20} {result['gt_mean']:>12.6f} {result['model_mean']:>12.6f} "
              f"{result['relative_error']:>10.4f} {result['n_samples']:>6}")

    # --- Match quality ---
    gt_model.report_match_quality()

    # --- Full generate_report (oracle + rollout + consistency) ---
    # This validates the ENTIRE pipeline, not just individual extractions.
    # Includes rollout extraction which autoregressively applies the model
    # for multiple steps — for GT model this should still give perfect results.
    print(f"\n{'=' * 70}")
    print("FULL PIPELINE: generate_report (oracle + rollout + consistency)")
    print("=" * 70)
    full_report = generate_report(gt_model, norm_stats, episodes, recurrent=False)
    print(format_console_report(full_report, run_name="GroundTruthModel"))

    # --- Verdict ---
    print(f"\n{'=' * 70}")
    all_errors = [r['relative_error'] for _, r in [
        ("g", grav), ("t", thrust), ("s", side), ("k", kin), ("d", damp), ("a", angle),
    ] if r['n_samples'] > 0 and not np.isnan(r['relative_error'])]

    if all_errors and max(all_errors) < 0.01:
        print("PASS: All constants extracted within 1% of GT.")
        print("The extraction pipeline is sound. Model results are trustworthy.")
    elif all_errors and max(all_errors) < 0.05:
        print("WARN: Some constants have 1-5% error on GT data.")
        print("Extraction has inherent noise. Interpret model errors relative to this baseline.")
    else:
        print("FAIL: Significant extraction errors on GT data.")
        print("The extraction methodology has problems. Model results may be misleading.")
        if all_errors:
            print(f"Max relative error: {max(all_errors):.4f}")


if __name__ == "__main__":
    main()
