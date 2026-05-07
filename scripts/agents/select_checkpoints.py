#!/usr/bin/env python
"""Select evaluation checkpoints for stable and unstable RL training runs.

Reads TensorBoard eval logs and selects two checkpoints per run:
  1. **final**: last checkpoint (model.zip) — retained performance
  2. **peak**: best rolling-5 eval average after competence burn-in — peak capability

The gap between peak and final landed% is the instability metric.

Burn-in gate: rolling-3 landed% must exceed threshold for 2 consecutive
windows before any checkpoint becomes eligible for peak selection.
Runs that never achieve competence are flagged (no peak selected).

See: e3-03-checkpoint-selection-Apr172026.md for methodology.

Usage:
    # Single run
    python scripts/agents/select_checkpoints.py \\
        --run-dir /path/to/condition/s42

    # All runs under a parent dir
    python scripts/agents/select_checkpoints.py \\
        --agents-dir /path/to/matched-encoder

    # Custom parameters
    python scripts/agents/select_checkpoints.py \\
        --run-dir /path/to/run --burn-in-k 3 --select-k 5 \\
        --burn-in-threshold 60 --burn-in-consecutive 2
"""

import argparse
import glob
import json
import os
import re
import sys

import numpy as np


def _load_eval_events(run_dir):
    """Load eval/landed_pct from all TensorBoard event files in run_dir.

    Merges events from multiple event files (e.g., from resumed runs),
    deduplicates by step (keeping the last-written value), and sorts by step.
    """
    from tensorboard.backend.event_processing.event_accumulator import (
        EventAccumulator,
    )

    tf_files = sorted(glob.glob(os.path.join(run_dir, "events.out.tfevents.*")))
    if not tf_files:
        return None, None

    all_events = {}
    for tf_file in tf_files:
        ea = EventAccumulator(tf_file)
        ea.Reload()
        if "eval/landed_pct" not in ea.Tags().get("scalars", []):
            continue
        for e in ea.Scalars("eval/landed_pct"):
            all_events[e.step] = e.value

    if not all_events:
        return None, None

    sorted_steps = sorted(all_events.keys())
    steps = np.array(sorted_steps)
    landed = np.array([all_events[s] for s in sorted_steps])
    return steps, landed


def _find_runs(agents_dir):
    """Find all seed directories under agents_dir (condition/s{seed}/)."""
    runs = []
    for condition in sorted(os.listdir(agents_dir)):
        cond_dir = os.path.join(agents_dir, condition)
        if not os.path.isdir(cond_dir):
            continue
        for seed_dir in sorted(os.listdir(cond_dir)):
            if not seed_dir.startswith("s"):
                continue
            full_path = os.path.join(cond_dir, seed_dir)
            if os.path.isdir(full_path):
                runs.append((condition, seed_dir, full_path))
    return runs


def _snap_to_checkpoint(target_step, checkpoint_dir):
    """Find the closest checkpoint file to target_step."""
    if not os.path.isdir(checkpoint_dir):
        return None, None
    pattern = re.compile(r"rl_model_(\d+)_steps\.zip")
    best_file = None
    best_step = None
    best_dist = float("inf")
    for f in os.listdir(checkpoint_dir):
        m = pattern.match(f)
        if m:
            ckpt_step = int(m.group(1))
            dist = abs(ckpt_step - target_step)
            if dist < best_dist:
                best_dist = dist
                best_file = f
                best_step = ckpt_step
    return best_file, best_step


def _rolling_mean(arr, k):
    """Compute rolling mean of length k. Returns array of length len(arr)-k+1."""
    if len(arr) < k:
        return np.array([])
    cumsum = np.cumsum(arr)
    cumsum = np.insert(cumsum, 0, 0)
    return (cumsum[k:] - cumsum[:-k]) / k


def select_checkpoints(
    run_dir,
    burn_in_k=3,
    burn_in_threshold=60.0,
    burn_in_consecutive=2,
    select_k=5,
    tail_n=10,
):
    """Select peak and final checkpoints for a run.

    Burn-in gate: compute rolling-burn_in_k over landed%. The burn-in
    is satisfied at the first eval index where burn_in_consecutive
    consecutive rolling windows all >= burn_in_threshold. Eligible
    region starts at the END of the last qualifying window (i.e., the
    first eval point where all preceding burn_in_consecutive windows
    have passed). This is conservative: we don't select from any eval
    that contributed to satisfying the gate.

    Peak selection: best rolling-select_k average within the eligible
    region after burn-in.

    If burn-in is never satisfied, no peak is selected and the run is
    flagged as never-competent.

    Returns dict with selection details, or None if TB data unavailable.
    """
    steps, landed = _load_eval_events(run_dir)
    if steps is None:
        return None

    n = len(steps)
    result = {
        "run_dir": run_dir,
        "total_evals": n,
        "final_step": int(steps[-1]),
        "final_landed": float(landed[-1]),
    }

    # Tail average (last tail_n evals)
    tail_start = max(0, n - tail_n)
    result["tail_landed"] = float(np.mean(landed[tail_start:]))
    result["tail_steps"] = (int(steps[tail_start]), int(steps[-1]))

    # Competence burn-in: rolling-burn_in_k must exceed threshold
    # for burn_in_consecutive consecutive windows.
    # Threshold <= 0 disables burn-in (eligible from the start).
    if burn_in_threshold <= 0:
        burn_in_eval_idx = 0
    else:
        burn_in_rolling = _rolling_mean(landed, burn_in_k)
        burn_in_eval_idx = None

        if len(burn_in_rolling) > 0:
            consecutive = 0
            for i in range(len(burn_in_rolling)):
                if burn_in_rolling[i] >= burn_in_threshold:
                    consecutive += 1
                    if consecutive >= burn_in_consecutive:
                        # Eligible region starts AFTER the last window
                        # that satisfied the gate. Rolling window i covers
                        # original indices [i, i+burn_in_k-1]. We start
                        # eligibility at the next eval after the end of
                        # window i.
                        gate_end_original = i + burn_in_k - 1
                        burn_in_eval_idx = gate_end_original + 1
                        break
                else:
                    consecutive = 0

    if burn_in_eval_idx is None or burn_in_eval_idx >= n:
        result["burn_in_step"] = None
        result["competent"] = False
        result["peak_step"] = None
        result["peak_landed"] = None
        result["peak_rolling_landed"] = None
        result["peak_checkpoint"] = None
        result["peak_checkpoint_step"] = None
        result["instability_gap"] = None
        return result

    result["competent"] = True
    result["burn_in_step"] = int(steps[burn_in_eval_idx])

    # Peak selection: best rolling-select_k average after burn-in
    eligible_landed = landed[burn_in_eval_idx:]

    select_rolling = _rolling_mean(eligible_landed, select_k)

    if len(select_rolling) == 0:
        # Eligible region too short for rolling window — use best single
        # eval point after burn-in as fallback.
        best_idx = burn_in_eval_idx + int(np.argmax(eligible_landed))
        result["peak_step"] = int(steps[best_idx])
        result["peak_landed"] = float(landed[best_idx])
        result["peak_rolling_landed"] = float(landed[best_idx])
        result["peak_checkpoint"] = "model.zip"
        result["peak_checkpoint_step"] = result["final_step"]
        result["instability_gap"] = float(landed[best_idx]) - result["tail_landed"]
        return result

    best_window = int(np.argmax(select_rolling))
    best_avg = float(select_rolling[best_window])
    # Center of the best window in original index space
    center_in_eligible = best_window + select_k // 2
    peak_eval_idx = burn_in_eval_idx + center_in_eligible
    peak_eval_idx = min(peak_eval_idx, n - 1)

    result["peak_step"] = int(steps[peak_eval_idx])
    result["peak_landed"] = float(landed[peak_eval_idx])
    result["peak_rolling_landed"] = float(best_avg)

    # Snap to nearest checkpoint file
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    ckpt_file, ckpt_step = _snap_to_checkpoint(
        result["peak_step"], checkpoint_dir
    )
    if ckpt_file:
        result["peak_checkpoint"] = os.path.join("checkpoints", ckpt_file)
        result["peak_checkpoint_step"] = ckpt_step
    else:
        result["peak_checkpoint"] = "model.zip"
        result["peak_checkpoint_step"] = result["final_step"]

    result["instability_gap"] = result["peak_rolling_landed"] - result["tail_landed"]

    return result


def print_result(result, condition=None, seed=None):
    """Print selection summary for one run."""
    label = ""
    if condition:
        label = f"{condition}/{seed}" if seed else condition
    else:
        label = os.path.basename(result["run_dir"])

    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"{'=' * 70}")

    if not result.get("competent"):
        burn_in_msg = "NEVER (run never reached competence)"
        print(f"  Evals: {result['total_evals']}  |  Burn-in: {burn_in_msg}")
        print(f"  Final: {result['final_landed']:.1f}%  at step {result['final_step']/1e6:.1f}M")
        print(f"  Tail:  {result['tail_landed']:.1f}%")
        print(f"  Peak:  N/A (no competent region)")
        return

    print(f"  Evals: {result['total_evals']}  |  Burn-in at: {result['burn_in_step']/1e6:.1f}M")
    print(f"  Peak:  rolling-avg={result['peak_rolling_landed']:.1f}%  "
          f"at step {result['peak_step']/1e6:.1f}M  "
          f"(checkpoint: {result.get('peak_checkpoint_step', result['peak_step'])/1e6:.1f}M)")
    print(f"  Final: {result['final_landed']:.1f}%  at step {result['final_step']/1e6:.1f}M")
    print(f"  Tail:  {result['tail_landed']:.1f}%  "
          f"(last {result['tail_steps'][0]/1e6:.1f}M-{result['tail_steps'][1]/1e6:.1f}M)")
    print(f"  Gap:   {result['instability_gap']:+.1f}pp")
    print(f"  Eval checkpoint: {result['peak_checkpoint']}")


def print_table(results):
    """Print summary table across all runs."""
    print(f"\n{'Condition':<35} {'Seed':<6} {'Peak':>6} {'Final':>6} "
          f"{'Tail':>6} {'Gap':>6}  {'Peak ckpt'}")
    print("-" * 100)
    for cond, seed, result in results:
        if result is None:
            print(f"{cond:<35} {seed:<6} {'(no TB data)':>30}")
            continue
        if not result.get("competent"):
            print(f"{cond:<35} {seed:<6} "
                  f"{'N/A':>6} "
                  f"{result['final_landed']:>5.1f}% "
                  f"{result['tail_landed']:>5.1f}% "
                  f"{'N/A':>6}  "
                  f"(never competent)")
            continue
        print(f"{cond:<35} {seed:<6} "
              f"{result['peak_rolling_landed']:>5.1f}% "
              f"{result['final_landed']:>5.1f}% "
              f"{result['tail_landed']:>5.1f}% "
              f"{result['instability_gap']:>+5.1f}  "
              f"{result['peak_checkpoint']}")


def save_result(result):
    """Save checkpoint selection JSON to the run directory."""
    run_dir = result["run_dir"]
    out = {k: v for k, v in result.items() if k != "run_dir"}
    # Convert tuples to lists for JSON
    if "tail_steps" in out and isinstance(out["tail_steps"], tuple):
        out["tail_steps"] = list(out["tail_steps"])
    out_path = os.path.join(run_dir, "checkpoint_selection.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Select evaluation checkpoints (peak + final) for RL runs."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-dir", help="Single seed directory")
    group.add_argument("--agents-dir", help="Parent dir with condition/seed subdirs")
    parser.add_argument("--burn-in-k", type=int, default=3,
                        help="Rolling window for burn-in gate (default: 3)")
    parser.add_argument("--select-k", type=int, default=5,
                        help="Rolling window for peak selection (default: 5)")
    parser.add_argument("--burn-in-threshold", type=float, default=60.0,
                        help="Landed%% threshold for burn-in gate (default: 60)")
    parser.add_argument("--burn-in-consecutive", type=int, default=2,
                        help="Consecutive burn-in rolling windows above threshold (default: 2)")
    parser.add_argument("--tail-n", type=int, default=10,
                        help="Number of final evals for tail average (default: 10)")
    parser.add_argument("--table-only", action="store_true",
                        help="Only show summary table")
    parser.add_argument("--save", action="store_true",
                        help="Save checkpoint_selection.json to each run dir")
    args = parser.parse_args()

    kwargs = dict(
        burn_in_k=args.burn_in_k,
        burn_in_threshold=args.burn_in_threshold,
        burn_in_consecutive=args.burn_in_consecutive,
        select_k=args.select_k,
        tail_n=args.tail_n,
    )

    if args.run_dir:
        result = select_checkpoints(args.run_dir, **kwargs)
        if result is None:
            print(f"No TB eval data in {args.run_dir}", file=sys.stderr)
            sys.exit(1)
        print_result(result)
        if args.save:
            path = save_result(result)
            print(f"\n  Saved: {path}")
    else:
        runs = _find_runs(args.agents_dir)
        if not runs:
            print(f"No runs found under {args.agents_dir}", file=sys.stderr)
            sys.exit(1)
        results = []
        saved = 0
        for condition, seed, path in runs:
            result = select_checkpoints(path, **kwargs)
            results.append((condition, seed, result))
            if not args.table_only and result:
                print_result(result, condition, seed)
            if args.save and result:
                save_result(result)
                saved += 1
        print_table(results)
        if args.save:
            print(f"\nSaved {saved} checkpoint_selection.json files.")


if __name__ == "__main__":
    main()
