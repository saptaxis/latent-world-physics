#!/usr/bin/env python3
"""Summarize a training profile log.

Reads profile.jsonl (written by ProfileLogger during training) and
produces a per-phase summary table. Can be run anytime — during
training (partial log) or after.

Usage:
    python scripts/profile_summary.py runs/my-run/profile.jsonl
    python scripts/profile_summary.py runs/my-run/profile.jsonl --step-min 100 --step-max 500
    python scripts/profile_summary.py runs/my-run/profile.jsonl --json runs/my-run/profile_summary.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_profile(path: str | Path) -> list[dict]:
    """Load JSONL profile log into a list of event dicts."""
    events = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def summarize(events: list[dict],
              step_min: int | None = None,
              step_max: int | None = None) -> dict:
    """Aggregate events by phase. Optionally filter by step range.

    Returns: {phase_name: {count, total_ms, mean_ms}}
    """
    totals: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)

    for e in events:
        step = e.get("step", 0)
        if step_min is not None and step < step_min:
            continue
        if step_max is not None and step > step_max:
            continue
        phase = e["phase"]
        totals[phase] += e["dur_ms"]
        counts[phase] += 1

    result = {}
    for phase in totals:
        total = totals[phase]
        count = counts[phase]
        result[phase] = {
            "count": count,
            "total_ms": round(total, 4),
            "mean_ms": round(total / count, 4) if count > 0 else 0,
        }
    return result


def format_table(summary: dict) -> str:
    """Format summary as a console table sorted by total time descending."""
    if not summary:
        return "(no profile data)"
    lines = []
    lines.append(f"{'Phase':<30} {'Total (s)':>10} {'Count':>8} {'Mean (ms)':>10} {'%':>6}")
    lines.append("-" * 68)
    grand_total_ms = sum(v["total_ms"] for v in summary.values())
    for name, stats in sorted(summary.items(), key=lambda x: -x[1]["total_ms"]):
        pct = stats["total_ms"] / grand_total_ms * 100 if grand_total_ms > 0 else 0
        total_sec = stats["total_ms"] / 1000
        lines.append(
            f"{name:<30} {total_sec:>9.2f}s {stats['count']:>8} "
            f"{stats['mean_ms']:>9.2f}ms {pct:>5.1f}%"
        )
    lines.append("-" * 68)
    lines.append(f"{'TOTAL':<30} {grand_total_ms / 1000:>9.2f}s")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Summarize training profile log")
    parser.add_argument("profile_log", help="Path to profile.jsonl")
    parser.add_argument("--step-min", type=int, default=None,
                        help="Only include steps >= this value")
    parser.add_argument("--step-max", type=int, default=None,
                        help="Only include steps <= this value")
    parser.add_argument("--json", type=str, default=None,
                        help="Write summary to JSON file")
    args = parser.parse_args()

    events = load_profile(args.profile_log)
    summary = summarize(events, step_min=args.step_min, step_max=args.step_max)
    print(format_table(summary))

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary written to {args.json}")


if __name__ == "__main__":
    main()
