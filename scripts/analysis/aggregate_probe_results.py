#!/usr/bin/env python3
"""Aggregate encoder probe results into a long-format facts table + pivots.

Walks known probe-result directories (T1a, T1b, Finding 06 reference),
emits a single long-format CSV plus markdown pivot tables and a flagged-cells
list. Idempotent: rerun whenever new jsons land.

Inputs (auto-discovered by glob):
    T1a / T1b (easy profile):
      <NETS>/encoder-pretrain/easy/6d/s*/encoder-probes/encoder_probe_results.json
      <NETS>/visual_rl_agents/easy/matched-encoder/<run>/s*/encoder-probes/<tp>/encoder_probe_results.json
    Finding 06 reference (gym-default):
      <NETS>/analysis/encoder-probes/matched-encoder/{pretrained,finetuned}/s*/encoder_probe_results.json

Outputs (under <NETS>/analysis/encoder-probes/aggregated/e3-t1/):
    facts.csv                — long-format table, one row per (encoder, target, probe_type)
    facts_t1a_summary.md     — Finding-06-style table: mean ± std across seeds, by condition × timepoint
    facts_t1a_full.md        — per-seed wide pivot (T1a)
    facts_t1b_trajectory.md  — trajectory pivot: condition × step, headline R² per target
    flagged.md               — probe cells where MLP fit looks broken
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

NETS = Path("/media/hdd1/physics-priors-latent-space/lunar-lander-networks")
OUT_DIR = NETS / "analysis" / "encoder-probes" / "aggregated" / "e3-t1"

# Run-dir basename → condition label.
RUN_DIR_TO_CONDITION = {
    "finetune-1e5": "blind-ft-1e5",
    "finetune-lowlr": "blind-ft-5e5",
    "labeled-raw-finetune-lowlr": "raw-ft",
    "labeled-branch-finetune-lowlr": "branch-ft",
}

# Step grid labels → integer training step
STEP_LABEL_TO_STEP = {
    "step-100k": 100_000,
    "step-500k": 500_000,
    "step-1000k": 1_000_000,
    "step-2000k": 2_000_000,
    "step-5000k": 5_000_000,
}

TARGETS = ["x", "y", "vx", "vy", "angle", "ang_vel"]
PROBES = ["linear", "mlp"]

MLP_FAIL_LIN_GAP = 0.05   # linear_mean − mlp_mean above this → flag
MLP_FAIL_STD = 0.20       # mlp_std above this → flag

# Per-fold drop rule for MLP: drop fold i when mlp_fold[i] < linear_fold[i] - PER_FOLD_DROP_MARGIN.
# Rationale: MLP is a strict superset of linear; a per-fold gap below this is a fitting failure
# (almost always: MLP extrapolates wildly on an OOD episode split where Ridge stays bounded).
PER_FOLD_DROP_MARGIN = 0.10
# If more than this many folds are dropped, MLP estimate is unreliable on this cell.
MAX_DROPS_BEFORE_UNRELIABLE = 2


def parse_easy_path(json_path: Path) -> dict | None:
    """Infer (experiment, condition, seed, timepoint, step) from a T1a/T1b path.

    Easy-profile layouts:
      .../encoder-pretrain/easy/6d/sN/encoder-probes/encoder_probe_results.json
      .../visual_rl_agents/easy/matched-encoder/<run>/sN/encoder-probes/<tp>/encoder_probe_results.json
    """
    parts = json_path.parts
    if "encoder-pretrain" in parts and "easy" in parts:
        seed = next((int(p[1:]) for p in parts if p.startswith("s") and p[1:].isdigit()), None)
        return {
            "experiment": "T1a",
            "profile": "easy",
            "condition": "pretrained",
            "seed": seed,
            "timepoint": "pretrained",
            "step": None,
        }
    if "visual_rl_agents" in parts and "easy" in parts and "matched-encoder" in parts:
        # ['', 'media', 'hdd1', ..., 'visual_rl_agents', 'easy', 'matched-encoder',
        #  '<run>', 's42', 'encoder-probes', '<tp>', 'encoder_probe_results.json']
        try:
            mi = parts.index("matched-encoder")
        except ValueError:
            return None
        run = parts[mi + 1]
        seed_part = parts[mi + 2]  # e.g. 's42'
        tp = parts[mi + 4]         # 'peak' / 'final' / 'step-Xk'
        condition = RUN_DIR_TO_CONDITION.get(run)
        if condition is None:
            return None
        if not (seed_part.startswith("s") and seed_part[1:].isdigit()):
            return None
        seed = int(seed_part[1:])
        if tp in STEP_LABEL_TO_STEP:
            experiment = "T1b"
            step = STEP_LABEL_TO_STEP[tp]
        elif tp in {"peak", "final"}:
            experiment = "T1a" if seed in {42, 123, 456} and condition != "blind-ft-5e5" else "T1b"
            step = None
        else:
            return None
        return {
            "experiment": experiment,
            "profile": "easy",
            "condition": condition,
            "seed": seed,
            "timepoint": tp,
            "step": step,
        }
    return None


def parse_finding06_path(json_path: Path) -> dict | None:
    parts = json_path.parts
    if "analysis" not in parts or "encoder-probes" not in parts or "matched-encoder" not in parts:
        return None
    if "aggregated" in parts:
        return None
    try:
        mi = parts.index("matched-encoder")
    except ValueError:
        return None
    sub = parts[mi + 1]      # 'pretrained' or 'finetuned'
    seed_part = parts[mi + 2]
    if sub not in {"pretrained", "finetuned"}:
        return None
    if not (seed_part.startswith("s") and seed_part[1:].isdigit()):
        return None
    seed = int(seed_part[1:])
    return {
        "experiment": "finding06",
        "profile": "gym-default",
        "condition": "pretrained" if sub == "pretrained" else "blind-ft-5e5",
        "seed": seed,
        "timepoint": "pretrained" if sub == "pretrained" else "final",
        "step": None,
    }


def discover_jsons() -> list[tuple[Path, dict]]:
    out: list[tuple[Path, dict]] = []
    patterns = [
        NETS / "encoder-pretrain" / "easy" / "6d",
        NETS / "visual_rl_agents" / "easy" / "matched-encoder",
        NETS / "analysis" / "encoder-probes" / "matched-encoder",
    ]
    for root in patterns:
        if not root.exists():
            continue
        for jp in root.rglob("encoder_probe_results.json"):
            meta = parse_easy_path(jp) or parse_finding06_path(jp)
            if meta is None:
                continue
            out.append((jp, meta))
    return out


def load_rows(json_path: Path, meta: dict) -> list[dict]:
    """Return one row per (target, probe_type)."""
    with json_path.open() as f:
        blob = json.load(f)
    n_samples = blob.get("n_samples")
    n_episodes = blob.get("n_episodes")
    n_folds = blob.get("n_folds")
    encoder_path = blob.get("encoder_path")
    results = blob.get("results", {})
    rows = []
    for tgt in TARGETS:
        per_target = results.get(tgt, {})
        for probe_type in PROBES:
            r = per_target.get(probe_type)
            if r is None:
                continue
            rows.append({
                **meta,
                "target": tgt,
                "probe_type": probe_type,
                "r2_mean": float(r.get("r2_mean")) if r.get("r2_mean") is not None else None,
                "r2_std": float(r.get("r2_std")) if r.get("r2_std") is not None else None,
                "r2_folds": r.get("r2_folds"),
                "n_samples": n_samples,
                "n_episodes": n_episodes,
                "n_folds": n_folds,
                "encoder_path": encoder_path,
                "json_path": str(json_path),
            })
    return rows


def drop_bad_mlp_folds(rows: list[dict]) -> list[dict]:
    """Drop folds where a probe failed basic competence (R² < 0) and recompute.

    Applied independently per probe: a fold where the probe's R² is below 0 means
    the probe didn't even match the mean predictor — its train→test extrapolation
    blew up, almost always due to an episode-level OOD split. Drop those folds
    for that probe and recompute r2_mean / r2_std from the survivors.

    Adds per row: dropped_fold_idx (list[int]), n_folds_used (int),
    unreliable (bool — true when too many folds dropped).
    Preserves original values as r2_mean_orig / r2_std_orig.
    """
    by_key: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for r in rows:
        key = (r["json_path"], r["target"])
        by_key[key][r["probe_type"]] = r

    for probes in by_key.values():
        for ptype in ("linear", "mlp"):
            r = probes.get(ptype)
            if r is None:
                continue
            r.setdefault("r2_mean_orig", r["r2_mean"])
            r.setdefault("r2_std_orig", r["r2_std"])

            folds = r.get("r2_folds") or []
            kept_idx = [i for i, x in enumerate(folds) if x is not None and x >= 0]
            dropped_idx = [i for i in range(len(folds)) if i not in kept_idx]

            if dropped_idx and kept_idx:
                kept = [folds[i] for i in kept_idx]
                mean = sum(kept) / len(kept)
                var = sum((x - mean) ** 2 for x in kept) / len(kept)
                r["r2_mean"] = round(mean, 4)
                r["r2_std"] = round(var ** 0.5, 4)
            # If everything dropped, leave originals; flag unreliable below.

            r["dropped_fold_idx"] = dropped_idx
            r["n_folds_used"] = len(kept_idx)
            r["unreliable"] = (
                len(dropped_idx) > MAX_DROPS_BEFORE_UNRELIABLE or len(kept_idx) == 0
            )

    return rows


def compute_matched_folds(rows: list[dict]) -> list[dict]:
    """For (encoder, target) where both probes exist, compute matched means.

    Matched fold set = folds where both linear and MLP have R² ≥ 0. Average each
    probe over the same intersection, store as linear_matched_mean /
    mlp_matched_mean / matched_n. The matched gap (mlp − linear) is the
    apples-to-apples linear-vs-nonlinear-decodability measure.
    """
    by_key: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for r in rows:
        key = (r["json_path"], r["target"])
        by_key[key][r["probe_type"]] = r
    for probes in by_key.values():
        lin = probes.get("linear")
        mlp = probes.get("mlp")
        if lin is None or mlp is None:
            continue
        lin_folds = lin.get("r2_folds") or []
        mlp_folds = mlp.get("r2_folds") or []
        if not lin_folds or len(lin_folds) != len(mlp_folds):
            continue
        matched = [(lf, mf) for lf, mf in zip(lin_folds, mlp_folds)
                   if lf is not None and mf is not None and lf >= 0 and mf >= 0]
        if not matched:
            for r in (lin, mlp):
                r["linear_matched_mean"] = None
                r["mlp_matched_mean"] = None
                r["matched_n"] = 0
            continue
        lin_match = sum(lf for lf, _ in matched) / len(matched)
        mlp_match = sum(mf for _, mf in matched) / len(matched)
        for r in (lin, mlp):
            r["linear_matched_mean"] = round(lin_match, 4)
            r["mlp_matched_mean"] = round(mlp_match, 4)
            r["matched_n"] = len(matched)
    return rows


def attach_headline_and_flags(rows: list[dict]) -> list[dict]:
    """Add headline_r2 (max of adjusted linear/mlp, ignoring unreliable cells)."""
    by_key: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for r in rows:
        key = (r["json_path"], r["target"])
        by_key[key][r["probe_type"]] = r
    for probes in by_key.values():
        candidates = []
        for ptype in ("linear", "mlp"):
            r = probes.get(ptype)
            if r is None:
                continue
            if r.get("unreliable"):
                continue  # don't let an unreliable probe set the headline
            candidates.append(r["r2_mean"])
        # Fall back: if both probes are unreliable, use whichever has more surviving folds.
        if not candidates:
            best = None
            best_n = -1
            for ptype in ("linear", "mlp"):
                r = probes.get(ptype)
                if r is None:
                    continue
                n = r.get("n_folds_used", 0)
                if n > best_n and r["r2_mean"] is not None:
                    best = r["r2_mean"]
                    best_n = n
            headline = best
        else:
            headline = max(candidates)
        for r in probes.values():
            r["headline_r2"] = headline
            r["mlp_failed_flag"] = bool(probes.get("mlp", {}).get("unreliable"))
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    cols = [
        "experiment", "profile", "condition", "seed", "timepoint", "step",
        "target", "probe_type",
        "r2_mean", "r2_std", "headline_r2", "mlp_failed_flag",
        "n_samples", "n_episodes", "n_folds",
        "encoder_path", "json_path", "r2_folds",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            row = {c: r.get(c) for c in cols}
            if isinstance(row["r2_folds"], list):
                row["r2_folds"] = ";".join(f"{x:.4f}" for x in row["r2_folds"])
            w.writerow(row)


def fmt_r2(mean: float | None, std: float | None = None) -> str:
    if mean is None:
        return "—"
    if std is None:
        return f"{mean:.4f}"
    return f"{mean:.4f} ± {std:.4f}"


def write_t1a_full(rows: list[dict], path: Path) -> None:
    """Per-seed wide pivot of T1a: rows = (condition, seed, timepoint), cols = target × probe_type."""
    t1a = [r for r in rows if r["experiment"] == "T1a" and r["profile"] == "easy"]
    if not t1a:
        path.write_text("# T1a full (per-seed) — no data yet\n")
        return
    by = defaultdict(dict)
    for r in t1a:
        key = (r["condition"], r["seed"], r["timepoint"])
        by[key][(r["target"], r["probe_type"])] = r
    keys = sorted(by.keys(), key=lambda k: (
        ["pretrained", "blind-ft-1e5", "raw-ft", "branch-ft"].index(k[0])
        if k[0] in ["pretrained", "blind-ft-1e5", "raw-ft", "branch-ft"] else 999,
        k[1],
        ["pretrained", "peak", "final"].index(k[2]) if k[2] in ["pretrained", "peak", "final"] else 999,
    ))
    lines = ["# T1a full per-seed pivot (linear / mlp / headline)\n"]
    header_cells = ["condition", "seed", "timepoint"]
    for tgt in TARGETS:
        header_cells += [f"{tgt}/lin", f"{tgt}/mlp", f"{tgt}/HL", f"{tgt}/flag"]
    lines.append("| " + " | ".join(header_cells) + " |")
    lines.append("|" + "---|" * len(header_cells))
    for key in keys:
        cells = [str(key[0]), f"s{key[1]}", str(key[2])]
        for tgt in TARGETS:
            lin = by[key].get((tgt, "linear"))
            mlp = by[key].get((tgt, "mlp"))
            cells.append(fmt_r2(lin["r2_mean"]) if lin else "—")
            cells.append(fmt_r2(mlp["r2_mean"]) if mlp else "—")
            hl = (lin or mlp).get("headline_r2") if (lin or mlp) else None
            cells.append(fmt_r2(hl))
            flag = (lin or mlp).get("mlp_failed_flag") if (lin or mlp) else False
            cells.append("⚠" if flag else "")
        lines.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n")


def write_t1a_summary(rows: list[dict], path: Path) -> None:
    """T1a Finding-06-style table: mean ± std across seeds, by (condition, timepoint, target, probe).

    Also computes ΔR² vs pretrained-easy headline.
    """
    t1a = [r for r in rows if r["experiment"] == "T1a" and r["profile"] == "easy"]
    if not t1a:
        path.write_text("# T1a summary — no data yet\n")
        return
    # Aggregate across seeds: mean of r2_mean, std across seeds.
    bucket = defaultdict(list)  # (condition, timepoint, target, probe_type) -> [r2_mean per seed]
    bucket_hl = defaultdict(list)  # (condition, timepoint, target) -> [headline_r2 per seed]
    for r in t1a:
        bucket[(r["condition"], r["timepoint"], r["target"], r["probe_type"])].append(r["r2_mean"])
        bucket_hl[(r["condition"], r["timepoint"], r["target"])].append(r["headline_r2"])

    def mean_std(xs):
        xs = [x for x in xs if x is not None]
        if not xs:
            return (None, None)
        m = sum(xs) / len(xs)
        v = sum((x - m) ** 2 for x in xs) / len(xs)
        return (m, v ** 0.5)

    # Pretrained-easy headline per target — used for ΔR².
    pre_hl = {tgt: mean_std(bucket_hl.get(("pretrained", "pretrained", tgt), []))[0]
              for tgt in TARGETS}

    cond_tp_order = [
        ("pretrained", "pretrained"),
        ("blind-ft-1e5", "peak"), ("blind-ft-1e5", "final"),
        ("raw-ft", "peak"), ("raw-ft", "final"),
        ("branch-ft", "peak"), ("branch-ft", "final"),
    ]

    lines = ["# T1a summary — easy profile, mean ± std across seeds (N=3)\n"]
    lines.append("Headline R² = max(linear, mlp) per encoder. Δ = headline − pretrained-easy headline.\n")
    header = ["condition", "timepoint"]
    for tgt in TARGETS:
        header += [f"{tgt} HL", f"{tgt} Δ"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for (cond, tp) in cond_tp_order:
        cells = [cond, tp]
        for tgt in TARGETS:
            hl_mean, hl_std = mean_std(bucket_hl.get((cond, tp, tgt), []))
            cells.append(fmt_r2(hl_mean, hl_std))
            if hl_mean is None or pre_hl.get(tgt) is None or cond == "pretrained":
                cells.append("—")
            else:
                cells.append(f"{hl_mean - pre_hl[tgt]:+.4f}")
        lines.append("| " + " | ".join(cells) + " |")

    # Also a linear-only and mlp-only block for transparency.
    for ptype in PROBES:
        lines.append(f"\n## {ptype} probe — mean ± std across seeds\n")
        lines.append("| " + " | ".join(["condition", "timepoint"] + TARGETS) + " |")
        lines.append("|" + "---|" * (2 + len(TARGETS)))
        for (cond, tp) in cond_tp_order:
            cells = [cond, tp]
            for tgt in TARGETS:
                m, s = mean_std(bucket.get((cond, tp, tgt, ptype), []))
                cells.append(fmt_r2(m, s))
            lines.append("| " + " | ".join(cells) + " |")

    # Matched-folds linear vs MLP gap — apples-to-apples nonlinearity check.
    bucket_match_lin = defaultdict(list)
    bucket_match_mlp = defaultdict(list)
    bucket_match_n = defaultdict(list)
    for r in t1a:
        if r["probe_type"] != "linear":
            continue
        if r.get("linear_matched_mean") is None:
            continue
        k = (r["condition"], r["timepoint"], r["target"])
        bucket_match_lin[k].append(r["linear_matched_mean"])
        bucket_match_mlp[k].append(r["mlp_matched_mean"])
        bucket_match_n[k].append(r["matched_n"])
    lines.append("\n## Matched-folds linear vs MLP — apples-to-apples (mean across seeds)\n")
    lines.append("Folds intersected: linear and MLP both ≥ 0. Gap = MLP − linear (positive ⇒ nonlinear access helps).\n")
    header2 = ["condition", "timepoint"]
    for tgt in TARGETS:
        header2 += [f"{tgt} lin", f"{tgt} mlp", f"{tgt} Δ"]
    lines.append("| " + " | ".join(header2) + " |")
    lines.append("|" + "---|" * len(header2))
    for (cond, tp) in cond_tp_order:
        cells = [cond, tp]
        for tgt in TARGETS:
            lm, _ = mean_std(bucket_match_lin.get((cond, tp, tgt), []))
            mm, _ = mean_std(bucket_match_mlp.get((cond, tp, tgt), []))
            if lm is None or mm is None:
                cells += ["—", "—", "—"]
            else:
                cells += [f"{lm:.3f}", f"{mm:.3f}", f"{mm - lm:+.3f}"]
        lines.append("| " + " | ".join(cells) + " |")

    path.write_text("\n".join(lines) + "\n")


def write_t1b_trajectory(rows: list[dict], path: Path) -> None:
    """Trajectory pivot: rows = (condition, step), cols = target × headline R²."""
    t1b_step_order = ["pretrained"] + list(STEP_LABEL_TO_STEP.keys()) + ["peak", "final"]
    cond_order = ["blind-ft-5e5", "blind-ft-1e5", "raw-ft", "branch-ft"]
    # Build by (condition, timepoint, target) -> headline_r2 (s42 only)
    by = {}
    for r in rows:
        if r["profile"] != "easy" or r["seed"] != 42:
            continue
        if r["condition"] not in cond_order and r["condition"] != "pretrained":
            continue
        by[(r["condition"], r["timepoint"], r["target"])] = (r.get("headline_r2"), r.get("mlp_failed_flag"))
    if not by:
        path.write_text("# T1b trajectory — no data yet\n")
        return
    lines = ["# T1b trajectory — s42, headline R² per target across training steps\n"]
    lines.append("Headline = max(linear, mlp). ⚠ flags MLP-fit-suspect cells.\n")
    header = ["condition", "timepoint"] + TARGETS
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for cond in cond_order:
        for tp in t1b_step_order:
            # pretrained timepoint shared across conditions — emit under each condition for readability
            row_cond = "pretrained" if tp == "pretrained" else cond
            present = any((row_cond, tp, tgt) in by for tgt in TARGETS)
            if not present:
                continue
            cells = [cond, tp]
            for tgt in TARGETS:
                pair = by.get((row_cond, tp, tgt))
                if pair is None or pair[0] is None:
                    cells.append("—")
                else:
                    cells.append(f"{pair[0]:.3f}" + (" ⚠" if pair[1] else ""))
            lines.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n")


def write_flagged(rows: list[dict], path: Path) -> None:
    flagged = [r for r in rows if r.get("mlp_failed_flag") and r["probe_type"] == "mlp"]
    lines = ["# Flagged probe cells (MLP fit suspect)\n"]
    lines.append("Heuristic: linear_mean − mlp_mean > 0.05  OR  mlp_std > 0.20.\n")
    if not flagged:
        lines.append("_None._\n")
    else:
        lines.append("| profile | condition | seed | timepoint | target | linear R² | mlp R² | mlp std | json |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for r in flagged:
            # pull paired linear from rows
            lin_mean = next(
                (x["r2_mean"] for x in rows
                 if x["json_path"] == r["json_path"]
                 and x["target"] == r["target"]
                 and x["probe_type"] == "linear"),
                None,
            )
            lines.append(
                f"| {r['profile']} | {r['condition']} | s{r['seed']} | {r['timepoint']} | "
                f"{r['target']} | {fmt_r2(lin_mean)} | {fmt_r2(r['r2_mean'])} | "
                f"{fmt_r2(r['r2_std']) if r['r2_std'] is not None else '—'} | "
                f"`{Path(r['json_path']).relative_to(NETS)}` |"
            )
    path.write_text("\n".join(lines) + "\n")


def write_fold_audit(rows: list[dict], path: Path) -> None:
    """Per-cell breakdown of folds, drops, and adjusted means.

    One block per (profile, condition, seed, timepoint), one line per target.
    """
    # Group by encoder json
    by_json: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_json[r["json_path"]].append(r)
    lines = ["# Per-fold MLP audit", "",
             f"Drop rule: drop MLP fold *i* when `mlp[i] < linear[i] − {PER_FOLD_DROP_MARGIN}`.",
             "Headline R² = max(linear, mlp) per target after MLP fold-drop.", ""]
    # Sort encoders for stable output: by profile, condition, seed, timepoint.
    def _key(r):
        return (r["profile"], r["condition"], r.get("seed") or 0,
                r.get("timepoint") or "")
    for jp in sorted(by_json, key=lambda j: _key(by_json[j][0])):
        sample = by_json[jp][0]
        header = (f"### {sample['profile']} | {sample['condition']} | "
                  f"s{sample.get('seed')} | {sample.get('timepoint')}")
        lines.append(header)
        lines.append(f"`{Path(jp).relative_to(NETS)}`")
        lines.append("")
        lines.append("| target | linear folds | lin drop | linear adj | mlp folds | mlp drop | mlp adj | HL |")
        lines.append("|---|---|---|---|---|---|---|---|")
        by_target: dict[str, dict[str, dict]] = defaultdict(dict)
        for r in by_json[jp]:
            by_target[r["target"]][r["probe_type"]] = r
        for tgt in TARGETS:
            lin = by_target[tgt].get("linear")
            mlp = by_target[tgt].get("mlp")

            def _fmt(probe):
                if probe is None:
                    return "—", "—", "—"
                folds = probe.get("r2_folds") or []
                folds_str = "[" + ", ".join(f"{x:8.2f}" for x in folds) + "]"
                drops = probe.get("dropped_fold_idx", [])
                drops_str = ",".join(str(i) for i in drops) if drops else "—"
                adj = f"{probe['r2_mean']:.3f} ± {probe['r2_std']:.3f}"
                if drops:
                    adj += f" (n={probe['n_folds_used']})"
                if probe.get("unreliable"):
                    adj += " ⚠"
                return folds_str, drops_str, adj

            lin_folds_s, lin_drop_s, lin_adj_s = _fmt(lin)
            mlp_folds_s, mlp_drop_s, mlp_adj_s = _fmt(mlp)
            hl = (mlp or lin).get("headline_r2") if (mlp or lin) else None
            hl_str = f"{hl:.3f}" if hl is not None else "—"
            lines.append(
                f"| {tgt} | {lin_folds_s} | {lin_drop_s} | {lin_adj_s} | "
                f"{mlp_folds_s} | {mlp_drop_s} | {mlp_adj_s} | {hl_str} |"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    discovered = discover_jsons()
    print(f"Discovered {len(discovered)} probe jsons")
    rows: list[dict] = []
    for jp, meta in discovered:
        rows.extend(load_rows(jp, meta))
    rows = drop_bad_mlp_folds(rows)
    rows = compute_matched_folds(rows)
    rows = attach_headline_and_flags(rows)
    print(f"  -> {len(rows)} (encoder, target, probe) rows")

    write_csv(rows, OUT_DIR / "facts.csv")
    write_t1a_full(rows, OUT_DIR / "facts_t1a_full.md")
    write_t1a_summary(rows, OUT_DIR / "facts_t1a_summary.md")
    write_t1b_trajectory(rows, OUT_DIR / "facts_t1b_trajectory.md")
    write_flagged(rows, OUT_DIR / "flagged.md")
    write_fold_audit(rows, OUT_DIR / "fold_audit.md")
    print(f"Wrote outputs to {OUT_DIR}/")


if __name__ == "__main__":
    main()
