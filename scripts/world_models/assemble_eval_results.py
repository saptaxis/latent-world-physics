#!/usr/bin/env python
"""Assemble all eval results from an eval-all run into comparison tables.

Reads physics_understanding.json, physics_tests reports, and pixel eval
outputs from a timestamped eval directory. Produces markdown tables
comparing across models, data conditions, and eval types.

Usage:
    python scripts/world_models/assemble_eval_results.py \
        --eval-dir /media/hdd1/.../world-model-evals/20260330_1503 \
        --output /media/hdd1/.../world-model-evals/20260330_1503/summary.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))


def load_json(path: Path) -> dict | None:
    """Load JSON file, return None if missing."""
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def parse_run_name(name: str) -> dict:
    """Extract architecture, training mode, and data condition from run name."""
    info = {"name": name, "arch": "?", "training": "?", "data": "?"}

    # State-space runs
    if "linear" in name:
        info["arch"] = "Linear"
    elif "mlp" in name:
        info["arch"] = "MLP"
    elif "gru" in name:
        info["arch"] = "GRU"
    elif "rssm" in name:
        info["arch"] = "RSSM"

    if "single_step" in name:
        info["training"] = "single-step"
    elif "multi_step" in name:
        info["training"] = "multi-step k=10"
    elif "elbo" in name:
        info["training"] = "ELBO k=10"
    elif "film" in name:
        info["training"] = "FiLM multi-step"
    elif "-ss05" in name:
        info["training"] = "sched-samp 0.5"
    elif "multistep" in name:
        info["training"] = "multi-step k=10"

    if "primitives" in name or "prims" in name:
        info["data"] = "primitives"
    elif "policy" in name or "state6-" in name:
        info["data"] = "gym-default"

    # Pixel-specific
    if "dynamics-" in name:
        if "film" in name:
            info["arch"] = "FiLM-GRU"
        elif "rssm" in name:
            info["arch"] = "RSSM"
            if "fb1" in name:
                info["training"] = "ELBO+free-bits"
                if "long" in name:
                    info["training"] = "ELBO+fb (long)"
        elif "gru" in name or "multistep" in name:
            info["arch"] = "GRU"

    return info


def extract_physics_understanding(json_path: Path) -> dict:
    """Extract key metrics from physics_understanding.json."""
    data = load_json(json_path)
    if data is None:
        return {}

    results = {}
    constants = data.get("constants", {})
    for const_name in ["gravity", "main_thrust", "side_thrust", "kinematics",
                        "angular_damping"]:
        c = constants.get(const_name, {})
        oracle = c.get("oracle", {})
        results[f"{const_name}_oracle_err"] = oracle.get("relative_error", None)
        results[f"{const_name}_oracle_n"] = oracle.get("n_samples", 0)

        # Consistency R² — keys are state dim names (x, y, vx, vy, angle)
        consistency = c.get("consistency", {})
        if consistency:
            max_r2 = 0
            for dim_name, r2_val in consistency.items():
                if isinstance(r2_val, (int, float)):
                    max_r2 = max(max_r2, r2_val)
            results[f"{const_name}_max_consistency_r2"] = max_r2

    # Compounding — fit_params has aggregate a, b (MSE ~ a * h^b)
    compounding = data.get("compounding", {})
    fit_params = compounding.get("fit_params", {})
    results["compound_b"] = fit_params.get("b", None)
    results["compound_a"] = fit_params.get("a", None)

    # Useful horizon per dim
    useful = compounding.get("useful_horizon", {})
    for dim in ["vy", "vx", "y", "x", "angle"]:
        results[f"useful_horizon_{dim}"] = useful.get(dim, None)

    return results


def extract_physics_tests(dir_path: Path) -> dict:
    """Extract key metrics from physics tests output."""
    json_path = dir_path / "physics_tests.json"
    data = load_json(json_path)
    if data is None:
        # Try alternative name
        for f in dir_path.glob("*.json"):
            data = load_json(f)
            if data and ("maneuvers" in data or "results" in data):
                break
    if data is None:
        return {}

    results = {}
    # Top-level has overall_pass_rate, total_tested, total_passed
    if "total_passed" in data and "total_tested" in data:
        results["maneuvers_pass"] = data["total_passed"]
        results["maneuvers_total"] = data["total_tested"]
        results["maneuvers_pass_rate"] = f"{data['total_passed']}/{data['total_tested']}"
        results["overall_pass_rate"] = data.get("overall_pass_rate", None)
    else:
        maneuvers = data.get("maneuvers", data.get("results", {}))
        if isinstance(maneuvers, dict):
            n_pass = sum(1 for m in maneuvers.values()
                         if isinstance(m, dict) and m.get("pass", False))
            n_total = len(maneuvers)
            results["maneuvers_pass"] = n_pass
            results["maneuvers_total"] = n_total
            results["maneuvers_pass_rate"] = f"{n_pass}/{n_total}"
    return results


def extract_pixel_eval(dir_path: Path) -> dict:
    """Extract key metrics from pixel eval output."""
    results = {}

    # Pixel eval writes per-layer JSON files
    # Layer 1: state head fidelity — per_dim_r2 is a list, dim_names maps indices
    fidelity = load_json(dir_path / "layer1_fidelity.json")
    if fidelity:
        dim_names = fidelity.get("dim_names", ["x", "y", "vx", "vy", "angle", "ang_vel"])
        r2_list = fidelity.get("per_dim_r2", [])
        mse_list = fidelity.get("per_dim_mse", [])
        for i, dim_name in enumerate(dim_names):
            if i < len(r2_list):
                results[f"state_head_r2_{dim_name}"] = r2_list[i]
            if i < len(mse_list):
                results[f"state_head_mse_{dim_name}"] = mse_list[i]

    # Layer 2: oracle physics constants
    oracle = load_json(dir_path / "layer2_oracle.json")
    if oracle:
        for const_name in ["gravity", "main_thrust", "side_thrust", "kinematics",
                            "angular_damping"]:
            c = oracle.get(const_name, {})
            if isinstance(c, dict):
                results[f"{const_name}_oracle_err"] = c.get("relative_error", None)
                results[f"{const_name}_oracle_model_mean"] = c.get("model_mean", None)
                results[f"{const_name}_oracle_gt_mean"] = c.get("gt_mean", None)

    # Layer 2: consistency
    consistency = load_json(dir_path / "layer2_consistency.json")
    if consistency:
        for const_name in ["gravity", "main_thrust", "side_thrust"]:
            c = consistency.get(const_name, {})
            if isinstance(c, dict):
                max_r2 = 0
                for _, r2_val in c.items():
                    if isinstance(r2_val, (int, float)):
                        max_r2 = max(max_r2, r2_val)
                results[f"pixel_{const_name}_max_consistency_r2"] = max_r2

    # Layer 3: rollout kinematics
    kinematics = load_json(dir_path / "layer3_kinematics.json")
    if kinematics:
        for horizon in ["1", "5", "10", "20"]:
            h_data = kinematics.get(f"h{horizon}", kinematics.get(horizon, {}))
            if isinstance(h_data, dict):
                for dim in ["vy", "angle"]:
                    results[f"rollout_h{horizon}_{dim}"] = h_data.get(dim, None)

    # Layer 4: action conditioning
    sensitivity = load_json(dir_path / "layer4_sensitivity.json")
    if sensitivity:
        sr = sensitivity.get("sensitivity_ratio", {})
        if isinstance(sr, dict):
            # Average across all action pairs
            vals = [v for v in sr.values() if isinstance(v, (int, float))]
            results["sensitivity_ratio"] = sum(vals) / len(vals) if vals else None
        elif isinstance(sr, (int, float)):
            results["sensitivity_ratio"] = sr
    ablation = load_json(dir_path / "layer4_ablation.json")
    if ablation:
        # Ablation values are dicts keyed by horizon. Use horizon=1 for summary.
        mse_rz = ablation.get("mse_real_vs_zero", {})
        mse_rr = ablation.get("mse_real_vs_random", {})
        results["ablation_mse_real_vs_zero"] = mse_rz.get("1", None)
        results["ablation_mse_real_vs_random"] = mse_rr.get("1", None)
        # Kin is nested: horizon -> dim -> value. Aggregate across dims.
        kin_rz = ablation.get("kin_real_vs_zero", {})
        h1_kin = kin_rz.get("1", {})
        if isinstance(h1_kin, dict):
            vals = [v for v in h1_kin.values() if isinstance(v, (int, float))]
            results["ablation_kin_real_vs_zero"] = sum(vals) / len(vals) if vals else None

    return results


def format_pct(val):
    """Format a relative error as percentage."""
    if val is None:
        return "—"
    return f"{val * 100:.1f}%"


def format_float(val, decimals=3):
    if val is None:
        return "—"
    return f"{val:.{decimals}f}"


def main():
    p = argparse.ArgumentParser(description="Assemble eval results into comparison tables")
    p.add_argument("--eval-dir", type=str, required=True)
    p.add_argument("--output", type=str, default=None,
                   help="Output markdown file (default: eval-dir/summary.md)")
    args = p.parse_args()

    eval_dir = Path(args.eval_dir)
    output_path = Path(args.output) if args.output else eval_dir / "summary.md"

    lines = []
    lines.append("# Evaluation Results Summary\n")
    lines.append(f"**Eval dir:** `{eval_dir}`\n")

    # ================================================================
    # STATE-SPACE RESULTS
    # ================================================================
    ss_dir = eval_dir / "state-space"
    if ss_dir.exists():
        lines.append("\n## State-Space Models\n")

        # Collect all results
        ss_results = []
        for run_dir in sorted(ss_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            info = parse_run_name(run_dir.name)
            pu = extract_physics_understanding(
                run_dir / "physics_understanding" / "physics_understanding.json"
            )
            pt = extract_physics_tests(run_dir / "physics_tests")
            ss_results.append({"info": info, "pu": pu, "pt": pt})

        # Table 1: Oracle physics constants (relative error)
        lines.append("### Oracle Physics Constants (relative error, lower = better)\n")
        lines.append("| Model | Data | Training | Gravity | Thrust | Side | Kinematics | Damping | N(grav) |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for r in ss_results:
            i = r["info"]
            pu = r["pu"]
            lines.append(
                f"| {i['arch']} | {i['data']} | {i['training']} "
                f"| {format_pct(pu.get('gravity_oracle_err'))} "
                f"| {format_pct(pu.get('main_thrust_oracle_err'))} "
                f"| {format_pct(pu.get('side_thrust_oracle_err'))} "
                f"| {format_pct(pu.get('kinematics_oracle_err'))} "
                f"| {format_pct(pu.get('angular_damping_oracle_err'))} "
                f"| {pu.get('gravity_oracle_n', '—')} |"
            )

        # Table 2: Consistency R² (max spurious, lower = better)
        lines.append("\n### Consistency R² (max spurious correlation, lower = better)\n")
        lines.append("| Model | Data | Gravity R² | Thrust R² | Side R² | Damping R² |")
        lines.append("|---|---|---|---|---|---|")
        for r in ss_results:
            i = r["info"]
            pu = r["pu"]
            lines.append(
                f"| {i['arch']} | {i['data']} "
                f"| {format_float(pu.get('gravity_max_consistency_r2'))} "
                f"| {format_float(pu.get('main_thrust_max_consistency_r2'))} "
                f"| {format_float(pu.get('side_thrust_max_consistency_r2'))} "
                f"| {format_float(pu.get('angular_damping_max_consistency_r2'))} |"
            )

        # Table 3: Compounding exponents + useful horizons
        lines.append("\n### Error Compounding & Useful Horizon\n")
        lines.append("| Model | Data | b (MSE~h^b) | horizon(vy) | horizon(y) | horizon(angle) |")
        lines.append("|---|---|---|---|---|---|")
        for r in ss_results:
            i = r["info"]
            pu = r["pu"]
            lines.append(
                f"| {i['arch']} | {i['data']} "
                f"| {format_float(pu.get('compound_b'), 2)} "
                f"| {pu.get('useful_horizon_vy', '—')} "
                f"| {pu.get('useful_horizon_y', '—')} "
                f"| {pu.get('useful_horizon_angle', '—')} |"
            )

        # Table 4: Physics tests (maneuver pass rates)
        lines.append("\n### Physics Tests (controlled maneuver pass rate)\n")
        lines.append("| Model | Data | Pass rate |")
        lines.append("|---|---|---|")
        for r in ss_results:
            i = r["info"]
            pt = r["pt"]
            lines.append(
                f"| {i['arch']} | {i['data']} "
                f"| {pt.get('maneuvers_pass_rate', '—')} |"
            )

    # ================================================================
    # PIXEL RESULTS
    # ================================================================
    px_dir = eval_dir / "pixel"
    if px_dir.exists():
        lines.append("\n---\n")
        lines.append("\n## Pixel Models\n")

        px_results = []
        for run_dir in sorted(px_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            info = parse_run_name(run_dir.name)
            px = extract_pixel_eval(run_dir)
            px_results.append({"info": info, "px": px})

        # Table 5: State head fidelity (perception quality)
        lines.append("### State Head Fidelity (R² per dim, higher = better)\n")
        dims = ["x", "y", "vx", "vy", "angle", "ang_vel"]
        header = "| Model | Data | Training | " + " | ".join(dims) + " |"
        sep = "|---|---|---|" + "|".join(["---"] * len(dims)) + "|"
        lines.append(header)
        lines.append(sep)
        for r in px_results:
            i = r["info"]
            px = r["px"]
            vals = [format_float(px.get(f"state_head_r2_{d}")) for d in dims]
            lines.append(
                f"| {i['arch']} | {i['data']} | {i['training']} | "
                + " | ".join(vals) + " |"
            )

        # Table 6: Oracle physics constants
        lines.append("\n### Oracle Physics Constants (from pixel pipeline)\n")
        lines.append("| Model | Data | Gravity err | Thrust err | Side err | Model grav | GT grav |")
        lines.append("|---|---|---|---|---|---|---|")
        for r in px_results:
            i = r["info"]
            px = r["px"]
            lines.append(
                f"| {i['arch']} | {i['data']} "
                f"| {format_pct(px.get('gravity_oracle_err'))} "
                f"| {format_pct(px.get('main_thrust_oracle_err'))} "
                f"| {format_pct(px.get('side_thrust_oracle_err'))} "
                f"| {format_float(px.get('gravity_oracle_model_mean'), 4)} "
                f"| {format_float(px.get('gravity_oracle_gt_mean'), 4)} |"
            )

        # Table 7: Action conditioning
        lines.append("\n### Action Conditioning (ablation: MSE with real vs zero/random actions)\n")
        lines.append("| Model | Data | MSE real vs zero | MSE real vs random | Kin real vs zero | Sensitivity ratio |")
        lines.append("|---|---|---|---|---|---|")
        for r in px_results:
            i = r["info"]
            px = r["px"]
            lines.append(
                f"| {i['arch']} | {i['data']} "
                f"| {format_float(px.get('ablation_mse_real_vs_zero'), 4)} "
                f"| {format_float(px.get('ablation_mse_real_vs_random'), 4)} "
                f"| {format_float(px.get('ablation_kin_real_vs_zero'), 4)} "
                f"| {format_float(px.get('sensitivity_ratio'), 4)} |"
            )

    # ================================================================
    # GROUND TRUTH BASELINE
    # ================================================================
    gt_dir = eval_dir / "ground-truth"
    if gt_dir.exists():
        lines.append("\n---\n")
        lines.append("\n## Ground Truth Baseline\n")
        gt_log = gt_dir / "stdout.log"
        if gt_log.exists():
            lines.append("```")
            # Include the summary section of GT output
            text = gt_log.read_text()
            in_summary = False
            for line in text.split("\n"):
                if "SUMMARY" in line or "Constant" in line:
                    in_summary = True
                if in_summary:
                    lines.append(line)
                if "PASS" in line or "FAIL" in line or "WARN" in line:
                    if in_summary:
                        break
            lines.append("```")

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Summary written to: {output_path}")
    print(f"  State-space models: {len(ss_results) if ss_dir.exists() else 0}")
    print(f"  Pixel models: {len(px_results) if px_dir.exists() else 0}")


if __name__ == "__main__":
    main()
