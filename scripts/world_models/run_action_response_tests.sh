#!/bin/bash
# Run action-response direction test on all E2 world models
# (state-space + pixel). Writes per-model JSON + an aggregate markdown table.
#
# Usage:
#   bash scripts/world_models/run_action_response_tests.sh [gpu]

set -e

GPU="${1:-cuda:0}"
TIMESTAMP=$(date +%Y%m%d_%H%M)
BASE="/media/hdd1/physics-priors-latent-space/lunar-lander-networks"
DATA="/media/hdd1/physics-priors-latent-space/lunar-lander-data"
OUT="$BASE/world-model-evals/action_response_$TIMESTAMP"
mkdir -p "$OUT"

echo "=============================================="
echo "Action-response tests"
echo "Output: $OUT"
echo "=============================================="

# -------- State-space (all trained ladder runs) --------
SS_RUNS=(
    linear-delta-single_step_k1-policy--v3
    linear-delta-single_step_k1-primitives
    mlp-delta-single_step_k1-policy--v3
    mlp-delta-single_step_k1-primitives
    mlp-delta-multi_step_k10-policy--v3
    mlp-delta-multi_step_k10-primitives
    gru-delta-multi_step_k10-policy--v3
    gru-delta-multi_step_k10-primitives
    rssm-delta-elbo_k10-policy--v3
    rssm-delta-elbo_k10-primitives
    mlp-delta-single_step_k1-policy--force3
    mlp-delta-single_step_k1-primitives--force3
    gru-delta-multi_step_k10-policy--force3
    gru-delta-multi_step_k10-primitives--force3
    rssm-delta-elbo_k10-policy--force3
    rssm-delta-elbo_k10-primitives--force3
)

for run in "${SS_RUNS[@]}"; do
    CKPT="$BASE/world-model-ladder-runs/$run/best.pt"
    [ -f "$CKPT" ] || { echo "  SKIP: $run (no ckpt)"; continue; }
    if [[ "$run" == *"primitives"* ]]; then
        DATA_DIR="$DATA/world_model_data/primitives-v1"
    else
        DATA_DIR="$DATA/world_model_data/gym-default"
    fi
    OUTDIR="$OUT/state-space/$run"
    mkdir -p "$OUTDIR"
    echo "[SS] $run"
    python scripts/world_models/test_action_response.py \
        --ladder-checkpoint "$CKPT" \
        --data-dir "$DATA_DIR" \
        --output-json "$OUTDIR/action_response.json" \
        --device cpu \
        > "$OUTDIR/stdout.log" 2>&1 || echo "  FAILED: $run"
done

# -------- Pixel (vae + dynamics pairs) --------
# Each entry: "dynamics_run vae_run"
PX_PAIRS=(
    "dynamics-state6-multistep           vae-fg50-state6"
    "dynamics-state6-ss05                vae-fg50-state6"
    "dynamics-state6-rssm-elbo           vae-fg50-state6"
    "dynamics-state6-rssm-elbo-fb1       vae-fg50-state6"
    "dynamics-state6-rssm-elbo-fb1-long  vae-fg50-state6"
    "dynamics-state6-prims-gru-multistep vae-fg50-state6-prims"
    "dynamics-state6prims-film-multistep vae-fg50-state6-prims"
)

for pair in "${PX_PAIRS[@]}"; do
    read -r DYN_RUN VAE_RUN <<< "$pair"
    DYN_CKPT="$BASE/pixel-world-model/$DYN_RUN/best.pt"
    VAE_CKPT="$BASE/pixel-world-model/$VAE_RUN/best.pt"
    [ -f "$DYN_CKPT" ] || { echo "  SKIP: $DYN_RUN (no dyn ckpt)"; continue; }
    [ -f "$VAE_CKPT" ] || { echo "  SKIP: $DYN_RUN (no vae ckpt)"; continue; }
    if [[ "$VAE_RUN" == *"prims"* ]]; then
        EVAL_DATA="$DATA/world_model_data/visual-gym-default-random-heuristic-prims"
    else
        EVAL_DATA="$DATA/encoder-pretrain/random"
    fi
    OUTDIR="$OUT/pixel/$DYN_RUN"
    mkdir -p "$OUTDIR"
    echo "[PX] $DYN_RUN  (vae=$VAE_RUN)"
    python scripts/world_models/test_action_response.py \
        --vae-checkpoint "$VAE_CKPT" \
        --dynamics-checkpoint "$DYN_CKPT" \
        --data-path "$EVAL_DATA" \
        --output-json "$OUTDIR/action_response.json" \
        --device "$GPU" \
        > "$OUTDIR/stdout.log" 2>&1 || echo "  FAILED: $DYN_RUN"
done

# -------- Aggregate to markdown table --------
python - <<'PY' "$OUT"
import json, sys
from pathlib import Path

out = Path(sys.argv[1])
rows = []
for jpath in sorted(out.rglob("action_response.json")):
    with open(jpath) as f:
        d = json.load(f)
    t = d["tests"]
    rsteps = d.get("rollout_tests", [])
    typ = "SS" if "/state-space/" in str(jpath) else "PX"
    # Last-step sustained + impulse accuracy (for delayed-effect check).
    last = rsteps[-1] if rsteps else {}
    rows.append({
        "name": d["run_name"],
        "type": typ,
        "n": d["n_transitions"],
        "main_vy": t["main_thrust_vy_up"]["accuracy"],
        "grav_vy": t["gravity_vy_down"]["accuracy"],
        "right_av": t["side_right_angvel_up"]["accuracy"],
        "left_av": t["side_left_angvel_down"]["accuracy"],
        "symm": t["side_symmetry"]["accuracy"],
        "main_vy_s5": last.get("main_vy_up", float("nan")),
        "imp_vy_s5":  last.get("impulse_vy_up", float("nan")),
        "imp_vy_s5_diff": last.get("impulse_vy_mean_diff", float("nan")),
    })

def fmt(x):
    return f"{x:.0%}"

lines = [
    "# Action-Response Direction Test — All E2 Models",
    "",
    "Direction accuracy: >80% = model responds correctly; ~50% = random; <50% = wrong direction.",
    "",
    "| Model | Type | N | Main→vy↑ | Gravity→vy↓ | Right→av↑ | Left→av↓ | L/R symm | Main@5 | Impulse@5 | ImpΔvy@5 |",
    "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]
def fmt_f(x):
    try:
        return f"{x:+.4f}" if x == x else "—"  # NaN check
    except Exception:
        return "—"

for r in rows:
    lines.append(
        f"| {r['name']} | {r['type']} | {r['n']} | "
        f"{fmt(r['main_vy'])} | {fmt(r['grav_vy'])} | "
        f"{fmt(r['right_av'])} | {fmt(r['left_av'])} | {fmt(r['symm'])} | "
        f"{fmt(r['main_vy_s5'])} | {fmt(r['imp_vy_s5'])} | {fmt_f(r['imp_vy_s5_diff'])} |"
    )

# Flag any model that passes >80% on at least 3 of 5 tests.
lines.append("")
lines.append("## Passing models (>80% on ≥3 of 5 tests)")
lines.append("")
passes = []
for r in rows:
    scores = [r["main_vy"], r["grav_vy"], r["right_av"], r["left_av"], r["symm"]]
    n_pass = sum(1 for s in scores if s > 0.80)
    if n_pass >= 3:
        passes.append((r["name"], n_pass))
if passes:
    for name, n in sorted(passes, key=lambda x: -x[1]):
        lines.append(f"- {name} ({n}/5)")
else:
    lines.append("- none")

(out / "summary.md").write_text("\n".join(lines) + "\n")
print(f"\nWrote {out}/summary.md ({len(rows)} models)")
PY

echo ""
echo "Done. Output: $OUT"
