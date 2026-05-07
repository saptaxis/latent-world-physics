#!/bin/bash
# Evaluate the 6 E2-25 force3 runs (physics understanding + physics tests).
# Runs all 6 in parallel on CPU. Takes a few minutes.
#
# Usage: bash scripts/world_models/eval_force3.sh [device]
#   bash scripts/world_models/eval_force3.sh cuda:0

set -e

DEVICE="${1:-cpu}"

BASE="/media/hdd1/physics-priors-latent-space/lunar-lander-networks/world-model-ladder-runs"
DATA_GD="/media/hdd1/physics-priors-latent-space/lunar-lander-data/world_model_data/gym-default"
DATA_PR="/media/hdd1/physics-priors-latent-space/lunar-lander-data/world_model_data/primitives-v1"

RUNS_POLICY=(
    mlp-delta-single_step_k1-policy--force3
    gru-delta-multi_step_k10-policy--force3
    rssm-delta-elbo_k10-policy--force3
)
RUNS_PRIMS=(
    mlp-delta-single_step_k1-primitives--force3
    gru-delta-multi_step_k10-primitives--force3
    rssm-delta-elbo_k10-primitives--force3
)

ts() { date +"%H:%M:%S"; }

run_eval() {
    local run="$1"
    local data_dir="$2"
    local ckpt="$BASE/$run/best.pt"

    if [ ! -f "$ckpt" ]; then
        echo "[$(ts)] SKIP: $run (no best.pt)"
        return
    fi

    echo "[$(ts)] START: $run"

    # Physics understanding (constants extraction)
    python scripts/analysis/physics_understanding_report.py \
        --ladder-checkpoint "$ckpt" \
        --data-dir "$data_dir" \
        --n-episodes 200 \
        --device "$DEVICE" \
        2>&1 | tail -5

    # Physics tests (controlled maneuvers)
    python scripts/world_models/physics_test_wm.py \
        --ladder-checkpoint "$ckpt" \
        --n-seeds 10 \
        2>&1 | tail -3

    echo "[$(ts)] DONE: $run"
    echo ""
}

echo "=============================="
echo "E2-25 Force3 Evaluation"
echo "=============================="
echo ""

PIDS=()

for run in "${RUNS_POLICY[@]}"; do
    run_eval "$run" "$DATA_GD" &
    PIDS+=($!)
done

for run in "${RUNS_PRIMS[@]}"; do
    run_eval "$run" "$DATA_PR" &
    PIDS+=($!)
done

# Wait for all
FAILED=0
for pid in "${PIDS[@]}"; do
    wait "$pid" || FAILED=$((FAILED + 1))
done

echo "=============================="
echo "All done. Failed: $FAILED"
echo ""

# Quick summary
echo "Results saved to:"
for run in "${RUNS_POLICY[@]}" "${RUNS_PRIMS[@]}"; do
    pu="$BASE/$run/physics_understanding/physics_understanding.json"
    pt="$BASE/$run/physics_tests"
    pu_ok="--"; pt_ok="--"
    [ -f "$pu" ] && pu_ok="OK"
    ls "$pt"/*.json >/dev/null 2>&1 && pt_ok="OK"
    printf "  %-55s understanding: %-4s  tests: %-4s\n" "$run" "$pu_ok" "$pt_ok"
done
