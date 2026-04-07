#!/bin/bash
# Run ALL evaluations on ALL models.
# Outputs go to a timestamped directory for clean comparison.
#
# Parallelism:
#   - State-space evals run in parallel on CPU (no GPU needed)
#   - Pixel evals run sequentially on GPU (VRAM constraint)
#   - State-space and pixel run concurrently
#
# Usage:
#   bash scripts/world_models/run_all_evals.sh [device]
#   bash scripts/world_models/run_all_evals.sh cuda:0
#   bash scripts/world_models/run_all_evals.sh cuda:1

set -e

GPU0="${1:-cuda:0}"
GPU1="${2:-cuda:1}"
TIMESTAMP=$(date +%Y%m%d_%H%M)
BASE="/media/hdd1/physics-priors-latent-space/lunar-lander-networks"
DATA="/media/hdd1/physics-priors-latent-space/lunar-lander-data"
OUT="$BASE/world-model-evals/$TIMESTAMP"
mkdir -p "$OUT"

echo "=============================================="
echo "Running ALL evaluations (parallel where possible)"
echo "Output: $OUT"
echo "GPU0: $GPU0, GPU1: $GPU1"
echo "=============================================="

# Max parallel CPU jobs for state-space evals
MAX_CPU_JOBS=2

ts() { date +"%H:%M:%S"; }
log() { echo "[$(ts)] $*"; }

# ============================================================
# Helper: run a single state-space eval (physics understanding + physics tests)
# Called in background for parallelism
# ============================================================
run_state_space_eval() {
    local run="$1"
    local CKPT="$BASE/world-model-ladder-runs/$run/best.pt"

    if [ ! -f "$CKPT" ]; then
        log "  SKIP: $run (no checkpoint)"
        return
    fi

    # Determine data dir
    local DATA_DIR
    if [[ "$run" == *"primitives"* ]]; then
        DATA_DIR="$DATA/world_model_data/primitives-v1"
    else
        DATA_DIR="$DATA/world_model_data/gym-default"
    fi

    # A. Physics understanding (constants extraction)
    local PU_OUT="$OUT/state-space/$run/physics_understanding"
    mkdir -p "$PU_OUT"
    log "  [SS] START physics understanding: $run"
    python scripts/analysis/physics_understanding_report.py \
        --ladder-checkpoint "$CKPT" \
        --data-dir "$DATA_DIR" \
        --n-episodes 200 \
        --output-dir "$PU_OUT" \
        --device cpu \
        > "$PU_OUT/stdout.log" 2>&1 || echo "  FAILED: $run physics_understanding"

    # B. Physics tests (controlled maneuvers)
    local PT_OUT="$OUT/state-space/$run/physics_tests"
    mkdir -p "$PT_OUT"
    log "  [SS] START physics tests: $run"
    python scripts/world_models/physics_test_wm.py \
        --ladder-checkpoint "$CKPT" \
        --n-seeds 10 \
        --output-dir "$PT_OUT" \
        > "$PT_OUT/stdout.log" 2>&1 || echo "  FAILED: $run physics_tests"

    # C. Generic rollout metrics (horizon curves, divergence, etc.)
    local EV_OUT="$OUT/state-space/$run/eval"
    mkdir -p "$EV_OUT"
    log "  [SS] START eval: $run"
    python scripts/world_models/eval.py \
        --checkpoint "$CKPT" \
        --device cpu \
        > "$EV_OUT/stdout.log" 2>&1 || echo "  FAILED: $run eval"

    log "  [SS] DONE: $run"
}

# ============================================================
# Helper: run a single pixel eval
# Runs sequentially (GPU bound)
# ============================================================
run_pixel_eval() {
    local DYN_RUN="$1"
    local VAE_RUN="$2"
    local GPU="$3"

    local DYN_CKPT="$BASE/pixel-world-model/$DYN_RUN/best.pt"
    local VAE_CKPT="$BASE/pixel-world-model/$VAE_RUN/best.pt"

    if [ ! -f "$DYN_CKPT" ]; then
        log "  SKIP: $DYN_RUN (no dynamics checkpoint)"
        return
    fi
    if [ ! -f "$VAE_CKPT" ]; then
        log "  SKIP: $DYN_RUN (no VAE: $VAE_RUN)"
        return
    fi

    # Determine eval data
    local EVAL_DATA
    if [[ "$VAE_RUN" == *"prims"* ]]; then
        EVAL_DATA="$DATA/world_model_data/visual-gym-default-random-heuristic-prims"
    else
        EVAL_DATA="$DATA/encoder-pretrain/random"
    fi

    local OUTDIR="$OUT/pixel/$DYN_RUN"
    mkdir -p "$OUTDIR"
    log "  [PX/$GPU] START: $DYN_RUN (VAE: $VAE_RUN)"
    python scripts/world_models/eval_pixel_wm_physics.py \
        --vae-checkpoint "$VAE_CKPT" \
        --dynamics-checkpoint "$DYN_CKPT" \
        --data-path "$EVAL_DATA" \
        --output-dir "$OUTDIR" \
        --n-episodes 200 \
        --device "$GPU" \
        --per-policy \
        > "$OUTDIR/stdout.log" 2>&1 || echo "  FAILED: $DYN_RUN"

    log "  [PX/$GPU] DONE: $DYN_RUN"
}

# ============================================================
# Launch state-space evals in parallel (CPU only)
# ============================================================
echo ""
echo "===== Launching state-space evals (parallel, CPU) ====="
echo ""

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
)

# Track background PIDs
SS_PIDS=()
for run in "${SS_RUNS[@]}"; do
    # Throttle: wait if we have MAX_CPU_JOBS running
    while [ $(jobs -rp | wc -l) -ge $MAX_CPU_JOBS ]; do
        sleep 2
    done
    run_state_space_eval "$run" &
    SS_PIDS+=($!)
done

# ============================================================
# Launch pixel evals sequentially (GPU bound)
# Runs concurrently with state-space evals
# ============================================================
echo ""
echo "===== Launching pixel evals (sequential, GPU) ====="
echo ""

# Split pixel evals across 2 GPUs, 2 jobs per GPU
# GPU0 gets 4 runs, GPU1 gets 3 runs
PX_PIDS=()

# GPU0 batch
run_pixel_eval "dynamics-state6-multistep"          "vae-fg50-state6"       "$GPU0" &
PX_PIDS+=($!)
run_pixel_eval "dynamics-state6-ss05"               "vae-fg50-state6"       "$GPU0" &
PX_PIDS+=($!)

# GPU1 batch
run_pixel_eval "dynamics-state6-prims-gru-multistep" "vae-fg50-state6-prims" "$GPU1" &
PX_PIDS+=($!)
run_pixel_eval "dynamics-state6prims-film-multistep" "vae-fg50-state6-prims" "$GPU1" &
PX_PIDS+=($!)

# Wait for first batch to finish before launching more
# (2 per GPU to avoid OOM)
for pid in "${PX_PIDS[@]}"; do wait "$pid" || true; done
PX_PIDS=()

# Second batch
run_pixel_eval "dynamics-state6-rssm-elbo"          "vae-fg50-state6"       "$GPU0" &
PX_PIDS+=($!)
run_pixel_eval "dynamics-state6-rssm-elbo-fb1"      "vae-fg50-state6"       "$GPU1" &
PX_PIDS+=($!)

for pid in "${PX_PIDS[@]}"; do wait "$pid" || true; done
PX_PIDS=()

run_pixel_eval "dynamics-state6-rssm-elbo-fb1-long" "vae-fg50-state6"       "$GPU0" &
PX_PIDS+=($!)

for pid in "${PX_PIDS[@]}"; do wait "$pid" || true; done

# ============================================================
# Ground truth baseline
# ============================================================
echo ""
echo "===== Ground truth baseline ====="
echo ""

GT_OUT="$OUT/ground-truth"
mkdir -p "$GT_OUT"
python scripts/world_models/validate_physics_extraction.py \
    --data-path "$DATA/world_model_data/gym-default" \
    --max-episodes 200 \
    > "$GT_OUT/stdout.log" 2>&1 || echo "  FAILED: GT validation"
echo "  [GT] Done"

# ============================================================
# Wait for all state-space background jobs
# ============================================================
echo ""
echo "Waiting for state-space background jobs..."
FAILED=0
for pid in "${SS_PIDS[@]}"; do
    wait "$pid" || FAILED=$((FAILED + 1))
done

# ============================================================
# Summary
# ============================================================
log ""
log "=============================================="
log "ALL EVALS COMPLETE"
log "Output: $OUT"
log ""

# --- Per-model status check ---
log "STATE-SPACE RESULTS:"
for run in "${SS_RUNS[@]}"; do
    PU="$OUT/state-space/$run/physics_understanding"
    PT="$OUT/state-space/$run/physics_tests"
    EV="$OUT/state-space/$run/eval"
    PU_OK="--"
    PT_OK="--"
    EV_OK="--"
    if [ -f "$PU/results.json" ] 2>/dev/null; then PU_OK="OK"; elif [ -f "$PU/stdout.log" ]; then PU_OK="??"; fi
    if ls "$PT"/*.json >/dev/null 2>&1; then PT_OK="OK"; elif [ -f "$PT/stdout.log" ]; then PT_OK="??"; fi
    if [ -f "$EV/stdout.log" ] && ! grep -q "FAILED" "$EV/stdout.log" 2>/dev/null; then EV_OK="OK"; elif [ -f "$EV/stdout.log" ]; then EV_OK="??"; fi
    printf "  %-50s  understanding: %-4s  tests: %-4s  eval: %-4s\n" "$run" "$PU_OK" "$PT_OK" "$EV_OK"
done

log ""
log "PIXEL RESULTS:"
for entry in \
    "dynamics-state6-multistep" \
    "dynamics-state6-ss05" \
    "dynamics-state6-prims-gru-multistep" \
    "dynamics-state6prims-film-multistep" \
    "dynamics-state6-rssm-elbo" \
    "dynamics-state6-rssm-elbo-fb1" \
    "dynamics-state6-rssm-elbo-fb1-long"; do
    PX_DIR="$OUT/pixel/$entry"
    PX_OK="--"
    if ls "$PX_DIR"/*.json >/dev/null 2>&1; then PX_OK="OK"; elif [ -f "$PX_DIR/stdout.log" ]; then PX_OK="??"; fi
    printf "  %-50s  eval: %-4s\n" "$entry" "$PX_OK"
done

log ""
log "GT BASELINE: $([ -f "$GT_OUT/stdout.log" ] && echo 'OK' || echo '--')"

if [ $FAILED -gt 0 ]; then
    log "WARNING: $FAILED state-space background jobs returned non-zero"
fi
log ""
log "Check individual logs: tail \$OUT/<model>/stdout.log"
log "=============================================="
