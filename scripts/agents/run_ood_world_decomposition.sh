#!/usr/bin/env bash
# Decompose world-only OOD into gravity / turbulence / wind components.
#
# Physics-only easy-trained agents evaluated on three isolated world shifts:
#   (a) gravity-widened: gravity at full range, no wind/turbulence (≈training control)
#   (b) turbulence-only: turbulence [0,2] introduced, no wind
#   (c) wind-only: wind [0,15] introduced, no turbulence
#
# Tests whether labeled collapse under physics-only-medium is driven
# specifically by wind, or by novel world perturbations broadly.
#
# All 6 batches run in parallel (3 profiles × 2 conditions).

set -euo pipefail

AGENTS_BASE="/media/hdd1/physics-priors-latent-space/lunar-lander-networks/rl_agents/parametric-vs-behavioral/physics-only"
EPISODES=100
LOGDIR="$HOME/vsr-tmp/ood-world-decomposition-logs"
mkdir -p "$LOGDIR"

echo "Starting 6 parallel world-OOD decomposition batches..."
echo "Logs in $LOGDIR/"

# (a) Gravity-widened (effectively a training-distribution control)
python scripts/agents/run_eval_pipeline.py \
    --agents-dir "$AGENTS_BASE/blind-ppo-easy-128-lowent" \
    --profiles physics-only-gravity-widened \
    --episodes $EPISODES \
    --trajectory-subdir trajectories-ood-gravity \
    --force \
    > "$LOGDIR/blind-gravity.log" 2>&1 &
PID1=$!
echo "  [PID $PID1] blind × gravity-widened"

python scripts/agents/run_eval_pipeline.py \
    --agents-dir "$AGENTS_BASE/labeled-ppo-easy-128-lowent" \
    --profiles physics-only-gravity-widened \
    --episodes $EPISODES \
    --trajectory-subdir trajectories-ood-gravity \
    --force \
    > "$LOGDIR/labeled-gravity.log" 2>&1 &
PID2=$!
echo "  [PID $PID2] labeled × gravity-widened"

# (b) Turbulence only
python scripts/agents/run_eval_pipeline.py \
    --agents-dir "$AGENTS_BASE/blind-ppo-easy-128-lowent" \
    --profiles physics-only-turbulence-only \
    --episodes $EPISODES \
    --trajectory-subdir trajectories-ood-turbulence \
    --force \
    > "$LOGDIR/blind-turbulence.log" 2>&1 &
PID3=$!
echo "  [PID $PID3] blind × turbulence-only"

python scripts/agents/run_eval_pipeline.py \
    --agents-dir "$AGENTS_BASE/labeled-ppo-easy-128-lowent" \
    --profiles physics-only-turbulence-only \
    --episodes $EPISODES \
    --trajectory-subdir trajectories-ood-turbulence \
    --force \
    > "$LOGDIR/labeled-turbulence.log" 2>&1 &
PID4=$!
echo "  [PID $PID4] labeled × turbulence-only"

# (c) Wind only
python scripts/agents/run_eval_pipeline.py \
    --agents-dir "$AGENTS_BASE/blind-ppo-easy-128-lowent" \
    --profiles physics-only-wind-only \
    --episodes $EPISODES \
    --trajectory-subdir trajectories-ood-wind \
    --force \
    > "$LOGDIR/blind-wind.log" 2>&1 &
PID5=$!
echo "  [PID $PID5] blind × wind-only"

python scripts/agents/run_eval_pipeline.py \
    --agents-dir "$AGENTS_BASE/labeled-ppo-easy-128-lowent" \
    --profiles physics-only-wind-only \
    --episodes $EPISODES \
    --trajectory-subdir trajectories-ood-wind \
    --force \
    > "$LOGDIR/labeled-wind.log" 2>&1 &
PID6=$!
echo "  [PID $PID6] labeled × wind-only"

echo ""
echo "Waiting for all 6 batches to finish..."
wait $PID1 && echo "  blind × gravity: DONE" || echo "  blind × gravity: FAILED"
wait $PID2 && echo "  labeled × gravity: DONE" || echo "  labeled × gravity: FAILED"
wait $PID3 && echo "  blind × turbulence: DONE" || echo "  blind × turbulence: FAILED"
wait $PID4 && echo "  labeled × turbulence: DONE" || echo "  labeled × turbulence: FAILED"
wait $PID5 && echo "  blind × wind: DONE" || echo "  blind × wind: FAILED"
wait $PID6 && echo "  labeled × wind: DONE" || echo "  labeled × wind: FAILED"

echo ""
echo "=== All batches complete ==="
echo "Next: run cross-config analysis with the new manifests."
