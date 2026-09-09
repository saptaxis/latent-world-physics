#!/usr/bin/env bash
# Run body-only and world-only OOD evals on easy-trained full-variation agents.
#
# Decomposes the easy→medium OOD shift into two components:
#   body-ood-medium: damping + TWR shift, no wind/turbulence
#   world-ood-medium: wind + turbulence introduced, damping/TWR stay easy
#
# All 4 batches run in parallel (independent output dirs).
# Each batch processes 10 seeds sequentially within run_eval_pipeline.py.
#
# After this script, run the cross-config analysis:
#   python scripts/analysis/compare_configs.py \
#       analysis-manifests/comparison/ood-generalization-body-medium.yaml
#   python scripts/analysis/compare_configs.py \
#       analysis-manifests/comparison/ood-generalization-world-medium.yaml

set -euo pipefail

AGENTS_BASE="/media/hdd1/physics-priors-latent-space/lunar-lander-networks/rl_agents/parametric-vs-behavioral/full-variation"
EPISODES=100
LOGDIR="$HOME/vsr-tmp/ood-decomposition-logs"
mkdir -p "$LOGDIR"

echo "Starting 4 parallel OOD decomposition batches..."
echo "Logs in $LOGDIR/"

python scripts/agents/run_eval_pipeline.py \
    --agents-dir "$AGENTS_BASE/blind-ppo-easy-128-lowent" \
    --profiles body-ood-medium \
    --episodes $EPISODES \
    --trajectory-subdir trajectories-ood-body-medium \
    --force \
    > "$LOGDIR/blind-body.log" 2>&1 &
PID1=$!
echo "  [PID $PID1] blind × body-ood-medium"

python scripts/agents/run_eval_pipeline.py \
    --agents-dir "$AGENTS_BASE/labeled-ppo-easy-128-lowent" \
    --profiles body-ood-medium \
    --episodes $EPISODES \
    --trajectory-subdir trajectories-ood-body-medium \
    --force \
    > "$LOGDIR/labeled-body.log" 2>&1 &
PID2=$!
echo "  [PID $PID2] labeled × body-ood-medium"

python scripts/agents/run_eval_pipeline.py \
    --agents-dir "$AGENTS_BASE/blind-ppo-easy-128-lowent" \
    --profiles world-ood-medium \
    --episodes $EPISODES \
    --trajectory-subdir trajectories-ood-world-medium \
    --force \
    > "$LOGDIR/blind-world.log" 2>&1 &
PID3=$!
echo "  [PID $PID3] blind × world-ood-medium"

python scripts/agents/run_eval_pipeline.py \
    --agents-dir "$AGENTS_BASE/labeled-ppo-easy-128-lowent" \
    --profiles world-ood-medium \
    --episodes $EPISODES \
    --trajectory-subdir trajectories-ood-world-medium \
    --force \
    > "$LOGDIR/labeled-world.log" 2>&1 &
PID4=$!
echo "  [PID $PID4] labeled × world-ood-medium"

echo ""
echo "Waiting for all 4 batches to finish..."
wait $PID1 && echo "  blind × body-ood-medium: DONE" || echo "  blind × body-ood-medium: FAILED (check $LOGDIR/blind-body.log)"
wait $PID2 && echo "  labeled × body-ood-medium: DONE" || echo "  labeled × body-ood-medium: FAILED (check $LOGDIR/labeled-body.log)"
wait $PID3 && echo "  blind × world-ood-medium: DONE" || echo "  blind × world-ood-medium: FAILED (check $LOGDIR/blind-world.log)"
wait $PID4 && echo "  labeled × world-ood-medium: DONE" || echo "  labeled × world-ood-medium: FAILED (check $LOGDIR/labeled-world.log)"

echo ""
echo "=== All batches complete ==="
echo "Next: run cross-config analysis:"
echo "  python scripts/analysis/compare_configs.py analysis-manifests/comparison/ood-generalization-body-medium.yaml"
echo "  python scripts/analysis/compare_configs.py analysis-manifests/comparison/ood-generalization-world-medium.yaml"
