#!/usr/bin/env bash
# OOD evals for body-only and world-only TRAINED agents.
#
# Vehicle-only easy-trained agents → vehicle-only-medium (body range widening)
# Physics-only easy-trained agents → physics-only-medium (world range widening, wind introduced)
#
# Tests whether the training-time behavioral split (Section 2) predicts
# OOD robustness within each condition:
#   - Vehicle-only: large blind/labeled gap at training → expect blind advantage OOD
#   - Physics-only: small blind/labeled gap at training → expect similar OOD performance
#
# All 4 batches run in parallel. Each processes 10 seeds sequentially.
#
# After this script, run cross-config analysis:
#   python scripts/analysis/compare_configs.py \
#       --manifest analysis-manifests/comparison/ood-vehicle-only-medium.yaml
#   python scripts/analysis/compare_configs.py \
#       --manifest analysis-manifests/comparison/ood-physics-only-medium.yaml

set -euo pipefail

AGENTS_BASE="/media/hdd1/physics-priors-latent-space/lunar-lander-networks/rl_agents/parametric-vs-behavioral"
EPISODES=100
LOGDIR="$HOME/vsr-tmp/ood-bodyworld-agents-logs"
mkdir -p "$LOGDIR"

echo "Starting 4 parallel OOD eval batches for body-only and world-only agents..."
echo "Logs in $LOGDIR/"

python scripts/agents/run_eval_pipeline.py \
    --agents-dir "$AGENTS_BASE/vehicle-only/blind-ppo-easy-128-lowent" \
    --profiles vehicle-only-medium \
    --episodes $EPISODES \
    --trajectory-subdir trajectories-ood-medium \
    --force \
    > "$LOGDIR/vehicle-blind.log" 2>&1 &
PID1=$!
echo "  [PID $PID1] vehicle-only blind → vehicle-only-medium"

python scripts/agents/run_eval_pipeline.py \
    --agents-dir "$AGENTS_BASE/vehicle-only/labeled-ppo-easy-128-lowent" \
    --profiles vehicle-only-medium \
    --episodes $EPISODES \
    --trajectory-subdir trajectories-ood-medium \
    --force \
    > "$LOGDIR/vehicle-labeled.log" 2>&1 &
PID2=$!
echo "  [PID $PID2] vehicle-only labeled → vehicle-only-medium"

python scripts/agents/run_eval_pipeline.py \
    --agents-dir "$AGENTS_BASE/physics-only/blind-ppo-easy-128-lowent" \
    --profiles physics-only-medium \
    --episodes $EPISODES \
    --trajectory-subdir trajectories-ood-medium \
    --force \
    > "$LOGDIR/physics-blind.log" 2>&1 &
PID3=$!
echo "  [PID $PID3] physics-only blind → physics-only-medium"

python scripts/agents/run_eval_pipeline.py \
    --agents-dir "$AGENTS_BASE/physics-only/labeled-ppo-easy-128-lowent" \
    --profiles physics-only-medium \
    --episodes $EPISODES \
    --trajectory-subdir trajectories-ood-medium \
    --force \
    > "$LOGDIR/physics-labeled.log" 2>&1 &
PID4=$!
echo "  [PID $PID4] physics-only labeled → physics-only-medium"

echo ""
echo "Waiting for all 4 batches to finish..."
wait $PID1 && echo "  vehicle-only blind: DONE" || echo "  vehicle-only blind: FAILED (check $LOGDIR/vehicle-blind.log)"
wait $PID2 && echo "  vehicle-only labeled: DONE" || echo "  vehicle-only labeled: FAILED (check $LOGDIR/vehicle-labeled.log)"
wait $PID3 && echo "  physics-only blind: DONE" || echo "  physics-only blind: FAILED (check $LOGDIR/physics-blind.log)"
wait $PID4 && echo "  physics-only labeled: DONE" || echo "  physics-only labeled: FAILED (check $LOGDIR/physics-labeled.log)"

echo ""
echo "=== All batches complete ==="
echo "Next: run cross-config analysis:"
echo "  python scripts/analysis/compare_configs.py --manifest analysis-manifests/comparison/ood-vehicle-only-medium.yaml"
echo "  python scripts/analysis/compare_configs.py --manifest analysis-manifests/comparison/ood-physics-only-medium.yaml"
