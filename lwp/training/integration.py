"""Shared hybrid state update for 3D force-target world models.

All rollout, loss, and eval code routes through hybrid_state_update()
when delta_dim != state_dim. This is the single source of truth for
how predicted force deltas are combined with the current state.

Integration constants derived from parametric-lunar-lander env
observation scaling (env.py:68,93-94,636-644). See E2-25 plan Q1
for full derivation.
"""
import torch

# Observation-space integration constants.
# Derived from: VIEWPORT_W=600, VIEWPORT_H=400, SCALE=30, FPS=50.
#   IC_X     = 1 / (VIEWPORT_W/SCALE/2)^2 = 1/100
#   IC_Y     = 1 / (VIEWPORT_H/SCALE/2)^2 = 1/44.444
#   IC_ANGLE = 1/20  (from 20/FPS angular velocity scaling)
INTEGRATION_CONSTANTS = {
    "IC_X": 0.01,
    "IC_Y": 0.0225,
    "IC_ANGLE": 0.05,
}

# State indices: [x=0, y=1, vx=2, vy=3, angle=4, ang_vel=5]
# Force target indices within the 6D state.
FORCE_TARGET_INDICES = [2, 3, 5]  # vx, vy, ang_vel


def hybrid_state_update(
    state: torch.Tensor,
    delta: torch.Tensor,
    subsample: int = 1,
) -> torch.Tensor:
    """Compute next 6D state from current state and predicted delta.

    Supports two modes based on delta dimension:
    - delta_dim == 6: plain addition (backward compatible with existing 6D models)
    - delta_dim == 3: hybrid integration — update velocities from delta,
      then integrate positions analytically using post-update velocities
      (matches Box2D semi-implicit Euler convention)

    Args:
        state: [batch, 6] current raw state [x, y, vx, vy, angle, ang_vel]
        delta: [batch, 3] force delta [Δvx, Δvy, Δang_vel]
               or [batch, 6] full delta (backward compatible)
        subsample: data subsample factor (default 1). Scales position
                   integration constants.

    Returns:
        next_state: [batch, 6] next raw state
    """
    delta_dim = delta.shape[-1]

    if delta_dim == state.shape[-1]:
        # Backward compatible: 6D delta, plain addition.
        return state + delta

    # 3D force-target hybrid integration.
    ic = INTEGRATION_CONSTANTS
    next_state = torch.empty_like(state)

    # 1. Update velocities from force deltas.
    next_state[:, 2] = state[:, 2] + delta[:, 0]  # vx
    next_state[:, 3] = state[:, 3] + delta[:, 1]  # vy
    next_state[:, 5] = state[:, 5] + delta[:, 2]  # ang_vel

    # 2. Integrate positions using POST-update velocities.
    next_state[:, 0] = state[:, 0] + next_state[:, 2] * ic["IC_X"] * subsample
    next_state[:, 1] = state[:, 1] + next_state[:, 3] * ic["IC_Y"] * subsample
    next_state[:, 4] = state[:, 4] + next_state[:, 5] * ic["IC_ANGLE"] * subsample

    return next_state
