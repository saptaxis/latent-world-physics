"""Env wrappers for data collection.

PostLandingWrapper: suppresses crash/landing termination so episodes can
  continue after the lander touches down. OOB still terminates.

override_initial_state(): sets Box2D body state after env.reset() for
  fresh-reset primitive episodes with custom initial conditions.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np


class PostLandingWrapper(gym.Wrapper):
    """Suppress crash/landing termination, keep OOB termination.

    After the wrapped env returns terminated=True, this wrapper checks
    whether it's an OOB termination (|x| >= 1.0 in state vector). If so,
    it passes terminated=True through. Otherwise, it records the event
    in self.termination_event and returns terminated=False so the episode
    continues.

    The Box2D simulation continues naturally — the lander stays on the
    ground with proper contact forces. This enables collecting ground
    contact dynamics data.

    Attributes:
        termination_event: Dict with {"type": str, "step": int} recording
            the first suppressed termination. None if no termination has
            been suppressed. Reset on env.reset().
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.termination_event: dict | None = None
        self._step_count: int = 0

    def reset(self, **kwargs):
        self.termination_event = None
        self._step_count = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._step_count += 1

        if terminated:
            # Check if it's OOB — let those through.
            # obs[0] is normalized x position; |x| >= 1.0 means OOB.
            if abs(obs[0]) >= 1.0:
                return obs, reward, True, truncated, info

            # Suppress landing/crash termination.
            # Only record the first termination event — subsequent contacts
            # (e.g., bouncing after initial crash) are ignored.
            if self.termination_event is None:
                outcome = info.get("outcome", "unknown")
                self.termination_event = {
                    "type": outcome,
                    "step": self._step_count,
                }

            # Continue the episode — don't pass terminated=True.
            # Reset the game_over flag on the unwrapped env so Box2D
            # doesn't keep triggering termination on subsequent steps.
            # The ContactDetector sets game_over=True on body contact,
            # which the env's step() checks each frame. Clearing it
            # lets the simulation proceed normally with the lander on
            # the ground.
            self.env.unwrapped.game_over = False
            return obs, reward, False, truncated, info

        return obs, reward, terminated, truncated, info


# ---------------------------------------------------------------------------
# Initial-state override for fresh-reset primitive episodes
# ---------------------------------------------------------------------------

# Coordinate conversion constants matching env.py.
VIEWPORT_W = 600
VIEWPORT_H = 400
SCALE = 30.0
LEG_DOWN = 18  # Leg attachment vertical offset (pixels), same as env.py.
FPS = 50


def override_initial_state(
    env: gym.Env,
    x: float | None = None,
    y: float | None = None,
    vx: float | None = None,
    vy: float | None = None,
    angle: float | None = None,
    angular_vel: float | None = None,
) -> None:
    """Set Box2D body state after env.reset().

    Call immediately after env.reset() and before any env.step(). Sets the
    lander's position, velocity, and angle in the Box2D world. Joints
    initialize fresh from reset(), so the override doesn't cause
    inconsistencies.

    All parameters use the same coordinate system as the state vector
    (normalized coordinates). Conversion to Box2D world coords happens
    internally.

    Coordinate conversions (derived from env.py step() observation code):
      - obs_x = (world_x - W/2) / (W/2)
        → world_x = obs_x * (W/2) + W/2
      - obs_y = (world_y - (helipad_y + LEG_DOWN/SCALE)) / (H/2)
        → world_y = obs_y * (H/2) + helipad_y + LEG_DOWN/SCALE
      - obs_vx = world_vx * (W/2) / FPS
        → world_vx = obs_vx * FPS / (W/2)
      - obs_vy = world_vy * (H/2) / FPS
        → world_vy = obs_vy * FPS / (H/2)
      - obs_angle = world_angle (no conversion)
      - obs_angvel = world_angvel * 20.0 / FPS
        → world_angvel = obs_angvel * FPS / 20.0

    Args:
        env: ParameterizedLunarLander (or wrapped). Will unwrap to find the
            base env.
        x: Normalized x position. State vector convention: 0 = center.
        y: Normalized y position. State vector convention: ~1.0 = top.
        vx: Normalized x velocity (state-vector scale).
        vy: Normalized y velocity (state-vector scale).
        angle: Angle in radians. Positive = counterclockwise.
        angular_vel: Angular velocity (state-vector scale, i.e. obs[5] value).

    Raises:
        ValueError: If the resulting position is below the terrain.
    """
    base_env = env.unwrapped
    lander = base_env.lander

    # Half-viewport dimensions in world coords — used for normalization.
    W = VIEWPORT_W / SCALE   # 20.0 world units total width
    H = VIEWPORT_H / SCALE   # ~13.33 world units total height

    # Record pre-override position so we can compute the delta for legs.
    old_x, old_y = lander.position[0], lander.position[1]

    if x is not None:
        # Invert: obs_x = (world_x - W/2) / (W/2)
        world_x = x * (W / 2) + W / 2
        lander.position = (world_x, lander.position[1])

    if y is not None:
        # Invert: obs_y = (world_y - (helipad_y + LEG_DOWN/SCALE)) / (H/2)
        world_y = y * (H / 2) + base_env.helipad_y + LEG_DOWN / SCALE
        # Check terrain height at the current x position.
        cur_x = lander.position[0]
        terrain_y = _get_terrain_height(base_env, cur_x)
        if world_y < terrain_y + 0.5:  # 0.5m margin for lander size
            raise ValueError(
                f"Requested y={y} (world_y={world_y:.2f}) is below terrain "
                f"height {terrain_y:.2f} at x={cur_x:.2f}"
            )
        lander.position = (lander.position[0], world_y)

    # Move legs by the same position delta so joint constraints stay valid.
    # Legs are separate Box2D bodies connected to the lander via revolute
    # joints — if we move the lander without the legs, the joints explode.
    dx = lander.position[0] - old_x
    dy = lander.position[1] - old_y
    if dx != 0 or dy != 0:
        for leg in base_env.legs:
            leg.position = (leg.position[0] + dx, leg.position[1] + dy)

    if vx is not None:
        # Invert: obs_vx = world_vx * (W/2) / FPS
        world_vx = vx * FPS / (W / 2)
        lander.linearVelocity = (world_vx, lander.linearVelocity[1])

    if vy is not None:
        # Invert: obs_vy = world_vy * (H/2) / FPS
        world_vy = vy * FPS / (H / 2)
        lander.linearVelocity = (lander.linearVelocity[0], world_vy)

    if angle is not None:
        # obs_angle = world_angle (direct, no conversion needed).
        lander.angle = angle

    if angular_vel is not None:
        # Invert: obs_angvel = world_angvel * 20.0 / FPS
        lander.angularVelocity = angular_vel * FPS / 20.0

    # Wake up all bodies (reset may have put them to sleep).
    lander.awake = True
    for leg in base_env.legs:
        leg.awake = True


def _get_terrain_height(env, world_x: float) -> float:
    """Query terrain height at a world x coordinate.

    Linearly interpolates between terrain segment endpoints. Falls back
    to 0.0 if no terrain data is available (shouldn't happen after reset).

    Args:
        env: Unwrapped ParameterizedLunarLander with .terrain_segments set.
        world_x: X coordinate in Box2D world space.

    Returns:
        Terrain surface height in world coords at the given x.
    """
    if not hasattr(env, 'terrain_segments') or not env.terrain_segments:
        return 0.0  # Fallback if no terrain data

    for seg in env.terrain_segments:
        x1, y1, x2, y2 = seg
        if x1 <= world_x <= x2:
            t = (world_x - x1) / max(x2 - x1, 1e-6)
            return y1 + t * (y2 - y1)

    # world_x outside terrain bounds — return minimum terrain height as
    # a conservative fallback.
    all_y = [seg[1] for seg in env.terrain_segments] + [seg[3] for seg in env.terrain_segments]
    return min(all_y) if all_y else 0.0
