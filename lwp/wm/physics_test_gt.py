# lunar_lander/src/wm/physics_test_gt.py
"""Box2D ground truth trajectory generation for physics unit tests.

Generates ground truth state trajectories by running controlled action
sequences through Box2D. Two initialization modes:

1. Teleport+settle: Set Box2D body state to the episode's branch-point state,
   run 5 zero-action settle steps (to dissipate joint constraint transients),
   then apply controlled actions. Fast, but settle steps introduce drift.

2. Replay: Create a fresh env, replay the episode's actual actions from
   step 0 to the branch point, then switch to controlled actions. Clean
   joint state, exact match to the episode. Slower.

Both modes return a (n_steps+1, 15) state trajectory: initial state at
the branch point + one state per controlled action step.

Design spec: traitful-docs/docs/research/.../specs/physics-unit-tests.md
"""
from __future__ import annotations

import json

import numpy as np

from parametric_lunar_lander.env import (
    ParameterizedLunarLander,
    FPS,
    SCALE,
    VIEWPORT_W,
    VIEWPORT_H,
    LEG_DOWN,
)
from parametric_lunar_lander.physics_config import LunarLanderPhysicsConfig


# Number of zero-action steps after teleporting the lander body.
# Teleportation violates joint constraints (legs are attached via revolute
# joints at positions relative to the original body pose). Box2D corrects
# these with large corrective forces in subsequent steps. 5 steps is
# empirically sufficient to dissipate these transients (calibration.py
# uses 1 step for angular-only override; full teleport needs more).
SETTLE_STEPS = 5


def _extract_physics_config(episode: dict) -> LunarLanderPhysicsConfig:
    """Get physics config from episode metadata or state vector.

    Episodes have physics params in state dims [8:15]. If metadata_json
    is available, prefer that (more readable). Otherwise, read from state.
    """
    if "metadata_json" in episode:
        meta = episode["metadata_json"]
        if isinstance(meta, bytes):
            meta = meta.decode()
        if isinstance(meta, str):
            meta = json.loads(meta)
        if "physics_config" in meta:
            return LunarLanderPhysicsConfig.from_dict(meta["physics_config"])

    # Fall back to reading from state vector.
    physics_vals = episode["states"][0, 8:15]
    return LunarLanderPhysicsConfig(
        gravity=float(physics_vals[0]),
        main_engine_power=float(physics_vals[1]),
        side_engine_power=float(physics_vals[2]),
        lander_density=float(physics_vals[3]),
        angular_damping=float(physics_vals[4]),
        wind_power=float(physics_vals[5]),
        turbulence_power=float(physics_vals[6]),
    )


def _denormalize_state(state_obs: np.ndarray) -> dict:
    """Convert normalized observation vector back to Box2D world coordinates.

    The observation vector uses normalized coordinates (see env.py lines 636-648).
    We need world coordinates to set Box2D body properties.

    Returns dict with: pos_x, pos_y, vel_x, vel_y, angle, angular_vel.
    """
    # Position: obs = (world - center) / half_viewport
    # world = obs * half_viewport + center
    half_w = VIEWPORT_W / SCALE / 2
    half_h = VIEWPORT_H / SCALE / 2
    helipad_y = VIEWPORT_H / SCALE / 4  # H/4 in world coords

    pos_x = state_obs[0] * half_w + half_w
    pos_y = state_obs[1] * half_h + (helipad_y + LEG_DOWN / SCALE)

    # Velocity: obs = world_vel * half_viewport / FPS
    # world_vel = obs * FPS / half_viewport
    vel_x = state_obs[2] * FPS / half_w
    vel_y = state_obs[3] * FPS / half_h

    # Angle: stored directly in radians.
    angle = float(state_obs[4])

    # Angular velocity: obs = 20 * omega / FPS
    # omega = obs * FPS / 20
    angular_vel = state_obs[5] * FPS / 20.0

    return {
        "pos_x": float(pos_x),
        "pos_y": float(pos_y),
        "vel_x": float(vel_x),
        "vel_y": float(vel_y),
        "angle": float(angle),
        "angular_vel": float(angular_vel),
    }


def _teleport_env(env: ParameterizedLunarLander, world_state: dict):
    """Set Box2D lander body + legs to the given world-coordinate state.

    Moves the legs by the same position delta as the lander body so that
    revolute joint constraints stay valid. Without this, the joints explode
    and the settle steps introduce spurious forces.
    """
    old_x, old_y = env.lander.position[0], env.lander.position[1]
    env.lander.position = (world_state["pos_x"], world_state["pos_y"])
    env.lander.linearVelocity = (world_state["vel_x"], world_state["vel_y"])
    env.lander.angle = world_state["angle"]
    env.lander.angularVelocity = world_state["angular_vel"]

    # Move legs by the same delta so joint constraints stay valid.
    dx = world_state["pos_x"] - old_x
    dy = world_state["pos_y"] - old_y
    if dx != 0 or dy != 0:
        for leg in env.legs:
            leg.position = (leg.position[0] + dx, leg.position[1] + dy)

    # Wake up all bodies (reset may have put them to sleep).
    env.lander.awake = True
    for leg in env.legs:
        leg.awake = True


def _step_and_record(
    env: ParameterizedLunarLander,
    actions: np.ndarray,
    save_frames: bool = False,
    subsample: int = 1,
) -> dict:
    """Step the env through actions and record observations + optional frames.

    Each action in ``actions`` is applied for ``subsample`` Box2D steps.
    Only the state after the last substep is recorded, so the output has
    len(actions)+1 states regardless of subsample. This lets the model
    (trained at subsampled FPS) be compared directly to GT at the same
    temporal resolution.

    Stops at env termination (landing, crash, out of bounds) and records the
    termination step. No padding — the returned trajectory has only the steps
    that actually ran, so callers know exactly how much valid data there is.

    Returns dict with:
      - "states": (actual_steps + 1, 15) array — initial obs + one per step.
      - "terminated_at": int or None — step index where env terminated.
      - "rgb_frames": (actual_steps + 1, H, W, 3) uint8 — only if save_frames=True.
    """
    states = [env._last_obs.copy()]
    frames = []
    if save_frames:
        frames.append(env.render())  # Initial frame

    terminated_at = None
    for t in range(len(actions)):
        # Step Box2D `subsample` times with the same action.
        # Record only the final substep state.
        obs = None
        for _ in range(subsample):
            obs, _, terminated, _, _ = env.step(actions[t])
            if terminated:
                break
        states.append(obs.copy())
        if save_frames:
            frames.append(env.render())
        if terminated:
            terminated_at = t + 1  # +1 because step 0 is initial state
            break

    result = {
        "states": np.stack(states).astype(np.float32),
        "terminated_at": terminated_at,
    }
    if save_frames:
        result["rgb_frames"] = np.stack(frames).astype(np.uint8)
    return result


def _extract_seed(episode: dict, fallback: int = 42) -> int:
    """Get the episode's collection seed from metadata.

    The seed determines terrain generation in Box2D. Using the correct seed
    is critical for replay mode — wrong seed = wrong terrain = replay state
    diverges from the recorded episode.
    """
    if "metadata_json" in episode:
        meta = episode["metadata_json"]
        if isinstance(meta, bytes):
            meta = meta.decode()
        if isinstance(meta, str):
            meta = json.loads(meta)
        if "seed" in meta:
            return int(meta["seed"])
    return fallback


def generate_gt_trajectory(
    episode: dict,
    branch_point: int,
    controlled_actions: np.ndarray,
    mode: str = "replay",
    seed: int | None = None,
    save_frames: bool = False,
    subsample: int = 1,
) -> dict:
    """Generate ground truth state trajectory from Box2D.

    Args:
        episode: Dict with "states" (T+1, 15) and "actions" (T, 2).
        branch_point: Timestep index to branch at (0-indexed).
        controlled_actions: (n_steps, 2) action sequence to apply after branching.
            These are at model FPS (i.e., already accounting for subsampling).
        mode: "teleport" (fast, settle transients) or "replay" (clean, replay
            from start).
        seed: Random seed for env reset. If None, extracted from episode
            metadata (recommended — ensures correct terrain for replay).
            Falls back to 42 if metadata unavailable.
        save_frames: If True, capture RGB frames via env.render() each step.
            Requires env created with render_mode="rgb_array". Generic
            capability — works for any Box2D stepping, not just physics tests.
        subsample: Number of Box2D steps per model step. subsample=5 means
            each action is applied for 5 env steps (50 FPS -> 10 FPS).

    Returns:
        Dict with:
          - "states": (n_steps + 1, 15) ground truth trajectory at model FPS.
          - "rgb_frames": (n_steps + 1, H, W, 3) uint8 — only if save_frames=True.
    """
    if seed is None:
        seed = _extract_seed(episode)
    physics_config = _extract_physics_config(episode)

    if mode == "teleport":
        return _generate_teleport(
            episode, branch_point, controlled_actions, physics_config, seed,
            save_frames=save_frames, subsample=subsample,
        )
    elif mode == "replay":
        return _generate_replay(
            episode, branch_point, controlled_actions, physics_config, seed,
            save_frames=save_frames, subsample=subsample,
        )
    else:
        raise ValueError(f"Unknown GT mode: {mode}. Use 'teleport' or 'replay'.")


def _generate_teleport(
    episode: dict,
    branch_point: int,
    controlled_actions: np.ndarray,
    physics_config: LunarLanderPhysicsConfig,
    seed: int,
    save_frames: bool = False,
    subsample: int = 1,
) -> dict:
    """Teleport+settle ground truth generation.

    1. Create env, reset.
    2. Teleport lander body to the episode's branch-point state.
    3. Run SETTLE_STEPS zero-action steps to dissipate joint transients.
    4. Record initial state, then apply controlled actions.
    """
    render_mode = "rgb_array" if save_frames else None
    env = ParameterizedLunarLander(render_mode=render_mode, physics_config=physics_config)
    env.reset(seed=seed)

    # Teleport to branch-point state.
    branch_state = episode["states"][branch_point]
    world_state = _denormalize_state(branch_state)
    _teleport_env(env, world_state)

    # Settle: zero-action steps to dissipate joint constraint transients.
    zero_action = np.array([0.0, 0.0], dtype=np.float32)
    for _ in range(SETTLE_STEPS):
        env.step(zero_action)

    # Now record the post-settle state and apply controlled actions.
    result = _step_and_record(env, controlled_actions, save_frames=save_frames,
                              subsample=subsample)
    env.close()
    return result


def _generate_replay(
    episode: dict,
    branch_point: int,
    controlled_actions: np.ndarray,
    physics_config: LunarLanderPhysicsConfig,
    seed: int,
    save_frames: bool = False,
    subsample: int = 1,
) -> dict:
    """Replay ground truth generation.

    1. Create env, reset with same seed as collection.
    2. Replay the episode's actual actions from step 0 to branch_point.
    3. Record initial state, then apply controlled actions.

    This gives clean joint state — no teleportation artifacts. The state
    at the branch point should exactly match the episode's recorded state
    (Box2D is deterministic for a given seed + action sequence).

    Note: render_mode="rgb_array" is set from the start even though we only
    capture frames during the controlled portion. The overhead during replay
    is negligible (render() is only called in _step_and_record).
    """
    render_mode = "rgb_array" if save_frames else None
    env = ParameterizedLunarLander(render_mode=render_mode, physics_config=physics_config)
    env.reset(seed=seed)

    # Replay episode actions up to branch point.
    for t in range(branch_point):
        obs, _, terminated, _, _ = env.step(episode["actions"][t])
        if terminated:
            env.close()
            raise RuntimeError(
                f"Episode terminated at step {t} before reaching "
                f"branch point {branch_point}."
            )

    # Now apply controlled actions and record.
    result = _step_and_record(env, controlled_actions, save_frames=save_frames,
                              subsample=subsample)
    env.close()
    return result


def create_reset_episode(
    n_padding: int = 20,
    physics_config: LunarLanderPhysicsConfig | None = None,
    seed: int = 42,
) -> dict:
    """Create a synthetic episode from the env's reset state.

    For physics unit tests, we want a clean initial condition: lander upright,
    centered, zero velocity. This function resets the env and creates a
    synthetic episode by repeating the reset state with zero actions for
    n_padding steps. The branch point should be set to n_padding so models
    with context windows have enough (neutral) history.

    The episode length is just the padding — reset episodes skip the
    episode-length check in run_physics_tests(). Actual eval duration is
    determined by the maneuver definition and GT termination/truncation.

    Args:
        n_padding: Number of context padding steps (must be >= model's context_k).
        physics_config: Physics config for the env. Defaults to gym-default.
        seed: Random seed for env reset (determines terrain).

    Returns:
        Dict with:
          - "states": (n_padding + 1, 15) — reset state repeated.
          - "actions": (n_padding, 2) — zero actions.
          - "metadata_json": JSON string with seed and physics_config.
          - "is_reset_episode": True — marker for downstream code.
    """
    if physics_config is None:
        physics_config = LunarLanderPhysicsConfig()

    env = ParameterizedLunarLander(render_mode=None, physics_config=physics_config)
    obs, _ = env.reset(seed=seed)
    env.close()

    # Repeat the reset state for padding. All transitions are identity.
    states = np.tile(obs, (n_padding + 1, 1)).astype(np.float32)
    actions = np.zeros((n_padding, 2), dtype=np.float32)

    metadata = {
        "seed": seed,
        "physics_config": physics_config.to_dict(),
        "is_reset_episode": True,
    }

    return {
        "states": states,
        "actions": actions,
        "metadata_json": json.dumps(metadata),
        "is_reset_episode": True,
    }


def render_states_to_frames(
    states: np.ndarray,
    physics_config: LunarLanderPhysicsConfig | None = None,
    seed: int = 42,
) -> np.ndarray:
    """Render predicted state vectors as RGB frames via Box2D teleportation.

    For each state in the trajectory, teleports the Box2D lander body to
    that position/velocity/angle and calls env.render(). This is purely for
    visualization — no physics stepping happens. Joint positions won't be
    perfectly accurate (teleportation artifacts), but the lander body
    position and angle are correct.

    Generic utility — works for any state trajectory (physics tests, eval
    rollouts, etc.). Render model predictions the same way GT is rendered.

    Args:
        states: (T, state_dim) state trajectory in normalized observation coords.
        physics_config: Physics config for the env. Defaults to gym-default.
        seed: Random seed for env reset (for deterministic terrain).

    Returns:
        rgb_frames: (T, H, W, 3) uint8 array.
    """
    if physics_config is None:
        physics_config = LunarLanderPhysicsConfig()

    env = ParameterizedLunarLander(
        render_mode="rgb_array", physics_config=physics_config,
    )
    env.reset(seed=seed)

    frames = []
    for t in range(len(states)):
        world_state = _denormalize_state(states[t])
        _teleport_env(env, world_state)
        frames.append(env.render())

    env.close()
    return np.stack(frames).astype(np.uint8)
