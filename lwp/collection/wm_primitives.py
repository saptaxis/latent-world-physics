"""Action generators for primitive data collection maneuvers.

Each maneuver type maps to a function that produces an (n_steps, 2)
action array. Constant maneuvers fill with fixed values. Patterned
maneuvers (impulse, reversal) use per-episode random timing parameters
drawn from the config ranges.

Hover is special: it computes the equilibrium thrust from the physics config.
"""

from __future__ import annotations

import numpy as np

from parametric_lunar_lander.env import ParameterizedLunarLander
from parametric_lunar_lander.physics_config import LunarLanderPhysicsConfig
from lwp.collection.wm_collection_config import ManeuverConfig


def generate_actions(
    config: ManeuverConfig,
    n_steps: int,
    rng: np.random.Generator,
    physics_config=None,
) -> tuple[np.ndarray, dict]:
    """Generate action sequence for a primitive maneuver.

    Args:
        config: ManeuverConfig specifying maneuver type and parameters.
        n_steps: Number of timesteps to generate.
        rng: NumPy random generator for sampling timing parameters.
        physics_config: LunarLanderPhysicsConfig, required for hover maneuver.

    Returns:
        Tuple of:
        - (n_steps, 2) float32 array. Column 0 = main, column 1 = side.
        - dict of sampled parameters actually used (for metadata recording).
    """
    gen = _GENERATORS.get(config.type)
    if gen is None:
        raise ValueError(f"No action generator for maneuver type '{config.type}'")
    return gen(config, n_steps, rng, physics_config)


def _resolve_thrust(val: float | tuple[float, float], rng: np.random.Generator) -> float:
    """Resolve a thrust value: scalar pass-through, range sampled per episode."""
    if isinstance(val, tuple):
        return float(rng.uniform(val[0], val[1]))
    return float(val)


def _constant(
    config: ManeuverConfig, n_steps: int, rng: np.random.Generator, physics_config,
) -> tuple[np.ndarray, dict]:
    """Constant thrust at config.main / config.side for all steps.

    When main or side is a (lo, hi) range, samples a value per episode.
    """
    main_val = _resolve_thrust(config.main, rng)
    side_val = _resolve_thrust(config.side, rng)
    actions = np.zeros((n_steps, 2), dtype=np.float32)
    actions[:, 0] = main_val
    actions[:, 1] = side_val
    return actions, {"main": main_val, "side": side_val}


def _zero(
    config: ManeuverConfig, n_steps: int, rng: np.random.Generator, physics_config,
) -> tuple[np.ndarray, dict]:
    """Zero action for all steps."""
    return np.zeros((n_steps, 2), dtype=np.float32), {"main": 0.0, "side": 0.0}


def _sample_int_range(
    range_tuple: tuple[int, int] | None, rng: np.random.Generator,
) -> int:
    """Sample an integer from a (min, max) inclusive range."""
    if range_tuple is None:
        return 0
    lo, hi = range_tuple
    if lo == hi:
        return lo
    return int(rng.integers(lo, hi + 1))


def _impulse(
    config: ManeuverConfig, n_steps: int, rng: np.random.Generator, physics_config,
) -> tuple[np.ndarray, dict]:
    """Alternating on/off cycles with per-episode random timing.

    Pattern: [on for pulse_dur] [off for gap_dur] x n_cycles, then zero.
    Timing parameters are sampled from config ranges per episode.
    """
    pulse_dur = _sample_int_range(config.pulse_duration, rng)
    gap_dur = _sample_int_range(config.gap_duration, rng)
    n_cyc = _sample_int_range(config.n_cycles, rng)

    col = 0 if config.channel == "main" else 1

    actions = np.zeros((n_steps, 2), dtype=np.float32)
    t = 0
    for _ in range(n_cyc):
        end = min(t + pulse_dur, n_steps)
        actions[t:end, col] = config.thrust_level
        t = end
        t = min(t + gap_dur, n_steps)

    return actions, {
        "channel": config.channel,
        "thrust_level": config.thrust_level,
        "pulse_duration": pulse_dur,
        "gap_duration": gap_dur,
        "n_cycles": n_cyc,
    }


def _direction_reversal(
    config: ManeuverConfig, n_steps: int, rng: np.random.Generator, physics_config,
) -> tuple[np.ndarray, dict]:
    """Thrust in one direction, optional coast gap, then opposite direction."""
    first_dur = _sample_int_range(config.first_duration, rng)
    gap_dur = _sample_int_range(config.gap_duration, rng)
    second_dur = _sample_int_range(config.second_duration, rng)

    col = 0 if config.channel == "main" else 1

    actions = np.zeros((n_steps, 2), dtype=np.float32)
    t = 0

    end = min(t + first_dur, n_steps)
    actions[t:end, col] = config.thrust_level
    t = end

    t = min(t + gap_dur, n_steps)

    end = min(t + second_dur, n_steps)
    actions[t:end, col] = -config.thrust_level
    t = end

    return actions, {
        "channel": config.channel,
        "thrust_level": config.thrust_level,
        "first_duration": first_dur,
        "gap_duration": gap_dur,
        "second_duration": second_dur,
    }


def _hover(
    config: ManeuverConfig, n_steps: int, rng: np.random.Generator, physics_config,
) -> tuple[np.ndarray, dict]:
    """Constant thrust computed to balance gravity exactly."""
    if physics_config is None:
        raise ValueError("hover maneuver requires physics_config to compute equilibrium thrust")

    from parametric_lunar_lander.physics_utils import compute_hover_thrust

    action_value, m_power = compute_hover_thrust(physics_config)

    actions = np.zeros((n_steps, 2), dtype=np.float32)
    actions[:, 0] = action_value
    return actions, {"main": action_value, "side": 0.0, "hover_m_power": m_power}


def replay_to_branch_point(
    episode: dict,
    branch_point: int,
    render_mode: str | None = None,
    allow_post_landing: bool = False,
) -> "ParameterizedLunarLander":
    """Replay a source episode up to branch_point, return the live env.

    Creates a new env with the same seed and physics config as the source
    episode, replays the recorded actions step by step, and returns the env
    at the branch point state. The caller can then step the env with
    controlled primitive actions.

    This is used by collect_primitive() for "replay" and "replay_to_landing"
    start modes. The idea: we have a previously-recorded episode and want to
    branch off at a specific timestep to collect new primitive maneuver data
    from that mid-episode state (e.g., mid-air, near landing, post-bounce).

    Determinism guarantee: because Box2D is deterministic given the same seed
    and action sequence, replaying produces the exact same states as the
    original episode. We verify this implicitly — the caller can compare
    the env state to the source episode's state at branch_point.

    Args:
        episode: Dict from load_episode() with "states", "actions", "metadata".
        branch_point: Timestep to branch at (1-indexed minimum). The env
            will have been stepped branch_point times. Must be >= 1 because
            we need at least one replay step (branch_point=0 would mean
            "just reset", which should use fresh-start instead).
        render_mode: Optional render mode for the env (e.g., "rgb_array").
        allow_post_landing: If True, wrap with PostLandingWrapper so the
            env continues after landing instead of terminating. Useful for
            collecting post-landing primitives like ground thrust sweeps.

    Returns:
        Live ParameterizedLunarLander (or PostLandingWrapper) at the branch
        point state. Caller is responsible for env.close().

    Raises:
        ValueError: If branch_point < 1 or > episode length.
        RuntimeError: If episode terminates before reaching branch_point
            (and allow_post_landing is False).
    """
    if branch_point < 1:
        raise ValueError("branch_point must be >= 1 (need at least 1 replay step)")

    n_steps = len(episode["actions"])
    if branch_point > n_steps:
        raise ValueError(
            f"branch_point {branch_point} exceeds episode length {n_steps}"
        )

    # Extract seed and physics config from episode metadata.
    # These are the exact parameters used when the source episode was collected,
    # so replaying with the same seed + config reproduces the same trajectory.
    metadata = episode["metadata"]
    seed = metadata.get("seed", 42)
    physics_config = LunarLanderPhysicsConfig.from_dict(metadata["physics_config"])

    env = ParameterizedLunarLander(
        physics_config=physics_config, render_mode=render_mode,
    )
    # Reset with the same seed to reproduce the exact initial state.
    env.reset(seed=seed)

    # Replay the recorded actions one by one up to the branch point.
    # After this loop, the env is in the same state the source episode was
    # at timestep branch_point — ready for the caller to inject new actions.
    for t in range(branch_point):
        obs, _, terminated, _, _ = env.step(episode["actions"][t])
        if terminated and not allow_post_landing:
            # The episode ended before we reached our desired branch point.
            # This shouldn't happen if the caller picked a valid branch_point,
            # but we surface it as an error rather than silently continuing.
            env.close()
            raise RuntimeError(
                f"Episode terminated at step {t} before reaching "
                f"branch point {branch_point}"
            )

    # Optionally wrap for post-landing continuation — allows stepping
    # the env even after the lander has "landed" (terminated=True).
    if allow_post_landing:
        from lwp.collection.env_wrappers_collection import PostLandingWrapper
        env = PostLandingWrapper(env)

    return env


# Registry mapping maneuver type -> generator function.
_GENERATORS: dict[str, callable] = {
    "free_fall": _zero,
    "constant_thrust": _constant,
    "ground_stationary": _zero,
    "ground_thrust_sweep": _constant,
    "ground_side_thrust": _constant,
    "ground_liftoff": _constant,
    "controlled_descent": _constant,
    "low_hover": _constant,
    "bounce_liftoff": _constant,
    "impulse": _impulse,
    "direction_reversal": _direction_reversal,
    "hover": _hover,
}
