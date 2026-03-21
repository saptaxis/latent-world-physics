"""Behavioral target computation for linear probes.

Behavioral targets capture the *consequences* of physics parameters,
not the parameters themselves. A network that encodes TWR has learned
physics functionally; one that encodes gravity has learned it parametrically.

See probing-tooling.md (Section 2) for the full target specification.
"""
import numpy as np

from parametric_lunar_lander.physics_config import LunarLanderPhysicsConfig

# Target name lists — used throughout the probe pipeline for column ordering.
PARAMETRIC_TARGET_NAMES: list[str] = list(LunarLanderPhysicsConfig.PARAM_NAMES)

BEHAVIORAL_TARGET_NAMES: list[str] = [
    "twr",
    "descent_rate",
    "angular_responsiveness",
    "hover_cost",
    "effective_weight",
]

ALL_TARGET_NAMES: list[str] = PARAMETRIC_TARGET_NAMES + BEHAVIORAL_TARGET_NAMES

# Kinematic state targets — the per-timestep state variables that vary
# within and across episodes. For visual agents, probing these answers:
# "can the CNN encoder see where the lander is and how it's moving?"
KINEMATIC_TARGET_NAMES: list[str] = [
    "x_pos",
    "y_pos",
    "vx",
    "vy",
    "angle",
    "angular_vel",
    "left_leg_contact",
    "right_leg_contact",
]

# Reference height for descent rate calculation (meters).
_REFERENCE_HEIGHT = 1.0

# Moment scale for angular responsiveness. Approximate moment of inertia
# relative to density — based on the lander's body geometry in Box2D.
# The exact value is side_engine_height * side_engine_distance / I_body,
# but we use a constant since body shape doesn't change.
_MOMENT_SCALE = 1.0


def compute_behavioral_targets(config: LunarLanderPhysicsConfig) -> np.ndarray:
    """Compute 5 behavioral targets from a physics config.

    Args:
        config: Physics configuration for the episode.

    Returns:
        np.ndarray of shape (5,) with dtype float32, in BEHAVIORAL_TARGET_NAMES order:
            [twr, descent_rate, angular_responsiveness, hover_cost, effective_weight]
    """
    twr = config.twr()

    # Free-fall speed from reference height: v = sqrt(2 * |g| * h)
    descent_rate = np.sqrt(2.0 * abs(config.gravity) * _REFERENCE_HEIGHT)

    # Rotational control authority
    angular_responsiveness = config.side_engine_power / (
        config.lander_density * _MOMENT_SCALE
    )

    # Fraction of max thrust needed to hover (clamped to avoid inf for TWR=0)
    hover_cost = 1.0 / max(twr, 1e-6)

    # Gravitational force the engine must overcome
    effective_weight = (
        config.lander_density * LunarLanderPhysicsConfig.BODY_AREA * abs(config.gravity)
    )

    return np.array(
        [twr, descent_rate, angular_responsiveness, hover_cost, effective_weight],
        dtype=np.float32,
    )
