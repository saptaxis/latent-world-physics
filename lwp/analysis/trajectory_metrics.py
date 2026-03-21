"""Per-episode trajectory metrics for Lunar Lander.

Computes ~30 scalar metrics from a single .npz trajectory file.
Metrics cover identity (outcome, reward), physics regime, spatial
behavior, action patterns, control quality, action distribution
shape, flight phase fractions, and control smoothness.

The main entry points are:

    compute_metrics(npz_path) -> dict
        Single episode — returns one flat dict of scalars.

    compute_collection_metrics(directory, workers) -> pd.DataFrame
        All .npz files in a directory — returns a DataFrame with one
        row per episode. Supports parallel computation.

State vector layout (15D):
    [0]  x_pos             normalized by half-viewport
    [1]  y_pos             normalized by half-viewport
    [2]  vx                x velocity (scaled)
    [3]  vy                y velocity (scaled)
    [4]  angle             radians
    [5]  angular_vel       angular velocity (scaled)
    [6]  left_leg_contact  0/1
    [7]  right_leg_contact 0/1
    [8]  gravity           raw physics param
    [9]  main_engine_power raw physics param
    [10] side_engine_power raw physics param
    [11] lander_density    raw physics param
    [12] angular_damping   raw physics param
    [13] wind_power        raw physics param
    [14] turbulence_power  raw physics param

Action vector (2D):
    [0]  main_thrust       continuous [0, 1]
    [1]  side_thrust       continuous [-1, 1]
"""

from __future__ import annotations

import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

# --- State vector index constants ---
# Kinematic state (first 8 dims)
X_POS = 0
Y_POS = 1
VX = 2
VY = 3
ANGLE = 4
ANGULAR_VEL = 5
LEFT_LEG = 6
RIGHT_LEG = 7

# Physics parameters (dims 8-14, constant within an episode)
GRAVITY = 8
MAIN_ENGINE_POWER = 9
SIDE_ENGINE_POWER = 10
LANDER_DENSITY = 11
ANGULAR_DAMPING = 12
WIND_POWER = 13
TURBULENCE_POWER = 14

# Action indices
MAIN_THRUST = 0
SIDE_THRUST = 1

# Body area constant from physics_config.py — used for TWR computation.
# Computed from the lander polygon: 867 px² / 30² ≈ 0.96333 world units².
BODY_AREA = 867.0 / 900.0

# --- Phase detection thresholds ---
# Used for flight phase fraction scalars. Fixed values for
# reproducibility across analyses.
PHASE_VY_DESCENDING = -0.1     # vy below this = descending
PHASE_VY_HOVER_ABS = 0.1       # |vy| below this = hovering (if also above ground)
PHASE_Y_ABOVE_GROUND = 0.1     # y above this = not on ground
PHASE_Y_APPROACHING = 0.3      # y below this = approaching landing zone
PHASE_SIDE_CORRECTING = 0.3    # |side_thrust| above this = correcting

# Action distribution thresholds
THRUST_FULL_THRESHOLD = 0.9    # main > this = full throttle
THRUST_ZERO_THRESHOLD = 0.05   # main < this = engine off


def _autocorrelation_lag1(x: np.ndarray) -> float:
    """Compute autocorrelation at lag 1 for a 1D signal.

    Measures how similar consecutive values are. 1.0 = perfectly
    smooth (each value predicts the next), 0.0 = no relationship
    (random noise), -1.0 = alternating.

    Returns 0.0 for signals with zero variance (constant) or
    fewer than 2 samples (undefined).
    """
    if len(x) < 2:
        return 0.0
    x_mean = np.mean(x)
    x_centered = x - x_mean
    var = np.sum(x_centered ** 2)
    if var == 0.0:
        return 0.0
    # Autocorrelation: sum(x[t] * x[t+1]) / sum(x[t]^2)
    return float(np.sum(x_centered[:-1] * x_centered[1:]) / var)


def compute_metrics(npz_path: str | Path) -> dict:
    """Compute scalar metrics from a single episode .npz file.

    Reads the trajectory data and metadata, then computes ~20 metrics
    organized into categories: identity, physics, spatial, action, and
    control. Landed episodes get additional precision metrics (landing
    error, fuel efficiency); these are None for crashed/timeout episodes.

    Args:
        npz_path: Path to the .npz file (saved by episode_io.save_episode).

    Returns:
        Flat dict of scalar values. Keys include:
            Identity: outcome, episode_steps, total_reward, profile, npz_path
            Physics: gravity, main_engine_power, ..., turbulence_power, twr
            Spatial: max_altitude, max_lateral_drift, landing_x_error, landing_vy
            Actions: mean_main_thrust, mean_abs_side_thrust, thrust_duty_cycle,
                     side_thrust_reversals, total_fuel, fuel_efficiency
            Control: mean_abs_angular_vel, angle_at_landing, hover_time,
                     time_to_first_contact
    """
    npz_path = str(npz_path)
    data = np.load(npz_path, allow_pickle=False)

    states = data["states"]       # (T+1, 15)
    actions = data["actions"]     # (T, 2)
    rewards = data["rewards"]     # (T,)
    metadata = json.loads(str(data["metadata_json"]))

    T = len(actions)
    physics_config = metadata.get("physics_config", {})
    outcome = metadata.get("outcome", "unknown")
    is_landed = outcome == "landed"

    # --- Physics regime ---
    # Extract from metadata (authoritative) rather than state vector,
    # since metadata preserves exact values without float rounding.
    gravity = physics_config.get("gravity", float(states[0, GRAVITY]))
    main_power = physics_config.get("main_engine_power", float(states[0, MAIN_ENGINE_POWER]))
    side_power = physics_config.get("side_engine_power", float(states[0, SIDE_ENGINE_POWER]))
    density = physics_config.get("lander_density", float(states[0, LANDER_DENSITY]))
    ang_damping = physics_config.get("angular_damping", float(states[0, ANGULAR_DAMPING]))
    wind = physics_config.get("wind_power", float(states[0, WIND_POWER]))
    turbulence = physics_config.get("turbulence_power", float(states[0, TURBULENCE_POWER]))

    # Thrust-to-weight ratio: engine impulse / gravity impulse per step.
    # Per step, engine applies impulse = main_power * throttle.
    # Per step, gravity applies impulse = mass * |gravity| / fps.
    # TWR at full throttle = main_power * fps / (mass * |gravity|).
    # Must match LunarLanderPhysicsConfig.twr() exactly.
    mass = density * BODY_AREA
    fps = 50  # Box2D step rate
    twr = main_power * fps / (mass * abs(gravity)) if (mass * abs(gravity)) > 0 else 0.0

    # --- Spatial metrics ---
    # Max altitude: highest y-position observed during the episode.
    max_altitude = float(np.max(states[:, Y_POS]))

    # Max lateral drift: largest absolute x-position from center.
    max_lateral_drift = float(np.max(np.abs(states[:, X_POS])))

    # Landing precision (only for landed episodes)
    if is_landed:
        # Landing x error: absolute x-position at final state.
        # x=0 is the center of the landing pad.
        landing_x_error = float(abs(states[-1, X_POS]))
        # Landing vertical velocity: vy at final state (should be ~0 for soft landing).
        landing_vy = float(states[-1, VY])
    else:
        landing_x_error = None
        landing_vy = None

    # --- Action metrics ---
    main_thrusts = actions[:, MAIN_THRUST]
    side_thrusts = actions[:, SIDE_THRUST]

    # Mean thrust levels
    mean_main_thrust = float(np.mean(main_thrusts))
    mean_abs_side_thrust = float(np.mean(np.abs(side_thrusts)))

    # Thrust duty cycle: fraction of timesteps with main thrust > 0.05.
    # Measures how often the engine is "on" vs coasting.
    thrust_duty_cycle = float(np.mean(main_thrusts > 0.05))

    # Side thrust reversals: number of sign changes in side thrust.
    # Measures lateral control oscillation.
    side_signs = np.sign(side_thrusts)
    side_thrust_reversals = int(np.sum(np.abs(np.diff(side_signs)) > 0))

    # Total fuel: sum of absolute thrust values (main + |side|).
    # Proportional to total impulse applied.
    total_fuel = float(np.sum(main_thrusts) + np.sum(np.abs(side_thrusts)))

    # Fuel efficiency: reward per unit fuel (only for landed episodes).
    # Higher = better — agent achieved good reward with less thrust.
    if is_landed and total_fuel > 0:
        fuel_efficiency = float(metadata.get("total_reward", np.sum(rewards))) / total_fuel
    else:
        fuel_efficiency = None

    # --- Control metrics ---
    # Mean absolute angular velocity: measures rotational stability.
    # Lower = smoother flight, higher = wobbling.
    mean_abs_angular_vel = float(np.mean(np.abs(states[:-1, ANGULAR_VEL])))

    # Angle at landing (only for landed episodes): how tilted the lander
    # was at the final step. Ideally ~0 radians.
    if is_landed:
        angle_at_landing = float(abs(states[-1, ANGLE]))
    else:
        angle_at_landing = None

    # Hover time: number of timesteps where the lander has near-zero
    # vertical velocity (|vy| < 0.1) while above the ground (y > 0.1).
    # Indicates controlled descent vs ballistic/crash trajectory.
    hover_mask = (np.abs(states[:-1, VY]) < 0.1) & (states[:-1, Y_POS] > 0.1)
    hover_time = int(np.sum(hover_mask))

    # Time to first leg contact: first step where either leg touches ground.
    # Measures how quickly the lander reaches the surface.
    leg_contacts = (states[:-1, LEFT_LEG] > 0.5) | (states[:-1, RIGHT_LEG] > 0.5)
    contact_indices = np.where(leg_contacts)[0]
    if len(contact_indices) > 0:
        time_to_first_contact = int(contact_indices[0])
    else:
        time_to_first_contact = T  # never contacted

    # --- Action distribution shape (Plan 17) ---
    # These capture *how* the agent uses its actuators, not just the mean.
    # A bang-bang controller has high std + high frac_full + high frac_zero.
    # A proportional controller has moderate std + low extremes.
    std_main_thrust = float(np.std(main_thrusts))
    std_side_thrust = float(np.std(side_thrusts))
    main_thrust_frac_full = float(np.mean(main_thrusts > THRUST_FULL_THRESHOLD))
    main_thrust_frac_zero = float(np.mean(main_thrusts < THRUST_ZERO_THRESHOLD))

    # --- Phase fractions (Plan 17) ---
    # What fraction of the episode is spent in each flight phase?
    # These are lightweight phase labels — each step can match multiple
    # phases (or none). Not mutually exclusive.
    vy = states[:-1, VY]
    y = states[:-1, Y_POS]
    frac_descending = float(np.mean(vy < PHASE_VY_DESCENDING))
    frac_hovering = float(np.mean(
        (np.abs(vy) < PHASE_VY_HOVER_ABS) & (y > PHASE_Y_ABOVE_GROUND)
    ))
    frac_approaching = float(np.mean((vy < 0) & (y < PHASE_Y_APPROACHING)))
    frac_correcting = float(np.mean(np.abs(side_thrusts) > PHASE_SIDE_CORRECTING))

    # --- Control smoothness (Plan 17) ---
    # Autocorrelation at lag 1: how predictable is each action from the
    # previous one? 1.0 = smooth ramps, 0.0 = random, -1.0 = oscillating.
    thrust_autocorr_lag1 = _autocorrelation_lag1(main_thrusts)
    side_thrust_autocorr_lag1 = _autocorrelation_lag1(side_thrusts)

    # --- Orientation quality (E3 perception ablation) ---
    angles = states[:-1, ANGLE]
    UPRIGHT_THRESHOLD = 0.1  # radians (~6°)

    # Fraction of steps where lander is upright (|angle| < threshold).
    # Directly measures angle perception quality — agents that can't
    # see their angle can't stay upright.
    frac_upright = float(np.mean(np.abs(angles) < UPRIGHT_THRESHOLD))

    # Worst tilt during the episode. How badly does the agent lose control?
    max_angle_excursion = float(np.max(np.abs(angles)))

    # Mean recovery time: average steps to return to upright after
    # exceeding the threshold. Measures perception→correction loop speed.
    # NaN if agent never exceeds threshold (perfect orientation).
    exceeded = np.abs(angles) >= UPRIGHT_THRESHOLD
    if np.any(exceeded):
        # Find runs of consecutive exceeded steps.
        diffs = np.diff(exceeded.astype(int))
        starts = np.where(diffs == 1)[0] + 1  # step into exceeded
        ends = np.where(diffs == -1)[0] + 1    # step back to upright
        # Handle case where episode ends while exceeded.
        if len(starts) > len(ends):
            ends = np.append(ends, T)
        if len(starts) > 0 and len(ends) > 0:
            durations = ends[:len(starts)] - starts
            angle_recovery_time = float(np.mean(durations))
        else:
            angle_recovery_time = None
    else:
        angle_recovery_time = None  # never exceeded = perfect

    # --- Thrust-angle correlation (E3 perception ablation) ---
    # Pearson correlation between |angle| and main thrust.
    # High = reactive (agent sees tilt and fires). Low = open-loop.
    if T > 2 and np.std(np.abs(angles)) > 1e-8 and np.std(main_thrusts) > 1e-8:
        thrust_angle_correlation = float(
            np.corrcoef(np.abs(angles), main_thrusts)[0, 1]
        )
    else:
        thrust_angle_correlation = 0.0

    # --- Correction lag (E3 perception ablation) ---
    # Cross-correlation peak lag between |angle| and main thrust.
    # Measures how many steps between seeing a tilt and responding.
    # Positive lag = thrust follows angle (reactive). Zero = simultaneous.
    if T > 10 and np.std(np.abs(angles)) > 1e-8 and np.std(main_thrusts) > 1e-8:
        a_centered = np.abs(angles) - np.mean(np.abs(angles))
        t_centered = main_thrusts - np.mean(main_thrusts)
        # Check lags 0-10 steps
        max_lag = min(10, T // 4)
        cors = []
        for lag in range(max_lag + 1):
            if lag == 0:
                c = np.sum(a_centered * t_centered)
            else:
                c = np.sum(a_centered[:-lag] * t_centered[lag:])
            cors.append(c)
        correction_lag = int(np.argmax(cors))
    else:
        correction_lag = 0

    # --- Descent quality (E3 perception ablation) ---
    y_positions = states[:-1, Y_POS]
    vy_values = states[:-1, VY]

    # Time to descend: steps before altitude first drops below 50% of
    # starting altitude. High = hover trap (agent won't descend).
    start_y = y_positions[0] if len(y_positions) > 0 else 1.0
    half_y = start_y * 0.5
    below_half = np.where(y_positions < half_y)[0]
    time_to_descend = int(below_half[0]) if len(below_half) > 0 else T

    # Descent steadiness: variance of vy during descent phase.
    # Low = smooth controlled descent, high = jerky/oscillating.
    descent_mask = vy_values < PHASE_VY_DESCENDING
    if np.sum(descent_mask) > 1:
        descent_steadiness = float(np.var(vy_values[descent_mask]))
    else:
        descent_steadiness = None  # never descended

    # Fraction approaching pad: fraction of steps where lateral distance
    # to pad center (x=0) is decreasing. Measures navigation ability —
    # agents need position perception to navigate toward the pad.
    x_positions = states[:-1, X_POS]
    if len(x_positions) > 1:
        dist_to_pad = np.abs(x_positions)
        dist_decreasing = np.diff(dist_to_pad) < 0
        frac_approaching_pad = float(np.mean(dist_decreasing))
    else:
        frac_approaching_pad = 0.0

    # --- Assemble result dict ---
    return {
        # Identity
        "npz_path": npz_path,
        "outcome": outcome,
        "episode_steps": T,
        "total_reward": float(metadata.get("total_reward", np.sum(rewards))),
        "profile": metadata.get("profile", "default"),

        # Physics regime
        "gravity": gravity,
        "main_engine_power": main_power,
        "side_engine_power": side_power,
        "lander_density": density,
        "angular_damping": ang_damping,
        "wind_power": wind,
        "turbulence_power": turbulence,
        "twr": twr,

        # Spatial
        "max_altitude": max_altitude,
        "max_lateral_drift": max_lateral_drift,
        "landing_x_error": landing_x_error,
        "landing_vy": landing_vy,

        # Actions
        "mean_main_thrust": mean_main_thrust,
        "mean_abs_side_thrust": mean_abs_side_thrust,
        "thrust_duty_cycle": thrust_duty_cycle,
        "side_thrust_reversals": side_thrust_reversals,
        "total_fuel": total_fuel,
        "fuel_efficiency": fuel_efficiency,

        # Control
        "mean_abs_angular_vel": mean_abs_angular_vel,
        "angle_at_landing": angle_at_landing,
        "hover_time": hover_time,
        "time_to_first_contact": time_to_first_contact,

        # Action distribution shape (Plan 17)
        "std_main_thrust": std_main_thrust,
        "std_side_thrust": std_side_thrust,
        "main_thrust_frac_full": main_thrust_frac_full,
        "main_thrust_frac_zero": main_thrust_frac_zero,

        # Phase fractions (Plan 17)
        "frac_descending": frac_descending,
        "frac_hovering": frac_hovering,
        "frac_approaching": frac_approaching,
        "frac_correcting": frac_correcting,

        # Control smoothness (Plan 17)
        "thrust_autocorr_lag1": thrust_autocorr_lag1,
        "side_thrust_autocorr_lag1": side_thrust_autocorr_lag1,

        # Orientation quality (E3 perception ablation)
        "frac_upright": frac_upright,
        "max_angle_excursion": max_angle_excursion,
        "angle_recovery_time": angle_recovery_time,

        # Thrust-angle coupling (E3 perception ablation)
        "thrust_angle_correlation": thrust_angle_correlation,
        "correction_lag": correction_lag,

        # Descent quality (E3 perception ablation)
        "time_to_descend": time_to_descend,
        "descent_steadiness": descent_steadiness,
        "frac_approaching_pad": frac_approaching_pad,
    }


def compute_collection_metrics(
    directory: str | Path,
    workers: int = 8,
) -> "pd.DataFrame":
    """Compute metrics for all .npz files in a directory.

    Finds all .npz files, computes per-episode metrics (optionally in
    parallel), and returns a DataFrame with one row per episode.

    Args:
        directory: Path to directory containing .npz files.
        workers: Number of parallel workers. Use 1 for sequential.

    Returns:
        pandas DataFrame with one row per episode, columns matching
        the keys from compute_metrics().
    """
    import pandas as pd

    npz_paths = sorted(Path(directory).glob("*.npz"))
    if not npz_paths:
        return pd.DataFrame()

    npz_strs = [str(p) for p in npz_paths]

    if workers <= 1:
        # Sequential — simpler for debugging
        results = [compute_metrics(p) for p in npz_strs]
    else:
        # Parallel — use multiprocessing.Pool for CPU-bound metric computation.
        # Each worker loads one .npz and computes metrics independently.
        with Pool(processes=workers) as pool:
            results = pool.map(compute_metrics, npz_strs, chunksize=max(1, len(npz_strs) // workers))

    return pd.DataFrame(results)
