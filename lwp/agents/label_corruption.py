"""Label corruption wrapper for eval-time observation manipulation.

Tests whether labeled RL agents actually use their physics labels by
corrupting the 7 physics observation dimensions (indices 8-14) at eval
time. Four corruption modes:

  - zero: Set all physics dims to 0. Tests: are labels load-bearing?
  - shuffle: Randomly permute physics dims within each episode.
    Tests: do specific values matter, or just having signal?
  - mean: Replace with training-set mean. Tests: is per-episode
    variation in labels informative?
  - noise: Add Gaussian noise scaled by sigma. Tests: how precise
    must labels be?

This is an eval-time-only modification — zero training cost. The wrapper
goes in the wrapper stack after RaycastWrapper but before VecNormalize,
so the agent sees corrupted physics dims through the same normalization
it trained with.

Usage:
    env = make_lunar_lander_env(variant="labeled", ...)
    env = LabelCorruptionWrapper(env, corruption_type="zero")

    # Or with noise sweep:
    env = LabelCorruptionWrapper(env, corruption_type="noise", sigma=0.1)

Scientific context: Phase A2 of the mechanistic investigation
(mechanistic-behavioral-study.md). If performance holds under
zero-out, the agent learned behavioral mode despite having labels.
If it drops, the agent is in parametric mode — actually reading
the physics parameters.
"""

import numpy as np
import gymnasium


# Physics parameter indices in the base Lunar Lander observation.
# The full observation is 15D:
#   dims 0-7: kinematic state (x, y, vx, vy, angle, angular_vel, left_leg, right_leg)
#   dims 8-14: physics params (gravity, main_engine_power, side_engine_power,
#              lander_density, angular_damping, wind_power, turbulence_power)
#
# After RaycastWrapper adds n_rays dims, the physics indices are unchanged
# (rays are appended at the end). After PhysicsBlindWrapper strips them,
# there are no physics dims to corrupt — so this wrapper is only meaningful
# for the "labeled" variant.
PHYSICS_START = 8
PHYSICS_END = 15  # exclusive — dims 8,9,10,11,12,13,14
N_PHYSICS_DIMS = PHYSICS_END - PHYSICS_START  # 7

# Named subsets of the 7 physics dims, indexed within the physics slice
# (i.e., 0..6 maps to absolute obs indices 8..14). Order matches
# LunarLanderPhysicsConfig.PARAM_NAMES:
#   0 gravity              (world)
#   1 main_engine_power    (body)
#   2 side_engine_power    (body)
#   3 lander_density       (body)
#   4 angular_damping      (body)
#   5 wind_power           (world)
#   6 turbulence_power     (world)
BODY_DIM_INDICES: tuple[int, ...] = (1, 2, 3, 4)
WORLD_DIM_INDICES: tuple[int, ...] = (0, 5, 6)
ALL_DIM_INDICES: tuple[int, ...] = tuple(range(N_PHYSICS_DIMS))

SUBSET_PRESETS: dict[str, tuple[int, ...]] = {
    "all": ALL_DIM_INDICES,
    "body": BODY_DIM_INDICES,
    "world": WORLD_DIM_INDICES,
}


def resolve_corruption_dims(spec: str | None) -> list[int]:
    """Parse a subset spec into a sorted list of physics-dim indices.

    Accepts:
      - None or "all": all 7 dims (default — backwards compatible).
      - "body": the 4 body params (main/side engine, density, damping).
      - "world": the 3 world params (gravity, wind, turbulence).
      - Comma-separated list of integers in [0, 7), e.g. "0,2,5".

    Returns a sorted list of unique ints. Raises ValueError on bad input.
    """
    if spec is None or spec == "all":
        return list(ALL_DIM_INDICES)
    if spec in SUBSET_PRESETS:
        return list(SUBSET_PRESETS[spec])
    # Comma-separated list path.
    try:
        parts = [int(p.strip()) for p in spec.split(",") if p.strip() != ""]
    except ValueError as e:
        raise ValueError(
            f"corruption dims spec '{spec}' is not a recognized preset "
            f"({sorted(SUBSET_PRESETS)}) or comma-list of ints."
        ) from e
    if not parts:
        raise ValueError(f"corruption dims spec '{spec}' is empty.")
    for d in parts:
        if not (0 <= d < N_PHYSICS_DIMS):
            raise ValueError(
                f"corruption dim {d} is out of range [0, {N_PHYSICS_DIMS})."
            )
    if len(set(parts)) != len(parts):
        raise ValueError(f"corruption dims spec '{spec}' contains duplicates.")
    return sorted(set(parts))


class LabelCorruptionWrapper(gymnasium.ObservationWrapper):
    """Corrupt physics observation dimensions to test label dependency.

    Applied at eval time to labeled agents (22D observation = 8 kinematic
    + 7 physics + 7 rays). Modifies dims 8-14 (the physics labels) while
    leaving kinematic state (0-7) and ray distances (15+) untouched.

    Corruption is deterministic given the seed, ensuring reproducible
    experiments. For shuffle mode, the permutation is fixed per episode
    (regenerated on each reset) so all steps within an episode see the
    same shuffled labels.

    Args:
        env: A wrapped Lunar Lander env (any obs dimensionality >= 15).
        corruption_type: One of "zero", "shuffle", "mean", "noise".
        seed: Random seed for reproducible corruption (shuffle + noise).
        training_means: Array of 7 floats — per-dim training-set means,
            in PARAM_NAMES order. Required for "mean" mode.
        sigma: Noise standard deviation as a fraction of each param's
            range. Only used for "noise" mode. Default 0.1 = 10% of range.
        dims: Optional list of physics-dim indices (in [0, 7)) to corrupt.
            None (default) corrupts all 7 dims — backwards compatible.
            Use named subsets via resolve_corruption_dims("body" / "world")
            or pass an explicit list (e.g. [1, 2, 3, 4] for body only).
            Untouched dims pass through unchanged. For "shuffle" mode the
            permutation is restricted to the selected dims.
    """

    VALID_TYPES = ("zero", "shuffle", "mean", "noise")

    def __init__(
        self,
        env: gymnasium.Env,
        corruption_type: str,
        seed: int = 0,
        training_means: np.ndarray | None = None,
        sigma: float = 0.1,
        dims: list[int] | tuple[int, ...] | None = None,
    ):
        super().__init__(env)

        if corruption_type not in self.VALID_TYPES:
            raise ValueError(
                f"corruption_type must be one of {self.VALID_TYPES}, "
                f"got '{corruption_type}'"
            )

        obs_dim = env.observation_space.shape[0]
        if obs_dim < 15:
            raise ValueError(
                f"LabelCorruptionWrapper requires obs dim >= 15 (got {obs_dim}). "
                f"This wrapper is only meaningful for labeled agents — blind "
                f"agents have physics dims already stripped."
            )

        if corruption_type == "mean" and training_means is None:
            raise ValueError(
                "corruption_type='mean' requires training_means array "
                "(7 floats, one per physics param in PARAM_NAMES order)."
            )

        # Resolve and validate the dim subset. None = all 7 dims.
        if dims is None:
            dim_indices = list(ALL_DIM_INDICES)
        else:
            dim_indices = sorted(set(int(d) for d in dims))
            if not dim_indices:
                raise ValueError("dims must be non-empty (or None for all 7 dims).")
            for d in dim_indices:
                if not (0 <= d < N_PHYSICS_DIMS):
                    raise ValueError(
                        f"dim {d} is out of range [0, {N_PHYSICS_DIMS})."
                    )

        self.corruption_type = corruption_type
        self._rng = np.random.default_rng(seed)
        self._training_means = training_means
        self._sigma = sigma

        # Relative indices within the 7-dim physics slice and absolute
        # indices into the full observation vector. Cached as np.ndarray
        # for fast fancy indexing in observation().
        self._dim_indices: np.ndarray = np.array(dim_indices, dtype=np.intp)
        self._abs_indices: np.ndarray = self._dim_indices + PHYSICS_START

        # Per-episode shuffle permutation — regenerated on each reset().
        # Stays fixed across all steps within one episode so the agent
        # sees a consistent (but wrong) set of physics labels. Identity
        # default (no shuffle yet) — populated on reset() for shuffle mode.
        self._shuffle_perm = np.arange(len(self._dim_indices), dtype=np.intp)

        # Param ranges for noise scaling. Imported here to avoid circular
        # imports at module level.
        from parametric_lunar_lander.physics_config import LunarLanderPhysicsConfig
        self._param_ranges = np.array([
            LunarLanderPhysicsConfig.RANGES[name][1] - LunarLanderPhysicsConfig.RANGES[name][0]
            for name in LunarLanderPhysicsConfig.PARAM_NAMES
        ], dtype=np.float32)
        self._param_lows = np.array([
            LunarLanderPhysicsConfig.RANGES[name][0]
            for name in LunarLanderPhysicsConfig.PARAM_NAMES
        ], dtype=np.float32)
        self._param_highs = np.array([
            LunarLanderPhysicsConfig.RANGES[name][1]
            for name in LunarLanderPhysicsConfig.PARAM_NAMES
        ], dtype=np.float32)

    def reset(self, **kwargs):
        """Reset env and regenerate per-episode corruption state."""
        obs, info = self.env.reset(**kwargs)

        # Generate new shuffle permutation for this episode, restricted
        # to the selected dim subset. A length-1 subset trivially permutes
        # to itself and is effectively a no-op for shuffle.
        if self.corruption_type == "shuffle":
            self._shuffle_perm = self._rng.permutation(len(self._dim_indices))

        return self.observation(obs), info

    def observation(self, observation):
        """Apply corruption to the selected physics dims.

        Returns a copy — never modifies the original observation array.
        Dims outside ``self._abs_indices`` (kinematic, unselected physics,
        rays, etc.) are left untouched.
        """
        corrupted = observation.copy()
        abs_idx = self._abs_indices
        rel_idx = self._dim_indices

        if self.corruption_type == "zero":
            # Zero-out: set selected physics dims to 0.
            # The simplest test: if the agent doesn't degrade, it never
            # looked at these dims. If it degrades, they're load-bearing.
            corrupted[abs_idx] = 0.0

        elif self.corruption_type == "shuffle":
            # Shuffle: permute physics values within the selected subset.
            # Tests whether the agent cares about which dim is which (within
            # this subset), or just that some signal is present.
            subset_values = observation[abs_idx]
            corrupted[abs_idx] = subset_values[self._shuffle_perm]

        elif self.corruption_type == "mean":
            # Mean-replace: set selected dims to their training-set mean.
            # Tests whether per-episode variation in those labels is informative.
            assert self._training_means is not None  # checked in __init__
            corrupted[abs_idx] = self._training_means[rel_idx]

        elif self.corruption_type == "noise":
            # Noise: add Gaussian noise scaled by sigma * param_range to
            # selected dims only. Clipped to each param's valid range so
            # downstream code never sees impossible physics values.
            subset_values = observation[abs_idx]
            ranges = self._param_ranges[rel_idx]
            lows = self._param_lows[rel_idx]
            highs = self._param_highs[rel_idx]
            noise = self._rng.normal(0, self._sigma * ranges).astype(np.float32)
            corrupted[abs_idx] = np.clip(subset_values + noise, lows, highs)

        return corrupted


def compute_training_means(trajectory_dir: str) -> np.ndarray:
    """Compute per-physics-param means from trajectory .npz files.

    Reads the physics_config from each episode's metadata and computes
    the mean value for each of the 7 physics parameters. Used to provide
    training_means for the "mean" corruption mode.

    The mean is computed across episodes (one physics config per episode),
    not across timesteps (physics is constant within an episode).

    Args:
        trajectory_dir: Directory containing episode_NNNN.npz files.

    Returns:
        np.ndarray of shape (7,) — per-param means in PARAM_NAMES order.

    Raises:
        FileNotFoundError: If no .npz files found in the directory.
    """
    import json as _json
    from pathlib import Path
    from parametric_lunar_lander.physics_config import LunarLanderPhysicsConfig

    npz_files = sorted(Path(trajectory_dir).glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(
            f"No .npz trajectory files found in {trajectory_dir}"
        )

    # Collect physics param values from each episode's metadata.
    # Each episode has one physics config (constant within the episode).
    all_params = []
    for npz_path in npz_files:
        data = np.load(str(npz_path), allow_pickle=True)
        metadata = _json.loads(str(data["metadata_json"]))
        physics = metadata["physics_config"]
        params = [physics[name] for name in LunarLanderPhysicsConfig.PARAM_NAMES]
        all_params.append(params)

    return np.mean(all_params, axis=0).astype(np.float32)
