"""YAML config parsing for world model data collection.

Two config levels:
  - CollectionConfig: one run = one source type, one physics sampling, N episodes.
  - BatchConfig: lists multiple CollectionConfigs to run sequentially.

See specs/data-collection.md for the full design.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


# Valid source types for collection. RL sources require a checkpoint_dir;
# simple sources (random, heuristic) do not. "primitive" uses scripted
# maneuvers with configurable start conditions.
VALID_SOURCE_TYPES = {
    "blind_agent", "labeled_agent", "heuristic", "random", "noisy_expert",
    "primitive",
}
RL_SOURCE_TYPES = {"blind_agent", "labeled_agent", "noisy_expert"}

# Valid maneuver types for primitive collection. Each type uses a different
# subset of ManeuverConfig fields to define the action pattern.
VALID_MANEUVER_TYPES = {
    "free_fall", "constant_thrust", "impulse", "direction_reversal",
    "hover", "ground_stationary", "ground_thrust_sweep",
    "ground_side_thrust", "ground_liftoff",
    "controlled_descent", "low_hover", "bounce_liftoff",
}

# Valid start modes for primitive collection. Determines how the env
# is initialized at the beginning of each episode.
VALID_START_MODES = {"fresh_reset", "replay", "replay_to_landing"}


@dataclass
class ManeuverConfig:
    """Config for a primitive maneuver's action pattern.

    Parsed from the 'maneuver' block in a primitive collection YAML.
    Different maneuver types use different subsets of fields:
      - constant_thrust: main, side
      - impulse: channel, thrust_level, pulse_duration, gap_duration, n_cycles
      - direction_reversal: channel, thrust_level, first_duration, gap_duration, second_duration
      - hover: no extra fields (thrust computed from physics config)
      - ground_*: main, side (action applied after landing)
    """

    type: str
    # Constant thrust levels (used by constant_thrust, ground_thrust_sweep, etc.)
    # Scalar = fixed value. Tuple = (lo, hi) range, sampled per episode.
    main: float | tuple[float, float] = 0.0
    side: float | tuple[float, float] = 0.0
    # Impulse / direction reversal fields
    channel: str | None = None  # "main" or "side"
    thrust_level: float = 1.0
    pulse_duration: tuple[int, int] | None = None  # (min, max) steps, sampled per episode
    gap_duration: tuple[int, int] | None = None
    n_cycles: tuple[int, int] | None = None
    first_duration: tuple[int, int] | None = None
    second_duration: tuple[int, int] | None = None

    @classmethod
    def from_dict(cls, d: dict) -> ManeuverConfig:
        """Parse a maneuver config from a YAML-loaded dict.

        Validates the maneuver type against VALID_MANEUVER_TYPES and converts
        list values to tuples where appropriate (e.g., [lo, hi] -> (lo, hi)
        for range-valued thrust levels and timing fields).

        Args:
            d: Dict from the 'maneuver' block of a primitive collection YAML.

        Returns:
            A validated ManeuverConfig.

        Raises:
            ValueError: If maneuver type is not in VALID_MANEUVER_TYPES.
        """
        mtype = d.get("type")
        if mtype not in VALID_MANEUVER_TYPES:
            raise ValueError(
                f"Unknown maneuver type '{mtype}'. "
                f"Valid: {sorted(VALID_MANEUVER_TYPES)}"
            )

        def _parse_range(val):
            """Convert [lo, hi] list to (lo, hi) tuple; pass through None."""
            if val is None:
                return None
            if isinstance(val, list):
                return (val[0], val[1])
            return val

        def _parse_thrust(val, default=0.0):
            """Parse thrust: scalar float or [lo, hi] range tuple."""
            if val is None:
                return default
            if isinstance(val, list):
                return (float(val[0]), float(val[1]))
            return float(val)

        return cls(
            type=mtype,
            main=_parse_thrust(d.get("main"), 0.0),
            side=_parse_thrust(d.get("side"), 0.0),
            channel=d.get("channel"),
            thrust_level=float(d.get("thrust_level", 1.0)),
            pulse_duration=_parse_range(d.get("pulse_duration")),
            gap_duration=_parse_range(d.get("gap_duration")),
            n_cycles=_parse_range(d.get("n_cycles")),
            first_duration=_parse_range(d.get("first_duration")),
            second_duration=_parse_range(d.get("second_duration")),
        )


@dataclass
class StartConfig:
    """Config for how primitive episodes start.

    Three modes:
      - fresh_reset: Reset env, override initial state from ranges.
        Each key in initial_state maps to a (lo, hi) range sampled per episode.
      - replay: Replay a source episode to a random branch point, then
        apply the maneuver from there.
      - replay_to_landing: Replay a source episode past the landing event,
        then apply the maneuver on the ground.
    """

    mode: str
    # fresh_reset fields: maps state variable names to (lo, hi) sampling ranges
    initial_state: dict[str, tuple[float, float]] | None = None
    # replay / replay_to_landing fields
    source_dir: str | None = None
    min_step: int = 20
    max_step_fraction: float = 0.8

    @classmethod
    def from_dict(cls, d: dict) -> StartConfig:
        """Parse a start config from a YAML-loaded dict.

        Validates the mode against VALID_START_MODES and converts initial_state
        list values to tuples.

        Args:
            d: Dict from the 'start' block of a primitive collection YAML.

        Returns:
            A validated StartConfig.

        Raises:
            ValueError: If mode is not in VALID_START_MODES.
        """
        mode = d.get("mode")
        if mode not in VALID_START_MODES:
            raise ValueError(
                f"Unknown start mode '{mode}'. Valid: {sorted(VALID_START_MODES)}"
            )

        # Parse initial_state: convert [lo, hi] lists to (lo, hi) tuples
        initial_state = None
        if "initial_state" in d:
            initial_state = {
                k: (float(v[0]), float(v[1]))
                for k, v in d["initial_state"].items()
            }

        return cls(
            mode=mode,
            initial_state=initial_state,
            source_dir=d.get("source_dir"),
            min_step=d.get("min_step", 20),
            max_step_fraction=d.get("max_step_fraction", 0.8),
        )


@dataclass
class CollectionConfig:
    """Config for a single collection run.

    Parsed from YAML. Defines what policy to collect from, how many episodes,
    and how to sample physics. See specs/data-collection.md for field docs.

    Attributes:
        source_type: One of VALID_SOURCE_TYPES. Determines what kind of policy
            generates the trajectory data.
        n_episodes: Number of episodes to collect in this run.
        physics_ranges: Dict mapping physics param names to (lo, hi) sampling
            ranges. Always resolved to a flat dict — even when the YAML uses
            a named profile, we extract the ranges so downstream code has a
            uniform interface.
        seed: RNG seed for reproducible collection. Default 0.
        checkpoint_dir: Path to the trained agent directory. Required for RL
            source types (blind_agent, labeled_agent, noisy_expert); None for
            simple sources (random, heuristic).
        deterministic: Whether to use deterministic (argmax) action selection
            for RL agents. Default True.
        save_frames: Whether to store RGB frames alongside states/actions.
            Default False (frames are large and not always needed).
        noise_sigma: (lo, hi) range for noise standard deviation, sampled
            per episode. Only meaningful for noisy_expert source type.
            Scalar YAML values are normalized to (val, val). Default (0.01, 0.5).
        twr_range: Optional (min_twr, max_twr) constraint from a named
            sampling profile. When set, physics configs are rejection-sampled
            until TWR falls within this range. None means no constraint.
        maneuver_config: Parsed ManeuverConfig for primitive source type.
            None for RL/simple source types.
        start_config: Parsed StartConfig for primitive source type.
            None for RL/simple source types.
        max_steps: Maximum steps per episode for primitive collection.
            Default 1000.
        allow_post_landing: Whether to continue collecting after the lander
            touches down. Required for ground_* maneuvers. Default False.
    """

    source_type: str
    n_episodes: int
    physics_ranges: dict[str, tuple[float, float]]
    seed: int = 0
    checkpoint_dir: str | None = None
    deterministic: bool = True
    save_frames: bool = False
    noise_sigma: tuple[float, float] = (0.01, 0.5)
    twr_range: tuple[float, float] | None = None
    maneuver_config: ManeuverConfig | None = None
    start_config: StartConfig | None = None
    max_steps: int = 1000
    allow_post_landing: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> CollectionConfig:
        """Parse from a YAML-loaded dict.

        Validates source_type membership and checkpoint_dir requirements.

        Physics sampling supports three modes (same pattern as training configs):
          1. Named profile: physics_sampling.profile = "easy"
             Loads the profile and extracts ranges from its overrides, falling
             back to LunarLanderPhysicsConfig.RANGES for unspecified params.
          2. Named profile + overrides: physics_sampling.profile + .overrides
             Same as (1), but merges explicit overrides on top of the profile.
          3. Explicit ranges: physics_sampling.ranges = {gravity: [-15, -3], ...}
             Uses the raw dict directly — no validation against physics config
             ranges (that happens at sampling time).

        If no physics_sampling block is present, falls back to the full
        LunarLanderPhysicsConfig.RANGES.

        noise_sigma accepts a scalar (normalized to (val, val) for uniform
        interface) or a [lo, hi] list. Defaults to (0.01, 0.5) if omitted.

        Args:
            d: Dict parsed from YAML via yaml.safe_load().

        Returns:
            A validated CollectionConfig.

        Raises:
            ValueError: If source_type is invalid or RL source lacks checkpoint_dir.
        """
        # -- Validate source_type --
        source_type = d.get("source_type")
        if source_type not in VALID_SOURCE_TYPES:
            raise ValueError(
                f"source_type must be one of {sorted(VALID_SOURCE_TYPES)}, got '{source_type}'"
            )

        # -- Validate checkpoint_dir requirement for RL sources --
        checkpoint_dir = d.get("checkpoint_dir")
        if source_type in RL_SOURCE_TYPES and checkpoint_dir is None:
            raise ValueError(
                f"source_type '{source_type}' requires checkpoint_dir"
            )

        # -- Parse physics sampling --
        # Three modes: named profile, profile + overrides, or explicit ranges.
        # All modes resolve to a physics_ranges dict + optional TWR constraint.
        sampling = d.get("physics_sampling", {})
        physics_ranges, twr_range = _parse_physics_sampling(sampling)

        # -- Parse noise_sigma --
        # Scalar -> (val, val), list -> (lo, hi). Default (0.01, 0.5).
        raw_sigma = d.get("noise_sigma", [0.01, 0.5])
        if isinstance(raw_sigma, (int, float)):
            noise_sigma = (float(raw_sigma), float(raw_sigma))
        else:
            noise_sigma = (float(raw_sigma[0]), float(raw_sigma[1]))

        # -- Parse primitive-specific fields --
        # Primitive source type requires both 'maneuver' and 'start' blocks
        # in the YAML config. These define the scripted action pattern and
        # how episodes are initialized.
        maneuver_config = None
        start_config = None
        max_steps = d.get("max_steps", 1000)
        allow_post_landing = d.get("allow_post_landing", False)

        if source_type == "primitive":
            if "maneuver" not in d:
                raise ValueError(
                    "source_type 'primitive' requires a 'maneuver' block"
                )
            if "start" not in d:
                raise ValueError(
                    "source_type 'primitive' requires a 'start' block"
                )
            maneuver_config = ManeuverConfig.from_dict(d["maneuver"])
            start_config = StartConfig.from_dict(d["start"])

        return cls(
            source_type=source_type,
            n_episodes=d["n_episodes"],
            physics_ranges=physics_ranges,
            seed=d.get("seed", 0),
            checkpoint_dir=checkpoint_dir,
            deterministic=d.get("deterministic", True),
            save_frames=d.get("save_frames", False),
            noise_sigma=noise_sigma,
            twr_range=twr_range,
            maneuver_config=maneuver_config,
            start_config=start_config,
            max_steps=max_steps,
            allow_post_landing=allow_post_landing,
        )

    @classmethod
    def load(cls, path: str) -> CollectionConfig:
        """Load from a YAML file on disk.

        Args:
            path: Path to the YAML config file.

        Returns:
            A CollectionConfig parsed from the file contents.
        """
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f))


def _parse_physics_sampling(
    sampling: dict,
) -> tuple[dict[str, tuple[float, float]], tuple[float, float] | None]:
    """Resolve a physics_sampling YAML block to ranges + optional TWR constraint.

    Handles three modes:
      1. profile key present -> load named profile, extract ranges + twr_range
      2. profile + overrides -> load profile, merge overrides, extract ranges + twr_range
      3. ranges key present -> use explicit ranges directly (no TWR constraint)

    Falls back to full LunarLanderPhysicsConfig.RANGES if neither key is present.

    Args:
        sampling: The physics_sampling dict from the YAML config.

    Returns:
        Tuple of (physics_ranges, twr_range) where:
          - physics_ranges maps param names to (lo, hi) tuples
          - twr_range is (min_twr, max_twr) or None if no constraint
    """
    from parametric_lunar_lander.sampling_profiles import SamplingProfile
    from parametric_lunar_lander.physics_config import LunarLanderPhysicsConfig

    if "profile" in sampling:
        # Mode 1 or 2: load named profile, optionally merge overrides.
        profile = SamplingProfile.load(sampling["profile"])

        # Merge explicit overrides on top of the profile (mode 2).
        if "overrides" in sampling:
            for k, v in sampling["overrides"].items():
                if isinstance(v, list):
                    profile.overrides[k] = tuple(v)
                else:
                    profile.overrides[k] = v

        # Extract ranges: use override if present, else full config range.
        physics_ranges = {}
        for name in LunarLanderPhysicsConfig.PARAM_NAMES:
            if name in profile.overrides:
                override = profile.overrides[name]
                if isinstance(override, tuple):
                    physics_ranges[name] = override
                else:
                    # Scalar override -> fixed range (lo == hi)
                    physics_ranges[name] = (override, override)
            else:
                physics_ranges[name] = LunarLanderPhysicsConfig.RANGES[name]

        # Preserve the TWR constraint from the profile (e.g., twr_min: 8.0).
        return physics_ranges, profile.twr_range

    elif "ranges" in sampling:
        # Mode 3: explicit ranges dict. Convert lists to tuples. No TWR constraint.
        raw_ranges = sampling["ranges"]
        return {k: tuple(v) for k, v in raw_ranges.items()}, None

    else:
        # No physics_sampling specified — use full default ranges.
        return dict(LunarLanderPhysicsConfig.RANGES), None


@dataclass
class BatchEntry:
    """One entry in a batch config: a collection config + output name.

    Attributes:
        config: The parsed CollectionConfig for this run.
        output_name: Directory name under the batch output_base where this
            run's trajectories will be written.
    """

    config: CollectionConfig
    output_name: str


@dataclass
class BatchConfig:
    """Batch config listing multiple collections to run sequentially.

    Each entry references a collection YAML and an output directory name.
    The batch runner creates {output_base}/{output_name}/ for each entry.

    Attributes:
        output_base: Root directory for all batch outputs.
        entries: List of BatchEntry objects, each with a config and output name.
    """

    output_base: str
    entries: list[BatchEntry] = field(default_factory=list)

    @classmethod
    def load(cls, path: str) -> BatchConfig:
        """Load from a batch YAML file.

        Each entry's config field is a path to a collection YAML. Relative
        paths are resolved against the batch file's directory, so configs
        can live alongside the batch file.

        Supports a batch-level ``physics_sampling`` block that overrides the
        physics ranges of every collection in the batch. This lets you reuse
        the same individual configs (which define source type, seed, n_episodes,
        checkpoint, etc.) across different physics regimes without duplicating
        them. For example::

            physics_sampling:
              profile: easy

            collections:
              - config: blind-s42.yaml   # reuses the original config
              ...

        The override replaces the per-collection physics_ranges entirely.

        Args:
            path: Path to the batch YAML file.

        Returns:
            A BatchConfig with all collection configs loaded and validated.
        """
        batch_dir = Path(path).parent
        with open(path) as f:
            d = yaml.safe_load(f)

        # Batch-level physics override: if present, resolve it once and apply
        # to every collection entry after loading. Includes TWR constraint
        # from named profiles (e.g., easy has twr_min=8).
        batch_physics = None
        batch_twr_range = None
        if "physics_sampling" in d:
            batch_physics, batch_twr_range = _parse_physics_sampling(d["physics_sampling"])

        entries = []
        for entry in d.get("collections", []):
            config_path = entry["config"]
            # Resolve relative paths against batch file location so configs
            # can reference sibling files with just a filename.
            if not Path(config_path).is_absolute():
                config_path = str(batch_dir / config_path)
            cfg = CollectionConfig.load(config_path)

            # Apply batch-level physics override if specified. Replaces both
            # the per-config ranges and TWR constraint entirely.
            if batch_physics is not None:
                cfg.physics_ranges = batch_physics
                cfg.twr_range = batch_twr_range

            # output_name can be explicit or derived from config filename (e.g. "blind-s42.yaml" -> "blind-s42")
            output_name = entry.get("output_name", Path(config_path).stem)
            entries.append(BatchEntry(config=cfg, output_name=output_name))

        return cls(output_base=d["output_base"], entries=entries)
