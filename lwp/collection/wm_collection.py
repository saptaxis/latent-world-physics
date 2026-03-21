"""Core collection functions for world model training data.

Two collection paths:
  - collect_simple(): For policies that take raw obs (random, heuristic).
    Uses run_episode() from episode_io — no VecNormalize needed.
  - collect_rl_agent(): For trained SB3 agents that need VecNormalize.
    Adapts the VecEnv pattern from collect_trajectories.py.

Both produce the same output: a directory of .npz episodes (via episode_io)
plus a collection_meta.json summarizing the run.
"""

from __future__ import annotations

import json
import os
import time
from collections import Counter
from datetime import datetime
from typing import Callable

import numpy as np

from parametric_lunar_lander.env import ParameterizedLunarLander
from parametric_lunar_lander.episode_io import save_episode, load_episode, run_episode
from parametric_lunar_lander.physics_config import LunarLanderPhysicsConfig
from lwp.collection.wm_collection_config import ManeuverConfig, StartConfig


def _sample_physics_config(
    rng: np.random.Generator,
    ranges: dict[str, tuple[float, float]],
    twr_range: tuple[float, float] | None = None,
    max_attempts: int = 1000,
) -> LunarLanderPhysicsConfig:
    """Draw a random physics config with each param uniform in its range.

    When twr_range is set, rejection-samples until TWR falls within
    [min_twr, max_twr]. This mirrors SamplingProfile.sample() behavior
    so simple sources (random, heuristic) respect the same TWR constraints
    as RL agent sources.

    Args:
        rng: NumPy random generator.
        ranges: Dict mapping param names to (min, max) tuples.
        twr_range: Optional (min_twr, max_twr) constraint. None = no constraint.
        max_attempts: Max rejection-sampling attempts before RuntimeError.

    Returns:
        LunarLanderPhysicsConfig with random params within ranges (and TWR
        within twr_range if specified).
    """
    for _ in range(max_attempts):
        params = {}
        for name in LunarLanderPhysicsConfig.PARAM_NAMES:
            lo, hi = ranges[name]
            params[name] = float(rng.uniform(lo, hi))
        config = LunarLanderPhysicsConfig(**params)

        if twr_range is None:
            return config

        lo_twr, hi_twr = twr_range
        if lo_twr <= config.twr() <= hi_twr:
            return config

    raise RuntimeError(
        f"Could not sample physics config with TWR in {twr_range} "
        f"after {max_attempts} attempts. The range constraints may be too narrow."
    )


def _write_collection_meta(
    output_dir: str,
    source_type: str,
    n_episodes: int,
    physics_ranges: dict,
    seed: int,
    results: list[dict],
    wall_time: float,
    extra: dict | None = None,
) -> None:
    """Write collection_meta.json summarizing the collection run.

    Args:
        output_dir: Directory where episodes were saved.
        source_type: Policy source type string.
        n_episodes: Requested number of episodes.
        physics_ranges: Dict of param -> (min, max) ranges.
        seed: Base seed used.
        results: List of per-episode result dicts from collection.
        wall_time: Total wall time in seconds.
        extra: Additional fields to include (checkpoint_dir, etc.).
    """
    outcomes = Counter(r["outcome"] for r in results)
    rewards = [r["reward"] for r in results]
    lengths = [r["steps"] for r in results]

    meta = {
        "source_type": source_type,
        "n_episodes": n_episodes,
        "physics_sampling": {
            "method": "uniform",
            "ranges": {k: list(v) for k, v in physics_ranges.items()},
        },
        "seed": seed,
        "created": datetime.now().isoformat(timespec="seconds"),
        "summary": {
            "n_episodes_collected": len(results),
            "outcomes": dict(outcomes),
            "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            "mean_episode_length": float(np.mean(lengths)) if lengths else 0.0,
            "wall_time_seconds": round(wall_time, 1),
        },
    }
    if extra:
        meta.update(extra)

    meta_path = os.path.join(output_dir, "collection_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


def collect_simple(
    policy_fn: Callable[[np.ndarray], np.ndarray],
    output_dir: str,
    n_episodes: int,
    physics_ranges: dict[str, tuple[float, float]],
    source_type: str,
    seed: int = 0,
    max_steps: int = 1000,
    save_frames: bool = False,
    n_workers: int = 1,
    twr_range: tuple[float, float] | None = None,
) -> list[dict]:
    """Collect episodes using a simple policy_fn (no VecNormalize).

    For random and heuristic policies that take raw 15D observations.
    Creates a fresh ParameterizedLunarLander per episode with randomly
    sampled physics, runs the policy, saves the trajectory.

    When n_workers > 1, episodes are collected in parallel using
    multiprocessing.Pool. Each worker creates its own env instance
    (no shared state). The policy_fn is recreated per worker since
    closures can't be pickled — source_type determines which policy
    to instantiate.

    Args:
        policy_fn: Callable(obs) -> action. Takes (15,) obs, returns (2,) action.
            Only used when n_workers=1. For parallel, workers recreate the policy.
        output_dir: Directory for .npz files + collection_meta.json.
        n_episodes: Number of episodes to collect.
        physics_ranges: Dict mapping param names to (min, max) tuples.
        source_type: Label for metadata (e.g., "random", "heuristic").
        seed: Base seed for physics sampling and env resets.
        max_steps: Maximum steps per episode.
        save_frames: Whether to capture RGB frames.
        n_workers: Number of parallel workers for episode collection.

    Returns:
        List of per-episode dicts: {npz_path, outcome, reward, steps}.
    """
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    t0 = time.time()

    # Pre-sample all physics configs (deterministic regardless of n_workers).
    # When twr_range is set (e.g., from an easy/medium profile), each config
    # is rejection-sampled until TWR falls within the range.
    physics_configs = [
        _sample_physics_config(rng, physics_ranges, twr_range=twr_range)
        for _ in range(n_episodes)
    ]

    if n_workers > 1:
        # Parallel collection via multiprocessing.Pool.
        from multiprocessing import Pool

        work_items = [
            (ep_idx, output_dir,
             {name: getattr(physics_configs[ep_idx], name)
              for name in LunarLanderPhysicsConfig.PARAM_NAMES},
             source_type, seed + ep_idx, max_steps, save_frames, source_type)
            for ep_idx in range(n_episodes)
        ]
        with Pool(n_workers) as pool:
            results = pool.map(_collect_single_episode, work_items)
        print(f"  [{source_type}] {n_episodes}/{n_episodes} episodes (parallel, {n_workers} workers)")
    else:
        # Serial collection — uses the provided policy_fn directly.
        results = []
        render_mode = "rgb_array" if save_frames else None

        for ep_idx in range(n_episodes):
            physics_config = physics_configs[ep_idx]
            env = ParameterizedLunarLander(
                physics_config=physics_config,
                render_mode=render_mode,
            )

            ep_seed = seed + ep_idx
            ep_data = run_episode(
                env=env, policy_fn=policy_fn, seed=ep_seed,
                max_steps=max_steps, save_frames=save_frames,
            )
            env.close()

            # Add source_type to metadata (not set by run_episode).
            ep_data["metadata"]["source_type"] = source_type
            npz_path = os.path.join(output_dir, f"episode_{ep_idx:05d}.npz")
            save_episode(
                path=npz_path, states=ep_data["states"], actions=ep_data["actions"],
                rewards=ep_data["rewards"], dones=ep_data["dones"],
                metadata=ep_data["metadata"], rgb_frames=ep_data["rgb_frames"],
            )
            results.append({
                "npz_path": npz_path,
                "outcome": ep_data["metadata"]["outcome"],
                "reward": ep_data["metadata"]["total_reward"],
                "steps": ep_data["metadata"]["n_steps"],
            })

            if (ep_idx + 1) % max(1, n_episodes // 20) == 0 or ep_idx == 0:
                print(f"  [{source_type}] {ep_idx + 1}/{n_episodes} episodes")

    wall_time = time.time() - t0
    _write_collection_meta(
        output_dir=output_dir,
        source_type=source_type,
        n_episodes=n_episodes,
        physics_ranges=physics_ranges,
        seed=seed,
        results=results,
        wall_time=wall_time,
    )

    return results


def _collect_single_episode(args_tuple):
    """Worker function for parallel collection. Runs one episode.

    Takes a single tuple arg for multiprocessing.Pool compatibility.
    Creates its own env instance (no shared state between workers).
    """
    (ep_idx, output_dir, physics_params, source_type,
     ep_seed, max_steps, save_frames, policy_type) = args_tuple

    physics_config = LunarLanderPhysicsConfig(**physics_params)
    render_mode = "rgb_array" if save_frames else None
    env = ParameterizedLunarLander(physics_config=physics_config, render_mode=render_mode)

    # Recreate policy in worker (can't pickle closures across processes).
    if policy_type == "random":
        from lwp.collection.wm_policies import make_random_policy
        policy_fn = make_random_policy(seed=ep_seed)
    elif policy_type == "heuristic":
        from parametric_lunar_lander.heuristic import heuristic_policy
        policy_fn = heuristic_policy
    else:
        raise ValueError(f"Unknown policy_type for simple collection: {policy_type}")

    ep_data = run_episode(env=env, policy_fn=policy_fn, seed=ep_seed,
                          max_steps=max_steps, save_frames=save_frames)
    env.close()

    ep_data["metadata"]["source_type"] = source_type
    npz_path = os.path.join(output_dir, f"episode_{ep_idx:05d}.npz")
    save_episode(
        path=npz_path, states=ep_data["states"], actions=ep_data["actions"],
        rewards=ep_data["rewards"], dones=ep_data["dones"],
        metadata=ep_data["metadata"], rgb_frames=ep_data["rgb_frames"],
    )
    return {
        "npz_path": npz_path,
        "outcome": ep_data["metadata"]["outcome"],
        "reward": ep_data["metadata"]["total_reward"],
        "steps": ep_data["metadata"]["n_steps"],
    }


def collect_rl_agent(
    model,
    output_dir: str,
    n_episodes: int,
    variant: str,
    physics_ranges: dict[str, tuple[float, float]],
    source_type: str,
    seed: int = 0,
    vec_normalize_path: str | None = None,
    deterministic: bool = True,
    save_frames: bool = False,
    noise_sigma_range: tuple[float, float] | None = None,
    n_rays: int = 7,
    history_k: int = 8,
    twr_range: tuple[float, float] | None = None,
) -> list[dict]:
    """Collect episodes from a trained SB3 agent with VecNormalize.

    For blind_agent, labeled_agent, and noisy_expert sources. Sets up the
    full wrapper stack (DomainRandom -> PhysicsBlind -> Raycast -> etc.) with
    VecNormalize so the agent gets correctly normalized observations. Captures
    raw 15D states from the unwrapped base env.

    Adapts the VecEnv-based collection pattern from collect_trajectories.py.

    Args:
        model: Trained SB3 model with .predict(obs, deterministic) -> (action, _).
        output_dir: Directory for .npz files + collection_meta.json.
        n_episodes: Number of episodes to collect.
        variant: Agent variant ("labeled", "blind", "history").
        physics_ranges: Dict mapping param names to (min, max) tuples.
        source_type: Label for metadata (e.g., "blind_agent", "noisy_expert").
        seed: Base seed.
        vec_normalize_path: Path to VecNormalize .pkl stats. None if not used.
        deterministic: Use deterministic policy (no SB3 exploration noise).
        save_frames: Whether to capture RGB frames.
        noise_sigma_range: (lo, hi) range for noise sigma (noisy_expert only).
            Sigma is sampled uniformly per episode from this range. Each step
            within an episode uses the same sigma, with fresh noise draws.
            None means no noise (for blind_agent, labeled_agent).
        n_rays: Number of terrain raycast rays.
        history_k: History stack depth (for history variant).

    Returns:
        List of per-episode dicts: {npz_path, outcome, reward, steps}.
    """
    from parametric_lunar_lander.wrappers import make_lunar_lander_env
    from parametric_lunar_lander.sampling_profiles import SamplingProfile
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    os.makedirs(output_dir, exist_ok=True)
    t0 = time.time()

    # Build a sampling profile from the uniform ranges so DomainRandomization
    # samples physics within the specified range. When twr_range is set
    # (e.g., from an easy/medium batch profile), the profile enforces it
    # via rejection sampling inside DomainRandomizationWrapper.
    profile = SamplingProfile(
        overrides={k: tuple(v) for k, v in physics_ranges.items()},
        twr_range=twr_range,
        name="wm-uniform",
    )

    render_mode = "rgb_array" if save_frames else None

    def env_fn():
        return make_lunar_lander_env(
            variant=variant,
            seed=seed,
            n_rays=n_rays,
            history_k=history_k,
            profile=profile,
            render_mode=render_mode,
        )

    # Single-env VecEnv with optional VecNormalize.
    vec_env = DummyVecEnv([env_fn])
    if vec_normalize_path is not None:
        vec_env = VecNormalize.load(vec_normalize_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False

    # Reach through wrappers to base env for raw state capture.
    inner = vec_env.venv if isinstance(vec_env, VecNormalize) else vec_env
    base_env = inner.envs[0].unwrapped

    # Noise RNG for noisy_expert.
    noise_rng = np.random.default_rng(seed + 999) if noise_sigma_range else None

    # Sample this episode's noise sigma from the range.
    ep_sigma = None
    if noise_sigma_range and noise_rng is not None:
        lo, hi = noise_sigma_range
        ep_sigma = float(noise_rng.uniform(lo, hi))

    results = []
    obs = vec_env.reset()

    # Per-episode accumulators.
    ep_states = [base_env._last_obs.copy()]
    ep_actions = []
    ep_rewards = []
    ep_dones = []
    ep_frames = [base_env.render()] if save_frames else []
    ep_reward_total = 0.0

    while len(results) < n_episodes:
        action, _ = model.predict(obs, deterministic=deterministic)

        # Noisy-expert: add Gaussian noise with this episode's sigma, clipped.
        if ep_sigma is not None and noise_rng is not None:
            noise = noise_rng.normal(0.0, ep_sigma, size=action.shape).astype(np.float32)
            action = np.clip(action + noise, -1.0, 1.0)

        obs, reward, done, infos = vec_env.step(action)

        ep_reward_total += float(reward[0])
        ep_states.append(base_env._last_obs.copy())
        ep_actions.append(action[0].copy())
        ep_rewards.append(float(reward[0]))
        ep_dones.append(bool(done[0]))
        if save_frames:
            ep_frames.append(base_env.render())

        if done[0]:
            ep_idx = len(results)

            # Classify outcome from final reward.
            final_reward = ep_rewards[-1]
            if final_reward >= 100:
                outcome = "landed"
            elif final_reward <= -100:
                outcome = "crashed"
            else:
                outcome = "timeout"

            metadata = {
                "physics_config": base_env._physics_config.to_dict(),
                "outcome": outcome,
                "seed": seed + ep_idx,
                "n_steps": len(ep_actions),
                "total_reward": ep_reward_total,
                "source_type": source_type,
            }
            if ep_sigma is not None:
                metadata["noise_sigma"] = ep_sigma

            npz_path = os.path.join(output_dir, f"episode_{ep_idx:05d}.npz")
            save_episode(
                path=npz_path,
                states=np.array(ep_states, dtype=np.float32),
                actions=np.array(ep_actions, dtype=np.float32),
                rewards=np.array(ep_rewards, dtype=np.float32),
                dones=np.array(ep_dones, dtype=bool),
                metadata=metadata,
                rgb_frames=np.array(ep_frames, dtype=np.uint8) if save_frames else None,
            )

            results.append({
                "npz_path": npz_path,
                "outcome": outcome,
                "reward": ep_reward_total,
                "steps": len(ep_actions),
            })

            if (ep_idx + 1) % max(1, n_episodes // 20) == 0 or ep_idx == 0:
                print(f"  [{source_type}] {ep_idx + 1}/{n_episodes} episodes")

            # Reset accumulators + sample fresh sigma for next episode.
            # VecEnv auto-resets, so base_env already has new state.
            ep_states = [base_env._last_obs.copy()]
            ep_actions = []
            ep_rewards = []
            ep_dones = []
            ep_frames = [base_env.render()] if save_frames else []
            ep_reward_total = 0.0
            if noise_sigma_range and noise_rng is not None:
                lo, hi = noise_sigma_range
                ep_sigma = float(noise_rng.uniform(lo, hi))

    vec_env.close()

    wall_time = time.time() - t0
    extra = {
        "variant": variant,
        "deterministic": deterministic,
    }
    if vec_normalize_path:
        extra["vec_normalize_path"] = vec_normalize_path
    if noise_sigma_range:
        extra["noise_sigma_range"] = list(noise_sigma_range)

    _write_collection_meta(
        output_dir=output_dir,
        source_type=source_type,
        n_episodes=n_episodes,
        physics_ranges=physics_ranges,
        seed=seed,
        results=results,
        wall_time=wall_time,
        extra=extra,
    )

    return results


# ---------------------------------------------------------------------------
# Primitive episode collection
# ---------------------------------------------------------------------------


def _load_source_episodes(source_dir: str) -> list[dict]:
    """Load all episodes from a source directory for replay.

    Scans for .npz files, loads each via load_episode(), and attaches
    the file path as a "_path" key so callers can reference it in
    branch_source metadata.

    Args:
        source_dir: Directory containing .npz episode files.

    Returns:
        List of episode dicts (from load_episode), each with an extra
        "_path" key containing the absolute file path.
    """
    import glob
    paths = sorted(glob.glob(os.path.join(source_dir, "*.npz")))
    episodes = []
    for p in paths:
        ep = load_episode(p)
        ep["_path"] = p
        episodes.append(ep)
    return episodes


def _pick_landed_episode(
    episodes: list[dict], rng: np.random.Generator,
) -> dict | None:
    """Pick a random episode that landed successfully.

    Filters episodes by outcome == "landed" and selects one uniformly
    at random. Returns None if no landed episodes are available.

    Args:
        episodes: List of episode dicts from _load_source_episodes().
        rng: NumPy random generator for selection.

    Returns:
        A single episode dict, or None if no landed episodes exist.
    """
    landed = [ep for ep in episodes if ep["metadata"].get("outcome") == "landed"]
    if not landed:
        return None
    idx = int(rng.integers(0, len(landed)))
    return landed[idx]


def _sample_initial_state(
    ranges: dict[str, tuple[float, float]] | None,
    rng: np.random.Generator,
) -> dict:
    """Sample initial state overrides from config ranges.

    Each key in ranges maps to a (lo, hi) tuple. Values are sampled
    uniformly from the range. Returns a dict suitable for passing
    as **kwargs to override_initial_state().

    Args:
        ranges: Dict mapping state variable names (x, y, vx, vy, angle,
            angular_vel) to (lo, hi) sampling ranges. None returns {}.
        rng: NumPy random generator.

    Returns:
        Dict of sampled state overrides (e.g., {"x": 0.1, "y": 0.6}).
    """
    if ranges is None:
        return {}
    return {name: float(rng.uniform(lo, hi)) for name, (lo, hi) in ranges.items()}


def collect_primitive(
    output_dir: str,
    n_episodes: int,
    physics_ranges: dict[str, tuple[float, float]],
    maneuver_config: ManeuverConfig,
    start_config: StartConfig,
    seed: int = 0,
    max_steps: int = 1000,
    save_frames: bool = False,
    allow_post_landing: bool = False,
    twr_range: tuple[float, float] | None = None,
) -> list[dict]:
    """Collect primitive episodes with controlled actions.

    For each episode:
      1. Set up starting state (fresh reset or replay-to-branch).
      2. Generate action sequence from maneuver config.
      3. Step env with those actions until termination or max_steps.
      4. Save episode with primitive-specific metadata.

    Three start modes:
      - fresh_reset: Create env with sampled physics, reset, override
        initial state from ranges, step once with zero action to populate
        _last_obs, optionally wrap with PostLandingWrapper.
      - replay: Pick a random source episode + branch point, replay
        actions up to that point, then apply the maneuver from there.
      - replay_to_landing: Pick a landed source episode, replay the
        full episode past landing, then apply the maneuver on the ground.

    Args:
        output_dir: Directory for .npz files + collection_meta.json.
        n_episodes: Number of episodes to collect.
        physics_ranges: Dict mapping param names to (min, max) tuples.
        maneuver_config: Maneuver type and action parameters.
        start_config: How to initialize each episode.
        seed: Base seed for RNG.
        max_steps: Maximum steps per episode (default 1000).
        save_frames: Whether to capture RGB frames.
        allow_post_landing: If True, suppress landing/crash termination
            so episodes continue past touchdown. Required for ground_*
            maneuvers with replay_to_landing start mode.
        twr_range: Optional TWR constraint for physics sampling.

    Returns:
        List of per-episode dicts: {npz_path, outcome, reward, steps}.
    """
    from lwp.collection.env_wrappers_collection import (
        override_initial_state, PostLandingWrapper,
    )
    from lwp.collection.wm_primitives import (
        generate_actions, replay_to_branch_point,
    )

    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    t0 = time.time()

    # Pre-load source episodes for replay modes (loaded once, reused).
    source_episodes = None
    if start_config.mode in ("replay", "replay_to_landing"):
        if start_config.source_dir is None:
            raise ValueError(
                f"start mode '{start_config.mode}' requires source_dir"
            )
        source_episodes = _load_source_episodes(start_config.source_dir)
        if not source_episodes:
            raise ValueError(
                f"No .npz episodes found in source_dir: {start_config.source_dir}"
            )

    render_mode = "rgb_array" if save_frames else None
    results = []

    for ep_idx in range(n_episodes):
        ep_seed = seed + ep_idx

        # ---------------------------------------------------------------
        # Step 1: Set up the environment based on start mode
        # ---------------------------------------------------------------

        # physics_config is needed for action generation (e.g., hover
        # computes equilibrium thrust from gravity and engine power).
        # For fresh_reset, we sample new physics. For replay modes,
        # we use the source episode's physics config.
        physics_config = None
        branch_source_meta = None
        termination_event = None

        if start_config.mode == "fresh_reset":
            # Sample physics for this episode. When twr_range is set,
            # rejection-samples until TWR falls within the range.
            physics_config = _sample_physics_config(
                rng, physics_ranges, twr_range=twr_range,
            )

            # Create env with the sampled physics and reset it.
            env = ParameterizedLunarLander(
                physics_config=physics_config, render_mode=render_mode,
            )
            env.reset(seed=ep_seed)

            # Override initial state from config ranges. This sets the
            # Box2D body position/velocity/angle before any simulation.
            overrides = _sample_initial_state(start_config.initial_state, rng)
            override_initial_state(env, **overrides)

            # Settle steps: run zero-action steps so Box2D resolves joint
            # constraint transients from the position override. The legs
            # are separate bodies connected via revolute joints — moving
            # them introduces internal stress that needs a few physics
            # ticks to dissipate. Without this, legs start crooked and
            # cause erratic dynamics. 5 steps matches physics_test_gt.py.
            _SETTLE_STEPS = 5
            _zero_action = np.array([0.0, 0.0], dtype=np.float32)
            for _ in range(_SETTLE_STEPS):
                env.step(_zero_action)

            # Wrap with PostLandingWrapper AFTER the initial setup step,
            # so the wrapper doesn't interfere with initialization.
            if allow_post_landing:
                env = PostLandingWrapper(env)

            # Record the initial state overrides for metadata.
            start_state = overrides

        elif start_config.mode == "replay":
            # Pick a random source episode and branch point within it.
            src_idx = int(rng.integers(0, len(source_episodes)))
            src_ep = source_episodes[src_idx]
            n_src_steps = len(src_ep["actions"])

            # Compute branch point within [min_step, max_step_fraction * n_steps].
            max_branch = max(
                start_config.min_step + 1,
                int(start_config.max_step_fraction * n_src_steps),
            )
            max_branch = min(max_branch, n_src_steps)
            branch_step = int(rng.integers(
                start_config.min_step, max_branch,
            ))

            # Replay source episode to branch point. This creates a new
            # env with the source episode's physics and seed, replays
            # the recorded actions, and returns the live env at that state.
            env = replay_to_branch_point(
                episode=src_ep,
                branch_point=branch_step,
                render_mode=render_mode,
                allow_post_landing=allow_post_landing,
            )

            # Use the source episode's physics for action generation
            # (e.g., hover thrust computation).
            physics_config = LunarLanderPhysicsConfig.from_dict(
                src_ep["metadata"]["physics_config"]
            )

            # Record branch source info for metadata.
            branch_source_meta = {
                "episode_path": src_ep.get("_path", "unknown"),
                "branch_step": branch_step,
                "source_seed": src_ep["metadata"].get("seed", -1),
            }

            # Start state records the branch point context.
            start_state = {
                "branch_step": branch_step,
                "source_episode": os.path.basename(
                    src_ep.get("_path", "unknown")
                ),
            }

        elif start_config.mode == "replay_to_landing":
            # Pick a source episode that landed successfully.
            src_ep = _pick_landed_episode(source_episodes, rng)
            if src_ep is None:
                # No landed episodes — skip this attempt but keep trying.
                # In practice, the test creates heuristic episodes that
                # almost always land, so this fallback is rare.
                print(
                    f"  [primitive] Warning: no landed episodes in source, "
                    f"skipping ep_idx={ep_idx}"
                )
                continue

            # Replay the entire episode past landing. allow_post_landing=True
            # wraps with PostLandingWrapper inside replay_to_branch_point,
            # so the env continues after the lander touches down.
            n_src_steps = len(src_ep["actions"])
            env = replay_to_branch_point(
                episode=src_ep,
                branch_point=n_src_steps,  # replay full episode
                render_mode=render_mode,
                allow_post_landing=True,  # always True for replay_to_landing
            )

            physics_config = LunarLanderPhysicsConfig.from_dict(
                src_ep["metadata"]["physics_config"]
            )

            branch_source_meta = {
                "episode_path": src_ep.get("_path", "unknown"),
                "branch_step": n_src_steps,
                "source_seed": src_ep["metadata"].get("seed", -1),
            }

            start_state = {
                "branch_step": n_src_steps,
                "source_episode": os.path.basename(
                    src_ep.get("_path", "unknown")
                ),
            }

            # Capture the termination event from replay (the landing that
            # happened during the source episode's replay).
            if hasattr(env, "termination_event") and env.termination_event:
                termination_event = env.termination_event

        else:
            raise ValueError(f"Unknown start mode: {start_config.mode}")

        # ---------------------------------------------------------------
        # Step 2: Generate actions and run the episode
        # ---------------------------------------------------------------

        actions_array, maneuver_params = generate_actions(
            config=maneuver_config,
            n_steps=max_steps,
            rng=rng,
            physics_config=physics_config,
        )

        # Collect trajectory data by stepping with the generated actions.
        # States come from env.unwrapped._last_obs (raw 15D) since the
        # env might be wrapped with PostLandingWrapper.
        states_list = [env.unwrapped._last_obs.copy()]
        actions_list = []
        rewards_list = []
        dones_list = []
        frames_list = []

        if save_frames:
            frame = env.unwrapped.render()
            frames_list.append(frame)

        outcome = "timeout"
        total_reward = 0.0

        for step_idx in range(max_steps):
            action = actions_array[step_idx]
            obs, reward, terminated, truncated, step_info = env.step(action)

            # Always capture raw 15D state from the unwrapped env,
            # not from obs (which may be transformed by wrappers).
            states_list.append(env.unwrapped._last_obs.copy())
            actions_list.append(action.copy())
            rewards_list.append(float(reward))
            dones_list.append(bool(terminated))
            total_reward += float(reward)

            if save_frames:
                frames_list.append(env.unwrapped.render())

            if terminated:
                # Classify outcome from step_info if available,
                # otherwise fall back to reward-based heuristics.
                if "outcome" in step_info:
                    outcome = step_info["outcome"]
                elif reward >= 100:
                    outcome = "landed"
                elif abs(env.unwrapped._last_obs[0]) >= 1.0:
                    outcome = "out_of_bounds"
                else:
                    outcome = "crashed"
                break

        # Capture termination event from PostLandingWrapper if present.
        # This covers cases where termination happened DURING the primitive
        # episode (not during replay). Must be done before env.close().
        if (
            allow_post_landing
            and hasattr(env, "termination_event")
            and env.termination_event is not None
            and termination_event is None
        ):
            termination_event = env.termination_event

        env.close()

        # ---------------------------------------------------------------
        # Step 3: Build metadata and save the episode
        # ---------------------------------------------------------------

        n_steps_actual = len(actions_list)
        metadata = {
            "physics_config": physics_config.to_dict(),
            "source_type": "primitive",
            "maneuver_type": maneuver_config.type,
            "maneuver_params": maneuver_params,
            "start_mode": start_config.mode,
            "start_state": start_state,
            "outcome": outcome,
            "seed": ep_seed,
            "n_steps": n_steps_actual,
            "total_reward": float(total_reward),
        }

        # Add branch source info for replay modes.
        if branch_source_meta is not None:
            metadata["branch_source"] = branch_source_meta

        # Add termination event info for post-landing episodes.
        # The PostLandingWrapper records the first suppressed termination
        # (e.g., "landed" at step N) so we can reconstruct the timeline.
        if termination_event is not None:
            metadata["termination_event"] = termination_event

        npz_path = os.path.join(output_dir, f"episode_{ep_idx:05d}.npz")
        save_episode(
            path=npz_path,
            states=np.array(states_list, dtype=np.float32),
            actions=np.array(actions_list, dtype=np.float32),
            rewards=np.array(rewards_list, dtype=np.float32),
            dones=np.array(dones_list, dtype=bool),
            metadata=metadata,
            rgb_frames=(
                np.array(frames_list, dtype=np.uint8) if save_frames else None
            ),
        )

        results.append({
            "npz_path": npz_path,
            "outcome": outcome,
            "reward": total_reward,
            "steps": n_steps_actual,
        })

        # Progress printing: every 5% of episodes.
        if (ep_idx + 1) % max(1, n_episodes // 20) == 0 or ep_idx == 0:
            print(f"  [primitive] {ep_idx + 1}/{n_episodes} episodes")

    # Write collection-level metadata summarizing the run.
    wall_time = time.time() - t0
    _write_collection_meta(
        output_dir=output_dir,
        source_type="primitive",
        n_episodes=n_episodes,
        physics_ranges=physics_ranges,
        seed=seed,
        results=results,
        wall_time=wall_time,
        extra={
            "maneuver_type": maneuver_config.type,
            "start_mode": start_config.mode,
            "max_steps": max_steps,
            "allow_post_landing": allow_post_landing,
        },
    )

    return results
