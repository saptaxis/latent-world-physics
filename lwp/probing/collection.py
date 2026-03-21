"""Probe data collection: run episodes with forward hooks to capture activations.

Follows the same env setup pattern as collect_trajectories.py but captures
hidden layer activations instead of saving per-episode .npz files.

Supports both state-vector agents (hook MLP hidden layers) and visual
agents (hook CNN encoder output + MLP hidden layers). The variant
parameter controls which layers are probed.

See probing-tooling.md (Sections 1-2) for the full specification.
"""
import json

import numpy as np
import torch.nn as nn
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecFrameStack

from lwp.agents.eval_utils import _make_env_thunk
from parametric_lunar_lander.physics_config import LunarLanderPhysicsConfig
from lwp.probing.hooks import ActivationCollector
from lwp.probing.targets import compute_behavioral_targets


def build_layer_specs(model, variant: str = "blind") -> list[tuple[str, nn.Module]]:
    """Build (name, module) pairs for activation hooking based on variant.

    For state-vector agents (blind, labeled, etc.):
        L1, L2 — post-ReLU MLP hidden layers (128D each)

    For visual agents (visual, visual-labeled):
        CNN — features_extractor output (512D from NatureCNN)
        L1, L2 — post-ReLU MLP hidden layers (128D each)

    Args:
        model: Trained SB3 model (PPO or SAC).
        variant: Agent variant string from config.json.

    Returns:
        List of (name, module) tuples for ActivationCollector.
    """
    specs = []

    # CNN encoder output for visual agents
    if variant.startswith("visual"):
        features_extractor = model.policy.features_extractor
        specs.append(("CNN", features_extractor))

    # MLP hidden layers (present in all variants)
    policy_net = model.policy.mlp_extractor.policy_net
    relu_indices = [
        i for i, layer in enumerate(policy_net)
        if isinstance(layer, (nn.ReLU, nn.Tanh, nn.ELU, nn.LeakyReLU))
    ]
    for j, idx in enumerate(relu_indices):
        specs.append((f"L{j+1}", policy_net[idx]))

    return specs


def collect_probe_data(
    model,
    env_fn,
    n_episodes: int,
    seed: int = 0,
    vec_normalize_path: str | None = None,
    deterministic: bool = True,
    variant: str = "blind",
    n_stack: int = 0,
) -> dict[str, np.ndarray | str]:
    """Collect activations and targets from a trained agent.

    Runs episodes through the full wrapper stack with forward hooks on
    the policy network. At each timestep, captures:
    - Layer activations (CNN output for visual, post-ReLU for MLP layers)
    - Physics parameters (from base env)
    - Behavioral targets (computed from physics config)
    - Episode IDs (for episode-level splitting)

    Args:
        model: Trained SB3 model (PPO or SAC).
        env_fn: Factory taking seed, returns wrapped env.
        n_episodes: Number of episodes to collect.
        seed: Base seed for env creation.
        vec_normalize_path: Path to VecNormalize .pkl stats.
        deterministic: Use deterministic policy.
        variant: Agent variant (blind, labeled, visual, visual-labeled).
            Controls which layers are hooked.
        n_stack: Frame stacking depth for visual agents (0 = no stacking).

    Returns:
        Dict with keys:
            activations_{name}: (N, dim) float32 for each hooked layer
            layer_names: JSON list of layer names (for downstream code)
            physics_params: (N, 7) float32
            behavioral: (N, 5) float32
            episode_ids: (N,) int32
            metadata_json: JSON string with collection params
    """
    # Set up env with VecFrameStack + VecNormalize (same stack as training)
    vec_env = DummyVecEnv([_make_env_thunk(env_fn, seed)])
    if n_stack > 0:
        vec_env = VecFrameStack(vec_env, n_stack=n_stack)
    if vec_normalize_path is not None:
        vec_env = VecNormalize.load(vec_normalize_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False

    # Get base env for physics config access.
    # Peel through VecNormalize, VecFrameStack, etc.
    inner = vec_env
    while hasattr(inner, "venv"):
        inner = inner.venv
    base_env = inner.envs[0].unwrapped

    # Set up activation hooks
    collector = None
    layer_names = []
    try:
        layer_specs = build_layer_specs(model, variant=variant)
        layer_names = [name for name, _ in layer_specs]
        collector = ActivationCollector(layer_specs=layer_specs)
    except (AttributeError, ValueError):
        # Mock model or unexpected structure — skip hooks
        pass

    # Accumulators: one list per hooked layer
    all_acts = {name: [] for name in layer_names}
    all_physics = []
    all_behavioral = []
    all_kinematic = []
    all_episode_ids = []

    obs = vec_env.reset()
    episode_count = 0

    # Get current episode's physics config
    current_physics = base_env._physics_config
    current_params = current_physics.as_array()
    current_behavioral = compute_behavioral_targets(current_physics)

    while episode_count < n_episodes:
        # Capture kinematic state BEFORE predict — this is the state that
        # generated the current observation (obs). The CNN processes obs,
        # producing activations. So activations[t] aligns with kinematics[t].
        # First 8 dims of _last_obs: x, y, vx, vy, angle, angular_vel,
        # left_leg_contact, right_leg_contact.
        all_kinematic.append(base_env._last_obs[:8].copy())

        action, _ = model.predict(obs, deterministic=deterministic)

        # Capture activations (hooks fire during predict)
        if collector is not None:
            acts = collector.get()
            for name in layer_names:
                all_acts[name].append(acts.get(name, np.zeros(1)))
        else:
            for name in layer_names:
                all_acts[name].append(np.zeros(1))

        all_physics.append(current_params)
        all_behavioral.append(current_behavioral)
        all_episode_ids.append(episode_count)

        obs, reward, done, infos = vec_env.step(action)

        if done[0]:
            episode_count += 1
            if episode_count < n_episodes:
                # New episode — get fresh physics config
                current_physics = base_env._physics_config
                current_params = current_physics.as_array()
                current_behavioral = compute_behavioral_targets(current_physics)

    vec_env.close()
    if collector is not None:
        collector.remove()

    n_timesteps = len(all_episode_ids)
    metadata = {
        "n_episodes": n_episodes,
        "n_timesteps": n_timesteps,
        "seed": seed,
        "deterministic": deterministic,
        "variant": variant,
        "n_stack": n_stack,
        "layer_names": layer_names,
    }

    result = {
        "physics_params": np.array(all_physics, dtype=np.float32),
        "behavioral": np.array(all_behavioral, dtype=np.float32),
        "kinematic": np.array(all_kinematic, dtype=np.float32),
        "episode_ids": np.array(all_episode_ids, dtype=np.int32),
        "layer_names": json.dumps(layer_names),
        "metadata_json": json.dumps(metadata),
    }

    # Add activation arrays keyed by layer name
    for name in layer_names:
        result[f"activations_{name}"] = np.array(all_acts[name], dtype=np.float32)

    # Backward compat: if L1/L2 exist, also write them as activations_L1/L2
    # (already handled by the loop above since MLP layers are named L1, L2)

    return result
