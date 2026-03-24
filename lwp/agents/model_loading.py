"""Convenience functions for loading trained RL agents.

Handles the full workflow: config.json → compat shim → SB3 model load →
VecNormalize → VecFrameStack. Works for both old (lwg) and new (lwp)
checkpoints, visual and state-vector agents.

Usage:
    from lwp.agents.model_loading import load_model, load_eval_env

    model, config = load_model("/path/to/agent/s42")
    env = load_eval_env(config, "/path/to/agent/s42")
    result = evaluate_agent(model, env_fn, ...)
"""
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize

from lwp.agents.eval_utils import (
    resolve_model_path,
    resolve_vec_normalize_path,
    load_training_config,
    make_env_factory,
    _make_env_thunk,
)


def load_model(
    checkpoint_dir: str,
    model_name: str | None = None,
    device: str = "auto",
) -> tuple:
    """Load a trained SB3 model from a checkpoint directory.

    Reads config.json for algo type, resolves the model .zip path,
    imports the compat shim for old checkpoints, and calls the
    appropriate SB3 AlgoClass.load().

    Args:
        checkpoint_dir: Path to the agent's seed directory (contains
            config.json, model.zip or best/model.zip, etc.)
        model_name: Specific model filename. None = best/model.zip.
        device: PyTorch device ("cpu", "cuda", "auto").

    Returns:
        (model, config) tuple where model is a loaded SB3 BaseAlgorithm
        and config is the parsed config.json dict.

    Raises:
        FileNotFoundError: If checkpoint_dir or model file doesn't exist.
    """
    # Ensure the compat shim is active before AlgoClass.load(). The shim
    # maps old lunar_lander.src.* paths to lwp.agents.* so cloudpickle can
    # deserialize pre-migration checkpoints. We call register_compat_modules()
    # directly (not just import lwp.compat) because import is a no-op if the
    # module is already cached, but sys.modules may have been cleaned up by
    # test teardown between the initial import and this call.
    from lwp.compat import register_compat_modules
    register_compat_modules()

    config = load_training_config(checkpoint_dir)
    model_path = resolve_model_path(checkpoint_dir, model_name)

    algo = config.get("algo", "ppo").lower()
    AlgoClass = PPO if algo == "ppo" else SAC

    model = AlgoClass.load(model_path, device=device)

    return model, config


def load_eval_env(
    config: dict,
    checkpoint_dir: str,
    seed: int = 42,
    model_name: str | None = None,
):
    """Build a ready-to-use eval VecEnv from an agent's config.

    Creates the correct wrapper stack:
        make_env_factory → DummyVecEnv → VecFrameStack (visual) → VecNormalize

    Args:
        config: Parsed config.json dict (from load_model).
        checkpoint_dir: Path to agent seed dir (for VecNormalize .pkl).
        seed: Env random seed.
        model_name: Specific model filename (for matching VecNormalize).

    Returns:
        A VecEnv ready for model.predict() / evaluate_agent().
    """
    variant = config["variant"]
    is_visual = variant.startswith("visual")
    n_stack = config.get("n_stack", 0) if is_visual else 0
    frame_size = config.get("frame_size", 84)
    n_rays = config.get("n_rays", 7)
    history_k = config.get("history_k", 8)
    profile = config.get("profile")

    env_fn = make_env_factory(
        variant=variant,
        n_rays=n_rays,
        history_k=history_k,
        frame_size=frame_size,
        profile=profile,
    )

    vec_env = DummyVecEnv([_make_env_thunk(env_fn, seed)])

    if n_stack > 0:
        vec_env = VecFrameStack(vec_env, n_stack=n_stack)

    # Load VecNormalize stats if available.
    model_path = resolve_model_path(checkpoint_dir, model_name)
    vec_norm_path = resolve_vec_normalize_path(checkpoint_dir, model_path)
    if vec_norm_path is not None:
        vec_env = VecNormalize.load(vec_norm_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False

    return vec_env
