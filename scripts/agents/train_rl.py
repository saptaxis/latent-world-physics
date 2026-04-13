#!/usr/bin/env python
"""Train a single RL agent on the Lunar Lander environment.

Supports three generalist variants (blind, labeled, history) that differ
in what physics information the agent receives. All variants get terrain
rays. Uses SB3 PPO or SAC via rl_common.

Usage:
    # Config-driven (preferred — one YAML defines the full run)
    python lunar_lander/scripts/train_rl.py --config labeled-ppo
    python lunar_lander/scripts/train_rl.py --config blind-sac --total-steps 5000000

    # Small-network baseline (RL Zoo 2x64 architecture)
    python lunar_lander/scripts/train_rl.py --config labeled-ppo-small

    # Gym-default baseline (fixed physics, no randomization)
    python lunar_lander/scripts/train_rl.py --config labeled-ppo-gym-default

    # CLI-only (all flags specified directly, no config file)
    python lunar_lander/scripts/train_rl.py --variant labeled --algo ppo --profile easy

    # Custom output directory (overrides YAML run_dir)
    python lunar_lander/scripts/train_rl.py --config labeled-ppo --run-dir /tmp/ll-test

    # Resume interrupted training
    python lunar_lander/scripts/train_rl.py --config labeled-ppo --resume
"""

import os
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
import json
import time
import argparse
import subprocess
from datetime import datetime
from pathlib import Path


import glob
import shutil
import numpy as np

from parametric_lunar_lander.wrappers import make_lunar_lander_env
from lwp.agents.eval_utils import evaluate_agent
from lwp.agents.training_config import TRAINING_DEFAULTS, load_training_config
from lwp.rl.training import train

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure as configure_logger
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

VARIANTS = ["blind", "labeled", "history", "visual"]


def build_checkpoint_dir(base_dir, variant, algo):
    return os.path.join(base_dir, "generalists", variant, algo)


def build_log_dir(base_dir, variant, algo):
    return os.path.join(base_dir, "generalists", variant, algo)


def _git_commit_hash():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, cwd=REPO_ROOT,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _jsonable_value(v):
    """Convert a value to a JSON-serializable form."""
    if isinstance(v, type):
        return v.__name__
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, dict):
        return {k: _jsonable_value(val) for k, val in v.items()}
    return v


def _jsonable_policy_kwargs(policy_kwargs):
    """Make policy_kwargs JSON-serializable for config.json."""
    if not policy_kwargs:
        return None
    return {k: _jsonable_value(v) for k, v in policy_kwargs.items()}


def save_training_config(checkpoint_dir, config, policy_kwargs, policy_class,
                         resumed_from=None):
    """Save resolved training config to config.json.

    Args:
        checkpoint_dir: Directory to save config.json in.
        config: Resolved training config dict (from _resolve_config).
        policy_kwargs: Actual policy_kwargs passed to the algorithm (may differ
            from config for visual variants).
        policy_class: Actual policy class string ("MlpPolicy" or "CnnPolicy").
        resumed_from: Provenance info dict when resuming from another run.
    """
    output = {
        "env": "lunar_lander",
        "variant": config["variant"],
        "algo": config["algo"],
        "total_steps": config["total_steps"],
        "n_envs": config["n_envs"],
        "history_k": config["history_k"],
        "n_rays": config["n_rays"],
        "seed": config["seed"],
        "net_arch": config["net_arch"],
        "policy_kwargs": _jsonable_policy_kwargs(policy_kwargs),
        "checkpoint_freq": config["checkpoint_freq"],
        "eval_freq": config["eval_freq"],
        "video_freq": config["video_freq"],
        "twr_min": config["twr_min"],
        "twr_max": config["twr_max"],
        "profile": config["profile"],
        "curriculum": config["curriculum"],
        "run_dir": config["run_dir"],
        "ent_coef": config["ent_coef"],
        "learning_rate": config.get("learning_rate"),
        "n_steps": config.get("n_steps"),
        "batch_size": config.get("batch_size"),
        "n_epochs": config.get("n_epochs"),
        "gamma": config.get("gamma"),
        "gae_lambda": config.get("gae_lambda"),
        "clip_range": config.get("clip_range"),
        "max_grad_norm": config.get("max_grad_norm"),
        "n_stack": config.get("n_stack", 0),
        "frame_size": config.get("frame_size", 84),
        "share_features_extractor": config.get("share_features_extractor", True),
        "cnn_backbone": config.get("cnn_backbone", "nature"),
        "cnn_channels": config.get("cnn_channels"),
        "cnn_pool_size": config.get("cnn_pool_size"),
        "cnn_features_dim": config.get("cnn_features_dim"),
        "lr_schedule": config.get("lr_schedule"),
        "angle_shaping_omega": config.get("angle_shaping_omega"),
        "time_penalty": config.get("time_penalty"),
        "aux_kinematic": config.get("aux_kinematic", False),
        "aux_coef": config.get("aux_coef", 0.5),
        "encoder_weights": config.get("encoder_weights"),
        "physics_branch_dim": config.get("physics_branch_dim"),
        "freeze_encoder": config.get("freeze_encoder", False),
        "policy_class": policy_class,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git_commit_hash(),
    }
    if resumed_from is not None:
        output["resumed_from"] = resumed_from
    path = os.path.join(checkpoint_dir, "config.json")
    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    return path


def _resolve_config(args):
    """Merge TRAINING_DEFAULTS -> YAML config -> CLI args into one dict.

    All argparse defaults are None, so any non-None value means the user
    explicitly passed that flag on the command line. This lets CLI args
    cleanly override YAML values without ambiguity.

    Args:
        args: Parsed argparse namespace (all defaults are None).

    Returns:
        Fully resolved config dict with all TRAINING_DEFAULTS keys.
    """
    # Start from YAML config or bare defaults
    if args.config:
        config = load_training_config(args.config)
    else:
        config = TRAINING_DEFAULTS.copy()

    # CLI overrides: non-None values win over YAML/defaults.
    cli_keys = [
        "variant", "algo", "total_steps", "n_envs", "seed",
        "history_k", "n_rays", "checkpoint_freq", "eval_freq",
        "video_freq", "profile", "curriculum", "twr_min", "twr_max",
        "run_dir", "early_stop_landed_pct", "early_stop_patience",
        "n_stack",
    ]
    for key in cli_keys:
        val = getattr(args, key, None)
        if val is not None:
            config[key] = val

    return config


def _find_latest_checkpoint(checkpoint_dir):
    """Find the most recent rl_model_*_steps.zip checkpoint.

    Looks in checkpoint_dir/checkpoints/ first (new layout), then falls
    back to checkpoint_dir/ (old flat layout) for backwards compatibility.

    Returns (path, step_count) or (None, 0) if no checkpoints found.
    """
    # New layout: intermediate checkpoints in checkpoints/ subdir
    subdir = os.path.join(checkpoint_dir, "checkpoints")
    pattern = os.path.join(subdir, "rl_model_*_steps.zip")
    checkpoints = glob.glob(pattern)

    # Fallback: old flat layout (checkpoints at top level)
    if not checkpoints:
        pattern = os.path.join(checkpoint_dir, "rl_model_*_steps.zip")
        checkpoints = glob.glob(pattern)

    if not checkpoints:
        return None, 0

    # Extract step count from filename: rl_model_100000_steps.zip -> 100000
    def _step_count(path):
        name = os.path.basename(path)
        # rl_model_100000_steps.zip
        parts = name.replace(".zip", "").split("_")
        # ['rl', 'model', '100000', 'steps']
        return int(parts[2])

    best = max(checkpoints, key=_step_count)
    return best, _step_count(best)


def _make_env_thunk(env_fn, seed):
    """Zero-arg callable for SubprocVecEnv (captures seed in closure)."""
    def thunk():
        return env_fn(seed)
    return thunk


def resume_training(
    env_fn,
    algo,
    total_steps,
    checkpoint_dir,
    log_dir,
    n_envs=8,
    seed=42,
    checkpoint_freq=100_000,
    eval_freq=50_000,
    eval_episodes=10,
    extra_callbacks=None,
    policy_kwargs=None,
    norm_obs=True,
    n_stack=0,
):
    """Resume training from the latest checkpoint in checkpoint_dir.

    Loads the model and VecNormalize stats, builds fresh envs, and
    continues training for the remaining steps. The SB3 timestep counter
    is preserved (reset_num_timesteps=False) so tensorboard curves are
    continuous.

    Returns (model, vec_env, steps_completed_before) or None if no
    checkpoint found.
    """
    checkpoint_path, steps_done = _find_latest_checkpoint(checkpoint_dir)
    if checkpoint_path is None:
        return None

    remaining = total_steps - steps_done
    if remaining <= 0:
        print(f"Training already complete ({steps_done:,} >= {total_steps:,} steps).")
        return None

    print(f"Resuming from {os.path.basename(checkpoint_path)} ({steps_done:,} steps done)")
    print(f"Remaining: {remaining:,} steps")

    algo = algo.lower()
    AlgoClass = PPO if algo == "ppo" else SAC

    # --- Build fresh vectorized envs ---
    train_vec_env = SubprocVecEnv(
        [_make_env_thunk(env_fn, seed + i) for i in range(n_envs)]
    )

    # Frame stacking for visual (must be before VecNormalize)
    if n_stack > 0:
        from stable_baselines3.common.vec_env import VecFrameStack
        train_vec_env = VecFrameStack(train_vec_env, n_stack=n_stack)

    # Load VecNormalize running stats from the checkpoint.
    # CheckpointCallback saves vec_normalize alongside each model checkpoint.
    # The naming convention is: rl_model_100000_steps.zip ->
    # rl_model_vecnormalize_100000_steps.pkl
    vec_norm_checkpoint = checkpoint_path.replace(
        "rl_model_", "rl_model_vecnormalize_"
    ).replace(".zip", ".pkl")

    if os.path.exists(vec_norm_checkpoint):
        train_vec_env = VecNormalize.load(vec_norm_checkpoint, train_vec_env)
        # Keep training=True so running stats continue updating
        train_vec_env.training = True
        train_vec_env.norm_reward = True
        print(f"Loaded VecNormalize stats from {os.path.basename(vec_norm_checkpoint)}")
    else:
        # Fallback: try vec_normalize.pkl in checkpoint dir
        fallback = os.path.join(checkpoint_dir, "vec_normalize.pkl")
        if os.path.exists(fallback):
            train_vec_env = VecNormalize.load(fallback, train_vec_env)
            train_vec_env.training = True
            train_vec_env.norm_reward = True
            print(f"Loaded VecNormalize stats from vec_normalize.pkl")
        else:
            print("WARNING: No VecNormalize stats found — starting fresh normalization")
            train_vec_env = VecNormalize(
                train_vec_env, norm_obs=norm_obs, norm_reward=True,
                clip_obs=10.0, clip_reward=10.0,
            )

    # --- Build eval env ---
    eval_vec_env = SubprocVecEnv(
        [_make_env_thunk(env_fn, seed + n_envs)]
    )
    if n_stack > 0:
        from stable_baselines3.common.vec_env import VecFrameStack
        eval_vec_env = VecFrameStack(eval_vec_env, n_stack=n_stack)
    eval_vec_env = VecNormalize(
        eval_vec_env, norm_obs=norm_obs, norm_reward=False, clip_obs=10.0,
    )

    # --- Load model with the training env attached ---
    model = AlgoClass.load(checkpoint_path, env=train_vec_env, device="auto")

    # Note: net_arch is baked into the model at creation time.
    # On resume, the loaded model keeps its original architecture.
    # If policy_kwargs specifies a different net_arch, we warn.
    if policy_kwargs and "net_arch" in policy_kwargs:
        print(f"Note: net_arch from config ignored on resume (model architecture is fixed)")

    # Re-configure logger to append to the same log dir
    logger = configure_logger(log_dir, ["stdout", "tensorboard"])
    model.set_logger(logger)

    # Dump architecture on resume too (confirms what was actually loaded)
    arch_str = str(model.policy)
    n_params = sum(p.numel() for p in model.policy.parameters())
    print(f"\n{'='*60}")
    print(f"Resumed architecture ({n_params:,} params)")
    print(f"{'='*60}")
    print(arch_str)
    print(f"{'='*60}\n")

    # --- Callbacks (same as fresh training, minus video) ---
    # Intermediate checkpoints go in checkpoints/ subdir (matches train()).
    intermediate_checkpoint_dir = os.path.join(checkpoint_dir, "checkpoints")
    checkpoint_cb = CheckpointCallback(
        save_freq=max(checkpoint_freq // n_envs, 1),
        save_path=intermediate_checkpoint_dir,
        name_prefix="rl_model",
        save_vecnormalize=True,
    )
    callbacks = [checkpoint_cb]
    if eval_freq > 0:
        eval_cb = EvalCallback(
            eval_vec_env,
            best_model_save_path=os.path.join(checkpoint_dir, "best/"),
            log_path=os.path.join(log_dir, "eval/"),
            eval_freq=max(eval_freq // n_envs, 1),
            n_eval_episodes=eval_episodes,
            deterministic=True,
        )
        callbacks.append(eval_cb)
    if extra_callbacks:
        callbacks.extend(extra_callbacks)

    # --- Continue training ---
    # reset_num_timesteps=False preserves the step counter so tensorboard
    # curves are continuous from where we left off.
    model.learn(
        total_timesteps=total_steps,
        callback=callbacks,
        reset_num_timesteps=False,
    )

    # --- Save final model ---
    model.save(os.path.join(checkpoint_dir, "model"))
    train_vec_env.save(os.path.join(checkpoint_dir, "vec_normalize.pkl"))

    eval_vec_env.close()
    return model, train_vec_env, steps_done


class AnnotatedVideoCallback(BaseCallback):
    """Record annotated Lunar Lander videos at regular intervals during training.

    Unlike the generic VideoRecorderCallback in rl_common (which produces raw
    gymnasium frames), this callback saves episodes as .npz files and renders
    them through visualize_trajectory's annotation pipeline — producing the
    same side-panel videos as eval_agent.py and collect_grid.py.

    This keeps rl_common env-agnostic while giving Lunar Lander training the
    same annotated videos (physics params, lander state, actions, rewards)
    that we get from lwp.evaluation.

    Args:
        env_fn: Factory that takes a seed and returns a wrapped env.
        video_dir: Base directory for video output.
        video_freq: Record every N timesteps.
        n_episodes: Number of episodes to record each time.
        checkpoint_dir: Where VecNormalize stats are saved.
        deterministic: Use deterministic actions for recording.
        fps: Video framerate (default 50 to match Box2D physics).
    """

    def __init__(
        self,
        env_fn,
        video_dir,
        video_freq,
        n_episodes=2,
        checkpoint_dir="checkpoints/",
        deterministic=True,
        fps=50,
        n_stack=0,
        verbose=0,
    ):
        super().__init__(verbose)
        self.env_fn = env_fn
        self.video_dir = video_dir
        self.video_freq = video_freq
        self.n_episodes = n_episodes
        self.checkpoint_dir = checkpoint_dir
        self.deterministic = deterministic
        self.fps = fps
        self.n_stack = n_stack

    def _on_step(self) -> bool:
        if self.num_timesteps % self.video_freq != 0:
            return True

        # Save current VecNormalize stats so the recording env uses them.
        # If no VecNormalize (e.g. SAC visual), pass None.
        vec_norm = self.model.get_vec_normalize_env()
        if vec_norm is not None:
            vec_norm_path = os.path.join(self.checkpoint_dir, "vec_normalize.pkl")
            vec_norm.save(vec_norm_path)
        else:
            vec_norm_path = None

        # Import here to avoid loading heavy deps at module level
        from scripts.agents.eval_agent import _record_annotated_videos

        step_video_dir = os.path.join(
            self.video_dir, f"step_{self.num_timesteps}"
        )
        results = _record_annotated_videos(
            model=self.model,
            env_fn=self.env_fn,
            video_dir=step_video_dir,
            n_episodes=self.n_episodes,
            seed=self.num_timesteps,
            vec_normalize_path=vec_norm_path,
            deterministic=self.deterministic,
            fps=self.fps,
            n_stack=self.n_stack,
        )

        # Log summary
        rewards = [r["reward"] for r in results]
        mean_rew = sum(rewards) / len(rewards) if rewards else 0
        if self.verbose:
            print(f"[AnnotatedVideo] step={self.num_timesteps}: "
                  f"{len(results)} episodes, mean_reward={mean_rew:.1f}, "
                  f"dir={step_video_dir}")

        # Log to tensorboard if available
        if self.logger:
            self.logger.record("video/mean_reward", mean_rew)
            self.logger.record("video/n_episodes", len(results))

        return True


class CurriculumCallback(BaseCallback):
    """Swap sampling profiles at step thresholds during training.

    Monitors the training timestep counter and, when a threshold from the
    curriculum schedule is crossed, updates the DomainRandomizationWrapper's
    profile on ALL parallel env workers via SubprocVecEnv.env_method().

    Each subprocess loads the profile YAML independently (string name sent
    across process boundary, not the full object). This means new profiles
    take effect on the next env.reset() — episodes already in progress
    continue with their current physics config, which is the desired behavior
    (no mid-episode physics changes).

    On resume: _on_training_start() reads the restored num_timesteps and
    immediately sets the correct profile for that point in the schedule,
    so the agent never trains with the wrong distribution.

    Args:
        schedule: CurriculumSchedule with (step, profile_name) stages.
        verbose: Print transitions to stdout when > 0.
    """

    def __init__(self, schedule, verbose=0):
        super().__init__(verbose)
        self.schedule = schedule
        # Track current profile to avoid redundant env_method calls.
        # set_profile() is cheap but unnecessary IPC is wasteful.
        self._current_profile = None

    def _on_training_start(self) -> None:
        """Set the correct profile for the current timestep.

        Critical for resume: if training restarts at step 600K with schedule
        easy:0,medium:500K,..., this immediately sets 'medium' without
        waiting for _on_step.
        """
        self._update_profile()

    def _on_step(self) -> bool:
        """Check for profile transitions on every step."""
        self._update_profile()
        return True

    def _update_profile(self):
        """Swap profile if the schedule says we should be on a different one."""
        target = self.schedule.get_active_profile(self.num_timesteps)
        if target != self._current_profile:
            # Send profile name (string) to each subprocess worker.
            # DomainRandomizationWrapper.set_profile() loads the YAML
            # independently in each subprocess.
            self.training_env.env_method("set_profile", target)

            if self.verbose:
                print(
                    f"[Curriculum] step={self.num_timesteps:,}: "
                    f"switching to profile '{target}'"
                )

            # Log to tensorboard if available
            if self.logger:
                self.logger.record(
                    "curriculum/profile",
                    self.schedule.profile_names.index(target),
                )

            self._current_profile = target


class OutcomeEvalCallback(BaseCallback):
    """Evaluate agent and log outcome metrics to tensorboard.

    Replaces SB3's generic EvalCallback for Lunar Lander. Runs eval
    episodes at regular intervals and logs:
      - eval/mean_reward, eval/std_reward (backward compatible)
      - eval/landed_pct, eval/crashed_pct, eval/out_of_bounds_pct, eval/timeout_pct
      - eval/mean_steps

    Uses evaluate_agent() from eval_utils.py — same code path as the
    CLI eval script, so training-time and post-hoc eval produce
    identical metrics.

    Also saves the best model based on landed_pct (not reward),
    because landing is the actual objective.

    Supports early stopping: if landed_pct >= early_stop_landed_pct for
    early_stop_patience consecutive evals, training is stopped (callback
    returns False). This saves GPU time in batch runs where some configs
    converge faster than others.

    Args:
        env_fn: Factory that takes a seed and returns a wrapped env.
        eval_freq: Run eval every N timesteps.
        n_eval_episodes: Number of episodes per evaluation.
        checkpoint_dir: Where to save the best model.
        vec_normalize_path: Path to VecNormalize stats (updated by caller).
        seed: Eval env seed.
        early_stop_landed_pct: Stop when landed_pct >= this for patience
            consecutive evals. None = disabled.
        early_stop_patience: Consecutive evals above threshold before stopping.
        verbose: Print one-line summary to stdout when > 0.
    """

    def __init__(
        self,
        env_fn,
        eval_freq=50_000,
        n_eval_episodes=20,
        checkpoint_dir=None,
        vec_normalize_path=None,
        seed=99,
        early_stop_landed_pct=None,
        early_stop_patience=3,
        n_stack=0,
        verbose=1,
    ):
        super().__init__(verbose)
        self.env_fn = env_fn
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.checkpoint_dir = checkpoint_dir
        self.vec_normalize_path = vec_normalize_path
        self.seed = seed
        self.early_stop_landed_pct = early_stop_landed_pct
        self.early_stop_patience = early_stop_patience
        self.n_stack = n_stack
        self._best_landed_pct = -1.0
        self._consecutive_above = 0
        # Set when early stopping triggers — main() reads this to record in config.json.
        self.early_stopped_at_step = None

    def _on_step(self) -> bool:
        # Only fire at eval_freq intervals.
        # Use num_timesteps (total steps across all envs), matching
        # AnnotatedVideoCallback and SB3's own conventions.
        if self.num_timesteps % self.eval_freq != 0:
            return True

        # Save current VecNormalize stats before eval so evaluate_agent
        # loads up-to-date normalization. Without this, the first eval
        # would fail (file doesn't exist yet) or use stale stats.
        # Use get_vec_normalize_env() instead of isinstance check — SB3
        # auto-wraps image envs with VecTransposeImage, so training_env
        # may not be VecNormalize even though VecNormalize is in the stack.
        if self.vec_normalize_path:
            vec_norm = self.model.get_vec_normalize_env()
            if vec_norm is not None:
                vec_norm.save(self.vec_normalize_path)

        # Run eval using the shared evaluation function.
        result = evaluate_agent(
            model=self.model,
            env_fn=self.env_fn,
            n_episodes=self.n_eval_episodes,
            seed=self.seed,
            vec_normalize_path=self.vec_normalize_path,
            deterministic=True,
            n_stack=self.n_stack,
        )
        summary = result["summary"]

        # Log to tensorboard.
        self.logger.record("eval/mean_reward", summary["mean_reward"])
        self.logger.record("eval/std_reward", summary["std_reward"])
        self.logger.record("eval/landed_pct", summary["landed_pct"])
        self.logger.record("eval/crashed_pct", summary["crashed_pct"])
        self.logger.record("eval/out_of_bounds_pct", summary["out_of_bounds_pct"])
        self.logger.record("eval/timeout_pct", summary["timeout_pct"])
        self.logger.record("eval/mean_steps", summary["mean_steps"])

        if self.verbose:
            print(
                f"[Eval] step={self.num_timesteps:,}: "
                f"{summary['n_landed']}/{summary['n_episodes']} landed "
                f"({summary['landed_pct']:.0f}%), "
                f"reward={summary['mean_reward']:.0f}\u00b1{summary['std_reward']:.0f}"
            )

        # Save best model by landed_pct.
        if (self.checkpoint_dir and
                summary["landed_pct"] > self._best_landed_pct):
            self._best_landed_pct = summary["landed_pct"]
            best_dir = os.path.join(self.checkpoint_dir, "best")
            os.makedirs(best_dir, exist_ok=True)
            self.model.save(os.path.join(best_dir, "model"))
            # Save VecNormalize stats alongside if available.
            # training_env is the VecEnv used for training — if it's a
            # VecNormalize, save the running stats so eval uses them too.
            if isinstance(self.training_env, VecNormalize):
                self.training_env.save(
                    os.path.join(best_dir, "vec_normalize.pkl")
                )

        # Early stopping: stop training when landed_pct is consistently
        # above threshold. Returns False to tell SB3 to stop .learn().
        if self.early_stop_landed_pct is not None:
            if summary["landed_pct"] >= self.early_stop_landed_pct:
                self._consecutive_above += 1
                if self._consecutive_above >= self.early_stop_patience:
                    self.early_stopped_at_step = self.num_timesteps
                    print(
                        f"[EarlyStop] landed_pct >= {self.early_stop_landed_pct}% "
                        f"for {self.early_stop_patience} consecutive evals. "
                        f"Stopping at step {self.num_timesteps:,}."
                    )
                    return False
            else:
                self._consecutive_above = 0

        return True


class TBPlotCallback(BaseCallback):
    """Save tensorboard scalar plots as PNG images during training.

    Reads ALL scalar tags from the tensorboard events file at regular
    intervals and saves:
      1. Individual PNGs: one per scalar tag in plots/{tag_name}.png
      2. Composite summary PNG: key metrics in a single figure at plots/summary.png

    The composite summary has 3 panels:
      - Outcome percentages: landed/crashed/timeout/oob % over training steps
      - Entropy loss: train/entropy_loss over training steps
      - Mean reward: rollout/ep_rew_mean over training steps

    Individual plots are overwritten each time (always show full history).
    This gives filesystem-browsable training curves without needing to
    launch tensorboard.

    Args:
        log_dir: Directory containing tensorboard events (same as run_dir).
        plot_freq: Save plots every N timesteps. Defaults to eval_freq.
        verbose: Print status messages when > 0.
    """

    # Tags for the 3-panel composite summary.
    # Panel 1: outcome percentages (stacked on one axes).
    OUTCOME_TAGS = [
        "eval/landed_pct",
        "eval/crashed_pct",
        "eval/timeout_pct",
        "eval/out_of_bounds_pct",
    ]
    # Panel 2: entropy loss (single line).
    ENTROPY_TAG = "train/entropy_loss"
    # Panel 3: mean reward (single line).
    REWARD_TAG = "rollout/ep_rew_mean"

    def __init__(self, log_dir, plot_freq=50_000, verbose=0):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.plot_freq = plot_freq
        self._plot_dir = os.path.join(log_dir, "plots")

    def _on_training_start(self) -> None:
        """Create plots directory on first call."""
        os.makedirs(self._plot_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps % self.plot_freq != 0:
            return True

        try:
            self._generate_plots()
        except Exception as e:
            # Never crash training because of a plotting error.
            # Print warning and continue.
            if self.verbose:
                print(f"[TBPlot] Warning: plot generation failed at step "
                      f"{self.num_timesteps:,}: {e}")

        return True

    def _load_events(self):
        """Load all scalar data from the tensorboard events file.

        Returns dict mapping tag_name -> list of (step, value) tuples.
        Uses EventAccumulator from tensorboard — the same backend that
        powers the TB web UI. We set size_guidance to 0 (load everything)
        so plots show the full training history.
        """
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )

        # EventAccumulator needs the directory containing events files.
        # size_guidance=0 means "load all events" (default caps at 10K).
        ea = EventAccumulator(self.log_dir, size_guidance={"scalars": 0})
        ea.Reload()

        scalars = {}
        for tag in ea.Tags().get("scalars", []):
            events = ea.Scalars(tag)
            scalars[tag] = [(e.step, e.value) for e in events]

        return scalars

    def _generate_plots(self):
        """Read TB events and save all individual + composite plots."""
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend (no display needed)
        import matplotlib.pyplot as plt

        scalars = self._load_events()
        if not scalars:
            return  # No data yet

        # --- Individual plots: one PNG per scalar tag ---
        for tag, data in scalars.items():
            if len(data) < 2:
                continue  # Need at least 2 points for a meaningful plot

            steps, values = zip(*data)

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(steps, values, linewidth=1.2)
            ax.set_xlabel("Steps")
            ax.set_ylabel(tag)
            ax.set_title(tag)
            ax.grid(True, alpha=0.3)

            # Format x-axis with K/M suffixes for readability
            from matplotlib.ticker import FuncFormatter
            ax.xaxis.set_major_formatter(
                FuncFormatter(lambda x, _: (
                    f"{x/1e6:.1f}M" if x >= 1e6 else
                    f"{x/1e3:.0f}K" if x >= 1e3 else
                    f"{x:.0f}"
                ))
            )

            # Sanitize tag for filename: eval/landed_pct -> eval_landed_pct
            safe_name = tag.replace("/", "_").replace(" ", "_")
            path = os.path.join(self._plot_dir, f"{safe_name}.png")
            fig.tight_layout()
            fig.savefig(path, dpi=100)
            plt.close(fig)

        # --- Composite summary: 3 panels ---
        self._generate_summary(scalars)

        if self.verbose:
            print(f"[TBPlot] step={self.num_timesteps:,}: "
                  f"saved {len(scalars)} plots to {self._plot_dir}")

    def _generate_summary(self, scalars):
        """Generate the 3-panel composite summary figure.

        Panel layout:
          [1] Outcome %: landed, crashed, timeout, oob as colored lines
          [2] Entropy loss: single line (entropy_loss is negative in SB3)
          [3] Mean reward: rollout/ep_rew_mean
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # --- Panel 1: Outcome percentages ---
        ax = axes[0]
        # Color scheme: green=landed, red=crashed, orange=timeout, purple=oob
        colors = {
            "eval/landed_pct": "#2ecc71",
            "eval/crashed_pct": "#e74c3c",
            "eval/timeout_pct": "#f39c12",
            "eval/out_of_bounds_pct": "#9b59b6",
        }
        labels = {
            "eval/landed_pct": "Landed %",
            "eval/crashed_pct": "Crashed %",
            "eval/timeout_pct": "Timeout %",
            "eval/out_of_bounds_pct": "OOB %",
        }
        has_outcome = False
        for tag in self.OUTCOME_TAGS:
            if tag in scalars and len(scalars[tag]) >= 2:
                steps, values = zip(*scalars[tag])
                ax.plot(steps, values, label=labels[tag],
                        color=colors[tag], linewidth=1.5)
                has_outcome = True
        if has_outcome:
            ax.set_ylim(-5, 105)
            ax.legend(loc="center right", fontsize=9)
        ax.set_title("Outcome %", fontsize=12, fontweight="bold")
        ax.set_xlabel("Steps")
        ax.set_ylabel("%")
        ax.grid(True, alpha=0.3)

        # --- Panel 2: Entropy loss ---
        ax = axes[1]
        if self.ENTROPY_TAG in scalars and len(scalars[self.ENTROPY_TAG]) >= 2:
            steps, values = zip(*scalars[self.ENTROPY_TAG])
            ax.plot(steps, values, color="#3498db", linewidth=1.5)
        ax.set_title("Entropy Loss", fontsize=12, fontweight="bold")
        ax.set_xlabel("Steps")
        ax.set_ylabel("entropy_loss")
        ax.grid(True, alpha=0.3)

        # --- Panel 3: Mean reward ---
        ax = axes[2]
        if self.REWARD_TAG in scalars and len(scalars[self.REWARD_TAG]) >= 2:
            steps, values = zip(*scalars[self.REWARD_TAG])
            ax.plot(steps, values, color="#e67e22", linewidth=1.5)
        ax.set_title("Mean Reward", fontsize=12, fontweight="bold")
        ax.set_xlabel("Steps")
        ax.set_ylabel("ep_rew_mean")
        ax.grid(True, alpha=0.3)

        # Format all x-axes
        for ax in axes:
            from matplotlib.ticker import FuncFormatter
            ax.xaxis.set_major_formatter(
                FuncFormatter(lambda x, _: (
                    f"{x/1e6:.1f}M" if x >= 1e6 else
                    f"{x/1e3:.0f}K" if x >= 1e3 else
                    f"{x:.0f}"
                ))
            )

        fig.suptitle(
            f"Training Summary — step {self.num_timesteps:,}",
            fontsize=13, fontweight="bold", y=1.02,
        )
        fig.tight_layout()
        fig.savefig(
            os.path.join(self._plot_dir, "summary.png"),
            dpi=120, bbox_inches="tight",
        )
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Train a single RL agent on Lunar Lander.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Config file (preferred -- defines the full run in one YAML)
    parser.add_argument("--config", type=str, default=None,
                        help="Training config YAML (builtin name or file path). "
                             "CLI args override YAML values.")

    # Agent
    parser.add_argument("--variant", default=None, choices=VARIANTS,
                        help="Agent variant: blind, labeled, or history")
    parser.add_argument("--algo", default=None, choices=["ppo", "sac"],
                        help="RL algorithm (default: ppo)")
    parser.add_argument("--history-k", type=int, default=None,
                        help="History stack depth for history variant (default: 8)")
    parser.add_argument("--n-rays", type=int, default=None,
                        help="Number of terrain rays (default: 7)")
    parser.add_argument("--n-stack", type=int, default=None,
                        help="Frame stack depth for visual variants (default: 4)")

    # Training
    parser.add_argument("--total-steps", type=int, default=None,
                        help="Total env steps (default: 3M)")
    parser.add_argument("--n-envs", type=int, default=None,
                        help="Parallel env workers (default: 8)")
    parser.add_argument("--seed", type=int, default=None)

    # Paths
    parser.add_argument("--run-dir", type=str, default=None,
                        help="Single directory for ALL outputs (overrides structured paths)")
    parser.add_argument("--data-root", type=str, default=None,
                        help="Root for auto-structured outputs (legacy, prefer --run-dir)")
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--log-dir", type=str, default=None)

    # Callback frequencies
    parser.add_argument("--checkpoint-freq", type=int, default=None)
    parser.add_argument("--eval-freq", type=int, default=None)
    parser.add_argument("--video-freq", type=int, default=None)

    # Physics constraints
    parser.add_argument("--twr-min", type=float, default=None,
                        help="Min thrust-to-weight ratio for domain randomization")
    parser.add_argument("--twr-max", type=float, default=None,
                        help="Max thrust-to-weight ratio for domain randomization")

    # Sampling profile
    parser.add_argument("--profile", type=str, default=None,
                        help="Sampling profile name or path to .yaml")

    # Curriculum
    parser.add_argument("--curriculum", type=str, default=None,
                        help="Curriculum schedule: 'profile:step,...' e.g. "
                             "'easy:0,medium:500K,hard:1.5M,full:2.5M'. "
                             "Mutually exclusive with --profile.")

    # Early stopping
    parser.add_argument("--early-stop-landed-pct", type=float, default=None,
                        help="Stop when landed%% >= this for N consecutive evals. "
                             "None = disabled (train for full total_steps).")
    parser.add_argument("--early-stop-patience", type=int, default=None,
                        help="Consecutive evals above threshold before stopping (default: 3)")

    # Resume
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest checkpoint in checkpoint dir")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Resume from checkpoint in a DIFFERENT run dir")
    parser.add_argument("--device", type=str, default=None,
                        help="CUDA device (e.g. 'cuda:0', 'cuda:1', 'cpu'). "
                             "Default: SB3 auto-selects (usually cuda:0).")
    parser.add_argument("--encoder-weights", type=str, default=None,
                        help="Override encoder_weights from config. "
                             "Path to pre-trained encoder .pt file.")

    args = parser.parse_args()

    # --- Resolve training config ---
    config = _resolve_config(args)

    # Validate required fields -- operational params must be explicit.
    if config["variant"] is None:
        parser.error("--variant is required (or specify it in --config YAML)")
    if config["algo"] is None:
        parser.error("--algo is required (or specify it in --config YAML)")
    if config["total_steps"] is None:
        parser.error("--total-steps is required (or specify it in --config YAML)")

    # Build policy_kwargs from net_arch.
    # None = use DEFAULT_POLICY_KWARGS (3x256 pi+vf).
    # List/dict = custom architecture (e.g., [64, 64] for RL Zoo baseline).
    policy_kwargs = None
    if config["net_arch"] is not None:
        policy_kwargs = {"net_arch": config["net_arch"]}

    # Build algo_kwargs from config overrides.
    # Only include keys that are explicitly set (not None) so rl_common
    # PPO_DEFAULTS / SAC_DEFAULTS are used for everything else.
    algo_kwargs = {}
    for algo_key in ["ent_coef", "learning_rate", "n_steps", "batch_size",
                     "n_epochs", "gamma", "gae_lambda", "clip_range",
                     "max_grad_norm"]:
        if config[algo_key] is not None:
            algo_kwargs[algo_key] = config[algo_key]

    # Device override: pass to SB3 via algo_kwargs.
    if args.device is not None:
        algo_kwargs["device"] = args.device

    # Encoder weights override: CLI takes precedence over config YAML.
    if args.encoder_weights is not None:
        config["encoder_weights"] = args.encoder_weights

    # Learning rate schedule: wrap the scalar LR with a schedule function.
    # SB3 accepts a callable(progress_remaining) -> float for learning_rate.
    # progress_remaining goes from 1.0 (start) to 0.0 (end of training).
    lr_schedule = config.get("lr_schedule")
    if lr_schedule == "linear" and "learning_rate" in algo_kwargs:
        base_lr = algo_kwargs["learning_rate"]
        algo_kwargs["learning_rate"] = lambda progress: progress * base_lr

    algo_kwargs = algo_kwargs or None  # None = no overrides

    # Visual variants use CNN policy, no obs normalization, frame stacking.
    # The CNN expects raw uint8 pixel observations — running-mean normalization
    # would destroy the image data.
    is_visual = config["variant"].startswith("visual")
    if is_visual:
        norm_obs = False
        n_stack = config["n_stack"]

        if policy_kwargs is None:
            policy_kwargs = {}

        cnn_backbone = config.get("cnn_backbone", "nature")
        if cnn_backbone not in ("nature", "impala"):
            print(f"ERROR: Unknown cnn_backbone '{cnn_backbone}'. Use 'nature' or 'impala'.")
            sys.exit(1)

        # Collect IMPALA-specific kwargs (used by both CnnPolicy and MultiInputPolicy).
        impala_kwargs = {}
        if cnn_backbone == "impala":
            if config.get("cnn_channels"):
                impala_kwargs["channels"] = config["cnn_channels"]
            if config.get("cnn_pool_size"):
                impala_kwargs["pool_size"] = config["cnn_pool_size"]

        if config["variant"] == "visual-labeled":
            # MultiInputPolicy handles Dict obs {"image": ..., "physics": ...}.
            # CombinedExtractor processes each sub-space with its own extractor
            # (CNN for images, branch MLP or Flatten for vectors) then concatenates.
            policy_class = "MultiInputPolicy"
            if cnn_backbone == "impala":
                physics_branch_dim = config.get("physics_branch_dim")
                if physics_branch_dim:
                    # Branch fusion: physics gets a dedicated projection MLP.
                    # The branch gives the 7D physics vector a real route into
                    # the policy despite the 256:7 dimensional mismatch with
                    # CNN features. See ImpalaBranchCombinedExtractor docstring.
                    #
                    # Compute normalization stats from the training profile:
                    # sample 10K physics configs, compute per-dim mean/std.
                    # This handles TWR rejection sampling correctly and works
                    # across profiles without hardcoding.
                    from parametric_lunar_lander.sampling_profiles import SamplingProfile
                    profile_obj = SamplingProfile.load(config["profile"])
                    # Seed the sampling for reproducibility across runs.
                    # Uses a fixed seed (0) independent of the RL seed so
                    # normalization stats are identical across all seeds.
                    norm_rng = np.random.default_rng(seed=0)
                    physics_samples = np.array(
                        [profile_obj.sample(rng=norm_rng).as_array() for _ in range(10_000)]
                    )
                    physics_mean = physics_samples.mean(axis=0)
                    physics_std = physics_samples.std(axis=0)
                    print(f"Physics normalization stats (from {config['profile']} profile):")
                    from parametric_lunar_lander.physics_config import LunarLanderPhysicsConfig
                    for i, name in enumerate(LunarLanderPhysicsConfig.PARAM_NAMES):
                        print(f"  {name:25s}: mean={physics_mean[i]:8.3f}  std={physics_std[i]:6.3f}")

                    # VecFrameStack repeats the physics vector n_stack times
                    # (e.g., 7 physics × 4 stacks = 28D). Tile normalization
                    # stats to match the stacked dimension.
                    n_stack = config.get("n_stack", 0) or 1
                    if n_stack > 1:
                        physics_mean = np.tile(physics_mean, n_stack)
                        physics_std = np.tile(physics_std, n_stack)

                    from lwp.agents.visual_backbones import ImpalaBranchCombinedExtractor
                    fe_kwargs = {
                        "cnn_output_dim": config.get("cnn_features_dim") or 256,
                        "physics_branch_dim": physics_branch_dim,
                        "physics_mean": physics_mean,
                        "physics_std": physics_std,
                    }
                    fe_kwargs.update(impala_kwargs)
                    policy_kwargs["features_extractor_class"] = ImpalaBranchCombinedExtractor
                    policy_kwargs["features_extractor_kwargs"] = fe_kwargs
                else:
                    # Raw concat: physics passes through nn.Flatten unchanged.
                    # This is the weak-fusion baseline (7D raw-concatenated
                    # to 256D CNN features → 263D total, physics is 2.7%).
                    from lwp.agents.visual_backbones import ImpalaCombinedExtractor
                    fe_kwargs = {"cnn_output_dim": config.get("cnn_features_dim") or 256}
                    fe_kwargs.update(impala_kwargs)
                    policy_kwargs["features_extractor_class"] = ImpalaCombinedExtractor
                    policy_kwargs["features_extractor_kwargs"] = fe_kwargs
            else:
                # NatureCNN default via SB3's CombinedExtractor.
                policy_kwargs["features_extractor_kwargs"] = {"cnn_output_dim": 512}
        else:
            # visual-blind: standard single-space CNN policy.
            policy_class = "CnnPolicy"
            if cnn_backbone == "impala":
                from lwp.agents.visual_backbones import ImpalaCNN
                fe_kwargs = {"features_dim": config.get("cnn_features_dim") or 256}
                fe_kwargs.update(impala_kwargs)
                policy_kwargs["features_extractor_class"] = ImpalaCNN
                policy_kwargs["features_extractor_kwargs"] = fe_kwargs

        # Separate actor/critic CNNs when share_features_extractor=False.
        if not config["share_features_extractor"]:
            policy_kwargs["share_features_extractor"] = False
    else:
        policy_class = "MlpPolicy"
        norm_obs = True
        n_stack = 0

    # --- Resolve output paths ---
    # run_dir is required in YAML configs (all presets set it).
    # For CLI-only usage, fall back to legacy auto-generated paths.
    if config["run_dir"]:
        checkpoint_dir = config["run_dir"]
        log_dir = config["run_dir"]
    elif args.config:
        # Using --config but no run_dir in YAML -- error
        parser.error("run_dir is required in training config YAML")
    elif args.data_root:
        ckpt_base = os.path.join(args.data_root, "checkpoints", "lunar_lander")
        log_base = os.path.join(args.data_root, "logs", "lunar_lander")
        checkpoint_dir = build_checkpoint_dir(ckpt_base, config["variant"], config["algo"])
        log_dir = build_log_dir(log_base, config["variant"], config["algo"])
    else:
        ckpt_base = args.checkpoint_dir or "checkpoints/lunar_lander"
        log_base = args.log_dir or "logs/lunar_lander"
        checkpoint_dir = build_checkpoint_dir(ckpt_base, config["variant"], config["algo"])
        log_dir = build_log_dir(log_base, config["variant"], config["algo"])

    # Handle --resume-from: copy checkpoint from a different run dir into this one.
    # This lets you branch a new training run from an existing checkpoint
    # (e.g., train easy -> resume-from easy into a new medium run dir).
    resumed_from_info = None
    if args.resume_from is not None:
        source_dir = args.resume_from
        source_checkpoint, source_steps = _find_latest_checkpoint(source_dir)
        if source_checkpoint is None:
            print(f"ERROR: No checkpoint found in --resume-from dir: {source_dir}")
            sys.exit(1)

        # Build the matching VecNormalize path
        source_vecnorm = source_checkpoint.replace(
            "rl_model_", "rl_model_vecnormalize_"
        ).replace(".zip", ".pkl")

        # Copy into checkpoints/ subdir (new layout) so _find_latest_checkpoint
        # and resume_training find them in the expected location.
        dest_subdir = os.path.join(checkpoint_dir, "checkpoints")
        os.makedirs(dest_subdir, exist_ok=True)

        # Copy model checkpoint
        dest_checkpoint = os.path.join(dest_subdir, os.path.basename(source_checkpoint))
        shutil.copy2(source_checkpoint, dest_checkpoint)
        print(f"Copied checkpoint: {source_checkpoint}")
        print(f"             -> {dest_checkpoint}")

        # Copy VecNormalize stats if available
        if os.path.exists(source_vecnorm):
            dest_vecnorm = os.path.join(dest_subdir, os.path.basename(source_vecnorm))
            shutil.copy2(source_vecnorm, dest_vecnorm)
            print(f"Copied VecNormalize: {os.path.basename(source_vecnorm)}")

        # Also copy source config.json for reference (as source_config.json)
        source_config = os.path.join(source_dir, "config.json")
        if os.path.exists(source_config):
            shutil.copy2(source_config, os.path.join(checkpoint_dir, "source_config.json"))

        # Record provenance for config.json
        resumed_from_info = {
            "source_dir": os.path.abspath(source_dir),
            "source_checkpoint": os.path.basename(source_checkpoint),
            "source_steps": source_steps,
        }

        # --resume-from implies --resume
        args.resume = True

    # Print config
    print("=" * 60)
    print(f"Training: generalist-{config['variant']} / {config['algo'].upper()}")
    print("=" * 60)
    print(f"  Total steps:    {config['total_steps']:,}")
    print(f"  Parallel envs:  {config['n_envs']}")
    print(f"  Seed:           {config['seed']}")
    print(f"  Terrain rays:   {config['n_rays']}")
    if config["variant"] == "history":
        print(f"  History K:      {config['history_k']}")
    if config["net_arch"] is not None:
        print(f"  Net arch:       {config['net_arch']}")
    if config["early_stop_landed_pct"] is not None:
        print(f"  Early stop:     landed_pct >= {config['early_stop_landed_pct']}% "
              f"for {config['early_stop_patience']} evals")
    if is_visual:
        print(f"  Policy:         CnnPolicy (Nature DQN CNN)")
        print(f"  Frame stack:    {n_stack}")
        print(f"  Obs norm:       disabled (raw pixels)")
    # Build TWR range from config (None if neither specified)
    twr_range = None
    if config["twr_min"] is not None or config["twr_max"] is not None:
        twr_range = (
            config["twr_min"] if config["twr_min"] is not None else 0.0,
            config["twr_max"] if config["twr_max"] is not None else float("inf"),
        )

    # Profile takes precedence over twr_range
    profile = config["profile"]
    if profile is not None and twr_range is not None:
        print("WARNING: --profile overrides --twr-min/--twr-max")
        twr_range = None

    # Curriculum schedule -- mutually exclusive with --profile
    curriculum = None
    if config["curriculum"] is not None:
        if profile is not None:
            print("ERROR: --curriculum and --profile are mutually exclusive.")
            sys.exit(1)
        from parametric_lunar_lander.sampling_profiles import CurriculumSchedule
        curriculum = CurriculumSchedule.from_string(config["curriculum"])
        # Use the first profile in the schedule for initial env creation.
        # The CurriculumCallback will swap to the correct profile
        # on _on_training_start (handles resume correctly).
        profile = curriculum.initial_profile

    print(f"  Checkpoints:    {checkpoint_dir}")
    print(f"  Logs:           {log_dir}")
    if curriculum:
        print(curriculum.describe())
    elif profile:
        from parametric_lunar_lander.sampling_profiles import SamplingProfile
        loaded = SamplingProfile.load(profile) if isinstance(profile, str) else profile
        print(f"  Profile:        {profile}")
        print(loaded.describe())
    elif twr_range:
        print(f"  TWR range:      [{twr_range[0]:.1f}, {twr_range[1]:.1f}]")
    print()

    config_path = save_training_config(
        checkpoint_dir, config,
        policy_kwargs=policy_kwargs,
        policy_class=policy_class,
        resumed_from=resumed_from_info,
    )
    print(f"  Config saved:   {config_path}")
    if resumed_from_info:
        print(f"  Resumed from:   {resumed_from_info['source_dir']}")
        print(f"  Source model:   {resumed_from_info['source_checkpoint']} "
              f"({resumed_from_info['source_steps']:,} steps)")

    # Build env factory
    def env_fn(seed):
        return make_lunar_lander_env(
            variant=config["variant"],
            seed=seed,
            history_k=config["history_k"],
            n_rays=config["n_rays"],
            twr_range=twr_range,
            profile=profile,
            frame_size=config["frame_size"],
            angle_shaping_omega=config.get("angle_shaping_omega"),
            time_penalty=config.get("time_penalty"),
            aux_kinematic=config.get("aux_kinematic", False),
        )

    # Build annotated video callback if video recording is requested.
    # We pass video_freq=0 to train() so its generic (un-annotated)
    # VideoRecorderCallback is NOT created, and instead supply our own
    # AnnotatedVideoCallback via extra_callbacks. This produces the same
    # annotated side-panel videos as eval_agent.py.
    extra_callbacks = []
    if config["video_freq"] > 0:
        video_cb = AnnotatedVideoCallback(
            env_fn=env_fn,
            video_dir=os.path.join(checkpoint_dir, "videos"),
            video_freq=config["video_freq"],
            checkpoint_dir=checkpoint_dir,
            verbose=1,
            fps=50,
            n_stack=n_stack,
        )
        extra_callbacks.append(video_cb)

    # Inject curriculum callback if schedule is set.
    # Must come after video callback -- order doesn't matter for SB3,
    # but grouping them logically is clearer.
    if curriculum is not None:
        curriculum_cb = CurriculumCallback(
            schedule=curriculum,
            verbose=1,
        )
        extra_callbacks.append(curriculum_cb)

    # Inject outcome eval callback for tensorboard outcome metrics.
    # Replaces rl_common's generic EvalCallback (disabled via eval_freq=0
    # in the train() call). OutcomeEvalCallback logs landed%, crashed%,
    # out_of_bounds%, timeout%, mean_reward -- everything in one callback.
    # vec_normalize_path tells the eval callback where to find the running
    # normalization stats. Without this, eval runs with raw (unnormalized)
    # observations but the model expects normalized inputs → garbage actions.
    # No VecNormalize for off-policy visual (SAC) — see rl_common/training.py.
    # Pass None so eval callback skips loading/saving normalization stats.
    if config["algo"].lower() == "sac" and is_visual:
        vec_normalize_path = None
    else:
        vec_normalize_path = os.path.join(checkpoint_dir, "vec_normalize.pkl")
    outcome_eval_cb = OutcomeEvalCallback(
        env_fn=env_fn,
        eval_freq=config["eval_freq"],
        n_eval_episodes=20,
        checkpoint_dir=checkpoint_dir,
        vec_normalize_path=vec_normalize_path,
        seed=99,
        early_stop_landed_pct=config["early_stop_landed_pct"],
        early_stop_patience=config["early_stop_patience"],
        n_stack=n_stack,
        verbose=1,
    )
    extra_callbacks.append(outcome_eval_cb)

    # Inject TBPlot callback to save training curves as PNG files.
    # Fires at eval_freq so plots update after each evaluation round.
    # Reads the tensorboard events file that SB3 is already writing to
    # log_dir and renders every scalar tag as an individual PNG, plus a
    # 3-panel composite summary (outcomes, entropy, reward).
    tb_plot_cb = TBPlotCallback(
        log_dir=log_dir,
        plot_freq=config["eval_freq"],
        verbose=1,
    )
    extra_callbacks.append(tb_plot_cb)

    # Train (or resume)
    print()
    start_time = time.time()
    steps_before = 0

    if args.resume:
        print("Looking for checkpoint to resume from...")
        result = resume_training(
            env_fn=env_fn,
            algo=config["algo"],
            total_steps=config["total_steps"],
            checkpoint_dir=checkpoint_dir,
            log_dir=log_dir,
            n_envs=config["n_envs"],
            seed=config["seed"],
            checkpoint_freq=config["checkpoint_freq"],
            eval_freq=0,  # disabled -- OutcomeEvalCallback handles eval
            extra_callbacks=extra_callbacks if extra_callbacks else None,
            policy_kwargs=policy_kwargs,
            norm_obs=norm_obs,
            n_stack=n_stack,
        )
        if result is not None:
            model, vec_env, steps_before = result
        elif _find_latest_checkpoint(checkpoint_dir)[0] is None:
            # No checkpoint at all — fall through to start fresh training.
            print("No checkpoint found. Starting fresh training...")
            args.resume = False  # so post-training config doesn't record resumed_from_step
        else:
            # Training already complete (steps_done >= total_steps).
            print("Training already complete. Exiting.")
            sys.exit(0)

    if not args.resume:
        # Use AuxPPO when aux_kinematic is enabled — adds an auxiliary
        # MLP head that predicts kinematic state from CNN features.
        algo_class_override = None
        if config.get("aux_kinematic", False):
            from lwp.agents.aux_ppo import AuxPPO
            aux_coef = config.get("aux_coef", 0.5)
            # Wrap AuxPPO to inject aux_coef via a factory function.
            # rl_common.training.train() calls PPOClass(policy, env, **defaults).
            # AuxPPO.__init__ accepts aux_coef as a kwarg.
            from functools import partial
            algo_class_override = partial(AuxPPO, aux_coef=aux_coef)

        print("Starting training...")
        model, vec_env = train(
            env_fn=env_fn,
            algo=config["algo"],
            total_steps=config["total_steps"],
            n_envs=config["n_envs"],
            seed=config["seed"],
            log_dir=log_dir,
            checkpoint_dir=checkpoint_dir,
            checkpoint_freq=config["checkpoint_freq"],
            eval_freq=0,  # disabled -- OutcomeEvalCallback handles eval
            video_freq=0,  # disabled -- using AnnotatedVideoCallback instead
            policy_kwargs=policy_kwargs,
            algo_kwargs=algo_kwargs,
            extra_callbacks=extra_callbacks,
            policy_class=policy_class,
            norm_obs=norm_obs,
            n_stack=n_stack,
            algo_class_override=algo_class_override,
            encoder_weights=config.get("encoder_weights"),
            freeze_encoder=config.get("freeze_encoder", False),
        )

    elapsed = time.time() - start_time
    throughput = config["total_steps"] / elapsed

    print()
    print("=" * 60)
    print(f"Training complete: generalist-{config['variant']} / {config['algo'].upper()}")
    print("=" * 60)
    print(f"  Wall time:      {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"  Throughput:     {throughput:,.0f} steps/sec")
    print(f"  Checkpoint:     {checkpoint_dir}/model.zip")

    # Update config.json with completion info
    with open(config_path) as f:
        saved_config = json.load(f)
    saved_config["completed_at"] = datetime.now().isoformat(timespec="seconds")
    saved_config["wall_time_seconds"] = round(elapsed, 1)
    saved_config["throughput_steps_per_sec"] = round(throughput, 1)
    if args.resume:
        saved_config["resumed_from_step"] = steps_before
    if outcome_eval_cb.early_stopped_at_step is not None:
        saved_config["early_stopped_at_step"] = outcome_eval_cb.early_stopped_at_step
    with open(config_path, "w") as f:
        json.dump(saved_config, f, indent=2)

    vec_env.close()


if __name__ == "__main__":
    main()
