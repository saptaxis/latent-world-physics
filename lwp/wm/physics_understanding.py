# lunar_lander/src/wm/physics_understanding.py
"""Physics understanding evaluation for world models.

Extracts physical constants (gravity, thrust, damping, kinematics,
angle-thrust coupling) from model predictions and measures consistency
across state space. Answers: "What does the model believe about physics?"

Uses 1-step oracle predictions (GT input) and short autoregressive
rollouts to separate physics knowledge from error compounding.

Design spec: traitful-docs/.../specs/physics-understanding-eval.md
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

# State dimension indices (6-dim normalized observation vector).
X, Y, VX, VY, ANGLE, ANGULAR_VEL = 0, 1, 2, 3, 4, 5


# --- Filter functions ---
# Each returns True if the transition qualifies for that measurement.
# General filter is applied first, then constant-specific filter.


def passes_general_filter(state: np.ndarray, timestep: int) -> bool:
    """General exclusions applied to all measurements.

    Excludes: near ground (y <= 0.3), near OOB (|x| >= 0.8),
    first 3 timesteps (settle transients).
    """
    if state[Y] <= 0.3:
        return False
    if abs(state[X]) >= 0.8:
        return False
    if timestep < 3:
        return False
    return True


def passes_gravity_filter(state: np.ndarray, action: np.ndarray) -> bool:
    """Gravity measurement: no engines, upright, not spinning.

    Side engine activates at |action_side| > 0.5, so we use that threshold.
    """
    if action[0] > 0.05:  # main engine
        return False
    if abs(action[1]) > 0.5:  # side engine activation threshold
        return False
    if abs(state[ANGLE]) > 0.1:
        return False
    if abs(state[ANGULAR_VEL]) > 0.3:
        return False
    return True


def passes_main_thrust_filter(state: np.ndarray, action: np.ndarray) -> bool:
    """Main thrust measurement: strong main, no side, upright, not spinning."""
    if action[0] < 0.5:
        return False
    if abs(action[1]) > 0.5:
        return False
    if abs(state[ANGLE]) > 0.1:
        return False
    if abs(state[ANGULAR_VEL]) > 0.3:
        return False
    return True


def passes_side_thrust_filter(state: np.ndarray, action: np.ndarray) -> bool:
    """Side thrust measurement: strong side, no main, upright, not spinning."""
    if abs(action[1]) < 0.5:
        return False
    if action[0] > 0.05:
        return False
    if abs(state[ANGLE]) > 0.1:
        return False
    if abs(state[ANGULAR_VEL]) > 0.3:
        return False
    return True


def passes_kinematic_filter(state: np.ndarray) -> bool:
    """Kinematic consistency: velocities not near zero (avoid /0)."""
    if abs(state[VX]) < 0.1:
        return False
    if abs(state[VY]) < 0.1:
        return False
    return True


def passes_angle_thrust_filter(
    state: np.ndarray, action: np.ndarray, gravity_model: float
) -> bool:
    """Angle-thrust coupling: tilted, strong main, no side.

    The denominator guard (|dvy - gravity| > 0.01) is checked per-transition
    after prediction, not here — we can't know dvy_pred at filter time.
    """
    if action[0] < 0.5:
        return False
    if abs(action[1]) > 0.5:
        return False
    if abs(state[ANGLE]) < 0.15:
        return False
    return True


def passes_angular_damping_filter(
    state: np.ndarray, action: np.ndarray
) -> bool:
    """Angular damping: no engines, enough rotation to measure decay."""
    if action[0] > 0.05:
        return False
    if abs(action[1]) > 0.5:
        return False
    if abs(state[ANGULAR_VEL]) < 0.3:
        return False
    return True


@dataclass
class ConstantResult:
    """Result of extracting a single physical constant.

    Stores the distribution of extracted values from both model and GT,
    plus the number of qualifying transitions.
    """
    model_values: np.ndarray  # per-transition extracted constant from model
    gt_values: np.ndarray     # per-transition extracted constant from GT
    # Associated state variables for consistency checks.
    associated_states: np.ndarray  # (n_samples, 6) states at each transition
    associated_timesteps: np.ndarray  # (n_samples,) timestep indices

    @property
    def n_samples(self) -> int:
        return len(self.model_values)

    @property
    def model_mean(self) -> float:
        return float(np.mean(self.model_values)) if self.n_samples > 0 else float("nan")

    @property
    def model_std(self) -> float:
        return float(np.std(self.model_values)) if self.n_samples > 0 else float("nan")

    @property
    def gt_mean(self) -> float:
        return float(np.mean(self.gt_values)) if self.n_samples > 0 else float("nan")

    @property
    def gt_std(self) -> float:
        return float(np.std(self.gt_values)) if self.n_samples > 0 else float("nan")

    @property
    def relative_error(self) -> float:
        if self.n_samples == 0 or self.gt_mean == 0:
            return float("nan")
        return abs(self.model_mean - self.gt_mean) / abs(self.gt_mean)

    def to_dict(self) -> dict:
        """Serialize for JSON report."""
        return {
            "n_samples": self.n_samples,
            "model_mean": self.model_mean,
            "model_std": self.model_std,
            "gt_mean": self.gt_mean,
            "gt_std": self.gt_std,
            "relative_error": self.relative_error,
        }


def _get_model_device(model) -> torch.device:
    """Get device from model, falling back to CPU for non-nn.Module models."""
    try:
        return next(model.parameters()).device
    except (StopIteration, AttributeError):
        return torch.device("cpu")


def _predict_oracle_delta(
    model, norm_stats, state: np.ndarray, action: np.ndarray
) -> np.ndarray:
    """Run one oracle prediction: GT state + action -> model delta (raw space).

    Normalizes state, calls model.step(), denormalizes delta.
    Returns raw-space delta as numpy array (state_dim,).
    """
    with torch.no_grad():
        # Detect device from model parameters.
        device = _get_model_device(model)
        s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        a = torch.tensor(action, dtype=torch.float32, device=device).unsqueeze(0)
        # Move norm_stats tensors to same device.
        state_mean = norm_stats.state_mean.to(device)
        state_std = norm_stats.state_std.to(device)
        delta_mean = norm_stats.delta_mean.to(device)
        delta_std = norm_stats.delta_std.to(device)
        # Normalize state.
        s_norm = (s - state_mean) / state_std
        # Model predicts normalized delta.
        delta_norm, _ = model.step(s_norm, a, None)
        # Denormalize delta.
        delta_raw = delta_norm * delta_std + delta_mean
    return delta_raw.squeeze(0).cpu().numpy()


def extract_gravity_oracle(
    model,
    norm_stats,
    episodes: list[dict],
) -> dict:
    """Extract gravity constant from 1-step oracle predictions.

    For each qualifying transition (no thrust, upright, away from ground),
    feeds GT state + action to the model and records predicted dvy.
    Also records GT dvy from episode deltas.

    Args:
        model: wm-ladder model with .step(obs, action, state) interface.
        norm_stats: NormStats with state_mean/std, delta_mean/std.
        episodes: List of episode dicts with 'states' (T+1, >=6) and 'actions' (T, 2).

    Returns:
        Dict with keys: n_samples, model_mean, model_std, gt_mean, gt_std,
        relative_error, model_values, gt_values, associated_states, timesteps.
    """
    model.eval()
    model_values = []
    gt_values = []
    assoc_states = []
    timesteps = []

    for episode in episodes:
        states = episode["states"][:, :6]  # 6-dim kinematic only
        actions = episode["actions"]
        n_transitions = min(len(states) - 1, len(actions))

        for t in range(n_transitions):
            if not passes_general_filter(states[t], timestep=t):
                continue
            if not passes_gravity_filter(states[t], actions[t]):
                continue

            # Oracle: feed GT state + action to model.
            delta_pred = _predict_oracle_delta(model, norm_stats, states[t], actions[t])
            model_dvy = delta_pred[VY]

            # GT: actual delta from recorded episode.
            gt_dvy = states[t + 1][VY] - states[t][VY]

            model_values.append(model_dvy)
            gt_values.append(gt_dvy)
            assoc_states.append(states[t].copy())
            timesteps.append(t)

    result = ConstantResult(
        model_values=np.array(model_values, dtype=np.float32),
        gt_values=np.array(gt_values, dtype=np.float32),
        associated_states=np.array(assoc_states, dtype=np.float32).reshape(-1, 6),
        associated_timesteps=np.array(timesteps, dtype=np.int32),
    )
    return result.to_dict() | {
        "model_values": result.model_values,
        "gt_values": result.gt_values,
        "associated_states": result.associated_states,
        "timesteps": result.associated_timesteps,
    }


def extract_main_thrust_oracle(
    model, norm_stats, episodes: list[dict],
    gravity_model: float, gravity_gt: float,
) -> dict:
    """Extract main thrust response: effective_thrust = dvy_pred - gravity.

    Model thrust uses model's gravity estimate. GT thrust uses GT gravity.
    This way each side is self-consistent.
    """
    model.eval()
    model_values, gt_values, assoc_states, timesteps = [], [], [], []

    for episode in episodes:
        states = episode["states"][:, :6]
        actions = episode["actions"]
        n_transitions = min(len(states) - 1, len(actions))

        for t in range(n_transitions):
            if not passes_general_filter(states[t], timestep=t):
                continue
            if not passes_main_thrust_filter(states[t], actions[t]):
                continue

            delta_pred = _predict_oracle_delta(model, norm_stats, states[t], actions[t])
            model_thrust = delta_pred[VY] - gravity_model
            gt_dvy = states[t + 1][VY] - states[t][VY]
            gt_thrust = gt_dvy - gravity_gt

            model_values.append(float(model_thrust))
            gt_values.append(float(gt_thrust))
            assoc_states.append(states[t].copy())
            timesteps.append(t)

    result = ConstantResult(
        model_values=np.array(model_values, dtype=np.float32),
        gt_values=np.array(gt_values, dtype=np.float32),
        associated_states=np.array(assoc_states, dtype=np.float32).reshape(-1, 6),
        associated_timesteps=np.array(timesteps, dtype=np.int32),
    )
    return result.to_dict() | {
        "model_values": result.model_values,
        "gt_values": result.gt_values,
        "associated_states": result.associated_states,
        "timesteps": result.associated_timesteps,
    }


def extract_side_thrust_oracle(
    model, norm_stats, episodes: list[dict]
) -> dict:
    """Extract side thrust response: dvx_pred on side-thrust transitions."""
    model.eval()
    model_values, gt_values, assoc_states, timesteps = [], [], [], []

    for episode in episodes:
        states = episode["states"][:, :6]
        actions = episode["actions"]
        n_transitions = min(len(states) - 1, len(actions))

        for t in range(n_transitions):
            if not passes_general_filter(states[t], timestep=t):
                continue
            if not passes_side_thrust_filter(states[t], actions[t]):
                continue

            delta_pred = _predict_oracle_delta(model, norm_stats, states[t], actions[t])
            model_values.append(float(delta_pred[VX]))
            gt_values.append(float(states[t + 1][VX] - states[t][VX]))
            assoc_states.append(states[t].copy())
            timesteps.append(t)

    result = ConstantResult(
        model_values=np.array(model_values, dtype=np.float32),
        gt_values=np.array(gt_values, dtype=np.float32),
        associated_states=np.array(assoc_states, dtype=np.float32).reshape(-1, 6),
        associated_timesteps=np.array(timesteps, dtype=np.int32),
    )
    return result.to_dict() | {
        "model_values": result.model_values,
        "gt_values": result.gt_values,
        "associated_states": result.associated_states,
        "timesteps": result.associated_timesteps,
    }


def extract_kinematics_oracle(
    model, norm_stats, episodes: list[dict]
) -> dict:
    """Extract kinematic consistency: ratio dx_pred / vx (effective dt).

    Measures whether position change is proportional to velocity.
    Reports the mean ratio (should be ~1.0 for normalized coords with dt=1).
    """
    model.eval()
    model_values, gt_values, assoc_states, timesteps = [], [], [], []

    for episode in episodes:
        states = episode["states"][:, :6]
        actions = episode["actions"]
        n_transitions = min(len(states) - 1, len(actions))

        for t in range(n_transitions):
            # Note: kinematic consistency holds everywhere (not just clean physics
            # region), so we skip passes_general_filter here. We only need
            # velocities to be nonzero to avoid division by zero.
            if not passes_kinematic_filter(states[t]):
                continue

            delta_pred = _predict_oracle_delta(model, norm_stats, states[t], actions[t])
            gt_dx = states[t + 1][X] - states[t][X]

            model_ratio = delta_pred[X] / states[t][VX]
            gt_ratio = gt_dx / states[t][VX]

            model_values.append(float(model_ratio))
            gt_values.append(float(gt_ratio))
            assoc_states.append(states[t].copy())
            timesteps.append(t)

    result = ConstantResult(
        model_values=np.array(model_values, dtype=np.float32),
        gt_values=np.array(gt_values, dtype=np.float32),
        associated_states=np.array(assoc_states, dtype=np.float32).reshape(-1, 6),
        associated_timesteps=np.array(timesteps, dtype=np.int32),
    )
    return result.to_dict() | {
        "model_values": result.model_values,
        "gt_values": result.gt_values,
        "associated_states": result.associated_states,
        "timesteps": result.associated_timesteps,
    }


def extract_damping_oracle(
    model, norm_stats, episodes: list[dict]
) -> dict:
    """Extract angular damping rate: 1 - (next/current angular velocity).

    On zero-thrust transitions with sufficient angular velocity, the
    ratio angular_vel[t+1] / angular_vel[t] should be a constant < 1.0
    (damping). We store the decay amount (1 - ratio) rather than the
    ratio itself, because the ratio is near 1.0 and relative error
    on a near-1.0 quantity is misleading.
    """
    model.eval()
    model_values, gt_values, assoc_states, timesteps = [], [], [], []

    for episode in episodes:
        states = episode["states"][:, :6]
        actions = episode["actions"]
        n_transitions = min(len(states) - 1, len(actions))

        for t in range(n_transitions):
            if not passes_general_filter(states[t], timestep=t):
                continue
            if not passes_angular_damping_filter(states[t], actions[t]):
                continue

            delta_pred = _predict_oracle_delta(model, norm_stats, states[t], actions[t])
            model_next_avel = states[t][ANGULAR_VEL] + delta_pred[ANGULAR_VEL]
            gt_next_avel = states[t + 1][ANGULAR_VEL]

            current_avel = states[t][ANGULAR_VEL]
            model_ratio = model_next_avel / current_avel
            gt_ratio = gt_next_avel / current_avel

            # Store decay amount (1 - ratio), not ratio.
            # GT decay is small but positive (e.g. 0.0002). This makes
            # relative error meaningful.
            model_values.append(float(1.0 - model_ratio))
            gt_values.append(float(1.0 - gt_ratio))
            assoc_states.append(states[t].copy())
            timesteps.append(t)

    result = ConstantResult(
        model_values=np.array(model_values, dtype=np.float32),
        gt_values=np.array(gt_values, dtype=np.float32),
        associated_states=np.array(assoc_states, dtype=np.float32).reshape(-1, 6),
        associated_timesteps=np.array(timesteps, dtype=np.int32),
    )
    return result.to_dict() | {
        "model_values": result.model_values,
        "gt_values": result.gt_values,
        "associated_states": result.associated_states,
        "timesteps": result.associated_timesteps,
    }


def extract_angle_thrust_oracle(
    model, norm_stats, episodes: list[dict], gravity_model: float,
) -> dict:
    """Extract angle-thrust coupling: ratio dvx / (dvy - gravity).

    For transitions with main thrust at nonzero angle, this ratio should
    equal -tan(angle) if the model correctly vectors thrust.

    Stores the model's ratio and the true -tan(angle) as GT.
    """
    model.eval()
    model_values, gt_values, assoc_states, timesteps = [], [], [], []

    for episode in episodes:
        states = episode["states"][:, :6]
        actions = episode["actions"]
        n_transitions = min(len(states) - 1, len(actions))

        for t in range(n_transitions):
            if not passes_general_filter(states[t], timestep=t):
                continue
            if not passes_angle_thrust_filter(states[t], actions[t], gravity_model):
                continue

            delta_pred = _predict_oracle_delta(model, norm_stats, states[t], actions[t])

            dvy_thrust = delta_pred[VY] - gravity_model
            if abs(dvy_thrust) < 0.01:
                continue

            model_ratio = delta_pred[VX] / dvy_thrust

            angle = states[t][ANGLE]
            gt_ratio = -np.tan(angle)

            model_values.append(float(model_ratio))
            gt_values.append(float(gt_ratio))
            assoc_states.append(states[t].copy())
            timesteps.append(t)

    result = ConstantResult(
        model_values=np.array(model_values, dtype=np.float32),
        gt_values=np.array(gt_values, dtype=np.float32),
        associated_states=np.array(assoc_states, dtype=np.float32).reshape(-1, 6),
        associated_timesteps=np.array(timesteps, dtype=np.int32),
    )
    return result.to_dict() | {
        "model_values": result.model_values,
        "gt_values": result.gt_values,
        "associated_states": result.associated_states,
        "timesteps": result.associated_timesteps,
    }


DIM_NAMES = ["x", "y", "vx", "vy", "angle", "angular_vel"]


# ---------------------------------------------------------------------------
# Rollout infrastructure
# ---------------------------------------------------------------------------


def _rollout_trajectory(
    model, norm_stats, start_state: np.ndarray, actions: np.ndarray,
    model_state=None,
) -> np.ndarray:
    """Autoregressive rollout: feed model's own predictions back.

    Args:
        model: wm-ladder model with .step() interface.
        norm_stats: NormStats for normalize/denormalize.
        start_state: (6,) initial state in raw space.
        actions: (horizon, 2) actions to apply.
        model_state: For recurrent models, the hidden state after warmup.

    Returns:
        trajectory: (horizon + 1, 6) states in raw space (including start).
    """
    device = _get_model_device(model)
    trajectory = [start_state.copy()]
    s_raw = torch.tensor(start_state, dtype=torch.float32, device=device).unsqueeze(0)
    state_mean = norm_stats.state_mean.to(device)
    state_std = norm_stats.state_std.to(device)
    delta_mean = norm_stats.delta_mean.to(device)
    delta_std = norm_stats.delta_std.to(device)

    with torch.no_grad():
        for t in range(len(actions)):
            s_norm = (s_raw - state_mean) / state_std
            a = torch.tensor(actions[t], dtype=torch.float32, device=device).unsqueeze(0)
            delta_norm, model_state = model.step(s_norm, a, model_state)
            delta_raw = delta_norm * delta_std + delta_mean
            s_raw = s_raw + delta_raw
            trajectory.append(s_raw.squeeze(0).cpu().numpy().copy())

    return np.stack(trajectory).astype(np.float32)


def _warmup_recurrent(
    model, norm_stats, episode: dict, warmup_steps: int,
):
    """Teacher-force warmup for recurrent models.

    Feeds GT states through the model to build hidden state.
    Returns model_state after warmup.
    """
    device = _get_model_device(model)
    states = episode["states"][:, :6]
    actions = episode["actions"]
    model_state = None
    state_mean = norm_stats.state_mean.to(device)
    state_std = norm_stats.state_std.to(device)

    with torch.no_grad():
        for t in range(min(warmup_steps, len(actions))):
            s = torch.tensor(states[t], dtype=torch.float32, device=device).unsqueeze(0)
            s_norm = (s - state_mean) / state_std
            a = torch.tensor(actions[t], dtype=torch.float32, device=device).unsqueeze(0)
            _, model_state = model.step(s_norm, a, model_state)

    return model_state


def _extract_from_window(
    traj: np.ndarray, window_actions: np.ndarray,
    gt_states: np.ndarray, window_start: int, horizon: int,
    accum: dict,
    gravity_model: float | None, gravity_gt: float | None,
):
    """Extract constants from a single rollout window into accumulators.

    traj: (horizon+1, 6) model's autoregressive trajectory.
    gt_states: full episode GT states (for GT deltas).
    window_start: timestep index where this window starts in the episode.
    """
    for h in range(horizon):
        s_h = traj[h]
        a_h = window_actions[h]
        t = window_start + h
        if not passes_general_filter(s_h, timestep=t):
            continue

        model_delta = traj[h + 1] - traj[h]
        gt_delta = gt_states[t + 1] - gt_states[t]

        # Gravity.
        if passes_gravity_filter(s_h, a_h):
            accum["gravity"]["model"].append(float(model_delta[VY]))
            accum["gravity"]["gt"].append(float(gt_delta[VY]))
            accum["gravity"]["states"].append(s_h.copy())
            accum["gravity"]["times"].append(t)

        # Main thrust.
        if passes_main_thrust_filter(s_h, a_h):
            g_m = gravity_model if gravity_model is not None else -0.13
            g_g = gravity_gt if gravity_gt is not None else -0.13
            accum["main_thrust"]["model"].append(float(model_delta[VY] - g_m))
            accum["main_thrust"]["gt"].append(float(gt_delta[VY] - g_g))
            accum["main_thrust"]["states"].append(s_h.copy())
            accum["main_thrust"]["times"].append(t)

        # Side thrust.
        if passes_side_thrust_filter(s_h, a_h):
            accum["side_thrust"]["model"].append(float(model_delta[VX]))
            accum["side_thrust"]["gt"].append(float(gt_delta[VX]))
            accum["side_thrust"]["states"].append(s_h.copy())
            accum["side_thrust"]["times"].append(t)

        # Kinematics.
        if passes_kinematic_filter(s_h):
            accum["kinematics"]["model"].append(float(model_delta[X] / s_h[VX]))
            accum["kinematics"]["gt"].append(float(gt_delta[X] / s_h[VX]))
            accum["kinematics"]["states"].append(s_h.copy())
            accum["kinematics"]["times"].append(t)

        # Angular damping.
        if passes_angular_damping_filter(s_h, a_h):
            m_next = s_h[ANGULAR_VEL] + model_delta[ANGULAR_VEL]
            g_next = gt_states[t + 1][ANGULAR_VEL]
            accum["angular_damping"]["model"].append(float(1.0 - m_next / s_h[ANGULAR_VEL]))
            accum["angular_damping"]["gt"].append(float(1.0 - g_next / s_h[ANGULAR_VEL]))
            accum["angular_damping"]["states"].append(s_h.copy())
            accum["angular_damping"]["times"].append(t)


def extract_constants_rollout(
    model, norm_stats, episodes: list[dict],
    horizon: int = 10, recurrent: bool = False,
    warmup_steps: int = 50,
    gravity_model: float | None = None,
    gravity_gt: float | None = None,
) -> dict:
    """Extract ALL constants from short autoregressive rollouts.

    For each episode, teacher-forces through the entire episode to maintain
    correct hidden state (for recurrent models). At every `horizon`-th step,
    snapshots the hidden state and branches off for a 10-step autoregressive
    rollout. The branch is independent — constant extraction happens on the
    model's own trajectory within the branch.

    For stateless models, no teacher-forcing needed — each window just starts
    from the GT state and runs the model autoregressively.

    Returns dict mapping constant name -> extraction result dict.
    """
    model.eval()

    accum = {
        "gravity": {"model": [], "gt": [], "states": [], "times": []},
        "main_thrust": {"model": [], "gt": [], "states": [], "times": []},
        "side_thrust": {"model": [], "gt": [], "states": [], "times": []},
        "kinematics": {"model": [], "gt": [], "states": [], "times": []},
        "angular_damping": {"model": [], "gt": [], "states": [], "times": []},
    }

    for episode in episodes:
        states = episode["states"][:, :6]
        actions = episode["actions"]
        n_transitions = min(len(states) - 1, len(actions))

        if recurrent:
            device = _get_model_device(model)
            state_mean = norm_stats.state_mean.to(device)
            state_std = norm_stats.state_std.to(device)
            model_state = None
            with torch.no_grad():
                for t in range(n_transitions):
                    s = torch.tensor(states[t], dtype=torch.float32, device=device).unsqueeze(0)
                    s_norm = (s - state_mean) / state_std
                    a = torch.tensor(actions[t], dtype=torch.float32, device=device).unsqueeze(0)
                    _, model_state = model.step(s_norm, a, model_state)

                    if t >= warmup_steps and (t - warmup_steps) % horizon == 0 \
                            and t + horizon < n_transitions:
                        window_actions = actions[t:t + horizon]
                        traj = _rollout_trajectory(
                            model, norm_stats, states[t], window_actions,
                            model_state=model_state,
                        )
                        _extract_from_window(
                            traj, window_actions, states, t, horizon,
                            accum, gravity_model, gravity_gt,
                        )
        else:
            t = 3  # Skip first 3 steps (general filter).
            while t + horizon < n_transitions:
                window_actions = actions[t:t + horizon]
                traj = _rollout_trajectory(
                    model, norm_stats, states[t], window_actions,
                )
                _extract_from_window(
                    traj, window_actions, states, t, horizon,
                    accum, gravity_model, gravity_gt,
                )
                t += horizon

    results = {}
    for name, acc in accum.items():
        cr = ConstantResult(
            model_values=np.array(acc["model"], dtype=np.float32),
            gt_values=np.array(acc["gt"], dtype=np.float32),
            associated_states=np.array(acc["states"], dtype=np.float32).reshape(-1, 6),
            associated_timesteps=np.array(acc["times"], dtype=np.int32),
        )
        results[name] = cr.to_dict()

    return results


# ---------------------------------------------------------------------------
# Compounding diagnostic
# ---------------------------------------------------------------------------


def compute_compounding_curve(
    model, norm_stats, episodes: list[dict],
    horizons: list[int] = None,
    recurrent: bool = False,
    warmup_steps: int = 50,
) -> dict:
    """Compute MSE-vs-horizon compounding diagnostic.

    For each episode, runs autoregressive rollouts at each horizon and
    computes per-dim MSE between model trajectory and GT.

    Args:
        model: wm-ladder model.
        norm_stats: NormStats.
        episodes: Validation episodes.
        horizons: List of horizon lengths to evaluate (default: [1,2,5,10,20,50]).
        recurrent: Whether model needs warmup.
        warmup_steps: Warmup steps for recurrent models.

    Returns:
        Dict with:
          - mse_by_horizon: {h: float} — mean MSE across episodes at each horizon.
          - per_dim_mse: {h: {dim_name: float}} — per-dim MSE at each horizon.
          - n_episodes_used: int — number of episodes with sufficient length.
    """
    if horizons is None:
        horizons = [1, 2, 5, 10, 20, 50]

    model.eval()
    max_h = max(horizons)

    mse_accum = {h: [] for h in horizons}
    dim_mse_accum = {h: {name: [] for name in DIM_NAMES} for h in horizons}

    for episode in episodes:
        states = episode["states"][:, :6]
        actions = episode["actions"]
        n_transitions = min(len(states) - 1, len(actions))

        # Adaptive warmup: use as much as the episode allows, up to the default.
        # Need at least 1 step after warmup for the shortest horizon.
        if recurrent:
            actual_warmup = min(warmup_steps, max(0, n_transitions - min(horizons) - 1))
            if actual_warmup < 0:
                continue
        else:
            actual_warmup = 0

        start_t = actual_warmup
        # How many steps are available after warmup?
        available = n_transitions - start_t
        if available < min(horizons):
            continue

        model_state = None
        if recurrent and actual_warmup > 0:
            model_state = _warmup_recurrent(
                model, norm_stats, episode, actual_warmup,
            )

        # Rollout up to max horizon that fits, or max_h.
        rollout_len = min(max_h, available)
        rollout_actions = actions[start_t:start_t + rollout_len]
        traj = _rollout_trajectory(
            model, norm_stats, states[start_t], rollout_actions,
            model_state=model_state,
        )
        gt_traj = states[start_t:start_t + rollout_len + 1]

        for h in horizons:
            if h > len(traj) - 1:
                continue
            pred_window = traj[1:h + 1]
            gt_window = gt_traj[1:h + 1]
            mse = float(np.mean((pred_window - gt_window) ** 2))
            mse_accum[h].append(mse)

            for i, name in enumerate(DIM_NAMES):
                dim_mse = float(np.mean((pred_window[:, i] - gt_window[:, i]) ** 2))
                dim_mse_accum[h][name].append(dim_mse)

    mse_by_horizon = {}
    per_dim_mse = {}
    for h in horizons:
        if mse_accum[h]:
            mse_by_horizon[h] = float(np.mean(mse_accum[h]))
            per_dim_mse[h] = {
                name: float(np.mean(dim_mse_accum[h][name]))
                for name in DIM_NAMES
            }
        else:
            mse_by_horizon[h] = float("nan")
            per_dim_mse[h] = {name: float("nan") for name in DIM_NAMES}

    fit_params = {"a": float("nan"), "b": float("nan")}
    valid_h = [(h, mse) for h, mse in mse_by_horizon.items()
               if not np.isnan(mse) and mse > 0 and h > 0]
    if len(valid_h) >= 2:
        log_h = np.array([np.log(h) for h, _ in valid_h])
        log_mse = np.array([np.log(mse) for _, mse in valid_h])
        A = np.stack([log_h, np.ones(len(log_h))], axis=1)
        coefs, _, _, _ = np.linalg.lstsq(A, log_mse, rcond=None)
        fit_params["b"] = float(coefs[0])
        fit_params["a"] = float(np.exp(coefs[1]))

    useful_horizon = {}
    mse_threshold = 0.1
    for name in DIM_NAMES:
        for h in sorted(horizons):
            if h in per_dim_mse and not np.isnan(per_dim_mse[h].get(name, float("nan"))):
                if per_dim_mse[h][name] > mse_threshold:
                    useful_horizon[name] = h
                    break
        if name not in useful_horizon:
            useful_horizon[name] = max(horizons)

    first_to_diverge = sorted(useful_horizon.items(), key=lambda x: x[1])

    return {
        "mse_by_horizon": mse_by_horizon,
        "per_dim_mse": per_dim_mse,
        "fit_params": fit_params,
        "useful_horizon": useful_horizon,
        "first_to_diverge": [(name, h) for name, h in first_to_diverge],
        "n_episodes_used": len(mse_accum[horizons[0]]) if mse_accum[horizons[0]] else 0,
    }


# ---------------------------------------------------------------------------
# Warmup length diagnostic
# ---------------------------------------------------------------------------


def compute_warmup_curve(
    model, norm_stats, episodes: list[dict],
    warmup_lengths: list[int] = None,
) -> dict:
    """Compute 1-step oracle MSE as a function of warmup length.

    For each warmup length, teacher-forces that many steps, then measures
    the 1-step prediction error on subsequent transitions. Shows how much
    context the model actually uses.

    Only meaningful for recurrent models (GRU, RSSM).

    Args:
        model: Recurrent wm-ladder model.
        norm_stats: NormStats.
        episodes: Validation episodes.
        warmup_lengths: List of warmup lengths to test (default: [0,1,5,10,20,50]).

    Returns:
        Dict with oracle_mse_by_warmup: {warmup_len: float}.
    """
    if warmup_lengths is None:
        warmup_lengths = [0, 1, 5, 10, 20, 50]

    model.eval()
    device = _get_model_device(model)
    state_mean = norm_stats.state_mean.to(device)
    state_std = norm_stats.state_std.to(device)
    delta_mean = norm_stats.delta_mean.to(device)
    delta_std = norm_stats.delta_std.to(device)
    mse_by_warmup = {}

    for wl in warmup_lengths:
        errors = []
        for episode in episodes:
            states = episode["states"][:, :6]
            actions = episode["actions"]
            n_transitions = min(len(states) - 1, len(actions))

            if wl + 10 >= n_transitions:
                continue  # Need warmup + at least 10 test transitions.

            # Teacher-force warmup.
            model_state = None
            with torch.no_grad():
                for t in range(wl):
                    s = torch.tensor(states[t], dtype=torch.float32, device=device).unsqueeze(0)
                    s_norm = (s - state_mean) / state_std
                    a = torch.tensor(actions[t], dtype=torch.float32, device=device).unsqueeze(0)
                    _, model_state = model.step(s_norm, a, model_state)

            # Measure 1-step oracle error on next 10 transitions.
            with torch.no_grad():
                for t in range(wl, min(wl + 10, n_transitions)):
                    s = torch.tensor(states[t], dtype=torch.float32, device=device).unsqueeze(0)
                    s_norm = (s - state_mean) / state_std
                    a = torch.tensor(actions[t], dtype=torch.float32, device=device).unsqueeze(0)
                    delta_norm, model_state = model.step(s_norm, a, model_state)
                    delta_raw = delta_norm * delta_std + delta_mean
                    delta_pred = delta_raw.squeeze(0).cpu().numpy()

                    gt_delta = states[t + 1] - states[t]
                    mse = float(np.mean((delta_pred - gt_delta) ** 2))
                    errors.append(mse)

        mse_by_warmup[wl] = float(np.mean(errors)) if errors else float("nan")

    return {"oracle_mse_by_warmup": mse_by_warmup}


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

import json as _json  # noqa: E402 — placed here to keep imports grouped by feature


def generate_report(
    model, norm_stats, episodes: list[dict],
    recurrent: bool = False,
    rollout_horizon: int = 10,
    warmup_steps: int = 50,
) -> dict:
    """Run all physics understanding measurements and return structured results.

    Extracts all 6 physical constants in both oracle and rollout modes,
    computes consistency R² for each, and runs the compounding diagnostic.

    Args:
        model: wm-ladder model with .step() interface.
        norm_stats: NormStats from lwp.training.
        episodes: Validation episodes.
        recurrent: Whether model is recurrent (needs warmup).
        rollout_horizon: Steps for rollout extraction (default: 10).
        warmup_steps: Warmup steps for recurrent models.

    Returns:
        Dict with keys: constants, compounding, metadata.
    """
    constants = {}

    import warnings

    # 1. Gravity (oracle) — extracted first because thrust depends on it.
    grav_oracle = extract_gravity_oracle(model, norm_stats, episodes)
    if grav_oracle["n_samples"] == 0:
        warnings.warn(
            "Gravity extraction got 0 qualifying samples. "
            "Thrust and angle-thrust results will be unreliable. "
            "Check that episodes have zero-thrust transitions in clean regions."
        )
    gravity_model = grav_oracle["model_mean"] if grav_oracle["n_samples"] > 0 else 0.0
    gravity_gt = grav_oracle["gt_mean"] if grav_oracle["n_samples"] > 0 else 0.0

    # 2. Main thrust (oracle, uses gravity estimates).
    thrust_oracle = extract_main_thrust_oracle(
        model, norm_stats, episodes, gravity_model, gravity_gt,
    )

    # 3. Side thrust (oracle).
    side_oracle = extract_side_thrust_oracle(model, norm_stats, episodes)

    # 4. Kinematics (oracle).
    kin_oracle = extract_kinematics_oracle(model, norm_stats, episodes)

    # 5. Angular damping (oracle).
    damp_oracle = extract_damping_oracle(model, norm_stats, episodes)

    # 6. Angle-thrust coupling (oracle).
    angle_oracle = extract_angle_thrust_oracle(
        model, norm_stats, episodes, gravity_model,
    )

    # Rollout extraction for all constants at once.
    rollout_results = extract_constants_rollout(
        model, norm_stats, episodes,
        horizon=rollout_horizon, recurrent=recurrent,
        warmup_steps=warmup_steps,
        gravity_model=gravity_model, gravity_gt=gravity_gt,
    )

    # Assemble per-constant results with oracle + rollout + consistency.
    oracle_map = {
        "gravity": grav_oracle,
        "main_thrust": thrust_oracle,
        "side_thrust": side_oracle,
        "kinematics": kin_oracle,
        "angular_damping": damp_oracle,
    }

    for name, oracle in oracle_map.items():
        consistency = {}
        if oracle["n_samples"] >= 3:
            consistency = compute_consistency_r2(
                oracle["model_values"], oracle["associated_states"],
            )
        constants[name] = {
            "oracle": _strip_arrays(oracle),
            "rollout": rollout_results.get(name, {"n_samples": 0}),
            "consistency": consistency,
        }

    # Angle-thrust coupling (special — reports fit R², not a scalar constant).
    angle_consistency = {}
    if angle_oracle["n_samples"] >= 3:
        ss_res = np.sum((angle_oracle["model_values"] - angle_oracle["gt_values"]) ** 2)
        ss_tot = np.sum((angle_oracle["gt_values"] - np.mean(angle_oracle["gt_values"])) ** 2)
        fit_r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
        angle_consistency = {"fit_r2": float(max(0.0, fit_r2))}
    constants["angle_thrust"] = {
        "oracle": _strip_arrays(angle_oracle),
        "rollout": {"n_samples": 0},
        "consistency": angle_consistency,
    }

    # Compounding diagnostic.
    compounding = compute_compounding_curve(
        model, norm_stats, episodes,
        recurrent=recurrent, warmup_steps=warmup_steps,
    )

    # Warmup length diagnostic (recurrent models only).
    warmup_curve = None
    if recurrent:
        warmup_curve = compute_warmup_curve(model, norm_stats, episodes)

    total_transitions = sum(
        min(len(ep["states"]) - 1, len(ep["actions"])) for ep in episodes
    )

    return {
        "constants": constants,
        "compounding": compounding,
        "warmup_curve": warmup_curve,
        "metadata": {
            "n_episodes": len(episodes),
            "n_total_transitions": total_transitions,
            "rollout_horizon": rollout_horizon,
            "recurrent": recurrent,
        },
    }


def _strip_arrays(result: dict) -> dict:
    """Remove numpy arrays from result dict for JSON serialization."""
    return {
        k: v for k, v in result.items()
        if not isinstance(v, np.ndarray)
    }


def format_console_report(results: dict, run_name: str = "") -> str:
    """Format results as a human-readable console report.

    Follows the format from the spec: constants table with oracle + rollout
    columns, consistency R², compounding summary.
    """
    lines = []
    lines.append(f"Physics Understanding Report: {run_name}")
    lines.append("=" * 64)

    meta = results["metadata"]
    lines.append(f"N episodes: {meta['n_episodes']}, "
                 f"N transitions: {meta['n_total_transitions']:,}")
    lines.append("")

    # Constants table.
    lines.append("Physical Constants:")
    lines.append(f"  {'':30s} {'Oracle (1-step)':>30s}   {'Rollout':>20s}")
    lines.append(f"  {'':30s} {'value':>10s} {'err':>8s} {'n':>6s}   "
                 f"{'value':>10s} {'err':>8s}")

    const_display = {
        "gravity": "Gravity (dvy/step)",
        "main_thrust": "Main thrust response",
        "side_thrust": "Side thrust response",
        "kinematics": "Kinematic dx/vx",
        "angular_damping": "Ang. damping (1-ratio)",
        "angle_thrust": "Angle-thrust coupling",
    }

    for key, label in const_display.items():
        if key not in results["constants"]:
            continue
        c = results["constants"][key]
        o = c["oracle"]
        r = c["rollout"]

        o_val = f"{o['model_mean']:+.4f}" if o["n_samples"] > 0 else "-"
        o_err = f"{o['relative_error']*100:.1f}%" if o["n_samples"] > 0 and not np.isnan(o["relative_error"]) else "-"
        o_n = str(o["n_samples"])
        r_val = f"{r['model_mean']:+.4f}" if r.get("n_samples", 0) > 0 else "-"
        r_err = f"{r['relative_error']*100:.1f}%" if r.get("n_samples", 0) > 0 and not np.isnan(r.get("relative_error", float("nan"))) else "-"

        lines.append(f"  {label:30s} {o_val:>10s} {o_err:>8s} {o_n:>6s}   "
                     f"{r_val:>10s} {r_err:>8s}")

    # GT reference line.
    lines.append("")
    gt_parts = []
    for key, label in [("gravity", "Gravity"), ("main_thrust", "Thrust"),
                       ("side_thrust", "Side"), ("angular_damping", "Damping")]:
        if key in results["constants"]:
            gt = results["constants"][key]["oracle"].get("gt_mean", float("nan"))
            if not np.isnan(gt):
                gt_parts.append(f"{label}: {gt:+.4f}")
    if gt_parts:
        lines.append(f"  GT reference: {', '.join(gt_parts)}")

    # Angle-thrust coupling (special section).
    if "angle_thrust" in results["constants"]:
        at = results["constants"]["angle_thrust"]
        lines.append("")
        lines.append("Angle-Thrust Coupling:")
        fit_r2 = at.get("consistency", {}).get("fit_r2", float("nan"))
        if not np.isnan(fit_r2):
            lines.append(f"  sin/cos fit R²:  {fit_r2:.2f}  (1.0 = perfect vectoring)")
        n = at["oracle"].get("n_samples", 0)
        lines.append(f"  N samples:       {n}")

    # Consistency.
    lines.append("")
    lines.append("Consistency (R² vs irrelevant variable, want < 0.05):")
    for key in ["gravity", "main_thrust", "side_thrust", "kinematics", "angular_damping"]:
        if key not in results["constants"]:
            continue
        consistency = results["constants"][key].get("consistency", {})
        for dim, r2 in consistency.items():
            if np.isnan(r2):
                continue
            flag = "*** SPURIOUS" if r2 > 0.05 else "ok"
            lines.append(f"  {const_display.get(key, key):20s} vs {dim:12s} {r2:.3f}  {flag}")

    # Compounding.
    comp = results.get("compounding", {})
    mse_h = comp.get("mse_by_horizon", {})
    if mse_h:
        lines.append("")
        lines.append("Compounding (MSE vs horizon):")
        for h in sorted(mse_h.keys()):
            lines.append(f"  h={h:<3d}  {mse_h[h]:.6f}")

        fit = comp.get("fit_params", {})
        a, b = fit.get("a", float("nan")), fit.get("b", float("nan"))
        if not np.isnan(b):
            lines.append(f"  Error growth:    MSE(h) ~ {a:.4f} * h^{b:.1f}")

        ftd = comp.get("first_to_diverge", [])
        if ftd:
            parts = [f"{name} (h={h})" for name, h in ftd[:3]]
            lines.append(f"  First to diverge: {', '.join(parts)}")

    # Warmup curve (recurrent only).
    wc = results.get("warmup_curve")
    if wc is not None:
        lines.append("")
        lines.append("Warmup Length Diagnostic (1-step oracle MSE vs warmup steps):")
        for wl in sorted(wc["oracle_mse_by_warmup"].keys()):
            mse = wc["oracle_mse_by_warmup"][wl]
            lines.append(f"  warmup={wl:<4d}  MSE={mse:.6f}")

    return "\n".join(lines)


def save_json_report(results: dict, output_path: str):
    """Save results to JSON file.

    Converts any remaining numpy types to Python natives for serialization.
    """
    def _convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, dict):
            return {str(k): _convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_convert(v) for v in obj]
        return obj

    with open(output_path, "w") as f:
        _json.dump(_convert(results), f, indent=2)


def compute_consistency_r2(
    values: np.ndarray,
    states: np.ndarray,
) -> dict[str, float]:
    """Compute R² of extracted constant vs each state dimension.

    A physical constant should NOT depend on irrelevant state variables.
    High R² (> 0.05) flags a spurious dependency learned by the model.

    Args:
        values: (n,) extracted constant values per transition.
        states: (n, 6) state at each transition.

    Returns:
        Dict mapping dim name -> R² value. NaN if < 3 samples.
    """
    n = len(values)
    if n < 3:
        return {name: float("nan") for name in DIM_NAMES}

    r2_dict = {}
    ss_tot = np.sum((values - np.mean(values)) ** 2)

    for i, name in enumerate(DIM_NAMES):
        if ss_tot < 1e-12:
            r2_dict[name] = 0.0
            continue

        x = states[:, i]
        A = np.stack([x, np.ones(n)], axis=1)
        coefs, _, _, _ = np.linalg.lstsq(A, values, rcond=None)
        predicted = A @ coefs
        ss_res = np.sum((values - predicted) ** 2)
        r2 = 1.0 - ss_res / ss_tot
        r2_dict[name] = float(max(0.0, r2))

    return r2_dict
