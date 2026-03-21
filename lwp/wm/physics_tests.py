# lunar_lander/src/wm/physics_tests.py
"""Physics unit test framework for world models.

Tests whether a world model has learned causal physics relationships by
running controlled maneuvers and comparing model predictions against Box2D
ground truth. Architecture-agnostic — works with any model that implements
the WorldModel interface.

Design spec: traitful-docs/docs/research/.../specs/physics-unit-tests.md

Maneuvers are split into two tiers:

  PRIMARY (default test suite — clean, visual, no heuristics needed):
    1. Free fall — gravity
    2. Full thrust — F=ma / thrust-to-weight ratio
    3. Side thrust — lateral acceleration
    4. Angle-thrust coupling — thrust vectoring
    5. Impulse thrust — temporal dynamics (alternating on/off cycles)
    6. Ramp thrust — continuous thrust response (linear 0->1)
    7. Thrust-then-coast — powered->ballistic transition
    8. Opposite thrust — action switching (main then side)

  SECONDARY (available via --maneuvers flag — real physics but measurement issues):
    9. Hover — force balance / equilibrium (heuristic thrust computation)
    10. Conservation — momentum without forces (Box2D damping artifacts)
    11. Angular decay — damping (precondition-dependent)

Each maneuver branches from a real episode with controlled actions, runs
those actions through both the model and Box2D, and compares the resulting
state trajectories.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from parametric_lunar_lander.physics_config import LunarLanderPhysicsConfig
from parametric_lunar_lander.env import FPS, SCALE, MAIN_ENGINE_Y_LOCATION


@dataclass
class Maneuver:
    """Definition of a single physics unit test.

    A maneuver specifies:
      - An action sequence to apply after branching from a real episode
      - A measurement window within that sequence (skip transients)
      - An optional precondition on the branch-point state
      - Tolerance for the model-vs-GT comparison

    Actions are defined via action_config: {"main": float, "side": float}.
    make_actions() fills an (n_steps, 2) array from these values. Maneuvers
    that need dynamic action values (e.g., hover computes thrust from physics)
    provide a _compute_actions override that returns the action_config dict.

    This design makes thrust levels explicit and configurable — you can test
    at 50% thrust, or vary side thrust, without writing new action generators.
    """

    name: str
    description: str
    n_steps: int
    window_start: int  # Measurement window start (inclusive, 0-indexed step)
    window_end: int    # Measurement window end (exclusive)
    tolerance: float   # Relative tolerance for pass/fail (0.0 = absolute mode)
    absolute_tolerance: float | None = None  # For near-zero targets (hover, conservation)
    action_config: dict = field(default_factory=lambda: {"main": 0.0, "side": 0.0})
    _compute_actions: Callable | None = None  # Override to compute action_config dynamically
    _action_pattern: Callable | None = None  # Override for time-varying action sequences
    _check_precondition: Callable | None = None

    def make_actions(
        self, physics_config: LunarLanderPhysicsConfig | None = None,
    ) -> np.ndarray:
        """Generate the controlled action sequence for this maneuver.

        If _compute_actions is set (e.g., hover), calls it to get the
        action_config dict dynamically. Otherwise uses the static action_config.

        Args:
            physics_config: Needed by maneuvers that compute actions from
                physics params (e.g., hover thrust).

        Returns:
            actions: (n_steps, 2) float32 array. Column 0 = main, column 1 = side.
        """
        # Time-varying patterns override the constant-fill path entirely.
        if self._action_pattern is not None:
            return self._action_pattern(self)

        if self._compute_actions is not None:
            cfg = self._compute_actions(self, physics_config)
        else:
            cfg = self.action_config

        actions = np.zeros((self.n_steps, 2), dtype=np.float32)
        actions[:, 0] = cfg.get("main", 0.0)
        actions[:, 1] = cfg.get("side", 0.0)
        return actions

    def check_precondition(self, state: np.ndarray) -> bool:
        """Check if the branch-point state meets this maneuver's precondition.

        Args:
            state: (15,) or (8,) state vector at the proposed branch point.

        Returns:
            True if the state is suitable for this maneuver.
        """
        if self._check_precondition is not None:
            return self._check_precondition(state)
        return True  # No precondition — any state works.


# --- Dynamic action computation ---
# Only needed for maneuvers where the action value depends on physics config.

def _compute_hover_actions(maneuver: Maneuver, config: LunarLanderPhysicsConfig | None = None) -> dict:
    """Compute the main thrust value that exactly balances gravity.

    From the Box2D engine physics:
      - Main engine fires when action[0] > 0
      - m_power = (clip(action[0], 0, 1) + 1) * 0.5  →  range [0.5, 1.0]
      - Impulse magnitude = main_engine_power * m_power
      - The impulse offsets are proportional to MAIN_ENGINE_Y_LOCATION / SCALE

    Approach (from calibration.py's measured relationship):
      impulse_factor = MAIN_ENGINE_Y_LOCATION / SCALE
      engine_accel = (impulse_factor * main_engine_power * m_power) / (mass * dt)
      Set engine_accel = |gravity| and solve for m_power:
        m_power = |gravity| * mass * dt / (impulse_factor * main_engine_power)
      Then solve for action[0]:
        m_power = (action[0] + 1) * 0.5
        action[0] = 2 * m_power - 1

    Returns:
        {"main": hover_action_value, "side": 0.0}
    """
    if config is None:
        raise ValueError("hover maneuver requires physics_config to compute hover thrust")

    dt = 1.0 / FPS
    mass = config.lander_density * LunarLanderPhysicsConfig.BODY_AREA
    impulse_factor = MAIN_ENGINE_Y_LOCATION / SCALE

    # Required m_power for hover equilibrium.
    m_power = abs(config.gravity) * mass * dt / (impulse_factor * config.main_engine_power)

    # Clamp m_power to [0.5, 1.0] range (engine's operating range).
    m_power = float(np.clip(m_power, 0.5, 1.0))

    # Invert the throttle mapping: m_power = (action + 1) * 0.5
    action_value = 2.0 * m_power - 1.0
    action_value = float(np.clip(action_value, 0.0, 1.0))

    return {"main": action_value, "side": 0.0}


# --- Action pattern generators ---
# Used by time-varying maneuvers. Each returns (n_steps, 2) array.

def _pulse_pattern(
    n_steps: int,
    action_config: dict,
    cycle_length: int = 30,
) -> np.ndarray:
    """Alternating on/off cycles of the action config.

    Even cycles (0, 2, 4, ...) apply the action_config values.
    Odd cycles (1, 3, 5, ...) apply zero thrust.
    Each cycle is cycle_length steps long.
    """
    actions = np.zeros((n_steps, 2), dtype=np.float32)
    for t in range(n_steps):
        cycle_idx = t // cycle_length
        if cycle_idx % 2 == 0:  # On cycle
            actions[t, 0] = action_config.get("main", 0.0)
            actions[t, 1] = action_config.get("side", 0.0)
        # Off cycle: already zero
    return actions


def _ramp_pattern(
    n_steps: int,
    action_config: dict,
) -> np.ndarray:
    """Linearly ramp action values from 0 to their config value.

    At step 0, thrust is 0. At step n_steps-1, thrust equals the
    action_config value. Linear interpolation in between.
    """
    actions = np.zeros((n_steps, 2), dtype=np.float32)
    for t in range(n_steps):
        frac = t / max(n_steps - 1, 1)
        actions[t, 0] = frac * action_config.get("main", 0.0)
        actions[t, 1] = frac * action_config.get("side", 0.0)
    return actions


def _split_pattern(
    n_steps: int,
    first_config: dict,
    second_config: dict,
) -> np.ndarray:
    """First half uses first_config, second half uses second_config.

    Split at n_steps // 2. Tests regime transitions.
    """
    mid = n_steps // 2
    actions = np.zeros((n_steps, 2), dtype=np.float32)
    actions[:mid, 0] = first_config.get("main", 0.0)
    actions[:mid, 1] = first_config.get("side", 0.0)
    actions[mid:, 0] = second_config.get("main", 0.0)
    actions[mid:, 1] = second_config.get("side", 0.0)
    return actions


# --- Precondition checkers ---
# State dim indices: [4]=angle, [5]=angular_vel (normalized by 20/FPS).

def _needs_high_angular_vel(state: np.ndarray) -> bool:
    """Angular decay needs the lander spinning fast enough to measure decay.

    The angular velocity in the state vector is normalized: obs[5] = 20 * omega / FPS.
    We need |omega| > 1 rad/s. In normalized units: |obs[5]| > 20 * 1.0 / 50 = 0.4.
    """
    return abs(state[5]) > 0.4


def _needs_significant_tilt(state: np.ndarray) -> bool:
    """Angle-thrust coupling needs the lander tilted to create a lateral component.

    obs[4] = angle in radians. Need |angle| > 0.1 rad (~5.7 degrees).
    """
    return abs(state[4]) > 0.1


# --- Action pattern wrappers ---
# These adapt the generic pattern generators (_pulse_pattern, etc.) to the
# Maneuver._action_pattern interface: fn(maneuver) -> (n_steps, 2) array.

def _impulse_thrust_pattern(maneuver: Maneuver) -> np.ndarray:
    """Alternating 30-step on/off main thrust cycles."""
    return _pulse_pattern(maneuver.n_steps, maneuver.action_config, cycle_length=30)

def _ramp_thrust_pattern(maneuver: Maneuver) -> np.ndarray:
    """Linear 0->1 main thrust ramp."""
    return _ramp_pattern(maneuver.n_steps, maneuver.action_config)

def _thrust_then_coast_pattern(maneuver: Maneuver) -> np.ndarray:
    """Full main for first half, zero for second half."""
    return _split_pattern(
        maneuver.n_steps,
        first_config={"main": 1.0, "side": 0.0},
        second_config={"main": 0.0, "side": 0.0},
    )

def _opposite_thrust_pattern(maneuver: Maneuver) -> np.ndarray:
    """Full main for first half, full side for second half."""
    return _split_pattern(
        maneuver.n_steps,
        first_config={"main": 1.0, "side": 0.0},
        second_config={"main": 0.0, "side": 1.0},
    )


# --- Maneuver registries ---

# Default maneuver duration: 300 steps (6 seconds at 50 FPS).
# Long enough to see meaningful divergence even for slow dynamics.
# Overridable via --duration CLI flag in physics_test_wm.py.
DEFAULT_DURATION = 300
# PRIMARY: clean, visual, no heuristics needed. These are the standard test suite.
# SECONDARY: real physics but measurement issues (heuristic thrust, Box2D damping,
#   precondition-dependent). Available via --maneuvers flag for targeted use.

PRIMARY_MANEUVERS: dict[str, Maneuver] = {
    "free_fall": Maneuver(
        name="free_fall",
        description="Free fall — tests gravity. Zero thrust.",
        n_steps=DEFAULT_DURATION,
        window_start=0,
        window_end=DEFAULT_DURATION,
        tolerance=0.05,  # 5% — clean measurement, only gravity acts
        action_config={"main": 0.0, "side": 0.0},
    ),
    "full_thrust": Maneuver(
        name="full_thrust",
        description="Full main thrust — tests F=ma / TWR. Main=1.0.",
        n_steps=DEFAULT_DURATION,
        window_start=5,
        window_end=DEFAULT_DURATION,
        tolerance=0.15,  # 15% — engine dispersion, angle drift
        action_config={"main": 1.0, "side": 0.0},
    ),
    "side_thrust": Maneuver(
        name="side_thrust",
        description="Full side thrust — tests lateral acceleration. Side=1.0.",
        n_steps=DEFAULT_DURATION,
        window_start=5,
        window_end=DEFAULT_DURATION,
        tolerance=0.10,  # 10% — off-center torque rotates lander
        action_config={"main": 0.0, "side": 1.0},
    ),
    "angle_thrust": Maneuver(
        name="angle_thrust",
        description="Angle-thrust coupling — tests thrust vectoring. Full main from tilted state.",
        n_steps=DEFAULT_DURATION,
        window_start=5,
        window_end=DEFAULT_DURATION,
        tolerance=0.15,  # 15% — angle changes during measurement
        action_config={"main": 1.0, "side": 0.0},
        _check_precondition=_needs_significant_tilt,
    ),
    "impulse_thrust": Maneuver(
        name="impulse_thrust",
        description="Impulse thrust -- alternating 30-step on/off main thrust. Tests temporal dynamics.",
        n_steps=DEFAULT_DURATION,
        window_start=0,
        window_end=DEFAULT_DURATION,
        tolerance=0.20,
        action_config={"main": 1.0, "side": 0.0},
        _action_pattern=_impulse_thrust_pattern,
    ),
    "ramp_thrust": Maneuver(
        name="ramp_thrust",
        description="Ramp thrust -- linear 0->1 main thrust. Tests continuous thrust response.",
        n_steps=DEFAULT_DURATION,
        window_start=0,
        window_end=DEFAULT_DURATION,
        tolerance=0.20,
        action_config={"main": 1.0, "side": 0.0},
        _action_pattern=_ramp_thrust_pattern,
    ),
    "thrust_then_coast": Maneuver(
        name="thrust_then_coast",
        description="Thrust then coast -- full main for half, zero for half. Tests powered->ballistic transition.",
        n_steps=DEFAULT_DURATION,
        window_start=0,
        window_end=DEFAULT_DURATION,
        tolerance=0.20,
        action_config={"main": 1.0, "side": 0.0},
        _action_pattern=_thrust_then_coast_pattern,
    ),
    "opposite_thrust": Maneuver(
        name="opposite_thrust",
        description="Opposite thrust -- full main for half, full side for half. Tests action switching.",
        n_steps=DEFAULT_DURATION,
        window_start=0,
        window_end=DEFAULT_DURATION,
        tolerance=0.20,
        action_config={"main": 1.0, "side": 1.0},
        _action_pattern=_opposite_thrust_pattern,
    ),
}

SECONDARY_MANEUVERS: dict[str, Maneuver] = {
    "hover": Maneuver(
        name="hover",
        description="Hover — tests force balance. Thrust that exactly balances gravity.",
        n_steps=DEFAULT_DURATION,
        window_start=0,
        window_end=DEFAULT_DURATION,
        tolerance=0.0,  # Uses absolute tolerance (near-zero target)
        absolute_tolerance=0.5,  # Absolute Δvy threshold in normalized units
        _compute_actions=_compute_hover_actions,  # Computes main thrust from physics config
    ),
    "conservation": Maneuver(
        name="conservation",
        description="Conservation — tests momentum. Zero thrust, check Δvx ≈ 0 (horizontal).",
        n_steps=DEFAULT_DURATION,
        window_start=0,
        window_end=DEFAULT_DURATION,
        tolerance=0.0,  # Uses absolute tolerance (near-zero target)
        absolute_tolerance=0.3,  # Absolute Δvx threshold in normalized units
        action_config={"main": 0.0, "side": 0.0},
    ),
    "angular_decay": Maneuver(
        name="angular_decay",
        description="Angular decay — tests damping. Zero thrust from spinning state.",
        n_steps=DEFAULT_DURATION,
        window_start=0,
        window_end=DEFAULT_DURATION,
        tolerance=0.10,  # 10% — leg joint motor torque offset
        action_config={"main": 0.0, "side": 0.0},
        _check_precondition=_needs_high_angular_vel,
    ),
}

# MANEUVERS = primary only (default test suite).
# ALL_MANEUVERS = both (for --maneuvers flag or explicit secondary use).
MANEUVERS: dict[str, Maneuver] = dict(PRIMARY_MANEUVERS)
ALL_MANEUVERS: dict[str, Maneuver] = {**PRIMARY_MANEUVERS, **SECONDARY_MANEUVERS}


def override_maneuver_duration(duration: int):
    """Override n_steps and window_end for all maneuvers (primary + secondary).

    Called from CLI when --duration flag is used. Updates each maneuver
    in-place: n_steps = duration, window_end = duration. Window_start
    is preserved (windowed maneuvers still skip their initial transients).
    """
    for maneuver in ALL_MANEUVERS.values():
        maneuver.n_steps = duration
        maneuver.window_end = duration


# --- Model adapters ---
# Architecture-specific context setup, architecture-agnostic prediction.

import torch


class PhysicsTestAdapter:
    """Base adapter for architecture-agnostic model prediction.

    Subclasses handle the architecture-specific part: building context
    from episode data. The predict() method is shared — it calls
    model.predict_sequence() with whatever context the subclass prepared.
    """

    def setup(
        self,
        model,
        episode: dict,
        branch_point: int,
    ) -> dict:
        """Prepare model context from episode data up to branch_point.

        Returns:
            context_kwargs: dict to pass as **kwargs to predict().
        """
        raise NotImplementedError

    def predict(
        self,
        model,
        branch_state: np.ndarray,
        actions: np.ndarray,
        **context_kwargs,
    ) -> np.ndarray:
        """Run model prediction for controlled action sequence.

        Args:
            model: WorldModel instance.
            branch_state: (state_dim,) state at branch point.
            actions: (n_steps, 2) controlled actions.
            **context_kwargs: From setup().

        Returns:
            predicted_states: (n_steps + 1, state_dim) including initial state.
        """
        raise NotImplementedError


class ContextMLPAdapter(PhysicsTestAdapter):
    """Adapter for ContextMLP: extracts K context transitions -> z.

    Context transitions are [s_t, a_t, s_{t+1}] tuples from the episode
    data before the branch point. The encoder processes these into a z
    vector that conditions the dynamics MLP.
    """

    def __init__(
        self,
        context_k: int,
        supervision: str = "labeled",
        prediction_target: str = "absolute",
    ):
        self.context_k = context_k
        self.supervision = supervision
        self.prediction_target = prediction_target

    def _apply_supervision(self, states: np.ndarray) -> np.ndarray:
        """Slice to kinematic dims for blind supervision.

        In 'blind' mode, the model only sees the first 8 state dims
        (kinematic: x, y, vx, vy, angle, angular_vel, left_leg, right_leg).
        In 'labeled' mode, all 15 dims are passed through.
        """
        if self.supervision == "blind":
            return states[:, :8].copy() if states.ndim == 2 else states[:8].copy()
        return states.copy()

    def setup(self, model, episode: dict, branch_point: int) -> dict:
        """Extract K context transitions from episode data before branch point.

        Context comes from the K transitions immediately before the branch
        point. Each transition is [s_t, a_t, s_{t+1}] concatenated.
        If fewer than K transitions are available (branch_point < K), the
        earliest available transition is repeated to pad up to K rows.
        """
        states = self._apply_supervision(episode["states"])
        actions = episode["actions"]

        # Extract K transitions ending at branch_point.
        # Transition t uses: states[t], actions[t], states[t+1].
        ctx_start = max(0, branch_point - self.context_k)
        ctx_rows = []
        for t in range(ctx_start, branch_point):
            row = np.concatenate([states[t], actions[t], states[t + 1]])
            ctx_rows.append(row)

        # Pad with copies of the first transition if not enough history.
        while len(ctx_rows) < self.context_k:
            ctx_rows.insert(0, ctx_rows[0].copy())

        context = np.stack(ctx_rows).astype(np.float32)
        context_tensor = torch.tensor(context).unsqueeze(0)  # (1, K, input_dim)
        return {"context": context_tensor}

    def predict(
        self,
        model,
        branch_state: np.ndarray,
        actions: np.ndarray,
        **context_kwargs,
    ) -> np.ndarray:
        """Run ContextMLP prediction with precomputed context.

        Handles both absolute and delta prediction targets. For delta mode,
        the model outputs delta-s and we accumulate: s_{t+1} = s_t + delta-s.
        For absolute mode, the model output IS the next state directly.

        The prediction loop is autoregressive: each predicted state becomes
        the input for the next step. This matches how the model would be
        used at inference time for multi-step rollouts.
        """
        model.eval()
        s = torch.tensor(branch_state, dtype=torch.float32).unsqueeze(0)
        actions_t = torch.tensor(actions, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            pred_list = [s.squeeze(0).numpy().copy()]
            s_current = s
            for t in range(len(actions)):
                # Forward pass: model predicts either next state (absolute)
                # or state delta (delta mode) conditioned on context.
                output = model(s_current, actions_t[:, t], **context_kwargs)
                if self.prediction_target == "delta":
                    # Delta mode: model output is the change in state.
                    # Accumulate: s_{t+1} = s_t + model_output.
                    s_current = s_current + output
                else:
                    # Absolute mode: model output IS the next state.
                    s_current = output
                pred_list.append(s_current.squeeze(0).numpy().copy())

        return np.stack(pred_list).astype(np.float32)


class GRUAdapter(PhysicsTestAdapter):
    """Adapter for GRUWorldModel: burn-in from episode history, then rollout.

    Setup runs the model through the episode data up to the branch point
    (teacher-forced), storing the resulting GRU hidden state. Predict
    then continues from that hidden state with autoregressive rollout
    using the controlled action sequence.

    This gives the GRU a warm start — it has seen the episode's dynamics
    leading up to the branch point, so its hidden state has accumulated
    evidence about the physics regime.
    """

    def __init__(self, supervision: str = "labeled",
                 input_mean: np.ndarray | None = None,
                 input_std: np.ndarray | None = None):
        self.supervision = supervision
        # Input normalization stats from lwp.training. When set, the adapter
        # normalizes states before feeding to the model and denormalizes
        # predictions back to raw space for comparison with GT.
        self.input_mean = input_mean
        self.input_std = input_std

    def _apply_supervision(self, states: np.ndarray) -> np.ndarray:
        """Slice to kinematic dims for blind supervision.

        In 'blind' mode, the model only sees the first 8 state dims
        (kinematic: x, y, vx, vy, angle, angular_vel, left_leg, right_leg).
        In 'labeled' mode, all 15 dims are passed through.
        """
        if self.supervision == "blind":
            return states[:, :8].copy() if states.ndim == 2 else states[:8].copy()
        return states.copy()

    def _normalize(self, s: torch.Tensor) -> torch.Tensor:
        """Normalize state tensor using training data stats."""
        if self.input_mean is not None:
            mean = torch.tensor(self.input_mean, dtype=s.dtype, device=s.device)
            std = torch.tensor(self.input_std, dtype=s.dtype, device=s.device)
            return (s - mean) / std
        return s

    def _denormalize(self, s: torch.Tensor) -> torch.Tensor:
        """Denormalize model output back to raw state space."""
        if self.input_mean is not None:
            mean = torch.tensor(self.input_mean, dtype=s.dtype, device=s.device)
            std = torch.tensor(self.input_std, dtype=s.dtype, device=s.device)
            return s * std + mean
        return s

    def setup(self, model, episode: dict, branch_point: int) -> dict:
        """Burn-in: run model through episode up to branch_point.

        Feeds the episode's true states and actions through the model
        in teacher-forced mode up to branch_point. The resulting GRU
        hidden state captures the model's understanding of the episode's
        dynamics at that point.

        Each step feeds (s_t, a_t) through the model to get the hidden
        state update. This is teacher-forced because we use the true
        episode states, not the model's predictions.

        Returns:
            {"hidden": (1, hidden_dim) tensor} — GRU hidden state after
            processing all timesteps up to branch_point.
        """
        states = self._apply_supervision(episode["states"])
        actions = episode["actions"]

        model.eval()
        with torch.no_grad():
            h = None  # Zero-init at start of sequence
            for t in range(branch_point):
                s_t = torch.tensor(states[t], dtype=torch.float32).unsqueeze(0)
                s_t = self._normalize(s_t)
                a_t = torch.tensor(actions[t], dtype=torch.float32).unsqueeze(0)
                _, h = model(s_t, a_t, hidden=h, return_hidden=True)

        return {"hidden": h}

    def predict(
        self,
        model,
        branch_state: np.ndarray,
        actions: np.ndarray,
        **context_kwargs,
    ) -> np.ndarray:
        """Autoregressive rollout from branch point with controlled actions.

        Each step: encode the current state (initially the true branch state,
        then the model's own prediction), run the GRU update with the hidden
        state from setup(), and decode to get the next predicted state.
        Predictions are fed back through the encoder (autoregressive).

        Args:
            model: GRUWorldModel instance.
            branch_state: (state_dim,) state at the branch point.
            actions: (n_steps, 2) controlled action sequence.
            **context_kwargs: Must contain "hidden" from setup().

        Returns:
            predicted_states: (n_steps + 1, state_dim) trajectory including
                the initial branch state as the first entry.
        """
        model.eval()
        h = context_kwargs.get("hidden")
        s = torch.tensor(branch_state, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            # First entry is the raw branch state (not a prediction).
            pred_list = [s.squeeze(0).numpy().copy()]
            # Normalize the initial state for the model.
            s = self._normalize(s)
            for t in range(len(actions)):
                a_t = torch.tensor(actions[t], dtype=torch.float32).unsqueeze(0)
                # Autoregressive: model outputs in normalized space,
                # which is fed back directly (model expects normalized).
                s, h = model(s, a_t, hidden=h, return_hidden=True)
                # Denormalize for the output trajectory (GT comparison
                # happens in raw space).
                pred_list.append(self._denormalize(s).squeeze(0).numpy().copy())

        return np.stack(pred_list).astype(np.float32)


class WMLadderAdapter(PhysicsTestAdapter):
    """Adapter for wm-ladder models (LinearModel, MLPModel, GRUModel, RSSMModel).

    wm-ladder models use:
        model.step(s_normalized, action, model_state) -> (delta_normalized, new_model_state)

    This adapter handles the normalize/denormalize boundary:
    - Normalizes raw states before model.step()
    - Denormalizes delta predictions before accumulation
    - For recurrent models (GRU, RSSM), does teacher-forced burn-in
      through episode history to warm up model_state

    The normalization math is done inline (not importing from wm-ladder's
    data.normalization) so this class can exist even when wm-ladder is not
    installed. Only the tests and CLI code path import from wm-ladder.

    Args:
        norm_stats: NormStats from training (state_mean/std, delta_mean/std).
        state_dim: Number of state dims the model sees (8 for blind, 15 for labeled).
        recurrent: Whether the model is recurrent (needs burn-in). Set True for GRU/RSSM.
    """

    def __init__(self, norm_stats, state_dim: int = 8, recurrent: bool = False,
                 subsample: int = 1):
        self.norm_stats = norm_stats
        self.state_dim = state_dim
        self.recurrent = recurrent
        self.subsample = subsample

    def _normalize_state(self, s: torch.Tensor) -> torch.Tensor:
        """Normalize a raw state tensor using training stats.

        Applies z-score normalization: (s - mean) / std, using the
        state_mean and state_std from the training NormStats.
        """
        ns = self.norm_stats
        mean = ns.state_mean.to(s.device)
        std = ns.state_std.to(s.device)
        return (s - mean) / std

    def _denormalize_delta(self, delta: torch.Tensor) -> torch.Tensor:
        """Denormalize a delta prediction back to raw space.

        Inverts z-score normalization: delta * std + mean, using the
        delta_mean and delta_std from the training NormStats.
        """
        ns = self.norm_stats
        mean = ns.delta_mean.to(delta.device)
        std = ns.delta_std.to(delta.device)
        return delta * std + mean

    def setup(self, model, episode: dict, branch_point: int) -> dict:
        """Burn-in for recurrent models; no-op for feedforward.

        For recurrent models: runs model.step() through episode data
        from t=0 to branch_point-1, accumulating model_state. Uses
        true states (teacher-forced) and true actions.

        For feedforward models (linear, mlp): returns model_state=None
        since these models don't have temporal state.

        Returns:
            {"model_state": model_state} -- None for feedforward models,
            GRU hidden / RSSMState for recurrent.
        """
        if not self.recurrent:
            return {"model_state": None}

        # Teacher-forced burn-in: feed true (state, action) pairs through
        # the model to build up the recurrent hidden state.
        states = episode["states"][:, :self.state_dim]
        actions = episode["actions"]

        # Subsample burn-in data to match model's training FPS.
        # Episode data is at 50 FPS; model trained at 50/subsample FPS.
        # Take every Nth state, average actions over each window.
        sub = self.subsample
        if sub > 1:
            n_burn = branch_point // sub
            burn_states = states[::sub][:n_burn + 1]
            burn_actions = np.zeros((n_burn, actions.shape[1]), dtype=np.float32)
            for j in range(n_burn):
                burn_actions[j] = actions[j * sub:(j + 1) * sub].mean(axis=0)
        else:
            n_burn = branch_point
            burn_states = states[:n_burn + 1]
            burn_actions = actions[:n_burn]

        model.eval()
        model_state = None
        with torch.no_grad():
            for t in range(n_burn):
                s_t = torch.tensor(burn_states[t], dtype=torch.float32).unsqueeze(0)
                s_n = self._normalize_state(s_t)
                a_t = torch.tensor(burn_actions[t], dtype=torch.float32).unsqueeze(0)
                # model.step returns (delta_normalized, new_model_state).
                # We discard the delta — only the model_state matters for burn-in.
                _, model_state = model.step(s_n, a_t, model_state)

        return {"model_state": model_state}

    def predict(
        self,
        model,
        branch_state: np.ndarray,
        actions: np.ndarray,
        **context_kwargs,
    ) -> np.ndarray:
        """Autoregressive rollout with normalize/denormalize at each step.

        Follows the same pattern as wm-ladder's _rollout_raw_space:
        1. Normalize raw state -> s_normalized
        2. model.step(s_normalized, action, model_state) -> (delta_normalized, new_model_state)
        3. Denormalize delta -> delta_raw
        4. s_raw = s_raw + delta_raw
        5. Repeat

        The model always works in normalized space, but we accumulate
        state updates in raw space so the output trajectory is directly
        comparable to Box2D ground truth.

        Returns:
            predicted_states: (n_steps + 1, state_dim) in raw space.
            First entry is the branch_state itself.
        """
        model.eval()
        model_state = context_kwargs.get("model_state")
        s_raw = torch.tensor(branch_state, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            pred_list = [branch_state.copy()]
            for t in range(len(actions)):
                # Normalize the current raw state for the model.
                s_n = self._normalize_state(s_raw)
                a_t = torch.tensor(actions[t], dtype=torch.float32).unsqueeze(0)
                # Model predicts a normalized delta and updates its internal state.
                delta_n, model_state = model.step(s_n, a_t, model_state)
                # Convert the delta back to raw space and accumulate.
                delta_raw = self._denormalize_delta(delta_n)
                s_raw = s_raw + delta_raw
                pred_list.append(s_raw.squeeze(0).numpy().copy())

        return np.stack(pred_list).astype(np.float32)


# --- Per-maneuver metric computation ---


def _detect_ground_contact(gt_states: np.ndarray) -> int | None:
    """Find the first timestep where the GT lander touches the ground.

    Ground contact is detected when either leg contact flag (dim 6 or 7)
    is active. Single-leg contact is sufficient — the lander is already
    in a landing/crash sequence where ground interaction dominates dynamics,
    not free flight.

    Args:
        gt_states: (T+1, state_dim) ground truth trajectory. Must have at
            least 8 dims (kinematic state with leg contacts).

    Returns:
        Step index of first ground contact, or None if no contact detected.
        Step 0 (initial state) is excluded from the search.
    """
    if gt_states.shape[1] < 8:
        return None

    # Check either leg contact from step 1 onward (step 0 is the branch state).
    left_leg = gt_states[1:, 6]
    right_leg = gt_states[1:, 7]
    any_contact = (left_leg >= 0.5) | (right_leg >= 0.5)

    if np.any(any_contact):
        # +1 because we searched from step 1
        return int(np.argmax(any_contact)) + 1
    return None


def compute_per_dim_trajectory_error(
    model_states: np.ndarray,
    gt_states: np.ndarray,
) -> np.ndarray:
    """Per-dim squared error at each timestep within a maneuver.

    Shows which dimensions diverge first and how error grows over time.
    Useful for diagnosing whether a model fails on position, velocity,
    or angle — and at what timestep the divergence begins.

    Args:
        model_states: (T+1, model_dim) predicted trajectory.
        gt_states: (T+1, gt_dim) ground truth trajectory.

    Returns:
        (T, min(model_dim, gt_dim)) per-step, per-dim squared error.
        Excludes step 0 (initial state, always matches).
    """
    n_dims = min(model_states.shape[1], gt_states.shape[1])
    # Skip step 0 (branch state, not a prediction).
    model = model_states[1:, :n_dims]
    gt = gt_states[1:, :n_dims]
    return (model - gt) ** 2


def compute_multi_horizon_error(
    model_states: np.ndarray,
    gt_states: np.ndarray,
    horizons: list[int] | None = None,
) -> dict:
    """Per-dim squared error at specific horizons within a maneuver.

    Samples error at a few timesteps for concise reporting (vs per_dim_trajectory_error
    which gives every step). Returns squared error per dim at each horizon —
    becomes MSE when averaged across episodes in run_physics_tests().

    Args:
        model_states: (T+1, model_dim) predicted trajectory.
        gt_states: (T+1, gt_dim) ground truth trajectory.
        horizons: List of timestep indices to measure. Defaults to
            [10, 30, 50, 100, 200, 300], filtered to those within T.

    Returns:
        Dict mapping horizon -> (n_dims,) per-dim squared error at that step.
    """
    if horizons is None:
        horizons = [10, 30, 50, 100, 200, 300]

    T = model_states.shape[0] - 1  # Exclude initial state
    n_dims = min(model_states.shape[1], gt_states.shape[1])

    result = {}
    for h in horizons:
        if h > T:
            continue
        diff = model_states[h, :n_dims] - gt_states[h, :n_dims]
        result[h] = (diff ** 2).astype(np.float32)

    return result


def compute_maneuver_metric(
    maneuver: Maneuver,
    model_states: np.ndarray,
    gt_states: np.ndarray,
) -> dict:
    """Compute the physics metric for a maneuver's predicted vs GT trajectory.

    Each maneuver tests a specific physical quantity. The metric extracts
    that quantity from both trajectories and computes the error.

    Metric definitions by maneuver:
      - free_fall: Δvy over measurement window (gravity effect)
      - full_thrust: Δvy over window (net acceleration from thrust + gravity)
      - side_thrust: Δvx over window (lateral acceleration)
      - angular_decay: ω decay ratio over window (damping)
      - hover: Δvy over full duration (should be ≈ 0)
      - conservation: Δvx over full duration (should be ≈ 0)
      - angle_thrust: ratio Δvx/Δvy vs GT ratio (thrust vectoring)

    Args:
        maneuver: Maneuver definition.
        model_states: (n_steps+1, model_state_dim) predicted trajectory.
        gt_states: (n_steps+1, 15) ground truth trajectory.

    Returns:
        Dict with: maneuver, measured, gt, relative_error (or absolute_error),
        passed, window_start, window_end.
    """
    ws = maneuver.window_start
    we = maneuver.window_end
    name = maneuver.name

    if name == "free_fall":
        # Measure Δvy over measurement window.
        # vy is dim 3 in both model (blind: 8D) and GT (15D).
        model_delta_vy = float(model_states[we, 3] - model_states[ws, 3])
        gt_delta_vy = float(gt_states[we, 3] - gt_states[ws, 3])
        return _relative_metric(maneuver, model_delta_vy, gt_delta_vy, "delta_vy")

    elif name == "full_thrust":
        # Measure Δvy in windowed region (skip transients, avoid angle drift).
        model_delta_vy = float(model_states[we, 3] - model_states[ws, 3])
        gt_delta_vy = float(gt_states[we, 3] - gt_states[ws, 3])
        return _relative_metric(maneuver, model_delta_vy, gt_delta_vy, "delta_vy")

    elif name == "side_thrust":
        # Measure Δvx in windowed region.
        # vx is dim 2.
        model_delta_vx = float(model_states[we, 2] - model_states[ws, 2])
        gt_delta_vx = float(gt_states[we, 2] - gt_states[ws, 2])
        return _relative_metric(maneuver, model_delta_vx, gt_delta_vx, "delta_vx")

    elif name == "angular_decay":
        # Measure angular velocity decay ratio.
        # angular_vel is dim 5.
        model_omega_start = float(model_states[ws, 5])
        model_omega_end = float(model_states[we, 5])
        gt_omega_start = float(gt_states[ws, 5])
        gt_omega_end = float(gt_states[we, 5])

        # Compare decay ratio: omega_end / omega_start.
        if abs(gt_omega_start) > 1e-6 and abs(model_omega_start) > 1e-6:
            model_ratio = model_omega_end / model_omega_start
            gt_ratio = gt_omega_end / gt_omega_start
            return _relative_metric(maneuver, model_ratio, gt_ratio, "decay_ratio")
        else:
            return {
                "maneuver": name,
                "measured": 0.0,
                "gt": 0.0,
                "relative_error": 0.0,
                "passed": True,
                "note": "angular_vel too small to measure decay",
                "window_start": ws,
                "window_end": we,
            }

    elif name == "hover":
        # Measure Δvy over full window — should be near zero.
        model_delta_vy = float(model_states[we, 3] - model_states[ws, 3])
        gt_delta_vy = float(gt_states[we, 3] - gt_states[ws, 3])
        return _absolute_metric(maneuver, model_delta_vy, gt_delta_vy, "delta_vy")

    elif name == "conservation":
        # Measure Δvx over full window — should be near zero.
        model_delta_vx = float(model_states[we, 2] - model_states[ws, 2])
        gt_delta_vx = float(gt_states[we, 2] - gt_states[ws, 2])
        return _absolute_metric(maneuver, model_delta_vx, gt_delta_vx, "delta_vx")

    elif name == "angle_thrust":
        # Measure ratio Δvx/Δvy — thrust vectoring.
        model_dvx = float(model_states[we, 2] - model_states[ws, 2])
        model_dvy = float(model_states[we, 3] - model_states[ws, 3])
        gt_dvx = float(gt_states[we, 2] - gt_states[ws, 2])
        gt_dvy = float(gt_states[we, 3] - gt_states[ws, 3])

        if abs(gt_dvy) > 1e-6 and abs(model_dvy) > 1e-6:
            model_ratio = model_dvx / model_dvy
            gt_ratio = gt_dvx / gt_dvy
            return _relative_metric(maneuver, model_ratio, gt_ratio, "vx_vy_ratio")
        else:
            # Fall back to comparing Δvx directly.
            return _relative_metric(maneuver, model_dvx, gt_dvx, "delta_vx_fallback")

    elif name in ("impulse_thrust", "ramp_thrust", "thrust_then_coast", "opposite_thrust"):
        # Trajectory-level metric: per-dim MSE across the measurement window.
        # These maneuvers test temporal dynamics -- we compare the full
        # trajectory shape, not a single endpoint quantity.
        # Use kinematic dims [0:6] (x, y, vx, vy, angle, angular_vel).
        model_window = model_states[ws:we, :6]
        gt_window = gt_states[ws:we, :6]
        per_dim_mse = np.mean((model_window - gt_window) ** 2, axis=0)
        trajectory_mse = float(np.mean(per_dim_mse))
        return {
            "maneuver": name,
            "quantity": "trajectory_mse_6d",
            "measured": trajectory_mse,
            "gt": 0.0,
            "absolute_error": trajectory_mse,
            "passed": trajectory_mse < maneuver.tolerance,
            "per_dim_mse": per_dim_mse.tolist(),
            "window_start": ws,
            "window_end": we,
        }

    raise ValueError(f"Unknown maneuver: {name}")


def _relative_metric(
    maneuver: Maneuver,
    measured: float,
    gt: float,
    quantity_name: str,
) -> dict:
    """Compute relative error metric.

    Relative error = |measured - gt| / |gt|.
    Falls back to absolute difference when GT is near zero.
    Pass/fail is determined by the maneuver's tolerance threshold.
    """
    if abs(gt) > 1e-6:
        rel_error = abs(measured - gt) / abs(gt)
    else:
        # GT is near zero — use absolute difference to avoid division issues.
        rel_error = abs(measured - gt)
    passed = rel_error < maneuver.tolerance
    return {
        "maneuver": maneuver.name,
        "quantity": quantity_name,
        "measured": measured,
        "gt": gt,
        "relative_error": rel_error,
        "passed": passed,
        "window_start": maneuver.window_start,
        "window_end": maneuver.window_end,
    }


def _absolute_metric(
    maneuver: Maneuver,
    measured: float,
    gt: float,
    quantity_name: str,
) -> dict:
    """Compute absolute error metric (for near-zero targets like hover/conservation).

    Absolute error = |measured - gt|.
    Uses maneuver.absolute_tolerance for pass/fail (defaults to 0.5 if not set).
    """
    abs_error = abs(measured - gt)
    tol = maneuver.absolute_tolerance or 0.5
    passed = abs_error < tol
    return {
        "maneuver": maneuver.name,
        "quantity": quantity_name,
        "measured": measured,
        "gt": gt,
        "absolute_error": abs_error,
        "passed": passed,
        "window_start": maneuver.window_start,
        "window_end": maneuver.window_end,
    }


# --- Test runner ---

def run_maneuver_test(
    model,
    episode: dict,
    maneuver: Maneuver,
    adapter: PhysicsTestAdapter,
    branch_point: int,
    gt_mode: str = "replay",
    physics_config: LunarLanderPhysicsConfig | None = None,
    save_frames: bool = False,
    subsample: int = 1,
) -> dict:
    """Run a single maneuver test on one episode.

    End-to-end pipeline for one maneuver × one episode:
      1. Generate controlled action sequence from maneuver definition.
      2. Setup model context from episode data up to branch_point.
      3. Run model prediction autoregressively from the branch-point state.
      4. Generate Box2D ground truth for the same actions from the same state.
      5. Compute the per-maneuver metric (relative or absolute error).

    Args:
        model: WorldModel instance (e.g., ContextMLP).
        episode: Dict with "states" (T+1, 15) and "actions" (T, 2).
        maneuver: Maneuver definition (from MANEUVERS dict).
        adapter: PhysicsTestAdapter for this model's architecture.
        branch_point: Timestep to branch at (0-indexed into episode).
        gt_mode: "teleport" or "replay" for Box2D GT generation.
        physics_config: Override physics config (else extracted from episode).
        save_frames: If True, capture RGB frames for GT and model trajectories.
        subsample: Number of Box2D steps per model step. subsample=5 means
            the model operates at 10 FPS (50/5). GT steps Box2D 5x per action.

    Returns:
        Result dict with: maneuver name, metric data (relative_error or
        absolute_error, passed, measured, gt), model_states, gt_states,
        and optionally gt_frames.
    """
    from lwp.wm.physics_test_gt import generate_gt_trajectory

    # Step 1: Generate controlled actions from maneuver definition.
    # For maneuvers like hover, this needs the physics config to compute
    # the thrust value that balances gravity.
    if physics_config is None:
        from lwp.wm.physics_test_gt import _extract_physics_config
        physics_config = _extract_physics_config(episode)
    controlled_actions = maneuver.make_actions(physics_config=physics_config)

    # Subsample actions to match model FPS. Maneuvers generate actions at
    # 50 FPS; with subsample=5 the model operates at 10 FPS. For constant-
    # action maneuvers this just shortens the array. For time-varying
    # patterns (impulse, ramp), we take every Nth action so the pattern
    # plays out over the same wall-clock duration.
    if subsample > 1:
        controlled_actions = controlled_actions[::subsample]
        # Scale maneuver window indices to model FPS for metric computation.
        maneuver = Maneuver(
            name=maneuver.name,
            description=maneuver.description,
            n_steps=maneuver.n_steps // subsample,
            window_start=maneuver.window_start // subsample,
            window_end=maneuver.window_end // subsample,
            tolerance=maneuver.tolerance,
            absolute_tolerance=maneuver.absolute_tolerance,
            action_config=maneuver.action_config,
        )

    # Step 2: Setup model context (architecture-specific).
    # For ContextMLP, this extracts K context transitions and encodes them.
    context_kwargs = adapter.setup(model, episode, branch_point)

    # Step 3: Get the branch-point state for the model.
    # WMLadderAdapter has an explicit state_dim (e.g. 8 for kinematic-only).
    # Old adapters use supervision: blind=8 dims, labeled=all dims.
    adapter_state_dim = getattr(adapter, "state_dim", None)
    if adapter_state_dim is not None:
        branch_state = episode["states"][branch_point, :adapter_state_dim]
    else:
        supervision = getattr(adapter, "supervision", "labeled")
        if supervision == "blind":
            branch_state = episode["states"][branch_point, :8]
        else:
            branch_state = episode["states"][branch_point]

    # Step 4: Run model prediction — autoregressive rollout from branch state.
    model_states = adapter.predict(
        model, branch_state, controlled_actions, **context_kwargs,
    )

    # Step 5: Generate Box2D ground truth for the same branch point + actions.
    gt_result = generate_gt_trajectory(
        episode=episode,
        branch_point=branch_point,
        controlled_actions=controlled_actions,
        mode=gt_mode,
        save_frames=save_frames,
        subsample=subsample,
    )
    gt_states = gt_result["states"]

    # Step 6: Truncate at GT termination.
    # The GT env may terminate mid-maneuver (landing, crash, out of bounds).
    # After termination, GT state is frozen and model predictions diverge into
    # meaningless territory. Use the env's termination signal — it covers all
    # terminal conditions, not just leg contact.
    # Also check leg contact as a fallback (catches soft landings where env
    # may not terminate immediately but dynamics are ground-dominated).
    terminated_at = gt_result.get("terminated_at")
    contact_step = _detect_ground_contact(gt_states)

    # Use whichever comes first: env termination or leg contact.
    truncate_at = None
    truncate_reason = None
    if terminated_at is not None and contact_step is not None:
        truncate_at = min(terminated_at, contact_step)
        truncate_reason = "terminated" if terminated_at <= contact_step else "contact"
    elif terminated_at is not None:
        truncate_at = terminated_at
        truncate_reason = "terminated"
    elif contact_step is not None:
        truncate_at = contact_step
        truncate_reason = "contact"

    original_n_steps = maneuver.n_steps

    # Minimum number of in-flight steps after window_start to produce a
    # meaningful metric. If the pre-termination window is too short, the
    # measurement is noise — skip the episode entirely.
    MIN_PRE_CONTACT_STEPS = 5

    if truncate_at is not None:
        effective_end = min(maneuver.window_end, truncate_at)

        # Not enough in-flight dynamics to measure — skip.
        if effective_end <= maneuver.window_start + MIN_PRE_CONTACT_STEPS:
            return {
                "maneuver": maneuver.name,
                "skipped": True,
                "skip_reason": (
                    f"GT {truncate_reason} at step {truncate_at}, "
                    f"insufficient pre-termination window "
                    f"({effective_end - maneuver.window_start} steps, "
                    f"need {MIN_PRE_CONTACT_STEPS})"
                ),
                "passed": False,
                "branch_point": branch_point,
                "gt_mode": gt_mode,
                "truncate_at": truncate_at,
                "truncate_reason": truncate_reason,
            }

        # Truncate trajectories at termination/contact step.
        model_states = model_states[:truncate_at + 1]
        gt_states = gt_states[:truncate_at + 1]

    # Build a truncated maneuver copy if the measurement window needs adjusting.
    # The window_end may extend past the truncated trajectory.
    effective_maneuver = maneuver
    if truncate_at is not None and truncate_at < maneuver.window_end:
        effective_maneuver = Maneuver(
            name=maneuver.name,
            description=maneuver.description,
            n_steps=truncate_at,
            window_start=maneuver.window_start,
            window_end=truncate_at,
            tolerance=maneuver.tolerance,
            absolute_tolerance=maneuver.absolute_tolerance,
            action_config=maneuver.action_config,
        )

    # Step 7: Compute the per-maneuver metric (model vs GT).
    metric = compute_maneuver_metric(effective_maneuver, model_states, gt_states)
    metric["model_states"] = model_states
    metric["gt_states"] = gt_states
    metric["controlled_actions"] = controlled_actions
    metric["branch_point"] = branch_point
    metric["gt_mode"] = gt_mode

    # Record truncation info for diagnostics.
    if truncate_at is not None:
        metric["truncated"] = True
        metric["truncate_at"] = truncate_at
        metric["truncate_reason"] = truncate_reason
        metric["original_n_steps"] = original_n_steps

    # Per-dim trajectory error for diagnostics.
    # (T, n_dims) squared error at each timestep — shows which dims diverge first.
    metric["per_dim_trajectory_error"] = compute_per_dim_trajectory_error(
        model_states, gt_states,
    )

    # Multi-horizon error at standard checkpoints within the maneuver.
    metric["multi_horizon_error"] = compute_multi_horizon_error(
        model_states, gt_states,
    )

    # Attach RGB frames if captured (for visualization / debugging).
    if "rgb_frames" in gt_result:
        metric["gt_frames"] = gt_result["rgb_frames"]

    return metric


def run_physics_tests(
    model,
    episodes: list[dict],
    adapter: PhysicsTestAdapter,
    maneuver_names: list[str] | None = None,
    gt_mode: str = "replay",
    branch_point_offset: int | None = None,
    subsample: int = 1,
) -> dict:
    """Run all physics tests across episodes.

    Outer loop over maneuvers, inner loop over episodes. For each combination:
      1. Choose branch point (default: right after context window).
      2. Check precondition at branch point — skip if not met, try later points.
      3. Run the maneuver test via run_maneuver_test().
      4. Collect per-episode results and aggregate statistics.

    Args:
        model: WorldModel instance.
        episodes: List of episode dicts (each with "states" and "actions").
        adapter: PhysicsTestAdapter for this model's architecture.
        maneuver_names: Which maneuvers to run (default: all from MANEUVERS).
        gt_mode: "teleport" or "replay" for Box2D GT generation.
        branch_point_offset: Fixed branch point. None = use context_k from adapter.
        subsample: Number of Box2D steps per model step (1 = 50 FPS, 5 = 10 FPS).

    Returns:
        Dict keyed by maneuver name, each containing:
          - n_tested: Number of episodes tested.
          - n_passed: Number of episodes that passed the tolerance.
          - pass_rate: n_passed / n_tested.
          - mean_error: Mean relative/absolute error across episodes.
          - max_error: Maximum error across episodes.
          - per_episode: List of per-episode result summaries.
    """
    if maneuver_names is None:
        maneuver_names = list(MANEUVERS.keys())

    # Determine default branch point from adapter's context window.
    # The model needs at least context_k transitions before branching.
    context_k = getattr(adapter, "context_k", 0) or 0
    default_branch = branch_point_offset if branch_point_offset is not None else context_k

    import time as _time

    results = {}
    n_maneuvers = len(maneuver_names)
    for m_idx, m_name in enumerate(maneuver_names):
        maneuver = ALL_MANEUVERS[m_name]
        per_episode = []
        m_start = _time.perf_counter()
        n_skipped = 0
        n_errors = 0

        effective_steps = maneuver.n_steps // subsample if subsample > 1 else maneuver.n_steps
        print(f"  [{m_idx+1}/{n_maneuvers}] {m_name} ({effective_steps} steps)...",
              flush=True)

        for ep_idx, episode in enumerate(episodes):
            T = len(episode["actions"])
            branch = max(default_branch, 1)  # Need at least 1 step of history

            # Check we have enough steps after branch for this maneuver.
            # Skip for reset episodes — they're synthetic, and actual duration
            # is determined by GT termination/truncation, not episode length.
            if not episode.get("is_reset_episode") and branch + maneuver.n_steps > T:
                n_skipped += 1
                continue

            # Check precondition at branch point. If it fails, search forward
            # up to 100 steps for a suitable branch point. Some maneuvers
            # (angular_decay, angle_thrust) need specific state conditions.
            if not maneuver.check_precondition(episode["states"][branch]):
                found = False
                for bp in range(branch, min(T - maneuver.n_steps, branch + 100)):
                    if maneuver.check_precondition(episode["states"][bp]):
                        branch = bp
                        found = True
                        break
                if not found:
                    n_skipped += 1
                    continue

            try:
                ep_start = _time.perf_counter()
                result = run_maneuver_test(
                    model=model,
                    episode=episode,
                    maneuver=maneuver,
                    adapter=adapter,
                    branch_point=branch,
                    gt_mode=gt_mode,
                    subsample=subsample,
                )
                ep_elapsed = _time.perf_counter() - ep_start
                result["episode_idx"] = ep_idx

                # Handle episodes skipped due to early ground contact.
                if result.get("skipped"):
                    n_skipped += 1
                    reason = result.get("skip_reason", "ground contact")
                    print(f"    ep {ep_idx:3d}: SKIP  {reason}  "
                          f"({ep_elapsed:.1f}s)", flush=True)
                    continue

                passed_str = "PASS" if result.get("passed") else "FAIL"
                err_val = result.get("relative_error", result.get("absolute_error", 0))
                trunc_tag = ""
                if result.get("truncated"):
                    trunc_tag = (f"  [truncated@{result['truncate_at']}"
                                 f":{result['truncate_reason']}]")
                print(f"    ep {ep_idx:3d}: {passed_str}  err={err_val:.4f}  "
                      f"({ep_elapsed:.1f}s){trunc_tag}", flush=True)
                # Don't store full state/frame arrays in the aggregate —
                # they're large and the summary stats are what we need.
                result_summary = {
                    k: v for k, v in result.items()
                    if k not in ("model_states", "gt_states", "controlled_actions",
                                 "gt_frames", "model_frames",
                                 "per_dim_trajectory_error", "multi_horizon_error")
                }
                per_episode.append(result_summary)
            except Exception as e:
                n_errors += 1
                print(f"    ep {ep_idx:3d}: ERROR  {e}", flush=True)
                per_episode.append({
                    "episode_idx": ep_idx,
                    "maneuver": m_name,
                    "error": str(e),
                    "passed": False,
                })

        m_elapsed = _time.perf_counter() - m_start
        n_tested = len(per_episode)
        n_passed = sum(1 for r in per_episode if r.get("passed", False))
        print(f"  [{m_idx+1}/{n_maneuvers}] {m_name}: "
              f"{n_passed}/{n_tested} passed, {n_skipped} skipped, "
              f"{n_errors} errors  ({m_elapsed:.1f}s)", flush=True)

        # Collect error values from successful results (skip exception entries).
        # Exception entries have an "error" key (string message), not
        # "relative_error" or "absolute_error" (floats).
        errors = [
            r.get("relative_error", r.get("absolute_error", 0.0))
            for r in per_episode
            if "relative_error" in r or "absolute_error" in r
        ]

        results[m_name] = {
            "n_tested": n_tested,
            "n_passed": n_passed,
            "pass_rate": n_passed / n_tested if n_tested > 0 else 0.0,
            "mean_error": float(np.mean(errors)) if errors else 0.0,
            "max_error": float(np.max(errors)) if errors else 0.0,
            "per_episode": per_episode,
        }

    return results
