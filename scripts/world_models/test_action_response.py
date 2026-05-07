#!/usr/bin/env python3
"""Action-response direction test for E2 world models.

For each model, predicts next state under 4 test actions
(zero, main_thrust, side_left, side_right) from the same starting state.
Measures direction accuracy:
  - Main thrust → vy should increase (vs zero action)
  - Zero action → vy should decrease vs current (gravity)
  - Side-right thrust → ang_vel should increase (vs zero)
  - Side-left thrust → ang_vel should decrease (vs zero)
  - Left/right should produce opposite ang_vel changes

Supports two backends:
  --ladder-checkpoint         state-space (linear/mlp/gru/rssm)
  --vae-checkpoint + --dynamics-checkpoint   pixel WM (vae + dynamics)

Output: console table + JSON at --output-json.

Usage (state-space):
  python scripts/world_models/test_action_response.py \
      --ladder-checkpoint /path/to/best.pt \
      --data-dir /path/to/world_model_data/gym-default \
      --output-json /path/to/action_response.json

Usage (pixel):
  python scripts/world_models/test_action_response.py \
      --vae-checkpoint /path/to/vae/best.pt \
      --dynamics-checkpoint /path/to/dynamics/best.pt \
      --data-path /path/to/episodes_with_rgb_frames \
      --output-json /path/to/action_response.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

KIN_NAMES = ["x", "y", "vx", "vy", "angle", "ang_vel"]
VX, VY, ANGLE, ANGULAR_VEL = 2, 3, 4, 5

TEST_ACTIONS = {
    "zero":       np.array([0.0, 0.0], dtype=np.float32),
    "main":       np.array([1.0, 0.0], dtype=np.float32),
    "side_left":  np.array([0.0, -1.0], dtype=np.float32),
    "side_right": np.array([0.0, +1.0], dtype=np.float32),
}

# Sustained rollout scenarios: constant action held for N steps,
# plus impulse (action at step 0 only, then zero).
def sustained_scenarios(n_steps):
    """Return dict of name -> (n_steps, 2) action sequence."""
    zero = np.array([0.0, 0.0], dtype=np.float32)
    out = {name: np.tile(act, (n_steps, 1)) for name, act in TEST_ACTIONS.items()}
    imp = np.tile(zero, (n_steps, 1))
    imp[0] = TEST_ACTIONS["main"]
    out["impulse_main"] = imp
    return out


def direction_accuracy(a, b, dim, sign):
    d = a[:, dim] - b[:, dim]
    return float((d > 0).mean()) if sign > 0 else float((d < 0).mean())


def mean_diff(a, b, dim):
    return float((a[:, dim] - b[:, dim]).mean())


def _clone_state(ms):
    if ms is None:
        return None
    if isinstance(ms, torch.Tensor):
        return ms.clone()
    if isinstance(ms, (tuple, list)):
        return type(ms)(_clone_state(x) for x in ms)
    if isinstance(ms, dict):
        return {k: _clone_state(v) for k, v in ms.items()}
    return ms


# =====================================================================
# State-space backend
# =====================================================================

def run_state_space(args):
    from lwp.utils.checkpoint import load_checkpoint
    from lwp.models.factory import build_model
    from lwp.data.normalization import NormStats
    from lwp.wm.diagnostics import _load_episodes_from_dir
    from lwp.training.integration import hybrid_state_update

    device = torch.device(args.device)
    ckpt = load_checkpoint(args.ladder_checkpoint)
    cfg = ckpt["config"]
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    norm_stats = NormStats.from_dict(ckpt["norm_stats"])
    state_mean = norm_stats.state_mean.to(device)
    state_std = norm_stats.state_std.to(device)
    delta_mean = norm_stats.delta_mean.to(device)
    delta_std = norm_stats.delta_std.to(device)
    subsample = getattr(cfg, "subsample", 1)
    arch = cfg.arch
    recurrent = arch in ("gru", "rssm")
    print(f"  model: arch={arch}, recurrent={recurrent}, state_dim={cfg.state_dim}, "
          f"subsample={subsample}")

    episodes = _load_episodes_from_dir(Path(args.data_dir), args.n_episodes)
    print(f"  loaded {len(episodes)} episodes from {args.data_dir}")

    def rollout(start_state_np, ms_init, actions_seq):
        """Autoregressive N-step rollout. Returns (n_steps, 6) numpy decoded states."""
        s = torch.as_tensor(start_state_np, dtype=torch.float32,
                            device=device).unsqueeze(0)
        ms = _clone_state(ms_init)
        out = []
        for i in range(len(actions_seq)):
            s_norm = (s - state_mean) / state_std
            a = torch.as_tensor(actions_seq[i], dtype=torch.float32,
                                device=device).unsqueeze(0)
            delta_norm, ms = model.step(s_norm, a, ms)
            delta_raw = delta_norm * delta_std + delta_mean
            s = hybrid_state_update(s, delta_raw, subsample=subsample)
            out.append(s.squeeze(0).cpu().numpy().astype(np.float32))
        return np.stack(out)

    current_kin = []
    scenarios = sustained_scenarios(args.sustained_steps)
    rollout_kin = {name: [] for name in scenarios}

    with torch.no_grad():
        for ep in episodes:
            states = ep["states"][:, :6]
            actions = ep["actions"]
            n_t = min(len(states) - 1, len(actions))
            if n_t <= args.warmup + args.sustained_steps + 1:
                continue

            model_state = None
            if recurrent:
                for t in range(args.warmup):
                    s = torch.as_tensor(states[t], dtype=torch.float32,
                                        device=device).unsqueeze(0)
                    a = torch.as_tensor(actions[t], dtype=torch.float32,
                                        device=device).unsqueeze(0)
                    s_norm = (s - state_mean) / state_std
                    _, model_state = model.step(s_norm, a, model_state)

            t_start = args.warmup if recurrent else 3
            step = max(1, (n_t - t_start) // max(1, args.per_episode))
            test_ts = list(range(t_start, n_t - args.sustained_steps, step))[:args.per_episode]

            for t in test_ts:
                current_kin.append(states[t].astype(np.float32).copy())
                for name, act_seq in scenarios.items():
                    traj = rollout(states[t], model_state, act_seq)
                    rollout_kin[name].append(traj)

                if recurrent:
                    s = torch.as_tensor(states[t], dtype=torch.float32,
                                        device=device).unsqueeze(0)
                    s_norm = (s - state_mean) / state_std
                    a_gt = torch.as_tensor(actions[t], dtype=torch.float32,
                                           device=device).unsqueeze(0)
                    _, model_state = model.step(s_norm, a_gt, model_state)

            if len(current_kin) >= args.n_transitions:
                break

    N = min(len(current_kin), args.n_transitions)
    current_kin = np.array(current_kin[:N], dtype=np.float32)
    rollout_kin = {k: np.stack(v[:N]).astype(np.float32) for k, v in rollout_kin.items()}
    # Single-step (step 1) view over the 4 main actions.
    decoded = {k: rollout_kin[k][:, 0, :] for k in TEST_ACTIONS}
    meta = {"arch": arch, "recurrent": recurrent, "subsample": subsample,
            "backend": "state-space", "sustained_steps": args.sustained_steps}
    return current_kin, decoded, rollout_kin, meta


# =====================================================================
# Pixel backend
# =====================================================================

def run_pixel(args):
    from lwp.wm.pixel_physics_eval import load_eval_episodes
    from scripts.world_models.eval_pixel_wm_physics import load_vae, load_dynamics

    device = torch.device(args.device)
    vae, vae_cfg = load_vae(args.vae_checkpoint, args.device)
    dynamics = load_dynamics(args.dynamics_checkpoint, vae_cfg["latent_dim"], args.device)
    frame_size = vae_cfg["frame_size"]

    # Collect episode paths — flat search under data_path.
    import os
    paths = []
    for root, _, files in os.walk(args.data_path, followlinks=True):
        if "cache" in root or "prepared" in root:
            continue
        for fn in sorted(files):
            if fn.endswith(".npz") and fn.startswith("episode_"):
                paths.append(os.path.join(root, fn))
        if len(paths) >= args.n_episodes:
            break
    paths = paths[:args.n_episodes]
    if not paths:
        raise RuntimeError(f"No episodes found under {args.data_path}")
    print(f"  loaded {len(paths)} episode paths from {args.data_path}")

    episodes = load_eval_episodes(paths, frame_size=frame_size)

    def rollout(z_start, actions_seq):
        """Autoregressive N-step rollout. Returns (n_steps, 6) decoded kinematics."""
        z = z_start
        hidden = None
        out = []
        for i in range(len(actions_seq)):
            a = torch.from_numpy(actions_seq[i]).unsqueeze(0).to(device)
            z, hidden = dynamics.forward(z, a, hidden)
            kin = vae.predict_state(z)
            assert kin is not None
            out.append(kin.cpu().numpy()[0].astype(np.float32))
        return np.stack(out)

    scenarios = sustained_scenarios(args.sustained_steps)
    current_kin = []
    rollout_kin = {name: [] for name in scenarios}

    with torch.no_grad():
        for ep in episodes:
            frames = ep["frames"].to(device)
            T = ep["actions"].shape[0]
            z_all = vae.encode(frames)

            step = max(1, T // max(1, args.per_episode))
            test_ts = list(range(3, T, step))[:args.per_episode]

            for t in test_ts:
                z_t = z_all[t:t+1]
                kin_t = vae.predict_state(z_t)
                assert kin_t is not None, "VAE has no state head"
                current_kin.append(kin_t.cpu().numpy()[0].astype(np.float32))
                for name, act_seq in scenarios.items():
                    traj = rollout(z_t, act_seq)
                    rollout_kin[name].append(traj)

            if len(current_kin) >= args.n_transitions:
                break

    N = min(len(current_kin), args.n_transitions)
    current_kin = np.array(current_kin[:N], dtype=np.float32)
    rollout_kin = {k: np.stack(v[:N]).astype(np.float32) for k, v in rollout_kin.items()}
    decoded = {k: rollout_kin[k][:, 0, :] for k in TEST_ACTIONS}
    meta = {"backend": "pixel", "latent_dim": vae_cfg["latent_dim"],
            "frame_size": frame_size, "sustained_steps": args.sustained_steps}
    return current_kin, decoded, rollout_kin, meta


# =====================================================================
# Reporting
# =====================================================================

def compute_tests(current_kin, decoded):
    N = len(current_kin)
    zero, main, left, right = (decoded["zero"], decoded["main"],
                               decoded["side_left"], decoded["side_right"])

    tests = {
        "main_thrust_vy_up": {
            "description": "main thrust → vy > zero action (thrust accelerates up)",
            "accuracy": direction_accuracy(main, zero, VY, +1),
            "mean_diff": mean_diff(main, zero, VY),
        },
        "gravity_vy_down": {
            "description": "zero action → vy < current (gravity pulls down)",
            "accuracy": direction_accuracy(zero, current_kin, VY, -1),
            "mean_diff": mean_diff(zero, current_kin, VY),
        },
        "side_right_angvel_up": {
            "description": "side-right thrust → ang_vel > zero",
            "accuracy": direction_accuracy(right, zero, ANGULAR_VEL, +1),
            "mean_diff": mean_diff(right, zero, ANGULAR_VEL),
        },
        "side_left_angvel_down": {
            "description": "side-left thrust → ang_vel < zero",
            "accuracy": direction_accuracy(left, zero, ANGULAR_VEL, -1),
            "mean_diff": mean_diff(left, zero, ANGULAR_VEL),
        },
    }
    # Symmetry: left and right produce opposite-sign ang_vel changes.
    dr = right[:, ANGULAR_VEL] - zero[:, ANGULAR_VEL]
    dl = left[:, ANGULAR_VEL] - zero[:, ANGULAR_VEL]
    opposite = ((dr > 0) & (dl < 0)) | ((dr < 0) & (dl > 0))
    tests["side_symmetry"] = {
        "description": "sign(Δang_vel|right) ≠ sign(Δang_vel|left)",
        "accuracy": float(opposite.mean()),
        "mean_diff": float((dr * dl).mean()),  # negative if opposite
    }
    return tests, N


def compute_rollout_tests(rollout_kin):
    """Per-step direction accuracy across sustained rollouts."""
    zero = rollout_kin["zero"]    # (N, n_steps, 6)
    main = rollout_kin["main"]
    right = rollout_kin["side_right"]
    left = rollout_kin["side_left"]
    impulse = rollout_kin["impulse_main"]
    n_steps = zero.shape[1]

    per_step = []
    for s in range(n_steps):
        row = {
            "step": s + 1,
            "main_vy_up": direction_accuracy(main[:, s], zero[:, s], VY, +1),
            "main_vy_mean_diff": mean_diff(main[:, s], zero[:, s], VY),
            "right_av_up": direction_accuracy(right[:, s], zero[:, s], ANGULAR_VEL, +1),
            "left_av_down": direction_accuracy(left[:, s], zero[:, s], ANGULAR_VEL, -1),
            "impulse_vy_up": direction_accuracy(impulse[:, s], zero[:, s], VY, +1),
            "impulse_vy_mean_diff": mean_diff(impulse[:, s], zero[:, s], VY),
        }
        per_step.append(row)
    return per_step


def print_rollout_table(per_step):
    print("\n" + "=" * 72)
    print("SUSTAINED & IMPULSE ROLLOUT TESTS")
    print("=" * 72)
    print("Main thrust sustained → vy should increase at every step.")
    print("Impulse main (step 0 only) — if no step-1 effect but step-2+ effect,")
    print("the predictor has an action delay.\n")
    print(f"  {'step':>4s}  {'main→vy':>10s}  {'Δvy':>10s}   "
          f"{'right→av':>10s}  {'left→-av':>10s}   "
          f"{'imp→vy':>10s}  {'impΔvy':>10s}")
    print("  " + "-" * 76)
    for r in per_step:
        print(f"  {r['step']:>4d}  {r['main_vy_up']:>9.1%}  {r['main_vy_mean_diff']:>+10.5f}   "
              f"{r['right_av_up']:>9.1%}  {r['left_av_down']:>9.1%}   "
              f"{r['impulse_vy_up']:>9.1%}  {r['impulse_vy_mean_diff']:>+10.5f}")


def print_table(tests, n, current_kin, decoded, run_name=""):
    print("\n" + "=" * 72)
    print(f"ACTION-RESPONSE DIRECTION TEST: {run_name}")
    print("=" * 72)
    print(f"N transitions: {n}")
    print()
    print(f"{'Test':30s} {'Accuracy':>10s} {'Mean Δ':>12s}  Description")
    print("-" * 72)
    for key, t in tests.items():
        flag = "PASS" if t["accuracy"] > 0.80 else ("weak" if t["accuracy"] > 0.60 else "random")
        print(f"{key:30s} {t['accuracy']:>9.1%}  {t['mean_diff']:>+12.5f}  {flag}")
    print()
    print("Decoded mean kinematics (absolute):")
    print(f"  {'':12s} " + " ".join(f"{k:>10s}" for k in KIN_NAMES))
    print(f"  {'current':12s} " + " ".join(
        f"{current_kin[:, d].mean():+10.4f}" for d in range(6)))
    for name in TEST_ACTIONS:
        print(f"  {name:12s} " + " ".join(
            f"{decoded[name][:, d].mean():+10.4f}" for d in range(6)))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--ladder-checkpoint", type=str)
    g.add_argument("--vae-checkpoint", type=str)
    p.add_argument("--dynamics-checkpoint", type=str,
                   help="Required with --vae-checkpoint")
    p.add_argument("--data-dir", type=str,
                   help="State-space: episode .npz dir (with 'states' + 'actions').")
    p.add_argument("--data-path", type=str,
                   help="Pixel: episode dir containing rgb_frames npz.")
    p.add_argument("--output-json", type=str, default=None)
    p.add_argument("--n-episodes", type=int, default=50)
    p.add_argument("--n-transitions", type=int, default=500)
    p.add_argument("--per-episode", type=int, default=20,
                   help="Max sampled transitions per episode.")
    p.add_argument("--warmup", type=int, default=20,
                   help="Teacher-force warmup steps for recurrent state-space models.")
    p.add_argument("--sustained-steps", type=int, default=5,
                   help="Rollout length for sustained + impulse tests.")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    # Ensure repo root importable (so scripts.world_models.* resolve).
    repo_root = str(Path(__file__).resolve().parents[2])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    if args.ladder_checkpoint:
        if not args.data_dir:
            p.error("--data-dir required with --ladder-checkpoint")
        current_kin, decoded, rollout_kin, meta = run_state_space(args)
        run_name = Path(args.ladder_checkpoint).parent.name
    else:
        if not args.dynamics_checkpoint or not args.data_path:
            p.error("--dynamics-checkpoint and --data-path required with --vae-checkpoint")
        current_kin, decoded, rollout_kin, meta = run_pixel(args)
        run_name = Path(args.dynamics_checkpoint).parent.name

    if len(current_kin) == 0:
        print("  ERROR: no transitions collected (episodes too short?)")
        sys.exit(1)

    tests, n = compute_tests(current_kin, decoded)
    print_table(tests, n, current_kin, decoded, run_name=run_name)
    per_step = compute_rollout_tests(rollout_kin)
    print_rollout_table(per_step)

    if args.output_json:
        out = {
            "run_name": run_name,
            "meta": meta,
            "n_transitions": n,
            "tests": tests,
            "rollout_tests": per_step,
            "per_action_mean_kin": {
                "current": current_kin.mean(axis=0).tolist(),
                **{name: decoded[name].mean(axis=0).tolist() for name in TEST_ACTIONS},
            },
        }
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n  wrote {args.output_json}")


if __name__ == "__main__":
    main()
