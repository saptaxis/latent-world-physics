#!/usr/bin/env python3
"""Watch Lunar Lander episodes with different physics configs.

Opens a pygame window and plays episodes back-to-back. Each episode uses
a different physics config so you can see how the lander behaves under
different gravity, thrust, wind, etc.

Usage:
  source ~/virtual_envs/lwg-lunar/bin/activate
  PYTHONPATH=. python lunar_lander/scripts/watch_episodes.py

Controls:
  - Episodes auto-play at 50 FPS (real-time physics)
  - Close the window or Ctrl+C to quit
"""

import numpy as np
import time

from parametric_lunar_lander.env import ParameterizedLunarLander
from parametric_lunar_lander.physics_config import LunarLanderPhysicsConfig


def heuristic_policy(obs):
    """Simple PD controller from Gymnasium — tries to land.

    Takes the base 8D state and computes thrust commands.
    Tuned for default physics — degrades under different configs.
    """
    s = obs[:8]
    angle_targ = s[0] * 0.5 + s[2] * 1.0
    angle_targ = np.clip(angle_targ, -0.4, 0.4)
    hover_targ = 0.55 * np.abs(s[0])

    angle_todo = (angle_targ - s[4]) * 0.5 - s[5] * 1.0
    hover_todo = (hover_targ - s[1]) * 0.5 - s[3] * 0.5

    if s[6] or s[7]:  # legs touching
        angle_todo = 0
        hover_todo = -s[3] * 0.5

    a = np.array([hover_todo * 20 - 1, -angle_todo * 20], dtype=np.float32)
    return np.clip(a, -1, 1)


# Configs to showcase — each highlights a different physics property.
CONFIGS = [
    ("DEFAULT (standard LunarLander)", LunarLanderPhysicsConfig(
        wind_power=0.0, turbulence_power=0.0,
    )),
    ("EASY: weak gravity (-3), strong thrust (22)", LunarLanderPhysicsConfig(
        gravity=-3.0, main_engine_power=22.0,
        wind_power=0.0, turbulence_power=0.0,
    )),
    ("HEAVY: density=10, struggles to hover", LunarLanderPhysicsConfig(
        lander_density=10.0,
        wind_power=0.0, turbulence_power=0.0,
    )),
    ("WINDY: wind=25, turbulence=4", LunarLanderPhysicsConfig(
        wind_power=25.0, turbulence_power=4.0,
    )),
    ("DAMPED: angular_damping=5, very stable rotation", LunarLanderPhysicsConfig(
        angular_damping=5.0,
        wind_power=0.0, turbulence_power=0.0,
    )),
    ("HARD: strong gravity, weak thrust, heavy, windy", LunarLanderPhysicsConfig(
        gravity=-12.0, main_engine_power=7.0, lander_density=8.0,
        wind_power=20.0, turbulence_power=3.0,
    )),
    ("MOON: weak gravity (-2), no wind", LunarLanderPhysicsConfig(
        gravity=-2.0, main_engine_power=10.0,
        wind_power=0.0, turbulence_power=0.0,
    )),
]


def main():
    max_steps = 400

    for i, (name, config) in enumerate(CONFIGS):
        print(f"\n{'='*60}")
        print(f"  Episode {i+1}/{len(CONFIGS)}: {name}")
        print(f"  Config: g={config.gravity}, thrust={config.main_engine_power}, "
              f"density={config.lander_density}, damp={config.angular_damping}, "
              f"wind={config.wind_power}")
        print(f"{'='*60}")

        env = ParameterizedLunarLander(
            render_mode="human",
            physics_config=config,
        )

        obs, info = env.reset(seed=42 + i)
        total_reward = 0.0

        for step in range(max_steps):
            action = heuristic_policy(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if terminated:
                outcome = "LANDED" if reward > 0 else "CRASHED/OOB"
                print(f"  -> {outcome} at step {step+1}, reward={total_reward:+.1f}")
                # Pause briefly so you can see the final state
                time.sleep(1.0)
                break
        else:
            print(f"  -> TIMEOUT at {max_steps} steps, reward={total_reward:+.1f}")
            time.sleep(0.5)

        env.close()

    print(f"\n{'='*60}")
    print("  Done! All episodes played.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
