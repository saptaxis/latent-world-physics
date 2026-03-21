# Latent World Physics

Tools for studying what neural networks learn about physics: world model training, RL agent experiments, and an evaluation framework that measures physics understanding rather than prediction accuracy.

> **Research repo: expect rough edges.** Code here supports an active research program and evolves with it.

Built on [parametric-lunar-lander](https://github.com/saptaxis/parametric-lunar-lander), a Gymnasium environment with 7 configurable physics parameters.

## What's in the repo

| Package | What it does |
|---------|-------------|
| `lwp/models/` | World model architectures: Linear, MLP, GRU, RSSM (state-space), PixelVAE, latent dynamics (pixel) |
| `lwp/training/` | Training loops, losses, callbacks, scheduled sampling, curriculum scheduling |
| `lwp/evaluation/` | Per-dimension MSE, horizon curves, divergence exponent, pixel metrics |
| `lwp/agents/` | RL agent training configs, label corruption, visual encoders |
| `lwp/wm/` | Physics evaluation: oracle extraction, consistency R-squared, physics unit tests |
| `lwp/analysis/` | Behavioral metrics, cross-config comparison, seed aggregation |
| `lwp/collection/` | Data collection from random, heuristic, agent, and primitive policies |
| `lwp/probing/` | Linear and MLP probes on trained representations |
| `lwp/rl/` | Shared RL training infrastructure (SB3 wrappers) |

## Install

```bash
pip install -e .
```

This installs `parametric-lunar-lander` as a dependency. For development:

```bash
pip install -e ".[dev]"
```

## Quick start

### Train a world model

```bash
# Linear baseline, single-step delta prediction
python scripts/world_models/train.py \
    --config configs/world_models/ladder/linear-gym-default.yaml

# GRU with multi-step rollout loss
python scripts/world_models/train.py \
    --config configs/world_models/ladder/gru-gym-default.yaml
```

### Evaluate physics understanding

```bash
# Physics unit tests: controlled interventions (free fall, thrust, hover)
python scripts/world_models/physics_test_wm.py \
    --ladder-checkpoint runs/<run>/best.pt \
    --data-dir /path/to/episodes \
    --n-seeds 20 --plot

# Oracle extraction and consistency R-squared
# (does the model learn constants or position-dependent curves?)
python scripts/analysis/physics_understanding_report.py \
    --ladder-checkpoint runs/<run>/best.pt \
    --data-dir /path/to/episodes
```

### Train an RL agent

```bash
# Blind agent: no physics labels, learns reactive control
python scripts/agents/train_rl.py \
    --config configs/agents/full-variation/blind-ppo-easy.yaml

# Labeled agent: receives physics parameters, learns parametric control
python scripts/agents/train_rl.py \
    --config configs/agents/full-variation/labeled-ppo-easy.yaml

# Evaluate and compare
python scripts/agents/run_eval_pipeline.py \
    --checkpoint-dir runs/<agent-run> --episodes 100
```

### Train a pixel world model

```bash
# Phase 1: VAE (frame encoder/decoder)
python scripts/world_models/train_pixel_vae.py \
    --data-path /path/to/episodes-with-rgb \
    --run-dir runs/pixel-vae \
    --fg-weight 50 --state-dim 6

# Phase 2: Latent dynamics (GRU on frozen VAE z-space)
python scripts/world_models/train_pixel_dynamics.py \
    --vae-checkpoint runs/pixel-vae/best.pt \
    --data-path /path/to/episodes-with-rgb \
    --run-dir runs/pixel-dynamics
```

## Repo structure

```
lwp/                    Python package
  models/               Linear, MLP, GRU, RSSM, PixelVAE, latent dynamics
  training/             Loops, losses, callbacks (state and pixel)
  data/                 Episode loading, normalization
  evaluation/           Metrics (per-dim MSE, horizon curves, SSIM)
  utils/                Config, checkpoints, logging, plotting
  viz/                  Dream sequence visualization
  rl/                   SB3 training loop, wrappers, inference
  agents/               RL experiment code (configs, corruption, visual encoders)
  wm/                   Physics tests, oracle extraction, consistency R-squared
  analysis/             Behavioral metrics, cross-config comparison
  collection/           Data collection (trajectories, WM data, primitives)
  probing/              Linear/MLP probes on trained representations

scripts/                Entry points organized by domain
  agents/               train_rl, eval_agent, check_runs, train_probes
  world_models/         train, eval, physics_test_wm, train_pixel_vae, train_pixel_dynamics
  perception/           pretrain_encoder, prepare_encoder_dataset
  collection/           collect_trajectories, collect_world_model_data
  analysis/             compare_configs, compute_metrics, physics_understanding_report
  viz/                  visualize_trajectory, render_clips

configs/                YAML configs organized by domain
  agents/               RL training configs (baselines, full-variation, physics-only, etc.)
  world_models/         WM ladder configs and examples
  collection/           Data collection configs and primitives

tests/                  988 tests mirroring lwp/ structure
```

## License

MIT
