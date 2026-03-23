# CLAUDE.md

## Project Overview

Research toolkit for studying what neural networks learn about physics. World model training (state + pixel), RL agent experiments, physics evaluation framework, data collection, analysis. Package name: `lwp`.

Depends on `parametric-lunar-lander` (`~/Dropbox/code/parametric-lunar-lander/`) for the Lunar Lander environment.

## Development Environment

**Working directory:** `~/Dropbox/code/latent-world-physics/`. Never `cd` away. Run everything from project root with relative paths.

**Virtualenv:** Check if running inside a scad container first. If yes, the container venv is already active. Otherwise, activate the host venv:

```bash
if [ -f /opt/venv/bin/activate ]; then
  echo "scad container — venv already active"
else
  source ~/virtual_envs/lwp/bin/activate
fi
```

**Scad containers:** If running in a scad container, the working directory is `/workspace/latent-world-physics` (not the Dropbox path). Data is at `/media/hdd1/physics-priors-latent-space`. All host `~/Dropbox/...` paths in this doc and in configs should be read as their `/workspace/` equivalents.

**GPU:** Use CUDA where possible. 4090 + 2080ti available.

**Cross-repo edits:** `traitful-docs` is typically added via `/add-dir ~/Dropbox/traitful-code/traitful-docs/`. When editing or committing files there, use full absolute paths.

## Documentation

Documentation (research questions, plans, implementation logs, literature) lives in a **separate repo**:

- **Docs repo:** `~/Dropbox/traitful-code/traitful-docs/`
- **Research notes:** `traitful-docs/docs/research/physics-priors-latent-space/`

**On session start:** Ask the user to run `/add-dir ~/Dropbox/traitful-code/traitful-docs/` to add the docs repo.

## Migration from lwg + wm-ladder

This repo was created on 2026-03-22 by merging code from two source repos:
- `latent-world-geometry` (lwg) commit `6e2502e`
- `world-model-ladder` (wm-ladder) commit `8af377f`

**No code logic changed.** Only imports and file paths differ.

### Import mapping (from lwg)

| Old (lwg) | New |
|---|---|
| `from lunar_lander.src.env import ...` | `from parametric_lunar_lander.env import ...` |
| `from lunar_lander.src.physics_config import ...` | `from parametric_lunar_lander.physics_config import ...` |
| `from lunar_lander.src.wrappers import ...` | `from parametric_lunar_lander.wrappers import ...` |
| `from lunar_lander.src.sampling_profiles import ...` | `from parametric_lunar_lander.sampling_profiles import ...` |
| `from lunar_lander.src.heuristic import ...` | `from parametric_lunar_lander.heuristic import ...` |
| `from lunar_lander.src.raycasting import ...` | `from parametric_lunar_lander.raycasting import ...` |
| `from lunar_lander.src.calibration import ...` | `from parametric_lunar_lander.calibration import ...` |
| `from lunar_lander.src.episode_io import ...` | `from parametric_lunar_lander.episode_io import ...` |
| `from lunar_lander.src.physics_utils import ...` | `from parametric_lunar_lander.physics_utils import ...` |
| `from lunar_lander.src.training_config import ...` | `from lwp.agents.training_config import ...` |
| `from lunar_lander.src.label_corruption import ...` | `from lwp.agents.label_corruption import ...` |
| `from lunar_lander.src.eval_utils import ...` | `from lwp.agents.eval_utils import ...` |
| `from lunar_lander.src.clip_recording import ...` | `from lwp.agents.clip_recording import ...` |
| `from lunar_lander.src.aux_ppo import ...` | `from lwp.agents.aux_ppo import ...` |
| `from lunar_lander.src.visual_backbones import ...` | `from lwp.agents.visual_backbones import ...` |
| `from lunar_lander.src.encoder_dataset import ...` | `from lwp.agents.encoder_dataset import ...` |
| `from lunar_lander.src.env_wrappers_collection import ...` | `from lwp.collection.env_wrappers_collection import ...` |
| `from lunar_lander.src.wm_collection import ...` | `from lwp.collection.wm_collection import ...` |
| `from lunar_lander.src.wm_collection_config import ...` | `from lwp.collection.wm_collection_config import ...` |
| `from lunar_lander.src.wm_policies import ...` | `from lwp.collection.wm_policies import ...` |
| `from lunar_lander.src.wm_primitives import ...` | `from lwp.collection.wm_primitives import ...` |
| `from lunar_lander.src.wm.X import ...` | `from lwp.wm.X import ...` |
| `from lunar_lander.src.analysis.X import ...` | `from lwp.analysis.X import ...` |
| `from lunar_lander.src.probing.X import ...` | `from lwp.probing.X import ...` |
| `from rl_common.X import ...` | `from lwp.rl.X import ...` |

### Import mapping (from wm-ladder)

| Old (wm-ladder) | New |
|---|---|
| `from models.X import ...` | `from lwp.models.X import ...` |
| `from training.X import ...` | `from lwp.training.X import ...` |
| `from data.X import ...` | `from lwp.data.X import ...` |
| `from evaluation.X import ...` | `from lwp.evaluation.X import ...` |
| `from utils.X import ...` | `from lwp.utils.X import ...` |
| `from viz.X import ...` | `from lwp.viz.X import ...` |

### Script locations

| Old (lwg) | New |
|---|---|
| `lunar_lander/scripts/train_rl.py` | `scripts/agents/train_rl.py` |
| `lunar_lander/scripts/eval_agent.py` | `scripts/agents/eval_agent.py` |
| `lunar_lander/scripts/check_runs.py` | `scripts/agents/check_runs.py` |
| `lunar_lander/scripts/train_all_agents.py` | `scripts/agents/train_all_agents.py` |
| `lunar_lander/scripts/run_eval_pipeline.py` | `scripts/agents/run_eval_pipeline.py` |
| `lunar_lander/scripts/train_probes.py` | `scripts/agents/train_probes.py` |
| `lunar_lander/scripts/physics_test_wm.py` | `scripts/world_models/physics_test_wm.py` |
| `lunar_lander/scripts/check_wm_runs.py` | `scripts/world_models/check_wm_runs.py` |
| `lunar_lander/scripts/viz_rollouts_wm.py` | `scripts/world_models/viz_rollouts_wm.py` |
| `lunar_lander/scripts/eval_pixel_wm_physics.py` | `scripts/world_models/eval_pixel_wm_physics.py` |
| `lunar_lander/scripts/check_pixel_wm_runs.py` | `scripts/world_models/check_pixel_wm_runs.py` |
| `lunar_lander/scripts/collect_world_model_data.py` | `scripts/collection/collect_world_model_data.py` |
| `lunar_lander/scripts/collect_trajectories.py` | `scripts/collection/collect_trajectories.py` |
| `lunar_lander/scripts/collect_grid.py` | `scripts/collection/collect_grid.py` |
| `lunar_lander/scripts/pretrain_encoder.py` | `scripts/perception/pretrain_encoder.py` |
| `lunar_lander/scripts/pretrain_encoder_reconstruction.py` | `scripts/perception/pretrain_encoder_reconstruction.py` |
| `lunar_lander/scripts/prepare_encoder_dataset.py` | `scripts/perception/prepare_encoder_dataset.py` |
| `lunar_lander/scripts/collect_probe_data.py` | `scripts/perception/collect_probe_data.py` |
| `lunar_lander/scripts/compute_metrics.py` | `scripts/analysis/compute_metrics.py` |
| `lunar_lander/scripts/analyze_behavior.py` | `scripts/analysis/analyze_behavior.py` |
| `lunar_lander/scripts/compare_configs.py` | `scripts/analysis/compare_configs.py` |
| `lunar_lander/scripts/aggregate_seeds.py` | `scripts/analysis/aggregate_seeds.py` |
| `lunar_lander/scripts/physics_understanding_report.py` | `scripts/analysis/physics_understanding_report.py` |
| `lunar_lander/scripts/visualize_trajectory.py` | `scripts/viz/visualize_trajectory.py` |
| `lunar_lander/scripts/render_clips.py` | `scripts/viz/render_clips.py` |

| Old (wm-ladder) | New |
|---|---|
| `scripts/train.py` | `scripts/world_models/train.py` |
| `scripts/eval.py` | `scripts/world_models/eval.py` |
| `scripts/train_pixel_vae.py` | `scripts/world_models/train_pixel_vae.py` |
| `scripts/train_pixel_dynamics.py` | `scripts/world_models/train_pixel_dynamics.py` |
| `scripts/train_pixel_world_model.py` | `scripts/world_models/train_pixel_world_model.py` |
| `scripts/dream_compare.py` | `scripts/world_models/dream_compare.py` |

### Config locations

| Old (lwg) | New |
|---|---|
| `lunar_lander/configs/baselines/` | `configs/agents/baselines/` |
| `lunar_lander/configs/full-variation/` | `configs/agents/full-variation/` |
| `lunar_lander/configs/gym-default/` | `configs/agents/gym-default/` |
| `lunar_lander/configs/physics-only/` | `configs/agents/physics-only/` |
| `lunar_lander/configs/vehicle-only/` | `configs/agents/vehicle-only/` |
| `lunar_lander/configs/wm-ladder/` | `configs/world_models/ladder/` |
| `lunar_lander/configs/wm-data-collection/` | `configs/collection/wm-data-collection/` |
| `lunar_lander/configs/wm-data-mix/` | `configs/collection/wm-data-mix/` |

### Dead code removed

These files from lwg depended on the old `world_models/` and `introspection/` modules (pre-wm-ladder) and were not migrated:
- `scripts/world_models/eval_wm.py`, `diagnose_wm.py`
- `tests/wm/test_wm_diagnostics.py`, `test_wm_integration.py`, `test_wm_train_config.py`, `test_physics_tests.py`, `test_rollout_io.py`
- `tests/test_integration.py`
- `diagnostics.py` was stripped to data loading utilities only

## Development

```bash
pip install -e ~/Dropbox/code/parametric-lunar-lander  # testbed dependency
pip install -e ".[dev]"
pytest tests/ -v
```

772 tests (1 pre-existing failure in DreamGridCallback).

## Scad setup

In scad containers, run this before any work (deps are pre-installed via requirements.txt, but the two packages need editable install at runtime since workspace mounts aren't available at build time):

```bash
pip install -e /workspace/parametric-lunar-lander && pip install -e "/workspace/latent-world-physics[dev]"
```

## Code Organization

```
lwp/                    Python package
  models/               World model architectures (from wm-ladder)
  training/             Training loops, losses, callbacks (from wm-ladder)
  data/                 Episode loading, normalization (from wm-ladder)
  evaluation/           Metrics (from wm-ladder)
  utils/                Config, checkpoints, logging (from wm-ladder)
  viz/                  Dream visualization (from wm-ladder)
  rl/                   SB3 training loop, wrappers, inference (from lwg rl_common/)
  agents/               RL experiment code (from lwg lunar_lander/src/)
  collection/           Data collection (from lwg lunar_lander/src/)
  wm/                   Physics tests, understanding, diagnostics (from lwg lunar_lander/src/wm/)
  analysis/             Behavioral metrics, comparisons (from lwg lunar_lander/src/analysis/)
  probing/              Linear/MLP probes (from lwg lunar_lander/src/probing/)

scripts/                Entry points by domain
  agents/               train_rl, eval_agent, check_runs, etc.
  world_models/         train, eval, physics_test_wm, train_pixel_vae, etc.
  perception/           pretrain_encoder, prepare_encoder_dataset, etc.
  collection/           collect_trajectories, collect_world_model_data, etc.
  analysis/             compute_metrics, compare_configs, etc.
  viz/                  visualize_trajectory, render_clips, etc.

configs/                YAML configs by domain
  agents/               RL training configs
  world_models/         WM ladder configs
  collection/           Data collection configs
```

## Key conventions

- **state_dim=6** for wm-ladder models: [x, y, vx, vy, angle, angular_vel] (kinematic only)
- **state_dim=8** for blind RL agents: 6 kinematic + 2 leg contacts
- **state_dim=15** for labeled RL agents: 8 kinematic + 7 physics params
- Scripts that import other scripts use `sys.path.insert(0, repo_root)` to find `scripts.*`
- `conftest.py` adds repo root to `sys.path` so tests can import scripts
