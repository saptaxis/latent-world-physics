"""Tests for world model core collection functions."""

import json
import os

import numpy as np
import pytest

from lwp.collection.wm_collection import collect_simple, collect_rl_agent
from lwp.collection.wm_policies import make_random_policy
from parametric_lunar_lander.episode_io import load_episode
from parametric_lunar_lander.physics_config import LunarLanderPhysicsConfig


# Uniform physics ranges for testing. Must be within LunarLanderPhysicsConfig.RANGES
# so that _sample_physics_config produces valid configs.
UNIFORM_RANGES = {
    "gravity": (-12.0, -3.0),
    "main_engine_power": (5.0, 20.0),
    "side_engine_power": (0.2, 1.5),
    "lander_density": (2.5, 8.0),
    "angular_damping": (0.5, 5.0),
    "wind_power": (0.0, 15.0),
    "turbulence_power": (0.0, 2.0),
}


class TestCollectSimple:
    """Collection with simple policy_fn (no VecNormalize)."""

    def test_collects_correct_number_of_episodes(self, tmp_path):
        policy_fn = make_random_policy(seed=42)
        results = collect_simple(
            policy_fn=policy_fn,
            output_dir=str(tmp_path),
            n_episodes=3,
            physics_ranges=UNIFORM_RANGES,
            source_type="random",
            seed=0,
        )
        assert len(results) == 3
        npz_files = list(tmp_path.glob("*.npz"))
        assert len(npz_files) == 3

    def test_episode_format_correct(self, tmp_path):
        policy_fn = make_random_policy(seed=42)
        results = collect_simple(
            policy_fn=policy_fn,
            output_dir=str(tmp_path),
            n_episodes=1,
            physics_ranges=UNIFORM_RANGES,
            source_type="random",
            seed=0,
        )
        ep = load_episode(results[0]["npz_path"])
        T = len(ep["actions"])
        assert ep["states"].shape == (T + 1, 15)
        assert ep["actions"].shape == (T, 2)
        assert ep["rewards"].shape == (T,)
        assert ep["dones"].shape == (T,)

    def test_metadata_has_source_type(self, tmp_path):
        policy_fn = make_random_policy(seed=42)
        results = collect_simple(
            policy_fn=policy_fn,
            output_dir=str(tmp_path),
            n_episodes=1,
            physics_ranges=UNIFORM_RANGES,
            source_type="random",
            seed=0,
        )
        ep = load_episode(results[0]["npz_path"])
        assert ep["metadata"]["source_type"] == "random"

    def test_metadata_has_physics_config(self, tmp_path):
        policy_fn = make_random_policy(seed=42)
        results = collect_simple(
            policy_fn=policy_fn,
            output_dir=str(tmp_path),
            n_episodes=1,
            physics_ranges=UNIFORM_RANGES,
            source_type="random",
            seed=0,
        )
        ep = load_episode(results[0]["npz_path"])
        pc = ep["metadata"]["physics_config"]
        assert len(pc) == 7
        assert "gravity" in pc

    def test_physics_varies_across_episodes(self, tmp_path):
        """Each episode should get fresh random physics."""
        policy_fn = make_random_policy(seed=42)
        collect_simple(
            policy_fn=policy_fn,
            output_dir=str(tmp_path),
            n_episodes=5,
            physics_ranges=UNIFORM_RANGES,
            source_type="random",
            seed=0,
        )
        gravities = []
        for npz_path in sorted(tmp_path.glob("*.npz")):
            ep = load_episode(npz_path)
            gravities.append(ep["metadata"]["physics_config"]["gravity"])
        assert len(set(f"{g:.4f}" for g in gravities)) > 1

    def test_physics_within_specified_ranges(self, tmp_path):
        policy_fn = make_random_policy(seed=42)
        collect_simple(
            policy_fn=policy_fn,
            output_dir=str(tmp_path),
            n_episodes=10,
            physics_ranges=UNIFORM_RANGES,
            source_type="random",
            seed=0,
        )
        for npz_path in sorted(tmp_path.glob("*.npz")):
            ep = load_episode(npz_path)
            pc = ep["metadata"]["physics_config"]
            for param, (lo, hi) in UNIFORM_RANGES.items():
                assert lo <= pc[param] <= hi, f"{param}={pc[param]} not in [{lo}, {hi}]"

    def test_results_have_expected_keys(self, tmp_path):
        policy_fn = make_random_policy(seed=42)
        results = collect_simple(
            policy_fn=policy_fn,
            output_dir=str(tmp_path),
            n_episodes=1,
            physics_ranges=UNIFORM_RANGES,
            source_type="random",
            seed=0,
        )
        r = results[0]
        assert "npz_path" in r
        assert "outcome" in r
        assert "reward" in r
        assert "steps" in r


class TestCollectionMeta:
    """collection_meta.json written alongside episodes."""

    def test_meta_written(self, tmp_path):
        policy_fn = make_random_policy(seed=42)
        collect_simple(
            policy_fn=policy_fn,
            output_dir=str(tmp_path),
            n_episodes=3,
            physics_ranges=UNIFORM_RANGES,
            source_type="random",
            seed=0,
        )
        meta_path = tmp_path / "collection_meta.json"
        assert meta_path.exists()
        with open(meta_path) as f:
            meta = json.load(f)
        assert meta["source_type"] == "random"
        assert meta["summary"]["n_episodes_collected"] == 3
        assert "outcomes" in meta["summary"]
