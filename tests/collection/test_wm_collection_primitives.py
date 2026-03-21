"""Integration tests for primitive episode collection."""

import json
import os

import numpy as np
import pytest

from lwp.collection.wm_collection import collect_primitive
from lwp.collection.wm_collection_config import (
    CollectionConfig, ManeuverConfig, StartConfig,
)
from parametric_lunar_lander.episode_io import load_episode
from parametric_lunar_lander.physics_config import LunarLanderPhysicsConfig


# Gym-default physics ranges (fixed values).
GYM_DEFAULT_RANGES = {
    "gravity": (-10.0, -10.0),
    "main_engine_power": (13.0, 13.0),
    "side_engine_power": (0.6, 0.6),
    "lander_density": (5.0, 5.0),
    "angular_damping": (2.5, 2.5),
    "wind_power": (0.0, 0.0),
    "turbulence_power": (0.0, 0.0),
}


class TestCollectPrimitiveFreshReset:
    """Primitive collection from fresh resets."""

    def test_free_fall_produces_episodes(self, tmp_path):
        results = collect_primitive(
            output_dir=str(tmp_path),
            n_episodes=3,
            physics_ranges=GYM_DEFAULT_RANGES,
            maneuver_config=ManeuverConfig(type="free_fall"),
            start_config=StartConfig(
                mode="fresh_reset",
                initial_state={
                    "x": (-0.3, 0.3), "y": (0.4, 0.8),
                    "vx": (-0.5, 0.5), "vy": (-0.5, 0.0),
                    "angle": (-0.3, 0.3), "angular_vel": (-0.2, 0.2),
                },
            ),
            seed=42,
            max_steps=200,
        )
        assert len(results) == 3
        npz_files = list(tmp_path.glob("*.npz"))
        assert len(npz_files) == 3

    def test_episode_metadata_correct(self, tmp_path):
        results = collect_primitive(
            output_dir=str(tmp_path),
            n_episodes=1,
            physics_ranges=GYM_DEFAULT_RANGES,
            maneuver_config=ManeuverConfig(type="constant_thrust", main=0.5),
            start_config=StartConfig(
                mode="fresh_reset",
                initial_state={
                    "x": (0.0, 0.0), "y": (0.5, 0.5),
                    "vx": (0.0, 0.0), "vy": (0.0, 0.0),
                    "angle": (0.0, 0.0), "angular_vel": (0.0, 0.0),
                },
            ),
            seed=42,
            max_steps=100,
        )
        ep = load_episode(results[0]["npz_path"])
        meta = ep["metadata"]
        assert meta["source_type"] == "primitive"
        assert meta["maneuver_type"] == "constant_thrust"
        assert meta["start_mode"] == "fresh_reset"
        assert "start_state" in meta

    def test_actions_match_maneuver(self, tmp_path):
        """Free fall episodes should have zero actions."""
        results = collect_primitive(
            output_dir=str(tmp_path),
            n_episodes=1,
            physics_ranges=GYM_DEFAULT_RANGES,
            maneuver_config=ManeuverConfig(type="free_fall"),
            start_config=StartConfig(
                mode="fresh_reset",
                initial_state={
                    "x": (0.0, 0.0), "y": (0.8, 0.8),
                    "vx": (0.0, 0.0), "vy": (0.0, 0.0),
                    "angle": (0.0, 0.0), "angular_vel": (0.0, 0.0),
                },
            ),
            seed=42,
            max_steps=100,
        )
        ep = load_episode(results[0]["npz_path"])
        np.testing.assert_array_equal(ep["actions"], 0.0)

    def test_collection_meta_written(self, tmp_path):
        collect_primitive(
            output_dir=str(tmp_path),
            n_episodes=2,
            physics_ranges=GYM_DEFAULT_RANGES,
            maneuver_config=ManeuverConfig(type="free_fall"),
            start_config=StartConfig(
                mode="fresh_reset",
                initial_state={
                    "x": (-0.3, 0.3), "y": (0.4, 0.8),
                    "vx": (0.0, 0.0), "vy": (0.0, 0.0),
                    "angle": (0.0, 0.0), "angular_vel": (0.0, 0.0),
                },
            ),
            seed=42,
        )
        meta_path = tmp_path / "collection_meta.json"
        assert meta_path.exists()
        with open(meta_path) as f:
            meta = json.load(f)
        assert meta["source_type"] == "primitive"
        assert meta["summary"]["n_episodes_collected"] == 2


class TestCollectPrimitiveReplay:
    """Primitive collection from replayed source episodes."""

    def _create_source_dir(self, tmp_path):
        """Collect a few source episodes to replay from."""
        from lwp.collection.wm_collection import collect_simple
        from lwp.collection.wm_policies import make_random_policy
        source_dir = tmp_path / "source"
        collect_simple(
            policy_fn=make_random_policy(seed=0),
            output_dir=str(source_dir),
            n_episodes=5,
            physics_ranges=GYM_DEFAULT_RANGES,
            source_type="random",
            seed=0,
            max_steps=200,
        )
        return str(source_dir)

    def test_replay_produces_episodes(self, tmp_path):
        source_dir = self._create_source_dir(tmp_path)
        output_dir = tmp_path / "primitives"
        results = collect_primitive(
            output_dir=str(output_dir),
            n_episodes=2,
            physics_ranges=GYM_DEFAULT_RANGES,
            maneuver_config=ManeuverConfig(type="constant_thrust", main=0.75),
            start_config=StartConfig(
                mode="replay",
                source_dir=source_dir,
                min_step=10,
                max_step_fraction=0.5,
            ),
            seed=42,
            max_steps=200,
        )
        assert len(results) == 2

    def test_replay_metadata_has_branch_source(self, tmp_path):
        source_dir = self._create_source_dir(tmp_path)
        output_dir = tmp_path / "primitives"
        results = collect_primitive(
            output_dir=str(output_dir),
            n_episodes=1,
            physics_ranges=GYM_DEFAULT_RANGES,
            maneuver_config=ManeuverConfig(type="free_fall"),
            start_config=StartConfig(
                mode="replay",
                source_dir=source_dir,
                min_step=10,
                max_step_fraction=0.5,
            ),
            seed=42,
            max_steps=100,
        )
        ep = load_episode(results[0]["npz_path"])
        meta = ep["metadata"]
        assert meta["start_mode"] == "replay"
        assert "branch_source" in meta
        assert "episode_path" in meta["branch_source"]
        assert "branch_step" in meta["branch_source"]


class TestCollectPrimitivePostLanding:
    """Primitive collection with post-landing continuation."""

    def _create_heuristic_source(self, tmp_path):
        """Collect heuristic episodes (likely to land) as sources."""
        from lwp.collection.wm_collection import collect_simple
        from parametric_lunar_lander.heuristic import heuristic_policy
        source_dir = tmp_path / "heuristic"
        collect_simple(
            policy_fn=heuristic_policy,
            output_dir=str(source_dir),
            n_episodes=5,
            physics_ranges=GYM_DEFAULT_RANGES,
            source_type="heuristic",
            seed=0,
            max_steps=500,
        )
        return str(source_dir)

    def test_ground_stationary_continues_past_landing(self, tmp_path):
        source_dir = self._create_heuristic_source(tmp_path)
        output_dir = tmp_path / "ground"
        results = collect_primitive(
            output_dir=str(output_dir),
            n_episodes=1,
            physics_ranges=GYM_DEFAULT_RANGES,
            maneuver_config=ManeuverConfig(type="ground_stationary"),
            start_config=StartConfig(
                mode="replay_to_landing",
                source_dir=source_dir,
            ),
            seed=42,
            max_steps=200,
            allow_post_landing=True,
        )
        assert len(results) >= 1
        ep = load_episode(results[0]["npz_path"])
        meta = ep["metadata"]
        assert "termination_event" in meta
