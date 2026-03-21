"""Tests for clip recording utilities (episode selection, recording, rendering)."""

import json

import numpy as np
import pandas as pd
import pytest

from lwp.agents.clip_recording import (
    select_episodes, extract_physics_config, record_clip, render_clean_clip,
)
from parametric_lunar_lander.physics_config import LunarLanderPhysicsConfig


def _make_metrics_df(n=20):
    """Create a fake metrics.csv DataFrame for testing."""
    rng = np.random.default_rng(42)
    n_landed = int(n * 0.75)
    return pd.DataFrame({
        "npz_path": [f"/fake/episode_{i:04d}.npz" for i in range(n)],
        "outcome": ["landed"] * n_landed + ["crashed"] * (n - n_landed),
        "thrust_autocorr_lag1": rng.uniform(0.5, 1.0, n),
        "thrust_duty_cycle": rng.uniform(0.1, 0.6, n),
        "fuel_efficiency": rng.uniform(1.0, 100.0, n),
        "gravity": rng.uniform(-12, -2, n),
        "twr": rng.uniform(2.0, 15.0, n),
        "total_reward": rng.uniform(-100, 300, n),
    })


class TestSelectEpisodes:
    def test_filters_by_outcome(self):
        df = _make_metrics_df()
        result = select_episodes(
            df, n=4, outcome="landed", sort_by="thrust_autocorr_lag1",
            ascending=False,
        )
        assert len(result) == 4
        assert all(r["outcome"] == "landed" for r in result)

    def test_sort_descending_returns_highest(self):
        df = _make_metrics_df()
        result = select_episodes(
            df, n=3, outcome="landed", sort_by="thrust_autocorr_lag1",
            ascending=False,
        )
        autocorrs = [r["thrust_autocorr_lag1"] for r in result]
        assert autocorrs == sorted(autocorrs, reverse=True)

    def test_sort_ascending_returns_lowest(self):
        df = _make_metrics_df()
        result = select_episodes(
            df, n=3, outcome="landed", sort_by="thrust_autocorr_lag1",
            ascending=True,
        )
        autocorrs = [r["thrust_autocorr_lag1"] for r in result]
        assert autocorrs == sorted(autocorrs)

    def test_returns_npz_path_and_metrics(self):
        df = _make_metrics_df()
        result = select_episodes(
            df, n=1, outcome="landed", sort_by="thrust_autocorr_lag1",
            ascending=False,
        )
        assert "npz_path" in result[0]
        assert "thrust_autocorr_lag1" in result[0]
        assert "gravity" in result[0]

    def test_min_max_filters(self):
        df = _make_metrics_df()
        result = select_episodes(
            df, n=10, outcome="landed", sort_by="thrust_autocorr_lag1",
            ascending=False,
            filters={"thrust_autocorr_lag1": (0.8, None)},
        )
        assert all(r["thrust_autocorr_lag1"] >= 0.8 for r in result)

    def test_diversity_spread(self):
        """When diversity_on is set, selected episodes should have varied values."""
        df = _make_metrics_df(50)
        result = select_episodes(
            df, n=4, outcome="landed", sort_by="thrust_autocorr_lag1",
            ascending=False,
            filters={"thrust_autocorr_lag1": (0.7, None)},
            diversity_on="gravity",
        )
        gravities = [r["gravity"] for r in result]
        # With diversity, gravities should span a range (not all clustered)
        assert max(gravities) - min(gravities) > 1.0

    def test_returns_empty_if_no_matches(self):
        df = _make_metrics_df()
        result = select_episodes(
            df, n=4, outcome="out_of_bounds", sort_by="thrust_autocorr_lag1",
            ascending=False,
        )
        assert len(result) == 0


class TestExtractPhysicsConfig:
    def test_extracts_config_from_npz(self, tmp_path):
        """Extract physics config from a real .npz file's metadata."""
        from parametric_lunar_lander.episode_io import save_episode
        config_dict = {
            "gravity": -5.0,
            "main_engine_power": 13.0,
            "side_engine_power": 0.6,
            "lander_density": 5.0,
            "angular_damping": 2.0,
            "wind_power": 10.0,
            "turbulence_power": 1.0,
        }
        metadata = {"physics_config": config_dict, "outcome": "landed", "seed": 42}
        npz_path = tmp_path / "test_episode.npz"
        save_episode(
            path=npz_path,
            states=np.zeros((11, 15), dtype=np.float32),
            actions=np.zeros((10, 2), dtype=np.float32),
            rewards=np.zeros(10, dtype=np.float32),
            dones=np.zeros(10, dtype=bool),
            metadata=metadata,
        )
        result = extract_physics_config(npz_path)
        assert isinstance(result, LunarLanderPhysicsConfig)
        assert result.gravity == -5.0
        assert result.wind_power == 10.0


class TestRecordClip:
    def test_records_episode_with_frames(self, tmp_path):
        """Record a clip and verify it has rgb_frames + correct physics."""
        config = LunarLanderPhysicsConfig(gravity=-5.0, wind_power=0.0)
        result = record_clip(
            checkpoint_dir=None,  # Will use heuristic policy
            physics_config=config,
            output_path=tmp_path / "clip.npz",
            variant="labeled",
            use_heuristic=True,
        )
        assert result["output_path"].exists()

        from parametric_lunar_lander.episode_io import load_episode
        ep = load_episode(result["output_path"])
        assert ep["rgb_frames"] is not None
        assert ep["rgb_frames"].shape[0] == ep["states"].shape[0]
        assert ep["metadata"]["physics_config"]["gravity"] == -5.0
        assert "outcome" in ep["metadata"]

    def test_output_metadata_has_required_fields(self, tmp_path):
        config = LunarLanderPhysicsConfig()
        result = record_clip(
            checkpoint_dir=None,
            physics_config=config,
            output_path=tmp_path / "clip.npz",
            variant="labeled",
            use_heuristic=True,
        )
        assert "outcome" in result
        assert "n_steps" in result
        assert "physics_config" in result


class TestRenderCleanClip:
    def test_renders_mp4_without_annotations(self, tmp_path):
        """Render a clip from an npz with rgb_frames — game view only."""
        from parametric_lunar_lander.episode_io import save_episode

        n_steps = 10
        config_dict = {
            "gravity": -5.0, "main_engine_power": 13.0,
            "side_engine_power": 0.6, "lander_density": 5.0,
            "angular_damping": 0.0, "wind_power": 0.0,
            "turbulence_power": 0.0,
        }
        metadata = {
            "physics_config": config_dict, "outcome": "landed",
            "seed": 42, "n_steps": n_steps, "total_reward": 250.0,
        }
        npz_path = tmp_path / "clip.npz"
        save_episode(
            path=npz_path,
            states=np.zeros((n_steps + 1, 15), dtype=np.float32),
            actions=np.zeros((n_steps, 2), dtype=np.float32),
            rewards=np.zeros(n_steps, dtype=np.float32),
            dones=np.zeros(n_steps, dtype=bool),
            metadata=metadata,
            rgb_frames=np.zeros((n_steps + 1, 400, 600, 3), dtype=np.uint8),
        )

        mp4_path, json_path = render_clean_clip(
            npz_path, tmp_path / "output.mp4",
        )
        assert mp4_path.exists()
        assert json_path.exists()

        with open(json_path) as f:
            meta = json.load(f)
        assert meta["physics_config"]["gravity"] == -5.0
        assert meta["outcome"] == "landed"
        assert "thrust_autocorr_lag1" in meta["behavioral_metrics"]

    def test_companion_json_has_required_fields(self, tmp_path):
        from parametric_lunar_lander.episode_io import save_episode

        n_steps = 10
        metadata = {
            "physics_config": {
                "gravity": -10.0, "main_engine_power": 13.0,
                "side_engine_power": 0.6, "lander_density": 5.0,
                "angular_damping": 0.0, "wind_power": 0.0,
                "turbulence_power": 0.0,
            },
            "outcome": "crashed", "seed": 99,
            "n_steps": n_steps, "total_reward": -50.0,
        }
        npz_path = tmp_path / "clip.npz"
        save_episode(
            path=npz_path,
            states=np.random.randn(n_steps + 1, 15).astype(np.float32),
            actions=np.random.randn(n_steps, 2).astype(np.float32),
            rewards=np.random.randn(n_steps).astype(np.float32),
            dones=np.zeros(n_steps, dtype=bool),
            metadata=metadata,
            rgb_frames=np.zeros((n_steps + 1, 400, 600, 3), dtype=np.uint8),
        )

        _, json_path = render_clean_clip(npz_path, tmp_path / "out.mp4")
        with open(json_path) as f:
            meta = json.load(f)

        required = [
            "physics_config", "outcome", "seed", "n_steps", "total_reward",
            "duration_seconds", "behavioral_metrics", "variant", "condition",
        ]
        for key in required:
            assert key in meta, f"Missing key: {key}"


class TestRecordClipCorruption:
    def test_records_with_zero_corruption(self, tmp_path):
        """Record a clip with zero corruption — physics dims should be zeroed."""
        config = LunarLanderPhysicsConfig(gravity=-5.0)
        result = record_clip(
            checkpoint_dir=None,
            physics_config=config,
            output_path=tmp_path / "clip.npz",
            variant="labeled",
            use_heuristic=True,
            corruption="zero",
        )
        assert result["output_path"].exists()
        # The episode should still complete (may crash/OOB due to zeroed labels)
        assert result["outcome"] in ("landed", "crashed", "timeout", "out_of_bounds")


class TestFullPipeline:
    def test_select_extract_record_render(self, tmp_path):
        """Full pipeline: select -> extract config -> record -> render."""
        from parametric_lunar_lander.episode_io import save_episode

        # Create a fake metrics.csv with npz paths.
        n_episodes = 10
        npz_dir = tmp_path / "trajectories"
        npz_dir.mkdir()

        rows = []
        for i in range(n_episodes):
            config_dict = {
                "gravity": float(-2 - i),
                "main_engine_power": 13.0,
                "side_engine_power": 0.6,
                "lander_density": 5.0,
                "angular_damping": 0.0,
                "wind_power": 0.0,
                "turbulence_power": 0.0,
            }
            metadata = {
                "physics_config": config_dict,
                "outcome": "landed" if i < 7 else "crashed",
                "seed": i,
            }
            npz_path = npz_dir / f"episode_{i:04d}.npz"
            save_episode(
                path=npz_path,
                states=np.zeros((11, 15), dtype=np.float32),
                actions=np.random.randn(10, 2).astype(np.float32),
                rewards=np.zeros(10, dtype=np.float32),
                dones=np.zeros(10, dtype=bool),
                metadata=metadata,
            )
            rows.append({
                "npz_path": str(npz_path),
                "outcome": metadata["outcome"],
                "thrust_autocorr_lag1": 0.9 + i * 0.01,
                "thrust_duty_cycle": 0.3,
                "gravity": config_dict["gravity"],
                "twr": 5.0 + i,
                "total_reward": 200 + i * 10,
                "fuel_efficiency": 10.0,
            })

        df = pd.DataFrame(rows)

        # Stage 1: Select
        selected = select_episodes(
            df, n=2, outcome="landed",
            sort_by="thrust_autocorr_lag1", ascending=False,
        )
        assert len(selected) == 2

        # Stage 2: Extract + Record (using heuristic — no real agent)
        clips_dir = tmp_path / "clips"
        clips_dir.mkdir()
        for i, ep in enumerate(selected):
            config = extract_physics_config(ep["npz_path"])
            result = record_clip(
                checkpoint_dir=None,
                physics_config=config,
                output_path=clips_dir / f"clip_{i:03d}.npz",
                variant="labeled",
                use_heuristic=True,
            )
            assert result["output_path"].exists()

        # Stage 3: Render
        rendered_dir = tmp_path / "rendered"
        rendered_dir.mkdir()
        for npz in sorted(clips_dir.glob("*.npz")):
            mp4_path, json_path = render_clean_clip(
                npz, rendered_dir / (npz.stem + ".mp4"),
                variant="labeled", condition="test",
            )
            assert mp4_path.exists()
            assert json_path.exists()
