"""Integration tests for collect_world_model_data.py CLI."""

import json
import subprocess
import sys

import yaml


def _write_random_config(tmp_path, n_episodes=5):
    """Write a minimal random-policy collection config."""
    cfg = {
        "source_type": "random",
        "n_episodes": n_episodes,
        "physics_sampling": {
            "method": "uniform",
            "ranges": {
                "gravity": [-12.0, -3.0],
                "main_engine_power": [5.0, 20.0],
                "side_engine_power": [0.2, 1.5],
                "lander_density": [2.5, 8.0],
                "angular_damping": [0.5, 5.0],
                "wind_power": [0.0, 15.0],
                "turbulence_power": [0.0, 2.0],
            },
        },
        "seed": 42,
    }
    config_path = tmp_path / "random.yaml"
    config_path.write_text(yaml.dump(cfg))
    return str(config_path)


class TestSingleCollection:
    """CLI --config mode: single collection run."""

    def test_random_collection_produces_episodes(self, tmp_path):
        config_path = _write_random_config(tmp_path, n_episodes=3)
        output_dir = str(tmp_path / "output")

        result = subprocess.run(
            [
                sys.executable,
                "lunar_lander/scripts/collect_world_model_data.py",
                "--config", config_path,
                "--output-dir", output_dir,
            ],
            capture_output=True, text=True, timeout=120,
            cwd="/home/vsr/Dropbox/traitful-code/latent-world-geometry",
        )
        assert result.returncode == 0, f"STDERR: {result.stderr}"

        npz_files = list((tmp_path / "output").glob("*.npz"))
        assert len(npz_files) == 3

        meta_path = tmp_path / "output" / "collection_meta.json"
        assert meta_path.exists()
        with open(meta_path) as f:
            meta = json.load(f)
        assert meta["source_type"] == "random"
        assert meta["summary"]["n_episodes_collected"] == 3

    def test_sample_flag_overrides_n_episodes(self, tmp_path):
        config_path = _write_random_config(tmp_path, n_episodes=100)
        output_dir = str(tmp_path / "output")

        result = subprocess.run(
            [
                sys.executable,
                "lunar_lander/scripts/collect_world_model_data.py",
                "--config", config_path,
                "--output-dir", output_dir,
                "--sample", "2",
            ],
            capture_output=True, text=True, timeout=120,
            cwd="/home/vsr/Dropbox/traitful-code/latent-world-geometry",
        )
        assert result.returncode == 0, f"STDERR: {result.stderr}"
        npz_files = list((tmp_path / "output").glob("*.npz"))
        assert len(npz_files) == 2

    def test_heuristic_collection(self, tmp_path):
        cfg = {
            "source_type": "heuristic",
            "n_episodes": 3,
            "physics_sampling": {
                "method": "uniform",
                "ranges": {
                    "gravity": [-12.0, -3.0],
                    "main_engine_power": [5.0, 20.0],
                    "side_engine_power": [0.2, 1.5],
                    "lander_density": [2.5, 8.0],
                    "angular_damping": [0.5, 5.0],
                    "wind_power": [0.0, 15.0],
                    "turbulence_power": [0.0, 2.0],
                },
            },
            "seed": 42,
        }
        config_path = tmp_path / "heuristic.yaml"
        config_path.write_text(yaml.dump(cfg))
        output_dir = str(tmp_path / "output")

        result = subprocess.run(
            [
                sys.executable,
                "lunar_lander/scripts/collect_world_model_data.py",
                "--config", str(config_path),
                "--output-dir", output_dir,
            ],
            capture_output=True, text=True, timeout=120,
            cwd="/home/vsr/Dropbox/traitful-code/latent-world-geometry",
        )
        assert result.returncode == 0, f"STDERR: {result.stderr}"
        npz_files = list((tmp_path / "output").glob("*.npz"))
        assert len(npz_files) == 3


class TestBatchCollection:
    """CLI --batch mode: multiple collections."""

    def test_batch_runs_multiple_collections(self, tmp_path):
        configs_dir = tmp_path / "configs"
        configs_dir.mkdir()

        for name in ["a", "b"]:
            cfg = {
                "source_type": "random",
                "n_episodes": 2,
                "physics_sampling": {
                    "method": "uniform",
                    "ranges": {
                        "gravity": [-12.0, -3.0],
                        "main_engine_power": [5.0, 20.0],
                        "side_engine_power": [0.2, 1.5],
                        "lander_density": [2.5, 8.0],
                        "angular_damping": [0.5, 5.0],
                        "wind_power": [0.0, 15.0],
                        "turbulence_power": [0.0, 2.0],
                    },
                },
                "seed": 42,
            }
            (configs_dir / f"{name}.yaml").write_text(yaml.dump(cfg))

        batch = {
            "output_base": str(tmp_path / "data"),
            "collections": [
                {"config": str(configs_dir / "a.yaml"), "output_name": "run-a"},
                {"config": str(configs_dir / "b.yaml"), "output_name": "run-b"},
            ],
        }
        batch_path = tmp_path / "batch.yaml"
        batch_path.write_text(yaml.dump(batch))

        result = subprocess.run(
            [
                sys.executable,
                "lunar_lander/scripts/collect_world_model_data.py",
                "--batch", str(batch_path),
            ],
            capture_output=True, text=True, timeout=120,
            cwd="/home/vsr/Dropbox/traitful-code/latent-world-geometry",
        )
        assert result.returncode == 0, f"STDERR: {result.stderr}"

        assert (tmp_path / "data" / "run-a").is_dir()
        assert (tmp_path / "data" / "run-b").is_dir()
        assert len(list((tmp_path / "data" / "run-a").glob("*.npz"))) == 2
        assert len(list((tmp_path / "data" / "run-b").glob("*.npz"))) == 2
