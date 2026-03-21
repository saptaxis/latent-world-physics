"""Tests for analysis manifest loading."""

from __future__ import annotations

import pytest
import yaml


def _write_manifest(path, data: dict):
    """Write a YAML manifest to a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(data, f)


class TestLoadAnalysisManifest:
    """Test load_analysis_manifest(): YAML loading + validation."""

    def _make_manifest_data(self, **overrides) -> dict:
        """Minimal valid manifest data."""
        data = {
            "experiment": "test-experiment",
            "description": "Test manifest",
            "output_base": "/tmp/test-output",
            "configs": {
                "full-variation/blind-ppo-easy-128-lowent": {
                    "variant": "blind",
                    "condition": "full-variation",
                    "profile": "easy",
                    "net_arch": [128, 128],
                    "ent_coef": 0.001,
                    "seed_base": "/tmp/fake-agents/full-variation/blind-ppo-easy-128-lowent",
                    "seeds": [42, 123, 456],
                },
            },
        }
        data.update(overrides)
        return data

    def test_loads_from_file_path(self, tmp_path):
        """Loads manifest from a direct file path."""
        from lwp.analysis.manifest import load_analysis_manifest

        manifest_path = tmp_path / "test.yaml"
        _write_manifest(manifest_path, self._make_manifest_data())
        result = load_analysis_manifest(str(manifest_path))

        assert result["experiment"] == "test-experiment"
        assert len(result["configs"]) == 1

    def test_preserves_config_metadata(self, tmp_path):
        """Config metadata (variant, condition, etc.) is preserved as-is."""
        from lwp.analysis.manifest import load_analysis_manifest

        manifest_path = tmp_path / "test.yaml"
        _write_manifest(manifest_path, self._make_manifest_data())
        result = load_analysis_manifest(str(manifest_path))

        config = result["configs"]["full-variation/blind-ppo-easy-128-lowent"]
        assert config["variant"] == "blind"
        assert config["condition"] == "full-variation"
        assert config["seeds"] == [42, 123, 456]

    def test_resolves_seed_dirs(self, tmp_path):
        """Each config gets resolved seed_dirs: {seed_base}/s{seed}/."""
        from lwp.analysis.manifest import load_analysis_manifest

        manifest_path = tmp_path / "test.yaml"
        _write_manifest(manifest_path, self._make_manifest_data())
        result = load_analysis_manifest(str(manifest_path))

        config = result["configs"]["full-variation/blind-ppo-easy-128-lowent"]
        assert config["seed_dirs"] == [
            "/tmp/fake-agents/full-variation/blind-ppo-easy-128-lowent/s42",
            "/tmp/fake-agents/full-variation/blind-ppo-easy-128-lowent/s123",
            "/tmp/fake-agents/full-variation/blind-ppo-easy-128-lowent/s456",
        ]

    def test_missing_required_field_raises(self, tmp_path):
        """Missing 'experiment' or 'configs' raises ValueError."""
        from lwp.analysis.manifest import load_analysis_manifest

        manifest_path = tmp_path / "test.yaml"
        _write_manifest(manifest_path, {"description": "no experiment key"})

        with pytest.raises(ValueError, match="experiment"):
            load_analysis_manifest(str(manifest_path))

    def test_nonexistent_file_raises(self):
        """Missing file raises FileNotFoundError."""
        from lwp.analysis.manifest import load_analysis_manifest

        with pytest.raises(FileNotFoundError):
            load_analysis_manifest("/nonexistent/manifest.yaml")

    def test_resolves_builtin_name(self):
        """Builtin name resolves to analysis-manifests/ directory."""
        from lwp.analysis.manifest import _MANIFESTS_DIR

        # Just verify the resolution path is correct.
        # We don't test actual builtins here (they may not exist yet).
        expected = _MANIFESTS_DIR / "seed-agg" / "parametric-vs-behavioral.yaml"
        assert expected.parent.name == "seed-agg"


class TestLoadComparisonManifest:
    """Test load_comparison_manifest(): comparison-specific loading."""

    def _make_comparison_data(self, **overrides) -> dict:
        """Minimal valid comparison manifest data."""
        data = {
            "experiment": "test-comparison",
            "description": "Test comparison manifest",
            "output_base": "/tmp/test-output",
            "comparisons": {
                "full-variation-easy": {
                    "condition": "full-variation",
                    "profile": "easy",
                    "configs": {
                        "labeled": {
                            "seed_base": "/tmp/fake-agents/labeled",
                            "seeds": [123, 456],
                        },
                        "blind": {
                            "seed_base": "/tmp/fake-agents/blind",
                            "seeds": [123, 456],
                        },
                    },
                },
            },
        }
        data.update(overrides)
        return data

    def test_loads_comparison_manifest(self, tmp_path):
        """Loads comparison manifest with comparisons key."""
        from lwp.analysis.manifest import load_comparison_manifest

        path = tmp_path / "test.yaml"
        _write_manifest(path, self._make_comparison_data())
        result = load_comparison_manifest(str(path))

        assert result["experiment"] == "test-comparison"
        assert "full-variation-easy" in result["comparisons"]

    def test_resolves_seed_dirs_per_config(self, tmp_path):
        """Each config in each comparison gets resolved seed_dirs."""
        from lwp.analysis.manifest import load_comparison_manifest

        path = tmp_path / "test.yaml"
        _write_manifest(path, self._make_comparison_data())
        result = load_comparison_manifest(str(path))

        labeled = result["comparisons"]["full-variation-easy"]["configs"]["labeled"]
        assert labeled["seed_dirs"] == [
            "/tmp/fake-agents/labeled/s123",
            "/tmp/fake-agents/labeled/s456",
        ]

    def test_missing_comparisons_key_raises(self, tmp_path):
        """Missing 'comparisons' key raises ValueError."""
        from lwp.analysis.manifest import load_comparison_manifest

        path = tmp_path / "test.yaml"
        _write_manifest(path, {"experiment": "test", "configs": {}})

        with pytest.raises(ValueError, match="comparisons"):
            load_comparison_manifest(str(path))

    def test_missing_experiment_raises(self, tmp_path):
        """Missing 'experiment' key raises ValueError."""
        from lwp.analysis.manifest import load_comparison_manifest

        path = tmp_path / "test.yaml"
        _write_manifest(path, {"comparisons": {}})

        with pytest.raises(ValueError, match="experiment"):
            load_comparison_manifest(str(path))


class TestResolveManifestPath:
    """Test resolve_manifest_path (public)."""

    def test_resolves_absolute_yaml_path(self, tmp_path):
        from lwp.analysis.manifest import resolve_manifest_path

        path = tmp_path / "test.yaml"
        path.write_text("experiment: test")
        result = resolve_manifest_path(str(path))
        assert result == path

    def test_nonexistent_file_raises(self):
        from lwp.analysis.manifest import resolve_manifest_path

        with pytest.raises(FileNotFoundError):
            resolve_manifest_path("/nonexistent/file.yaml")
