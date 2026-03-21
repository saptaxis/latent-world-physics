"""Tests for forward hook activation capture on SB3 policy networks."""
import numpy as np
import pytest
import torch
import torch.nn as nn

from lwp.probing.hooks import ActivationCollector


class TestActivationCollectorLegacy:
    """Tests for the legacy (sequential, relu_indices) constructor."""

    def _make_simple_model(self):
        """A minimal Sequential that mimics SB3's policy_net structure."""
        return nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
        )

    def test_collects_activations_from_relu_layers(self):
        model = self._make_simple_model()
        collector = ActivationCollector(model, relu_indices=[1, 3])
        x = torch.randn(1, 8)
        model(x)
        acts = collector.get()
        assert "L1" in acts
        assert "L2" in acts

    def test_activation_shapes_match_layer_width(self):
        model = self._make_simple_model()
        collector = ActivationCollector(model, relu_indices=[1, 3])
        x = torch.randn(1, 8)
        model(x)
        acts = collector.get()
        assert acts["L1"].shape == (16,)
        assert acts["L2"].shape == (16,)

    def test_activations_are_numpy_float32(self):
        model = self._make_simple_model()
        collector = ActivationCollector(model, relu_indices=[1, 3])
        x = torch.randn(1, 8)
        model(x)
        acts = collector.get()
        assert isinstance(acts["L1"], np.ndarray)
        assert acts["L1"].dtype == np.float32

    def test_activations_are_post_relu(self):
        """All values should be >= 0 after ReLU."""
        model = self._make_simple_model()
        collector = ActivationCollector(model, relu_indices=[1, 3])
        torch.manual_seed(42)
        x = torch.randn(1, 8) * 5.0
        model(x)
        acts = collector.get()
        assert np.all(acts["L1"] >= 0)
        assert np.all(acts["L2"] >= 0)

    def test_overwrites_on_subsequent_forward_pass(self):
        model = self._make_simple_model()
        collector = ActivationCollector(model, relu_indices=[1, 3])

        x1 = torch.ones(1, 8)
        model(x1)
        acts1_L1 = collector.get()["L1"].copy()

        x2 = torch.ones(1, 8) * 10.0
        model(x2)
        acts2_L1 = collector.get()["L1"].copy()

        assert not np.array_equal(acts1_L1, acts2_L1)

    def test_remove_hooks_stops_collection(self):
        model = self._make_simple_model()
        collector = ActivationCollector(model, relu_indices=[1, 3])

        x = torch.randn(1, 8)
        model(x)
        assert "L1" in collector.get()

        collector.remove()

        old_acts = collector.get()["L1"].copy()
        model(torch.randn(1, 8) * 100.0)
        new_acts = collector.get()["L1"]
        np.testing.assert_array_equal(old_acts, new_acts)

    def test_get_returns_empty_before_forward_pass(self):
        model = self._make_simple_model()
        collector = ActivationCollector(model, relu_indices=[1, 3])
        acts = collector.get()
        assert len(acts) == 0

    def test_custom_layer_names(self):
        model = self._make_simple_model()
        collector = ActivationCollector(
            model, relu_indices=[1, 3], layer_names=["hidden1", "hidden2"],
        )
        x = torch.randn(1, 8)
        model(x)
        acts = collector.get()
        assert "hidden1" in acts
        assert "hidden2" in acts


class TestActivationCollectorLayerSpecs:
    """Tests for the new layer_specs constructor."""

    def _make_simple_model(self):
        return nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
        )

    def test_layer_specs_collects_activations(self):
        model = self._make_simple_model()
        specs = [("L1", model[1]), ("L2", model[3])]
        collector = ActivationCollector(layer_specs=specs)
        model(torch.randn(1, 8))
        acts = collector.get()
        assert "L1" in acts
        assert "L2" in acts

    def test_layer_specs_shapes(self):
        model = self._make_simple_model()
        specs = [("L1", model[1]), ("L2", model[3])]
        collector = ActivationCollector(layer_specs=specs)
        model(torch.randn(1, 8))
        acts = collector.get()
        assert acts["L1"].shape == (16,)
        assert acts["L2"].shape == (16,)

    def test_layer_specs_hooks_entire_module(self):
        """Hook the whole Sequential as one layer — gets final output."""
        model = self._make_simple_model()
        collector = ActivationCollector(layer_specs=[("full", model)])
        model(torch.randn(1, 8))
        acts = collector.get()
        assert "full" in acts
        assert acts["full"].shape == (16,)  # output of last ReLU

    def test_layer_specs_mixed_modules(self):
        """Hook modules from different parts of a network."""
        encoder = nn.Sequential(nn.Linear(8, 32), nn.ReLU())
        head = nn.Sequential(nn.Linear(32, 16), nn.ReLU())
        specs = [("enc", encoder), ("head_relu", head[1])]
        collector = ActivationCollector(layer_specs=specs)

        x = torch.randn(1, 8)
        z = encoder(x)
        head(z)

        acts = collector.get()
        assert "enc" in acts
        assert "head_relu" in acts
        assert acts["enc"].shape == (32,)
        assert acts["head_relu"].shape == (16,)

    def test_layer_specs_remove(self):
        model = self._make_simple_model()
        collector = ActivationCollector(layer_specs=[("L1", model[1])])
        model(torch.randn(1, 8))
        old = collector.get()["L1"].copy()
        collector.remove()
        model(torch.randn(1, 8) * 100.0)
        np.testing.assert_array_equal(old, collector.get()["L1"])

    def test_must_provide_one_constructor_form(self):
        """Raise if neither layer_specs nor sequential+relu_indices given."""
        with pytest.raises(ValueError, match="Must provide"):
            ActivationCollector()

    def test_layer_specs_takes_priority(self):
        """If layer_specs is given, sequential/relu_indices are ignored."""
        model = self._make_simple_model()
        # Pass both — layer_specs should win
        collector = ActivationCollector(
            sequential=model, relu_indices=[1, 3],
            layer_specs=[("only_one", model[1])],
        )
        model(torch.randn(1, 8))
        acts = collector.get()
        assert "only_one" in acts
        assert "L1" not in acts
