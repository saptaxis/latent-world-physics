"""Forward hook activation capture for SB3 policy networks.

Registers hooks on specified layers to capture post-activation values at
each forward pass. Used by collect_probe_data.py to build (activation,
target) datasets.

Supports two modes:
  1. layer_specs: list of (name, module) pairs — hooks the output of each
     module. Works for any architecture (CNN features_extractor, MLP layers).
  2. Legacy: (sequential, relu_indices) — hooks specific indices within an
     nn.Sequential. Kept for backward compatibility with Expt 1 callers.

See probing-tooling.md (Section 1.1) for the hook design.
"""
import numpy as np
import torch.nn as nn


class ActivationCollector:
    """Captures activations from specified layers via forward hooks.

    Usage (new — layer_specs):
        specs = [
            ("CNN", model.policy.features_extractor),
            ("L1", policy_net[1]),
            ("L2", policy_net[3]),
        ]
        collector = ActivationCollector(layer_specs=specs)

    Usage (legacy — sequential + relu_indices):
        collector = ActivationCollector(policy_net, relu_indices=[1, 3])

    After model.predict(obs), call collector.get() to retrieve activations
    as a dict mapping layer name -> numpy array.
    """

    def __init__(
        self,
        sequential: nn.Sequential | None = None,
        relu_indices: list[int] | None = None,
        layer_names: list[str] | None = None,
        *,
        layer_specs: list[tuple[str, nn.Module]] | None = None,
    ):
        self._activations: dict[str, np.ndarray] = {}
        self._handles = []

        if layer_specs is not None:
            # New path: hook output of each (name, module) pair.
            for name, module in layer_specs:
                handle = module.register_forward_hook(self._make_hook(name))
                self._handles.append(handle)
        elif sequential is not None and relu_indices is not None:
            # Legacy path: hook specific indices within an nn.Sequential.
            if layer_names is None:
                layer_names = [f"L{i+1}" for i in range(len(relu_indices))]
            if len(layer_names) != len(relu_indices):
                raise ValueError(
                    f"layer_names length ({len(layer_names)}) must match "
                    f"relu_indices length ({len(relu_indices)})"
                )
            for idx, name in zip(relu_indices, layer_names):
                handle = sequential[idx].register_forward_hook(self._make_hook(name))
                self._handles.append(handle)
        else:
            raise ValueError(
                "Must provide either layer_specs=[(name, module), ...] or "
                "(sequential, relu_indices)"
            )

    def _make_hook(self, name: str):
        def hook(module, input, output):
            self._activations[name] = (
                output.detach().cpu().numpy().squeeze().astype(np.float32)
            )
        return hook

    def get(self) -> dict[str, np.ndarray]:
        """Return the most recent activations. Empty dict before first forward pass."""
        return self._activations

    def remove(self):
        """Remove all hooks. Activations dict becomes stale (not cleared)."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
