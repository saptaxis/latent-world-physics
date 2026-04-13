"""Tests for ImpalaBranchCombinedExtractor.

Validates:
1. Output shape = cnn_output_dim + physics_branch_dim
2. Normalization buffers checkpoint and device-transfer correctly
3. CNN state_dict keys match standalone ImpalaCNN (weight loading compat)
4. Physics branch is separate from extractors dict (not frozen by encoder freeze)
5. Forward pass works with and without normalization stats
"""
import numpy as np
import pytest
import torch
from gymnasium import spaces

from lwp.agents.visual_backbones import ImpalaBranchCombinedExtractor, ImpalaCNN


def _make_dict_obs_space(img_shape=(4, 128, 128), n_physics=7):
    """Create a Dict observation space matching visual-labeled env output."""
    return spaces.Dict({
        "image": spaces.Box(low=0, high=255, shape=img_shape, dtype=np.uint8),
        "physics": spaces.Box(low=-np.inf, high=np.inf, shape=(n_physics,), dtype=np.float32),
    })


class TestOutputShape:

    def test_default_dims(self):
        """Output dim = 256 (CNN) + 32 (branch) = 288 with defaults."""
        obs_space = _make_dict_obs_space()
        extractor = ImpalaBranchCombinedExtractor(
            obs_space, cnn_output_dim=256, physics_branch_dim=32,
            normalized_image=True,
        )
        assert extractor.features_dim == 288

    def test_custom_branch_dim(self):
        """Output dim changes with physics_branch_dim."""
        obs_space = _make_dict_obs_space()
        extractor = ImpalaBranchCombinedExtractor(
            obs_space, cnn_output_dim=256, physics_branch_dim=64,
            normalized_image=True,
        )
        assert extractor.features_dim == 320  # 256 + 64

    def test_forward_output_shape(self):
        """Forward pass produces correct shape (batch, features_dim)."""
        obs_space = _make_dict_obs_space()
        extractor = ImpalaBranchCombinedExtractor(
            obs_space, cnn_output_dim=256, physics_branch_dim=32,
            normalized_image=True,
        )
        extractor.eval()
        batch = {
            "image": torch.zeros(2, 4, 128, 128),
            "physics": torch.zeros(2, 7),
        }
        with torch.no_grad():
            out = extractor(batch)
        assert out.shape == (2, 288)


class TestNormalizationBuffers:

    def test_buffers_registered(self):
        """physics_mean and physics_std are registered as buffers."""
        obs_space = _make_dict_obs_space()
        mean = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=np.float32)
        std = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5], dtype=np.float32)
        extractor = ImpalaBranchCombinedExtractor(
            obs_space, physics_mean=mean, physics_std=std,
            normalized_image=True,
        )
        # Buffers should appear in state_dict but not in parameters
        buf_names = [name for name, _ in extractor.named_buffers()]
        assert "physics_mean" in buf_names
        assert "physics_std" in buf_names
        param_names = [name for name, _ in extractor.named_parameters()]
        assert "physics_mean" not in param_names
        assert "physics_std" not in param_names

    def test_buffers_survive_state_dict_roundtrip(self):
        """Buffers are saved and restored via state_dict.

        Both source and target use always-tensor buffers, so state_dict
        roundtrip works cleanly without None-buffer compatibility issues.
        """
        obs_space = _make_dict_obs_space()
        mean = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=np.float32)
        std = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5], dtype=np.float32)
        extractor = ImpalaBranchCombinedExtractor(
            obs_space, physics_mean=mean, physics_std=std,
            normalized_image=True,
        )
        sd = extractor.state_dict()

        # Build a fresh extractor (gets identity norm buffers by default)
        extractor2 = ImpalaBranchCombinedExtractor(
            obs_space, normalized_image=True,
        )
        # Before loading: identity buffers
        torch.testing.assert_close(extractor2.physics_mean, torch.zeros(7))
        torch.testing.assert_close(extractor2.physics_std, torch.ones(7))

        # After loading: buffers from the source extractor
        extractor2.load_state_dict(sd)
        torch.testing.assert_close(extractor2.physics_mean, torch.tensor(mean))
        torch.testing.assert_close(extractor2.physics_std, torch.tensor(std))

    def test_normalization_applied(self):
        """Physics input is normalized before the branch MLP.

        Verifies that (physics=4.0, mean=0, std=2) produces the same
        branch output as (physics=2.0, mean=0, std=1) — both normalize
        to 2.0 before the branch MLP. This catches bugs where
        _normalize_physics() is accidentally bypassed.
        """
        obs_space = _make_dict_obs_space()

        # Extractor A: mean=0, std=2. Input physics=4.0 → normalized=2.0
        ext_a = ImpalaBranchCombinedExtractor(
            obs_space,
            physics_mean=np.zeros(7, dtype=np.float32),
            physics_std=np.full(7, 2.0, dtype=np.float32),
            normalized_image=True,
        )
        ext_a.eval()

        # Extractor B: identity normalization (mean=0, std=1). Input physics=2.0 → normalized=2.0
        ext_b = ImpalaBranchCombinedExtractor(
            obs_space, normalized_image=True,
        )
        ext_b.eval()

        # Copy branch weights from A to B so the Linear+ReLU is identical
        ext_b.physics_branch.load_state_dict(ext_a.physics_branch.state_dict())
        # Copy CNN weights too so image features match
        ext_b.extractors["image"].load_state_dict(ext_a.extractors["image"].state_dict())

        batch_a = {"image": torch.zeros(1, 4, 128, 128), "physics": torch.full((1, 7), 4.0)}
        batch_b = {"image": torch.zeros(1, 4, 128, 128), "physics": torch.full((1, 7), 2.0)}

        with torch.no_grad():
            out_a = ext_a(batch_a)  # physics=4, normalized by std=2 → 2.0 → branch
            out_b = ext_b(batch_b)  # physics=2, no normalization → 2.0 → branch

        # Same normalized input to the same branch → same output
        torch.testing.assert_close(out_a, out_b)

    def test_no_normalization_stats_uses_identity(self):
        """Without explicit stats, buffers default to identity (mean=0, std=1)."""
        obs_space = _make_dict_obs_space()
        extractor = ImpalaBranchCombinedExtractor(
            obs_space, normalized_image=True,
        )
        assert not extractor.has_physics_norm
        torch.testing.assert_close(extractor.physics_mean, torch.zeros(7))
        torch.testing.assert_close(extractor.physics_std, torch.ones(7))

        # Forward pass still works
        extractor.eval()
        batch = {"image": torch.zeros(1, 4, 128, 128), "physics": torch.ones(1, 7)}
        with torch.no_grad():
            out = extractor(batch)
        assert out.shape == (1, 288)


class TestWeightLoadingCompat:

    def test_cnn_state_dict_keys_match_standalone(self):
        """The CNN inside extractors['image'] has the same state_dict keys
        as a standalone ImpalaCNN — pretrained weights load without remapping."""
        obs_space = _make_dict_obs_space()
        branch_ext = ImpalaBranchCombinedExtractor(
            obs_space, cnn_output_dim=256, normalized_image=True,
        )
        standalone_space = spaces.Box(low=0, high=255, shape=(4, 128, 128), dtype=np.uint8)
        standalone_cnn = ImpalaCNN(standalone_space, features_dim=256, normalized_image=True)

        branch_cnn_keys = set(branch_ext.extractors["image"].state_dict().keys())
        standalone_keys = set(standalone_cnn.state_dict().keys())

        assert branch_cnn_keys == standalone_keys, (
            f"Key mismatch.\n"
            f"Only in branch CNN: {branch_cnn_keys - standalone_keys}\n"
            f"Only in standalone: {standalone_keys - branch_cnn_keys}"
        )

    def test_pretrained_weights_load_into_cnn_only(self):
        """Loading pretrained encoder weights updates CNN but not physics branch."""
        obs_space = _make_dict_obs_space()
        extractor = ImpalaBranchCombinedExtractor(
            obs_space, cnn_output_dim=256, normalized_image=True,
        )

        # Snapshot physics branch weights before loading
        branch_before = {k: v.clone() for k, v in extractor.physics_branch.state_dict().items()}

        # Create a pretrained state_dict (from standalone CNN)
        standalone_space = spaces.Box(low=0, high=255, shape=(4, 128, 128), dtype=np.uint8)
        pretrained = ImpalaCNN(standalone_space, features_dim=256, normalized_image=True)

        # Load into the CNN sub-module (same path as lwp/rl/training.py:_find_cnn)
        extractor.extractors["image"].load_state_dict(pretrained.state_dict())

        # Physics branch should be unchanged
        for key in branch_before:
            torch.testing.assert_close(
                extractor.physics_branch.state_dict()[key],
                branch_before[key],
                msg=f"Physics branch weight {key} changed after CNN weight loading"
            )


class TestEncoderFreezeCompat:

    def test_freezing_cnn_leaves_branch_trainable(self):
        """Freezing extractors['image'] params doesn't freeze physics branch."""
        obs_space = _make_dict_obs_space()
        extractor = ImpalaBranchCombinedExtractor(
            obs_space, cnn_output_dim=256, normalized_image=True,
        )

        # Freeze CNN (same logic as lwp/rl/training.py:397-401)
        for param in extractor.extractors["image"].parameters():
            param.requires_grad_(False)

        # CNN should be frozen
        cnn_trainable = [p for p in extractor.extractors["image"].parameters() if p.requires_grad]
        assert len(cnn_trainable) == 0, "CNN should be fully frozen"

        # Physics branch should still be trainable
        branch_trainable = [p for p in extractor.physics_branch.parameters() if p.requires_grad]
        assert len(branch_trainable) > 0, "Physics branch should remain trainable"
