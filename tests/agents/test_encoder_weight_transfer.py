"""Test that standalone encoder weights load into SB3's ImpalaCNN.

This is the critical integration point: pre-trained weights from
StandaloneImpalaCNN must load into SB3's ImpalaCNN (which inherits
from BaseFeaturesExtractor). The state_dict keys and tensor shapes
must match exactly.
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pytest
import torch

from lwp.agents.visual_backbones import ImpalaCNN
from scripts.perception.pretrain_encoder import StandaloneImpalaCNN


class TestEncoderWeightTransfer:

    def test_state_dict_keys_match(self):
        """Standalone and SB3 ImpalaCNN have identical state_dict keys."""
        # Standalone
        standalone = StandaloneImpalaCNN(in_channels=4, features_dim=256)

        # SB3 — needs a Box observation space
        obs_space = spaces.Box(
            low=0, high=255, shape=(4, 128, 128), dtype=np.uint8
        )
        sb3_cnn = ImpalaCNN(obs_space, features_dim=256, normalized_image=True)

        standalone_keys = set(standalone.state_dict().keys())
        sb3_keys = set(sb3_cnn.state_dict().keys())

        assert standalone_keys == sb3_keys, (
            f"Key mismatch.\n"
            f"Only in standalone: {standalone_keys - sb3_keys}\n"
            f"Only in SB3: {sb3_keys - standalone_keys}"
        )

    def test_state_dict_shapes_match(self):
        """All corresponding tensors have the same shape."""
        standalone = StandaloneImpalaCNN(in_channels=4, features_dim=256)
        obs_space = spaces.Box(
            low=0, high=255, shape=(4, 128, 128), dtype=np.uint8
        )
        sb3_cnn = ImpalaCNN(obs_space, features_dim=256, normalized_image=True)

        for key in standalone.state_dict():
            s_shape = standalone.state_dict()[key].shape
            sb3_shape = sb3_cnn.state_dict()[key].shape
            assert s_shape == sb3_shape, (
                f"Shape mismatch for {key}: standalone={s_shape}, sb3={sb3_shape}"
            )

    def test_weight_transfer_produces_same_output(self):
        """Loading standalone weights into SB3 CNN produces identical outputs."""
        standalone = StandaloneImpalaCNN(in_channels=4, features_dim=256)
        obs_space = spaces.Box(
            low=0, high=255, shape=(4, 128, 128), dtype=np.uint8
        )
        sb3_cnn = ImpalaCNN(obs_space, features_dim=256, normalized_image=True)

        # Transfer weights
        sb3_cnn.load_state_dict(standalone.state_dict())

        # Same input, same output
        dummy = torch.zeros(1, 4, 128, 128)
        with torch.no_grad():
            out_standalone = standalone(dummy)
            out_sb3 = sb3_cnn(dummy)
        torch.testing.assert_close(out_standalone, out_sb3)

    def test_weight_transfer_with_custom_channels(self):
        """Weight transfer works with non-default channel widths."""
        channels = [32, 64, 64]
        standalone = StandaloneImpalaCNN(
            in_channels=4, features_dim=256, channels=channels, pool_size=8
        )
        obs_space = spaces.Box(
            low=0, high=255, shape=(4, 128, 128), dtype=np.uint8
        )
        sb3_cnn = ImpalaCNN(
            obs_space, features_dim=256, normalized_image=True,
            channels=channels, pool_size=8,
        )

        sb3_cnn.load_state_dict(standalone.state_dict())

        dummy = torch.zeros(1, 4, 128, 128)
        with torch.no_grad():
            out_standalone = standalone(dummy)
            out_sb3 = sb3_cnn(dummy)
        torch.testing.assert_close(out_standalone, out_sb3)

    def test_save_load_roundtrip(self, tmp_path):
        """Save standalone weights to file, load into SB3 CNN."""
        standalone = StandaloneImpalaCNN(in_channels=4, features_dim=256)

        # Save
        path = tmp_path / "encoder.pt"
        torch.save(standalone.state_dict(), path)

        # Load into SB3
        obs_space = spaces.Box(
            low=0, high=255, shape=(4, 128, 128), dtype=np.uint8
        )
        sb3_cnn = ImpalaCNN(obs_space, features_dim=256, normalized_image=True)
        sb3_cnn.load_state_dict(torch.load(path, weights_only=True))

        dummy = torch.zeros(1, 4, 128, 128)
        with torch.no_grad():
            out_standalone = standalone(dummy)
            out_sb3 = sb3_cnn(dummy)
        torch.testing.assert_close(out_standalone, out_sb3)
