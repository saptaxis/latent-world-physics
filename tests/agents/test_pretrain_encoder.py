"""Tests for encoder pre-training script."""
import json
import numpy as np
import pytest
import torch
from pathlib import Path

from lwp.agents.encoder_dataset import EncoderPretrainDataset
from scripts.perception.pretrain_encoder import (
    build_model,
    train_one_epoch,
    evaluate,
)


class TestPretrainEncoder:
    """Test the pre-training pipeline components."""

    @pytest.fixture
    def fake_data_dir(self, tmp_path):
        """Create 5 small fake episodes for training tests."""
        for i in range(5):
            n_steps = 15
            states = np.random.randn(n_steps + 1, 15).astype(np.float32)
            actions = np.random.randn(n_steps, 2).astype(np.float32)
            rewards = np.random.randn(n_steps).astype(np.float32)
            dones = np.zeros(n_steps, dtype=bool)
            dones[-1] = True
            rgb_frames = np.random.randint(
                0, 255, (n_steps + 1, 64, 96, 3), dtype=np.uint8
            )
            metadata = {
                "physics_config": {"gravity": -10.0},
                "outcome": "crashed",
                "seed": i,
            }
            np.savez_compressed(
                str(tmp_path / f"ep_{i:04d}.npz"),
                states=states,
                actions=actions,
                rewards=rewards,
                dones=dones,
                rgb_frames=rgb_frames,
                metadata_json=json.dumps(metadata),
            )
        return tmp_path

    def test_build_model(self):
        """build_model returns encoder + head with correct output dim."""
        encoder, head = build_model(
            n_stack=4, frame_size=128, features_dim=256, n_targets=6
        )
        # Encoder is ImpalaCNN-like (without SB3 BaseFeaturesExtractor)
        # Head maps features_dim → n_targets
        dummy = torch.zeros(1, 4, 128, 128)
        features = encoder(dummy)
        assert features.shape == (1, 256)
        predictions = head(features)
        assert predictions.shape == (1, 6)

    def test_train_one_epoch(self, fake_data_dir):
        """One training epoch runs without error and returns loss."""
        ds = EncoderPretrainDataset(
            fake_data_dir, n_stack=4, frame_size=64
        )
        loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=True)
        encoder, head = build_model(
            n_stack=4, frame_size=64, features_dim=256, n_targets=6
        )
        optimizer = torch.optim.Adam(
            list(encoder.parameters()) + list(head.parameters()), lr=1e-3
        )
        loss = train_one_epoch(encoder, head, loader, optimizer, device="cpu")
        assert isinstance(loss, float)
        assert loss > 0

    def test_evaluate_returns_metrics(self, fake_data_dir):
        """evaluate() returns loss and per-target R² dict."""
        ds = EncoderPretrainDataset(
            fake_data_dir, n_stack=4, frame_size=64
        )
        loader = torch.utils.data.DataLoader(ds, batch_size=8)
        encoder, head = build_model(
            n_stack=4, frame_size=64, features_dim=256, n_targets=6
        )
        metrics = evaluate(encoder, head, loader, device="cpu")
        assert "loss" in metrics
        assert "r2" in metrics
        # R² dict has 6 keys for the 6 kinematic targets
        assert len(metrics["r2"]) == 6
        for name in ["x", "y", "vx", "vy", "angle", "ang_vel"]:
            assert name in metrics["r2"]

    def test_saved_encoder_weights_loadable(self, fake_data_dir, tmp_path):
        """Saved encoder .pt file can be loaded into a fresh ImpalaCNN."""
        encoder, head = build_model(
            n_stack=4, frame_size=128, features_dim=256, n_targets=6
        )
        # Save just the encoder (no head)
        save_path = tmp_path / "encoder.pt"
        torch.save(encoder.state_dict(), save_path)

        # Load into a fresh encoder
        encoder2, _ = build_model(
            n_stack=4, frame_size=128, features_dim=256, n_targets=6
        )
        encoder2.load_state_dict(torch.load(save_path, weights_only=True))

        # Verify weights match
        dummy = torch.zeros(1, 4, 128, 128)
        with torch.no_grad():
            out1 = encoder(dummy)
            out2 = encoder2(dummy)
        torch.testing.assert_close(out1, out2)
