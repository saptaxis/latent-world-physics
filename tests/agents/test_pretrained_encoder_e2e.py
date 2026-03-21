"""End-to-end smoke test: collect → pretrain → load into PPO."""
import json
import numpy as np
import pytest
import torch
from pathlib import Path
from gymnasium import spaces

from lwp.agents.encoder_dataset import EncoderPretrainDataset
from scripts.perception.pretrain_encoder import (
    build_model,
    train_one_epoch,
    evaluate,
    StandaloneImpalaCNN,
)
from lwp.agents.visual_backbones import ImpalaCNN


class TestEndToEnd:

    @pytest.fixture
    def mini_dataset(self, tmp_path):
        """Create 5 tiny episodes mimicking collected data."""
        data_dir = tmp_path / "episodes"
        data_dir.mkdir()
        for i in range(5):
            n_steps = 10
            np.savez_compressed(
                str(data_dir / f"ep_{i:04d}.npz"),
                states=np.random.randn(n_steps + 1, 15).astype(np.float32),
                actions=np.random.randn(n_steps, 2).astype(np.float32),
                rewards=np.random.randn(n_steps).astype(np.float32),
                dones=np.zeros(n_steps, dtype=bool),
                rgb_frames=np.random.randint(
                    0, 255, (n_steps + 1, 64, 96, 3), dtype=np.uint8
                ),
                metadata_json=json.dumps({
                    "physics_config": {"gravity": -10.0},
                    "outcome": "crashed",
                    "seed": i,
                }),
            )
        return data_dir

    def test_pretrain_then_load_into_sb3(self, mini_dataset, tmp_path):
        """Full pipeline: dataset → train → save → load into SB3 ImpalaCNN."""
        # Phase 2: Pre-train
        ds = EncoderPretrainDataset(
            mini_dataset, n_stack=4, frame_size=64
        )
        loader = torch.utils.data.DataLoader(ds, batch_size=8)
        encoder, head = build_model(
            n_stack=4, frame_size=64, features_dim=256, n_targets=6
        )

        optimizer = torch.optim.Adam(
            list(encoder.parameters()) + list(head.parameters()), lr=1e-3
        )

        # Train 2 epochs
        for _ in range(2):
            train_one_epoch(encoder, head, loader, optimizer, device="cpu")

        # Evaluate
        metrics = evaluate(encoder, head, loader, device="cpu")
        assert "loss" in metrics
        assert "r2" in metrics

        # Save encoder
        encoder_path = tmp_path / "encoder.pt"
        torch.save(encoder.state_dict(), encoder_path)

        # Phase 3: Load into SB3 ImpalaCNN
        obs_space = spaces.Box(
            low=0, high=255, shape=(4, 64, 64), dtype=np.uint8
        )
        sb3_cnn = ImpalaCNN(obs_space, features_dim=256, normalized_image=True)
        sb3_cnn.load_state_dict(
            torch.load(encoder_path, weights_only=True)
        )

        # Verify it produces features
        dummy = torch.zeros(1, 4, 64, 64)
        with torch.no_grad():
            features = sb3_cnn(dummy)
        assert features.shape == (1, 256)

    def test_freeze_encoder_grad_flow(self, mini_dataset, tmp_path):
        """When encoder is frozen, only head params have gradients."""
        encoder, head = build_model(
            n_stack=4, frame_size=64, features_dim=256, n_targets=6
        )

        # Freeze encoder
        for param in encoder.parameters():
            param.requires_grad_(False)

        # Forward + backward
        ds = EncoderPretrainDataset(mini_dataset, n_stack=4, frame_size=64)
        frames, targets = ds[0]
        frames = frames.unsqueeze(0)
        targets = targets.unsqueeze(0)

        features = encoder(frames)
        preds = head(features)
        loss = torch.nn.functional.mse_loss(preds, targets)
        loss.backward()

        # Encoder params should have no grad
        for param in encoder.parameters():
            assert param.grad is None

        # Head params should have grad
        for param in head.parameters():
            assert param.grad is not None
