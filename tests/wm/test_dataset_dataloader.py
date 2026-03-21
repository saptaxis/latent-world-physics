"""Tests for IterableDataset + DataLoader integration."""
import json
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from lwp.wm.dataset import EpisodeDataset, WMIterableDataset


class TestWMIterableDataset:

    @pytest.fixture
    def dummy_episodes(self, tmp_path):
        index = {}
        for i in range(10):
            ep_path = tmp_path / f"ep_{i:03d}.npz"
            T = 50
            np.savez(
                ep_path,
                states=np.random.randn(T + 1, 15).astype(np.float32),
                actions=np.random.randn(T, 2).astype(np.float32),
                rewards=np.random.randn(T).astype(np.float32),
            )
            index[str(ep_path)] = "train"
        index_path = tmp_path / "split_index.json"
        index_path.write_text(json.dumps(index))
        return index_path

    def test_single_step_context_yields_correct_shapes(self, dummy_episodes):
        ds = EpisodeDataset(dummy_episodes, "train")
        iterable = WMIterableDataset(ds, mode="context", context_k=5)
        loader = DataLoader(iterable, batch_size=8)
        batch = next(iter(loader))
        s, a, target, r, ctx = batch
        assert s.shape == (8, 15)
        assert a.shape == (8, 2)
        assert target.shape == (8, 15)
        assert ctx.shape == (8, 5, 32)

    def test_sequence_context_yields_correct_shapes(self, dummy_episodes):
        ds = EpisodeDataset(dummy_episodes, "train")
        iterable = WMIterableDataset(ds, mode="sequence_context", context_k=5, seq_len=10)
        loader = DataLoader(iterable, batch_size=4)
        batch = next(iter(loader))
        states, actions, ctx, rewards = batch
        assert states.shape == (4, 11, 15)
        assert actions.shape == (4, 10, 2)
        assert ctx.shape == (4, 5, 32)

    def test_works_with_num_workers(self, dummy_episodes):
        ds = EpisodeDataset(dummy_episodes, "train")
        iterable = WMIterableDataset(ds, mode="context", context_k=5)
        loader = DataLoader(iterable, batch_size=4, num_workers=2)
        batch = next(iter(loader))
        assert batch[0].shape[0] == 4
