import torch
from torch.utils.data import DataLoader

from lwp.data.loader import EpisodeDataset
from lwp.data.normalization import NormStats, compute_norm_stats
from lwp.models.mlp import MLPModel
from lwp.training.loop import train_epoch, validate


def _make_norm_stats():
    return NormStats(
        state_mean=torch.zeros(8), state_std=torch.ones(8),
        delta_mean=torch.zeros(8), delta_std=torch.ones(8),
    )


def test_train_epoch_single_step(episode_dir):
    ds = EpisodeDataset(episode_dir, state_dim=8, mode="single_step")
    loader = DataLoader(ds, batch_size=16, shuffle=True)
    model = MLPModel(state_dim=8, action_dim=2, hidden_dims=[32])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    ns = _make_norm_stats()
    metrics = train_epoch(model, loader, optimizer, ns,
                          training_mode="single_step", rollout_k=1)
    assert "train_loss" in metrics
    assert metrics["train_loss"] > 0
    assert metrics["train_loss"] < 100  # sanity — not exploded


def test_train_epoch_multi_step(episode_dir):
    ds = EpisodeDataset(episode_dir, state_dim=8, mode="sequence", seq_len=10)
    loader = DataLoader(ds, batch_size=8, shuffle=True)
    model = MLPModel(state_dim=8, action_dim=2, hidden_dims=[32])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    ns = _make_norm_stats()
    metrics = train_epoch(model, loader, optimizer, ns,
                          training_mode="multi_step", rollout_k=5)
    assert "train_loss" in metrics
    assert metrics["train_loss"] > 0


def test_validate(episode_dir):
    ds = EpisodeDataset(episode_dir, state_dim=8, mode="single_step")
    loader = DataLoader(ds, batch_size=16)
    model = MLPModel(state_dim=8, action_dim=2, hidden_dims=[32])
    ns = _make_norm_stats()
    metrics = validate(model, loader, ns, training_mode="single_step", rollout_k=1)
    assert "val_loss" in metrics
    assert metrics["val_loss"] > 0


from lwp.models.gru import GRUModel


def test_train_epoch_gru_multi_step(episode_dir):
    ds = EpisodeDataset(episode_dir, state_dim=8, mode="sequence", seq_len=10)
    loader = DataLoader(ds, batch_size=8, shuffle=True, drop_last=True)
    model = GRUModel(state_dim=8, action_dim=2, hidden_dim=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    norm_stats = compute_norm_stats(ds.episode_dicts())
    metrics = train_epoch(model, loader, optimizer, norm_stats,
                          training_mode="multi_step", rollout_k=5)
    assert metrics["train_loss"] > 0
    assert metrics["train_loss"] < 1000


def test_train_epoch_scheduled_sampling(episode_dir):
    ds = EpisodeDataset(episode_dir, state_dim=8, mode="sequence", seq_len=10)
    loader = DataLoader(ds, batch_size=8, shuffle=True, drop_last=True)
    model = GRUModel(state_dim=8, action_dim=2, hidden_dim=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    norm_stats = compute_norm_stats(ds.episode_dicts())
    metrics = train_epoch(model, loader, optimizer, norm_stats,
                          training_mode="scheduled_sampling",
                          rollout_k=5, sampling_prob=0.3)
    assert metrics["train_loss"] > 0
    assert metrics["train_loss"] < 1000


from lwp.models.rssm import RSSMModel


def test_train_epoch_rssm_multi_step(episode_dir):
    ds = EpisodeDataset(episode_dir, state_dim=8, mode="sequence", seq_len=10)
    loader = DataLoader(ds, batch_size=8, shuffle=True, drop_last=True)
    model = RSSMModel(state_dim=8, action_dim=2, deter_dim=32, stoch_dim=8,
                      hidden_dim=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    norm_stats = compute_norm_stats(ds.episode_dicts())
    metrics = train_epoch(model, loader, optimizer, norm_stats,
                          training_mode="multi_step", rollout_k=5)
    assert metrics["train_loss"] > 0
    assert metrics["train_loss"] < 1000


def test_train_epoch_rssm_elbo(episode_dir):
    ds = EpisodeDataset(episode_dir, state_dim=8, mode="sequence", seq_len=10)
    loader = DataLoader(ds, batch_size=8, shuffle=True, drop_last=True)
    model = RSSMModel(state_dim=8, action_dim=2, deter_dim=32, stoch_dim=8,
                      hidden_dim=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    norm_stats = compute_norm_stats(ds.episode_dicts())
    metrics = train_epoch(model, loader, optimizer, norm_stats,
                          training_mode="elbo", rollout_k=5, kl_weight=0.5)
    assert metrics["train_loss"] > 0
    assert metrics["train_loss"] < 1000


import json
from lwp.training.profiler import ProfileLogger
from lwp.training.callbacks import (
    CallbackContext, GradNormCallback, NaNDetectionCallback,
)


def test_train_epoch_with_profiler(episode_dir, tmp_path):
    """ProfileLogger writes per-phase JSONL including per-callback breakdown."""
    from torch.utils.tensorboard import SummaryWriter

    ds = EpisodeDataset(episode_dir, state_dim=8, mode="single_step")
    loader = DataLoader(ds, batch_size=16, shuffle=True)
    model = MLPModel(state_dim=8, action_dim=2, hidden_dims=[32])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    ns = compute_norm_stats(ds.episode_dicts())

    tb_dir = str(tmp_path / "tb")
    writer = SummaryWriter(log_dir=tb_dir)
    ctx = CallbackContext(
        model=model, optimizer=optimizer, writer=writer,
        global_step=0, epoch=0, run_dir=str(tmp_path), device="cpu",
    )
    callbacks = [GradNormCallback(every_n_steps=1), NaNDetectionCallback()]

    log_path = tmp_path / "profile.jsonl"
    profiler = ProfileLogger(log_path)
    result = train_epoch(model, loader, optimizer, ns,
                         ctx=ctx, callbacks=callbacks, profiler=profiler)
    profiler.close()
    writer.close()

    # Parse the log
    lines = log_path.read_text().strip().split("\n")
    events = [json.loads(l) for l in lines]
    phases = {e["phase"] for e in events}

    # Core phases present
    assert "dataloader" in phases
    assert "data_to_device" in phases
    assert "forward" in phases
    assert "backward" in phases
    assert "optimizer" in phases
    assert "tb_logging" in phases
    # Per-callback breakdown
    assert "cb/GradNormCallback" in phases
    assert "cb/NaNDetectionCallback" in phases
    # Every event has step and epoch
    for e in events:
        assert "step" in e
        assert "epoch" in e
        assert "dur_ms" in e
        assert "ts" in e


def test_train_epoch_with_torch_profiler(episode_dir):
    """torch.profiler steps per batch when provided."""
    from unittest.mock import MagicMock

    ds = EpisodeDataset(episode_dir, state_dim=8, mode="single_step")
    loader = DataLoader(ds, batch_size=16, shuffle=True)
    model = MLPModel(state_dim=8, action_dim=2, hidden_dims=[32])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    ns = compute_norm_stats(ds.episode_dicts())

    # Mock torch profiler to verify .step() is called per batch
    mock_tp = MagicMock()
    metrics = train_epoch(model, loader, optimizer, ns,
                          training_mode="single_step", torch_profiler=mock_tp)
    assert mock_tp.step.call_count > 0
    assert metrics["train_loss"] > 0
