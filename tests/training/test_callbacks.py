"""Tests for training callback system."""
import os
import torch
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from lwp.data.loader import EpisodeDataset
from lwp.data.normalization import compute_norm_stats
from lwp.models.mlp import MLPModel
from lwp.models.gru import GRUModel
from lwp.training.loop import validate
from lwp.utils.config import RunConfig
from lwp.training.callbacks import (
    CallbackContext, TrainCallback, ValidationCallback, CheckpointCallback,
    PerDimLossCallback, RolloutMetricsCallback, GradNormCallback,
    PerTimestepLossCallback, HiddenStateHealthCallback, WarmupRolloutCallback,
    NaNDetectionCallback, ProgressCallback,
)


# ---------------------------------------------------------------------------
# Task 1: CallbackContext and TrainCallback base class
# ---------------------------------------------------------------------------

def test_callback_context_creation():
    model = MLPModel(state_dim=4, action_dim=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    ctx = CallbackContext(
        model=model,
        optimizer=optimizer,
        writer=None,
        global_step=0,
        epoch=0,
        run_dir="/tmp/test_run",
        device="cpu",
    )
    assert ctx.model is model
    assert ctx.optimizer is optimizer
    assert ctx.global_step == 0
    assert ctx.epoch == 0
    assert ctx.run_dir == "/tmp/test_run"
    assert ctx.device == "cpu"
    assert ctx.extras == {}


def test_callback_context_extras_mutable():
    model = MLPModel(state_dim=4, action_dim=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    ctx = CallbackContext(
        model=model, optimizer=optimizer, writer=None,
        global_step=0, epoch=0, run_dir="/tmp/test_run", device="cpu",
    )
    ctx.extras["foo"] = 42
    assert ctx.extras["foo"] == 42


def test_train_callback_defaults_return_true():
    cb = TrainCallback()
    model = MLPModel(state_dim=4, action_dim=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    ctx = CallbackContext(
        model=model, optimizer=optimizer, writer=None,
        global_step=0, epoch=0, run_dir="/tmp/test_run", device="cpu",
    )
    assert cb.on_step(ctx) is True
    assert cb.on_epoch_end(ctx) is True
    cb.on_train_start(ctx)  # should not raise
    cb.on_train_end(ctx)    # should not raise


def test_custom_callback_on_step():
    class Counter(TrainCallback):
        def __init__(self):
            self.count = 0

        def on_step(self, ctx):
            self.count += 1
            return True

    cb = Counter()
    model = MLPModel(state_dim=4, action_dim=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    ctx = CallbackContext(
        model=model, optimizer=optimizer, writer=None,
        global_step=0, epoch=0, run_dir="/tmp/test_run", device="cpu",
    )
    for _ in range(5):
        cb.on_step(ctx)
    assert cb.count == 5


def test_callback_can_signal_stop():
    class Stopper(TrainCallback):
        def on_step(self, ctx):
            return ctx.global_step < 3

    cb = Stopper()
    model = MLPModel(state_dim=4, action_dim=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    ctx = CallbackContext(
        model=model, optimizer=optimizer, writer=None,
        global_step=0, epoch=0, run_dir="/tmp/test_run", device="cpu",
    )
    assert cb.on_step(ctx) is True
    ctx.global_step = 3
    assert cb.on_step(ctx) is False


# ---------------------------------------------------------------------------
# Task 2: ValidationCallback with early stopping
# ---------------------------------------------------------------------------

def _make_loader_and_norm(episode_dir, mode="single_step", seq_len=None, split=None):
    ds = EpisodeDataset(episode_dir, state_dim=8, mode=mode, seq_len=seq_len, split=split)
    norm_stats = compute_norm_stats(ds.episode_dicts())
    loader = DataLoader(ds, batch_size=32, shuffle=False)
    return ds, loader, norm_stats


def test_validation_callback_runs_and_logs(tmp_path, episode_dir):
    _, val_loader, norm_stats = _make_loader_and_norm(episode_dir)
    model = MLPModel(state_dim=8, action_dim=2, hidden_dims=[32, 32])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    writer = SummaryWriter(log_dir=str(tmp_path / "tb"))
    config = RunConfig(arch="mlp", data_path=str(episode_dir))

    ckpt_dir = str(tmp_path / "checkpoints")
    cb = ValidationCallback(
        val_loader, norm_stats, every_n_steps=1, patience=100,
        checkpoint_dir=ckpt_dir,
    )
    ctx = CallbackContext(
        model=model, optimizer=optimizer, writer=writer,
        global_step=0, epoch=0, run_dir=str(tmp_path), device="cpu",
        extras={"config": config},
    )
    cb.on_train_start(ctx)

    # Step 0 should be skipped
    cb.on_step(ctx)
    assert "val_loss" not in ctx.extras

    # Step 1 should run validation
    ctx.global_step = 1
    cb.on_step(ctx)
    assert "val_loss" in ctx.extras
    assert isinstance(ctx.extras["val_loss"], float)
    assert (Path(ckpt_dir) / "best.pt").exists()

    writer.close()


def test_validation_callback_early_stops(tmp_path, episode_dir):
    _, val_loader, norm_stats = _make_loader_and_norm(episode_dir)
    model = MLPModel(state_dim=8, action_dim=2, hidden_dims=[32, 32])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    cb = ValidationCallback(
        val_loader, norm_stats, every_n_steps=1, patience=3,
    )
    ctx = CallbackContext(
        model=model, optimizer=optimizer, writer=None,
        global_step=0, epoch=0, run_dir=str(tmp_path), device="cpu",
    )
    cb.on_train_start(ctx)

    # Artificially set best_val_loss very low so every validation worsens
    cb.best_val_loss = -1e10

    stopped = False
    for step in range(1, 21):
        ctx.global_step = step
        cont = cb.on_step(ctx)
        if not cont:
            stopped = True
            break

    assert stopped is True
    assert cb.patience_counter >= 3


# ---------------------------------------------------------------------------
# Task 3: CheckpointCallback
# ---------------------------------------------------------------------------

def test_checkpoint_callback_saves_at_interval(tmp_path):
    model = MLPModel(state_dim=4, action_dim=2, hidden_dims=[16])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    ckpt_dir = str(tmp_path / "ckpts")

    cb = CheckpointCallback(checkpoint_dir=ckpt_dir, every_n_steps=10)
    ctx = CallbackContext(
        model=model, optimizer=optimizer, writer=None,
        global_step=0, epoch=0, run_dir=str(tmp_path), device="cpu",
    )
    cb.on_train_start(ctx)

    for step in [10, 15, 20]:
        ctx.global_step = step
        cb.on_step(ctx)

    assert (Path(ckpt_dir) / "step_00010.pt").exists()
    assert not (Path(ckpt_dir) / "step_00015.pt").exists()
    assert (Path(ckpt_dir) / "step_00020.pt").exists()


def test_checkpoint_callback_epoch_end_saves_latest(tmp_path):
    model = MLPModel(state_dim=4, action_dim=2, hidden_dims=[16])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    ckpt_dir = str(tmp_path / "ckpts")

    cb = CheckpointCallback(checkpoint_dir=ckpt_dir, every_n_steps=10000)
    ctx = CallbackContext(
        model=model, optimizer=optimizer, writer=None,
        global_step=5, epoch=1, run_dir=str(tmp_path), device="cpu",
    )
    cb.on_train_start(ctx)
    cb.on_epoch_end(ctx)

    assert (Path(ckpt_dir) / "latest.pt").exists()


# ---------------------------------------------------------------------------
# Task 4: PerDimLossCallback
# ---------------------------------------------------------------------------

def test_per_dim_loss_callback_logs(tmp_path, episode_dir):
    _, val_loader, norm_stats = _make_loader_and_norm(episode_dir)
    model = MLPModel(state_dim=8, action_dim=2, hidden_dims=[32, 32])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    writer = SummaryWriter(log_dir=str(tmp_path / "tb"))

    cb = PerDimLossCallback(val_loader, norm_stats, every_n_steps=1)
    ctx = CallbackContext(
        model=model, optimizer=optimizer, writer=writer,
        global_step=1, epoch=0, run_dir=str(tmp_path), device="cpu",
    )
    cb.on_step(ctx)

    assert "per_dim_mse" in ctx.extras
    assert len(ctx.extras["per_dim_mse"]) == 8

    writer.close()


def test_per_dim_loss_callback_no_writer(tmp_path, episode_dir):
    _, val_loader, norm_stats = _make_loader_and_norm(episode_dir)
    model = MLPModel(state_dim=8, action_dim=2, hidden_dims=[32, 32])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    cb = PerDimLossCallback(val_loader, norm_stats, every_n_steps=1)
    ctx = CallbackContext(
        model=model, optimizer=optimizer, writer=None,
        global_step=1, epoch=0, run_dir=str(tmp_path), device="cpu",
    )
    cb.on_step(ctx)

    assert "per_dim_mse" in ctx.extras
    assert len(ctx.extras["per_dim_mse"]) == 8


# ---------------------------------------------------------------------------
# Task 5: RolloutMetricsCallback
# ---------------------------------------------------------------------------

def test_rollout_metrics_callback(tmp_path, episode_dir):
    ds = EpisodeDataset(episode_dir, state_dim=8, mode="sequence", seq_len=10)
    norm_stats = compute_norm_stats(ds.episode_dicts())
    model = MLPModel(state_dim=8, action_dim=2, hidden_dims=[32, 32])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    cb = RolloutMetricsCallback(
        ds, norm_stats, horizons=[1, 5, 10],
        every_n_steps=1, n_rollouts=3,
    )
    ctx = CallbackContext(
        model=model, optimizer=optimizer, writer=None,
        global_step=1, epoch=0, run_dir=str(tmp_path), device="cpu",
    )
    cb.on_step(ctx)

    assert "horizon_errors" in ctx.extras
    assert set(ctx.extras["horizon_errors"].keys()) == {1, 5, 10}
    for h in [1, 5, 10]:
        assert isinstance(ctx.extras["horizon_errors"][h], float)


def test_rollout_metrics_callback_logs_cumulative(tmp_path, episode_dir):
    model = MLPModel(state_dim=8, action_dim=2, hidden_dims=[16])
    optimizer = torch.optim.Adam(model.parameters())
    writer = SummaryWriter(log_dir=str(tmp_path / "tb"))

    val_ds = EpisodeDataset(episode_dir, state_dim=8)
    norm_stats = compute_norm_stats(val_ds.episode_dicts())

    cb = RolloutMetricsCallback(
        dataset=val_ds,
        norm_stats=norm_stats,
        horizons=[1, 5],
        every_n_steps=1,
        n_rollouts=3,
    )

    ctx = CallbackContext(
        model=model, optimizer=optimizer, writer=writer,
        global_step=1, epoch=0, run_dir=str(tmp_path), device="cpu",
    )
    cb.on_step(ctx)
    assert "cumul_horizon_errors" in ctx.extras

    writer.close()


def test_rollout_metrics_callback_skip_step_0(tmp_path):
    # Minimal dataset not needed because step 0 should be skipped
    model = MLPModel(state_dim=4, action_dim=2, hidden_dims=[16])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    cb = RolloutMetricsCallback(
        dataset=None, norm_stats=None,
        horizons=[1, 5], every_n_steps=1, n_rollouts=3,
    )
    ctx = CallbackContext(
        model=model, optimizer=optimizer, writer=None,
        global_step=0, epoch=0, run_dir=str(tmp_path), device="cpu",
    )
    result = cb.on_step(ctx)
    assert result is True
    assert "horizon_errors" not in ctx.extras


# ---------------------------------------------------------------------------
# Task 6: GradNormCallback
# ---------------------------------------------------------------------------

def test_grad_norm_callback_logs_modules(tmp_path):
    model = GRUModel(state_dim=4, action_dim=2, hidden_dim=16, num_layers=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Forward + backward to populate gradients
    obs = torch.randn(2, 4)
    act = torch.randn(2, 2)
    delta, _ = model.step(obs, act)
    loss = delta.pow(2).mean()
    loss.backward()

    cb = GradNormCallback(every_n_steps=1)
    ctx = CallbackContext(
        model=model, optimizer=optimizer, writer=None,
        global_step=1, epoch=0, run_dir=str(tmp_path), device="cpu",
    )
    cb.on_step(ctx)

    assert "grad_norms" in ctx.extras
    norms = ctx.extras["grad_norms"]
    # GRUModel has encoder, gru, decoder modules
    assert "encoder" in norms
    assert "gru" in norms
    assert "decoder" in norms
    assert "total" in norms
    for v in norms.values():
        assert isinstance(v, float)
        assert v >= 0.0


def test_grad_norm_callback_no_gradients(tmp_path):
    model = GRUModel(state_dim=4, action_dim=2, hidden_dim=16, num_layers=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # No backward pass -- no gradients
    cb = GradNormCallback(every_n_steps=1)
    ctx = CallbackContext(
        model=model, optimizer=optimizer, writer=None,
        global_step=1, epoch=0, run_dir=str(tmp_path), device="cpu",
    )
    # Should not crash
    result = cb.on_step(ctx)
    assert result is True
    assert "grad_norms" in ctx.extras


# ---------------------------------------------------------------------------
# Task 8: PerTimestepLossCallback
# ---------------------------------------------------------------------------

def test_per_timestep_loss_callback(tmp_path, episode_dir):
    model = GRUModel(state_dim=8, action_dim=2, hidden_dim=16)
    optimizer = torch.optim.Adam(model.parameters())
    writer = SummaryWriter(log_dir=str(tmp_path / "tb"))

    val_ds = EpisodeDataset(episode_dir, state_dim=8, mode="sequence", seq_len=10)
    val_loader = DataLoader(val_ds, batch_size=8)
    norm_stats = compute_norm_stats(val_ds.episode_dicts())

    cb = PerTimestepLossCallback(
        val_loader=val_loader, norm_stats=norm_stats,
        every_n_steps=1, positions=[0, 4, 9],
    )

    ctx = CallbackContext(
        model=model, optimizer=optimizer, writer=writer,
        global_step=1, epoch=0, run_dir=str(tmp_path), device="cpu",
    )
    result = cb.on_step(ctx)
    assert result is True
    assert "per_timestep_mse" in ctx.extras
    assert 0 in ctx.extras["per_timestep_mse"]
    assert 4 in ctx.extras["per_timestep_mse"]
    writer.close()


def test_per_timestep_loss_callback_skip_step_0(tmp_path, episode_dir):
    model = GRUModel(state_dim=8, action_dim=2, hidden_dim=16)
    optimizer = torch.optim.Adam(model.parameters())

    val_ds = EpisodeDataset(episode_dir, state_dim=8, mode="sequence", seq_len=10)
    val_loader = DataLoader(val_ds, batch_size=8)
    norm_stats = compute_norm_stats(val_ds.episode_dicts())

    cb = PerTimestepLossCallback(val_loader=val_loader, norm_stats=norm_stats, every_n_steps=1)

    ctx = CallbackContext(
        model=model, optimizer=optimizer, writer=None,
        global_step=0, epoch=0, run_dir=str(tmp_path), device="cpu",
    )
    cb.on_step(ctx)
    assert "per_timestep_mse" not in ctx.extras


# ---------------------------------------------------------------------------
# Task 9: HiddenStateHealthCallback
# ---------------------------------------------------------------------------

def test_hidden_state_health_callback_gru(tmp_path, episode_dir):
    model = GRUModel(state_dim=8, action_dim=2, hidden_dim=16)
    optimizer = torch.optim.Adam(model.parameters())
    writer = SummaryWriter(log_dir=str(tmp_path / "tb"))

    val_ds = EpisodeDataset(episode_dir, state_dim=8, mode="sequence", seq_len=10)
    norm_stats = compute_norm_stats(val_ds.episode_dicts())

    cb = HiddenStateHealthCallback(dataset=val_ds, norm_stats=norm_stats, every_n_steps=1, n_episodes=3)

    ctx = CallbackContext(
        model=model, optimizer=optimizer, writer=writer,
        global_step=1, epoch=0, run_dir=str(tmp_path), device="cpu",
    )
    result = cb.on_step(ctx)
    assert result is True
    assert "hidden_health" in ctx.extras
    assert "magnitude" in ctx.extras["hidden_health"]
    assert "saturation" in ctx.extras["hidden_health"]
    assert "effective_dim" in ctx.extras["hidden_health"]
    writer.close()


def test_hidden_state_health_skips_stateless(tmp_path, episode_dir):
    model = MLPModel(state_dim=8, action_dim=2, hidden_dims=[16])
    optimizer = torch.optim.Adam(model.parameters())

    val_ds = EpisodeDataset(episode_dir, state_dim=8, mode="sequence", seq_len=10)
    norm_stats = compute_norm_stats(val_ds.episode_dicts())

    cb = HiddenStateHealthCallback(dataset=val_ds, norm_stats=norm_stats, every_n_steps=1)

    ctx = CallbackContext(
        model=model, optimizer=optimizer, writer=None,
        global_step=1, epoch=0, run_dir=str(tmp_path), device="cpu",
    )
    result = cb.on_step(ctx)
    assert result is True
    assert "hidden_health" not in ctx.extras


# ---------------------------------------------------------------------------
# Task 10: WarmupRolloutCallback
# ---------------------------------------------------------------------------

def test_warmup_rollout_callback_gru(tmp_path, episode_dir):
    model = GRUModel(state_dim=8, action_dim=2, hidden_dim=16)
    optimizer = torch.optim.Adam(model.parameters())
    writer = SummaryWriter(log_dir=str(tmp_path / "tb"))

    val_ds = EpisodeDataset(episode_dir, state_dim=8, mode="sequence", seq_len=10)
    norm_stats = compute_norm_stats(val_ds.episode_dicts())

    cb = WarmupRolloutCallback(
        dataset=val_ds, norm_stats=norm_stats,
        warmup_steps=5, horizons=[1, 5],
        every_n_steps=1, n_rollouts=3,
    )

    ctx = CallbackContext(
        model=model, optimizer=optimizer, writer=writer,
        global_step=1, epoch=0, run_dir=str(tmp_path), device="cpu",
    )
    result = cb.on_step(ctx)
    assert result is True
    assert "warmup_horizon_errors" in ctx.extras
    writer.close()


def test_warmup_rollout_skips_stateless(tmp_path, episode_dir):
    model = MLPModel(state_dim=8, action_dim=2, hidden_dims=[16])
    optimizer = torch.optim.Adam(model.parameters())

    val_ds = EpisodeDataset(episode_dir, state_dim=8, mode="sequence", seq_len=10)
    norm_stats = compute_norm_stats(val_ds.episode_dicts())

    cb = WarmupRolloutCallback(
        dataset=val_ds, norm_stats=norm_stats,
        warmup_steps=5, horizons=[1, 5],
        every_n_steps=1, n_rollouts=2,
    )

    ctx = CallbackContext(
        model=model, optimizer=optimizer, writer=None,
        global_step=1, epoch=0, run_dir=str(tmp_path), device="cpu",
    )
    result = cb.on_step(ctx)
    assert result is True
    assert "warmup_horizon_errors" not in ctx.extras


# ---------------------------------------------------------------------------
# Task 11: NaNDetectionCallback
# ---------------------------------------------------------------------------

def test_nan_detection_callback_passes_on_normal_loss(tmp_path):
    model = MLPModel(state_dim=8, action_dim=2, hidden_dims=[16])
    optimizer = torch.optim.Adam(model.parameters())
    cb = NaNDetectionCallback()
    ctx = CallbackContext(model=model, optimizer=optimizer, writer=None,
        global_step=1, epoch=0, run_dir=str(tmp_path), device="cpu")
    ctx.extras["train_loss_step"] = 0.5
    result = cb.on_step(ctx)
    assert result is True


def test_nan_detection_callback_stops_on_nan(tmp_path):
    model = MLPModel(state_dim=8, action_dim=2, hidden_dims=[16])
    optimizer = torch.optim.Adam(model.parameters())
    cb = NaNDetectionCallback()
    ctx = CallbackContext(model=model, optimizer=optimizer, writer=None,
        global_step=1, epoch=0, run_dir=str(tmp_path), device="cpu")
    ctx.extras["train_loss_step"] = float("nan")
    result = cb.on_step(ctx)
    assert result is False


def test_nan_detection_callback_stops_on_inf(tmp_path):
    model = MLPModel(state_dim=8, action_dim=2, hidden_dims=[16])
    optimizer = torch.optim.Adam(model.parameters())
    cb = NaNDetectionCallback()
    ctx = CallbackContext(model=model, optimizer=optimizer, writer=None,
        global_step=1, epoch=0, run_dir=str(tmp_path), device="cpu")
    ctx.extras["train_loss_step"] = float("inf")
    result = cb.on_step(ctx)
    assert result is False


# ---------------------------------------------------------------------------
# Task 12: ProgressCallback
# ---------------------------------------------------------------------------

def test_progress_callback_prints(tmp_path, capsys):
    model = MLPModel(state_dim=8, action_dim=2, hidden_dims=[16])
    optimizer = torch.optim.Adam(model.parameters())
    cb = ProgressCallback(every_n_steps=5, total_epochs=10)
    ctx = CallbackContext(model=model, optimizer=optimizer, writer=None,
        global_step=5, epoch=0, run_dir=str(tmp_path), device="cpu")
    ctx.extras["train_loss_step"] = 0.123
    ctx.extras["val_loss"] = 0.456
    cb.on_train_start(ctx)  # initialize start_time
    cb.on_step(ctx)
    captured = capsys.readouterr()
    assert "step=5" in captured.out
    assert "0.123" in captured.out


def test_progress_callback_skips_non_interval(tmp_path, capsys):
    model = MLPModel(state_dim=8, action_dim=2, hidden_dims=[16])
    optimizer = torch.optim.Adam(model.parameters())
    cb = ProgressCallback(every_n_steps=10, total_epochs=10)
    ctx = CallbackContext(model=model, optimizer=optimizer, writer=None,
        global_step=3, epoch=0, run_dir=str(tmp_path), device="cpu")
    ctx.extras["train_loss_step"] = 0.5
    cb.on_step(ctx)
    captured = capsys.readouterr()
    assert captured.out == ""
