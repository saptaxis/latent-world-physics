import torch
from lwp.models.mlp import MLPModel
from lwp.data.normalization import NormStats
from lwp.utils.config import RunConfig
from lwp.utils.checkpoint import save_checkpoint, load_checkpoint


def test_save_load_roundtrip(tmp_path):
    model = MLPModel(state_dim=8, action_dim=2, hidden_dims=[32])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    norm_stats = NormStats(
        state_mean=torch.randn(8), state_std=torch.rand(8) + 0.1,
        delta_mean=torch.randn(8), delta_std=torch.rand(8) + 0.1,
    )
    config = RunConfig(arch="mlp", arch_params={"hidden_dims": [32]},
                       data_path="/data", state_dim=8, action_dim=2)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, optimizer, norm_stats, config, epoch=5,
                    metrics={"val_loss": 0.01})
    loaded = load_checkpoint(path)
    assert loaded["epoch"] == 5
    assert loaded["metrics"]["val_loss"] == 0.01
    assert loaded["config"].arch == "mlp"

    # Model weights should match
    model2 = MLPModel(state_dim=8, action_dim=2, hidden_dims=[32])
    model2.load_state_dict(loaded["model_state_dict"])
    obs = torch.randn(2, 8)
    act = torch.randn(2, 2)
    model.eval(); model2.eval()
    with torch.no_grad():
        d1, _ = model.step(obs, act)
        d2, _ = model2.step(obs, act)
    assert torch.allclose(d1, d2)


def test_norm_stats_preserved(tmp_path):
    norm_stats = NormStats(
        state_mean=torch.tensor([1.0, 2.0]),
        state_std=torch.tensor([0.5, 0.3]),
        delta_mean=torch.tensor([0.01, 0.02]),
        delta_std=torch.tensor([0.1, 0.2]),
    )
    model = MLPModel(state_dim=2, action_dim=2, hidden_dims=[8])
    optimizer = torch.optim.Adam(model.parameters())
    config = RunConfig(arch="mlp", data_path="/data", state_dim=2)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, optimizer, norm_stats, config, epoch=0)
    loaded = load_checkpoint(path)
    ns = NormStats.from_dict(loaded["norm_stats"])
    assert torch.allclose(ns.state_mean, norm_stats.state_mean)
    assert torch.allclose(ns.delta_std, norm_stats.delta_std)
