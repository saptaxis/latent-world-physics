import pytest
import torch
from lwp.models.base import WorldModel


def test_base_class_initial_state():
    """Stateless default returns None."""
    model = WorldModel()
    assert model.initial_state(batch_size=4) is None


def test_base_class_step_raises():
    model = WorldModel()
    obs = torch.randn(4, 8)
    action = torch.randn(4, 2)
    try:
        model.step(obs, action)
        assert False, "Should raise NotImplementedError"
    except NotImplementedError:
        pass


from lwp.models.linear import LinearModel


def test_linear_output_shape():
    model = LinearModel(state_dim=8, action_dim=2)
    obs = torch.randn(4, 8)
    action = torch.randn(4, 2)
    delta, ms = model.step(obs, action)
    assert delta.shape == (4, 8)
    assert ms is None


def test_linear_is_world_model():
    model = LinearModel(state_dim=8, action_dim=2)
    assert isinstance(model, WorldModel)


def test_linear_gradient_flows():
    model = LinearModel(state_dim=8, action_dim=2)
    obs = torch.randn(4, 8)
    action = torch.randn(4, 2)
    delta, _ = model.step(obs, action)
    loss = delta.pow(2).mean()
    loss.backward()
    assert model.linear.weight.grad is not None
    assert model.linear.weight.grad.abs().sum() > 0


from lwp.models.mlp import MLPModel


def test_mlp_output_shape():
    model = MLPModel(state_dim=8, action_dim=2, hidden_dims=[64, 64])
    obs = torch.randn(4, 8)
    action = torch.randn(4, 2)
    delta, ms = model.step(obs, action)
    assert delta.shape == (4, 8)
    assert ms is None


def test_mlp_is_world_model():
    model = MLPModel(state_dim=8, action_dim=2, hidden_dims=[32])
    assert isinstance(model, WorldModel)


def test_mlp_different_architectures():
    """Various hidden_dims configs produce correct output."""
    for dims in [[64], [128, 128], [256, 256, 256]]:
        model = MLPModel(state_dim=8, action_dim=2, hidden_dims=dims)
        delta, _ = model.step(torch.randn(2, 8), torch.randn(2, 2))
        assert delta.shape == (2, 8)


def test_mlp_gradient_flows_through_all_layers():
    model = MLPModel(state_dim=8, action_dim=2, hidden_dims=[32, 32])
    obs = torch.randn(4, 8)
    action = torch.randn(4, 2)
    delta, _ = model.step(obs, action)
    loss = delta.pow(2).mean()
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"


from lwp.models.gru import GRUModel


def test_gru_output_shape():
    model = GRUModel(state_dim=8, action_dim=2, hidden_dim=32, num_layers=1,
                     encoder_dims=[16], decoder_dims=[16])
    obs = torch.randn(4, 8)
    action = torch.randn(4, 2)
    delta, ms = model.step(obs, action)
    assert delta.shape == (4, 8)
    assert ms is not None
    assert ms.shape == (1, 4, 32)  # [num_layers, batch, hidden_dim]


def test_gru_is_world_model():
    model = GRUModel(state_dim=8, action_dim=2, hidden_dim=32)
    assert isinstance(model, WorldModel)


def test_gru_initial_state():
    model = GRUModel(state_dim=8, action_dim=2, hidden_dim=32, num_layers=2)
    ms = model.initial_state(batch_size=4)
    assert ms.shape == (2, 4, 32)
    assert (ms == 0).all()


def test_gru_initial_state_device():
    model = GRUModel(state_dim=8, action_dim=2, hidden_dim=32)
    ms = model.initial_state(batch_size=4, device="cpu")
    assert ms.device == torch.device("cpu")


def test_gru_hidden_state_evolves():
    """Consecutive steps should produce different hidden states."""
    model = GRUModel(state_dim=8, action_dim=2, hidden_dim=32)
    obs = torch.randn(4, 8)
    action = torch.randn(4, 2)
    _, ms1 = model.step(obs, action)
    _, ms2 = model.step(obs, action, ms1)
    assert not torch.allclose(ms1, ms2)


def test_gru_gradient_flows():
    model = GRUModel(state_dim=8, action_dim=2, hidden_dim=32,
                     encoder_dims=[16], decoder_dims=[16])
    obs = torch.randn(4, 8)
    action = torch.randn(4, 2)
    delta, ms = model.step(obs, action)
    # Step again to test gradients through hidden state
    delta2, _ = model.step(obs, action, ms)
    loss = delta2.pow(2).mean()
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"


def test_gru_multi_layer():
    model = GRUModel(state_dim=8, action_dim=2, hidden_dim=64, num_layers=3,
                     encoder_dims=[32], decoder_dims=[32])
    obs = torch.randn(4, 8)
    action = torch.randn(4, 2)
    delta, ms = model.step(obs, action)
    assert delta.shape == (4, 8)
    assert ms.shape == (3, 4, 64)


def test_gru_default_encoder_decoder():
    """GRU with no encoder/decoder dims should still work (direct projection)."""
    model = GRUModel(state_dim=8, action_dim=2, hidden_dim=32)
    obs = torch.randn(4, 8)
    action = torch.randn(4, 2)
    delta, ms = model.step(obs, action)
    assert delta.shape == (4, 8)
    assert ms.shape == (1, 4, 32)


from lwp.models.factory import build_model
from lwp.utils.config import RunConfig


def test_build_linear():
    cfg = RunConfig(arch="linear", arch_params={}, data_path="/tmp",
                    state_dim=8, action_dim=2)
    model = build_model(cfg)
    assert isinstance(model, LinearModel)
    delta, _ = model.step(torch.randn(2, 8), torch.randn(2, 2))
    assert delta.shape == (2, 8)


def test_build_mlp():
    cfg = RunConfig(arch="mlp", arch_params={"hidden_dims": [64, 64]},
                    data_path="/tmp", state_dim=8, action_dim=2)
    model = build_model(cfg)
    assert isinstance(model, MLPModel)


def test_build_mlp_defaults():
    cfg = RunConfig(arch="mlp", data_path="/tmp", state_dim=8, action_dim=2)
    model = build_model(cfg)
    assert isinstance(model, MLPModel)


def test_build_gru():
    cfg = RunConfig(arch="gru", arch_params={"hidden_dim": 64, "num_layers": 1,
                    "encoder_dims": [32], "decoder_dims": [32]},
                    data_path="/tmp", state_dim=8, action_dim=2)
    model = build_model(cfg)
    assert isinstance(model, GRUModel)
    delta, ms = model.step(torch.randn(2, 8), torch.randn(2, 2))
    assert delta.shape == (2, 8)
    assert ms.shape == (1, 2, 64)


def test_build_gru_defaults():
    cfg = RunConfig(arch="gru", data_path="/tmp", state_dim=8, action_dim=2)
    model = build_model(cfg)
    assert isinstance(model, GRUModel)


from lwp.models.rssm import RSSMModel


def test_build_rssm():
    cfg = RunConfig(arch="rssm", arch_params={"deter_dim": 64, "stoch_dim": 16,
                    "hidden_dim": 32, "encoder_dims": [32]},
                    data_path="/tmp", state_dim=8, action_dim=2)
    model = build_model(cfg)
    assert isinstance(model, RSSMModel)
    delta, ms = model.step(torch.randn(2, 8), torch.randn(2, 2))
    assert delta.shape == (2, 8)


def test_build_rssm_defaults():
    cfg = RunConfig(arch="rssm", data_path="/tmp", state_dim=8, action_dim=2)
    model = build_model(cfg)
    assert isinstance(model, RSSMModel)


def test_build_unknown_arch():
    cfg = RunConfig(arch="transformer", data_path="/tmp")
    with pytest.raises(ValueError, match="Unknown architecture"):
        build_model(cfg)


from lwp.models.copy import CopyStateModel


def test_copy_state_output_shape():
    model = CopyStateModel(state_dim=8)
    obs = torch.randn(4, 8)
    action = torch.randn(4, 2)
    delta, ms = model.step(obs, action)
    assert delta.shape == (4, 8)
    assert ms is None


def test_copy_state_predicts_zero():
    model = CopyStateModel(state_dim=8)
    obs = torch.randn(4, 8)
    action = torch.randn(4, 2)
    delta, _ = model.step(obs, action)
    assert torch.allclose(delta, torch.zeros_like(delta))


def test_copy_state_is_world_model():
    from lwp.models.base import WorldModel
    model = CopyStateModel(state_dim=8)
    assert isinstance(model, WorldModel)


def test_build_copy():
    cfg = RunConfig(arch="copy", data_path="/tmp", state_dim=8, action_dim=2)
    model = build_model(cfg)
    assert isinstance(model, CopyStateModel)
