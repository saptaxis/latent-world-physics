import torch
from lwp.data.loader import EpisodeDataset, detect_dims


def test_load_episodes(episode_dir):
    ds = EpisodeDataset(episode_dir, state_dim=8)
    assert ds.n_episodes == 10
    assert ds.state_dim == 8
    assert ds.action_dim == 2


def test_single_step_mode(episode_dir):
    ds = EpisodeDataset(episode_dir, state_dim=8, mode="single_step")
    assert len(ds) > 0  # total transitions across all episodes
    s, a, delta = ds[0]
    assert s.shape == (8,)
    assert a.shape == (2,)
    assert delta.shape == (8,)


def test_sequence_mode(episode_dir):
    ds = EpisodeDataset(episode_dir, state_dim=8, mode="sequence", seq_len=10)
    assert len(ds) > 0
    states, actions = ds[0]
    assert states.shape == (11, 8)   # seq_len + 1 states
    assert actions.shape == (10, 2)  # seq_len actions


def test_train_val_split(episode_dir):
    train_ds = EpisodeDataset(episode_dir, state_dim=8, split="train", val_fraction=0.2)
    val_ds = EpisodeDataset(episode_dir, state_dim=8, split="val", val_fraction=0.2)
    assert train_ds.n_episodes == 8
    assert val_ds.n_episodes == 2
    assert train_ds.n_episodes + val_ds.n_episodes == 10


def test_episode_dicts_for_norm_stats(episode_dir):
    """Dataset exposes episode data as dicts for computing norm stats."""
    ds = EpisodeDataset(episode_dir, state_dim=8)
    ep_dicts = ds.episode_dicts()
    assert len(ep_dicts) == 10
    assert "states" in ep_dicts[0]
    assert "deltas" in ep_dicts[0]
    assert ep_dicts[0]["states"].shape[1] == 8
    assert ep_dicts[0]["deltas"].shape[1] == 8


def test_dataloader_integration(episode_dir):
    ds = EpisodeDataset(episode_dir, state_dim=8, mode="single_step")
    loader = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=True)
    batch = next(iter(loader))
    s, a, delta = batch
    assert s.shape == (16, 8)
    assert a.shape == (16, 2)
    assert delta.shape == (16, 8)


def test_detect_dims(episode_dir):
    state_dim, action_dim = detect_dims(episode_dir)
    assert state_dim == 8
    assert action_dim == 2


def test_detect_dims_missing_dir():
    import pytest
    with pytest.raises(FileNotFoundError):
        detect_dims("/nonexistent/path")
