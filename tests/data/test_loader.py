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


def test_force_target_slicing(episode_dir):
    """Dataset with force_target_indices returns 3D deltas."""
    ds = EpisodeDataset(
        str(episode_dir), state_dim=6, mode="single_step",
        force_target_indices=[2, 3, 5],
    )
    state, action, delta = ds[0]
    assert state.shape == (6,)
    assert delta.shape == (3,)


def test_force_target_values_match_slice(episode_dir):
    """3D deltas should be the [vx, vy, ang_vel] slice of 6D deltas."""
    ds_6d = EpisodeDataset(str(episode_dir), state_dim=6, mode="single_step")
    ds_3d = EpisodeDataset(
        str(episode_dir), state_dim=6, mode="single_step",
        force_target_indices=[2, 3, 5],
    )
    _, _, delta_6d = ds_6d[0]
    _, _, delta_3d = ds_3d[0]
    torch.testing.assert_close(delta_3d[0], delta_6d[2])  # vx
    torch.testing.assert_close(delta_3d[1], delta_6d[3])  # vy
    torch.testing.assert_close(delta_3d[2], delta_6d[5])  # ang_vel
