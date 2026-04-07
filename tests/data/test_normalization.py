import torch
from lwp.data.normalization import NormStats, compute_norm_stats, normalize, denormalize


def test_normalize_denormalize_roundtrip():
    x = torch.randn(100, 8)
    mean = x.mean(dim=0)
    std = x.std(dim=0)
    x_n = normalize(x, mean, std)
    x_recovered = denormalize(x_n, mean, std)
    assert torch.allclose(x, x_recovered, atol=1e-5)


def test_normalize_zero_mean_unit_var():
    x = torch.randn(10000, 4)
    mean = x.mean(dim=0)
    std = x.std(dim=0)
    x_n = normalize(x, mean, std)
    assert torch.allclose(x_n.mean(dim=0), torch.zeros(4), atol=0.05)
    assert torch.allclose(x_n.std(dim=0), torch.ones(4), atol=0.05)


def test_compute_norm_stats():
    """Stats from list of (states, actions) episode tensors."""
    episodes = []
    rng = torch.Generator().manual_seed(0)
    for _ in range(5):
        s = torch.randn(50, 8, generator=rng) * 3 + 1  # non-zero mean, non-unit std
        a = torch.randn(49, 2, generator=rng)
        d = s[1:] - s[:-1]
        episodes.append({"states": s, "actions": a, "deltas": d})
    stats = compute_norm_stats(episodes)
    assert stats.state_mean.shape == (8,)
    assert stats.state_std.shape == (8,)
    assert stats.delta_mean.shape == (8,)
    assert stats.delta_std.shape == (8,)
    # Std should be positive
    assert (stats.state_std > 0).all()
    assert (stats.delta_std > 0).all()


def test_norm_stats_serialization():
    stats = NormStats(
        state_mean=torch.randn(8),
        state_std=torch.rand(8) + 0.1,
        delta_mean=torch.randn(8),
        delta_std=torch.rand(8) + 0.1,
    )
    d = stats.to_dict()
    restored = NormStats.from_dict(d)
    assert torch.allclose(stats.state_mean, restored.state_mean)
    assert torch.allclose(stats.delta_std, restored.delta_std)


def test_norm_stats_3d_delta(episode_dir):
    """NormStats with 3D deltas should have delta_mean/std of shape (3,)."""
    from lwp.data.loader import EpisodeDataset
    ds = EpisodeDataset(
        str(episode_dir), state_dim=6, mode="single_step",
        force_target_indices=[2, 3, 5],
    )
    ns = compute_norm_stats(ds.episode_dicts())
    assert ns.state_mean.shape == (6,)
    assert ns.delta_mean.shape == (3,)
    assert ns.delta_std.shape == (3,)
