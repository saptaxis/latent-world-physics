# tests/test_pixel_integration.py
"""Integration tests for PixelWorldModel (VAE + dynamics combined)."""
import torch
import pytest
from lwp.models.pixel_vae import PixelVAE
from lwp.models.pixel_dynamics import LatentDynamicsModel
from lwp.models.pixel_rssm import LatentRSSM
from lwp.models.pixel_world_model import PixelWorldModel
from lwp.training.pixel_loop import pixel_dynamics_train_epoch


# ---------------------------------------------------------------------------
# End-to-end training integration tests
# ---------------------------------------------------------------------------

COMBOS = [
    ("gru", "latent_mse", {}),
    ("gru", "multi_step_latent", {"rollout_k": 3}),
    ("rssm", "multi_step_latent", {"rollout_k": 3}),
    ("rssm", "latent_elbo", {"rollout_k": 3, "kl_weight": 1.0}),
]


@pytest.mark.parametrize("model_type,training_mode,kwargs", COMBOS,
                         ids=[f"{m}-{t}" for m, t, _ in COMBOS])
def test_e2e_training_combo(model_type, training_mode, kwargs):
    """End-to-end: train 2 steps with each valid (model, mode) combo."""
    vae = PixelVAE(in_channels=1, latent_dim=8, frame_size=64, channels=[8, 16, 32, 64])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    if model_type == "gru":
        dynamics = LatentDynamicsModel(latent_dim=8, action_dim=2, hidden_size=16)
    else:
        dynamics = LatentRSSM(latent_dim=8, action_dim=2, deter_dim=16, stoch_dim=4, hidden_dim=16)

    optimizer = torch.optim.Adam(dynamics.parameters(), lr=1e-3)
    frames = torch.rand(2, 6, 1, 64, 64)
    # latent_mse uses predict_sequence which expects T actions matching T frames;
    # rollout-based modes expect T-1 actions (transitions between frames)
    n_actions = 6 if training_mode == "latent_mse" else 5
    actions = torch.randn(2, n_actions, 2)
    # Two batches so we train for 2 steps
    loader = [(frames, actions), (frames, actions)]

    result = pixel_dynamics_train_epoch(
        dynamics, vae, loader, optimizer,
        training_mode=training_mode,
        device="cpu",
        **kwargs,
    )
    assert "train_loss" in result
    assert result["train_loss"] > 0


class TestPixelWorldModel:
    """End-to-end tests for combined pixel world model."""

    @pytest.fixture
    def model(self):
        vae = PixelVAE(in_channels=1, latent_dim=32, frame_size=84,
                       channels=[16, 32, 64, 128])
        dynamics = LatentDynamicsModel(latent_dim=32, action_dim=2, hidden_size=64)
        return PixelWorldModel(vae, dynamics)

    def test_predict_next_teacher_forced(self, model):
        """predict_next encodes real frame, predicts, decodes."""
        frame = torch.rand(2, 1, 84, 84)
        action = torch.randn(2, 2)
        pred_frame, z_next, hidden = model.predict_next(frame, action)
        assert pred_frame.shape == (2, 1, 84, 84)
        assert z_next.shape == (2, 32)

    def test_dream_autoregressive(self, model):
        """dream produces frame sequence from seed."""
        seed = torch.rand(2, 1, 84, 84)
        actions = torch.randn(2, 5, 2)
        frames = model.dream(seed, actions)
        assert frames.shape == (2, 6, 1, 84, 84)

    def test_dream_from_latent(self, model):
        """dream_from_latent operates purely in latent space."""
        z_seed = torch.randn(2, 32)
        actions = torch.randn(2, 5, 2)
        frames, z_seq = model.dream_from_latent(z_seed, actions)
        assert frames.shape == (2, 6, 1, 84, 84)
        assert z_seq.shape == (2, 6, 32)

    def test_encode_decode_roundtrip(self, model):
        """Encode then decode preserves shape."""
        frame = torch.rand(2, 1, 84, 84)
        z = model.encode(frame)
        recon = model.decode(z)
        assert recon.shape == frame.shape

    def test_4channel_stacked_dream(self):
        """Dream with 4-channel stacked frames uses frame buffer."""
        vae = PixelVAE(in_channels=4, latent_dim=32, frame_size=84,
                       channels=[16, 32, 64, 128])
        dynamics = LatentDynamicsModel(latent_dim=32, action_dim=2, hidden_size=64)
        model = PixelWorldModel(vae, dynamics)

        seed = torch.rand(1, 4, 84, 84)
        actions = torch.randn(1, 3, 2)
        frames = model.dream(seed, actions)
        assert frames.shape == (1, 4, 4, 84, 84)


def test_dream_with_rssm_dynamics():
    """PixelWorldModel.dream() works with LatentRSSM dynamics."""
    from lwp.models.pixel_rssm import LatentRSSM
    vae = PixelVAE(in_channels=1, latent_dim=16, frame_size=64, channels=[8, 16, 32, 64])
    rssm = LatentRSSM(latent_dim=16, action_dim=2, deter_dim=32, stoch_dim=8, hidden_dim=32)
    model = PixelWorldModel(vae, rssm)
    seed = torch.rand(1, 1, 64, 64)
    actions = torch.randn(1, 5, 2)
    frames = model.dream(seed, actions)
    assert frames.shape == (1, 6, 1, 64, 64)

def test_dream_from_latent_with_rssm():
    """dream_from_latent works with LatentRSSM."""
    from lwp.models.pixel_rssm import LatentRSSM
    vae = PixelVAE(in_channels=1, latent_dim=16, frame_size=64, channels=[8, 16, 32, 64])
    rssm = LatentRSSM(latent_dim=16, action_dim=2, deter_dim=32, stoch_dim=8, hidden_dim=32)
    model = PixelWorldModel(vae, rssm)
    z = torch.randn(1, 16)
    actions = torch.randn(1, 5, 2)
    frames, z_seq = model.dream_from_latent(z, actions)
    assert frames.shape == (1, 6, 1, 64, 64)
    assert z_seq.shape == (1, 6, 16)


class TestPixelSmokeTest:
    """Smoke test with a real Lunar Lander episode."""

    def test_vae_on_real_frame(self):
        """VAE can encode and decode a real Lunar Lander frame."""
        import numpy as np
        episode_path = "/media/hdd1/physics-priors-latent-space/lunar-lander-data/encoder-pretrain/random/episode_00000.npz"
        data = np.load(episode_path)
        frame = data["rgb_frames"][0]  # (400, 600, 3)

        import cv2
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(resized).float().unsqueeze(0).unsqueeze(0) / 255.0

        vae = PixelVAE(in_channels=1, latent_dim=64, frame_size=84)
        recon, mu, logvar, _ = vae(tensor)
        assert recon.shape == (1, 1, 84, 84)
        assert 0 <= recon.min() and recon.max() <= 1

    def test_dream_on_real_episode(self):
        """Dream 10 steps from a real episode seed."""
        import numpy as np
        import cv2
        episode_path = "/media/hdd1/physics-priors-latent-space/lunar-lander-data/encoder-pretrain/random/episode_00000.npz"
        data = np.load(episode_path)
        frame = data["rgb_frames"][0]
        actions = data["actions"][:10]

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        seed = torch.from_numpy(resized).float().unsqueeze(0).unsqueeze(0) / 255.0
        act_tensor = torch.from_numpy(actions).float().unsqueeze(0)

        vae = PixelVAE(in_channels=1, latent_dim=32, frame_size=84,
                       channels=[16, 32, 64, 128])
        dynamics = LatentDynamicsModel(latent_dim=32, action_dim=2, hidden_size=64)
        model = PixelWorldModel(vae, dynamics)

        frames = model.dream(seed, act_tensor)
        assert frames.shape == (1, 11, 1, 84, 84)
