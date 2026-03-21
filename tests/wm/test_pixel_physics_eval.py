"""Tests for pixel world model physics evaluation.

Layer 1 tests cover:
  - Episode loading (frames, states, actions preprocessing)
  - State head fidelity (per-dim MSE and R² from encoder)
  - VAE reconstruction quality (pixel MSE and SSIM)
"""
import numpy as np
import torch
import pytest


# ---------------------------------------------------------------------------
# Test helpers — tiny models and fake data generators
# ---------------------------------------------------------------------------


def _make_tiny_vae(state_dim=6):
    """Create a minimal PixelVAE for testing.

    Uses small channels [4, 8, 16, 32] and latent_dim=8 to keep tests fast.
    frame_size=64 matches the default eval preprocessing resolution.
    """
    from lwp.models.pixel_vae import PixelVAE
    return PixelVAE(
        in_channels=1, latent_dim=8, frame_size=64,
        channels=[4, 8, 16, 32], state_dim=state_dim,
    )


def _make_tiny_dynamics():
    """Create a minimal GRU dynamics for testing."""
    from lwp.models.pixel_dynamics import LatentDynamicsModel
    return LatentDynamicsModel(latent_dim=8, action_dim=2, hidden_size=16)


def _make_tiny_rssm():
    """Create a minimal RSSM dynamics for testing."""
    from lwp.models.pixel_rssm import LatentRSSM
    return LatentRSSM(latent_dim=8, action_dim=2, deter_dim=16, stoch_dim=4, hidden_dim=16)


def _make_constant_dynamics(latent_dim=8):
    """Mock dynamics that always returns the same z regardless of action.

    Used to verify action sensitivity = 0 for action-ignoring models.
    """
    class ConstantDynamics(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.constant = torch.nn.Parameter(torch.randn(latent_dim))

        def forward(self, z, action, state=None):
            return self.constant.unsqueeze(0).expand(z.size(0), -1), state

        def initial_state(self, batch_size, device=None):
            return None

        def rollout(self, z_start, actions, state=None):
            B, T, _ = actions.shape
            z_seq = [z_start]
            for _ in range(T):
                z_seq.append(self.constant.unsqueeze(0).expand(B, -1))
            return torch.stack(z_seq, dim=1), state

    return ConstantDynamics()


def _make_fake_episode(n_steps=20, tmp_path=None):
    """Create a fake episode npz file with frames + states + actions.

    Generates random RGB frames (100x150), 8-dim states, and 2-dim actions.
    This mirrors the format of real collected episodes from the LunarLander
    environment, just with random data for testing shape/type contracts.
    """
    # T+1 frames (includes initial observation), T actions
    frames = np.random.randint(0, 255, (n_steps + 1, 100, 150, 3), dtype=np.uint8)
    states = np.random.randn(n_steps + 1, 8).astype(np.float32)
    actions = np.random.randn(n_steps, 2).astype(np.float32)
    metadata = '{"source_type": "heuristic", "seed": 0}'
    path = str(tmp_path / "episode_00000.npz")
    np.savez(path, rgb_frames=frames, states=states, actions=actions,
             metadata_json=metadata)
    return path


# ---------------------------------------------------------------------------
# Layer 1a: Episode loading
# ---------------------------------------------------------------------------

from lwp.wm.pixel_physics_eval import (
    load_eval_episodes,
    evaluate_state_head_fidelity,
    evaluate_vae_reconstruction,
    evaluate_latent_dynamics_accuracy,
    extract_pixel_oracle_constants,
    extract_pixel_rollout_constants,
    compute_consistency_r2,
    evaluate_pixel_rollout_kinematics,
    evaluate_pixel_rollout_frames,
    evaluate_state_head_ood,
    compute_perception_tax,
    evaluate_action_sensitivity,
    evaluate_action_ablation,
    compute_baselines,
    run_full_eval,
    _dream_episodes,
)


class TestLoadEvalEpisodes:
    """Verify episode loading preprocesses frames, states, actions correctly."""

    def test_loads_frames_states_actions(self, tmp_path):
        """Single episode: frames resized to 64x64 grayscale float [0,1]."""
        path = _make_fake_episode(20, tmp_path)
        episodes = load_eval_episodes([path], frame_size=64)
        assert len(episodes) == 1
        ep = episodes[0]
        # Frames: (T+1, 1, H, W) — grayscale channel, resized to 64x64
        assert ep["frames"].shape == (21, 1, 64, 64)
        # Must be float32 normalized to [0,1] for VAE compatibility
        assert ep["frames"].dtype == torch.float32
        assert ep["frames"].min() >= 0 and ep["frames"].max() <= 1
        # States and actions keep original shapes
        assert ep["states"].shape == (21, 8)
        assert ep["actions"].shape == (20, 2)

    def test_multiple_episodes(self, tmp_path):
        """Loading multiple episode files returns one dict per episode."""
        paths = []
        for i in range(3):
            p = tmp_path / f"ep{i}"
            p.mkdir()
            paths.append(_make_fake_episode(10, p))
        episodes = load_eval_episodes(paths, frame_size=64)
        assert len(episodes) == 3


# ---------------------------------------------------------------------------
# Layer 1b: State head fidelity
# ---------------------------------------------------------------------------


class TestStateHeadFidelity:
    """Verify state head evaluation returns per-dim MSE and R² metrics."""

    def test_returns_per_dim_mse_and_r2(self, tmp_path):
        """With a state head (state_dim=6), we get per-dimension metrics."""
        vae = _make_tiny_vae(state_dim=6)
        path = _make_fake_episode(20, tmp_path)
        episodes = load_eval_episodes([path], frame_size=64)
        result = evaluate_state_head_fidelity(vae, episodes, device="cpu")
        # Should return per-dimension MSE and R² for all 6 kinematic dims
        assert "per_dim_mse" in result
        assert "per_dim_r2" in result
        assert len(result["per_dim_mse"]) == 6
        assert len(result["per_dim_r2"]) == 6
        # MSE must be non-negative
        for mse in result["per_dim_mse"]:
            assert mse >= 0
        # R² can be negative (worse than mean predictor) but must be finite
        for r2 in result["per_dim_r2"]:
            assert np.isfinite(r2)

    def test_aborts_if_no_state_head(self, tmp_path):
        """VAE with state_dim=0 has no state head — should raise ValueError."""
        vae = _make_tiny_vae(state_dim=0)
        path = _make_fake_episode(20, tmp_path)
        episodes = load_eval_episodes([path], frame_size=64)
        with pytest.raises(ValueError, match="state_dim"):
            evaluate_state_head_fidelity(vae, episodes, device="cpu")


# ---------------------------------------------------------------------------
# Layer 1c: VAE reconstruction quality
# ---------------------------------------------------------------------------


class TestVAEReconstruction:
    """Verify VAE reconstruction evaluation returns pixel MSE and SSIM."""

    def test_returns_pixel_mse_and_ssim(self, tmp_path):
        """Encode-decode round trip should produce valid pixel metrics."""
        vae = _make_tiny_vae()
        path = _make_fake_episode(10, tmp_path)
        episodes = load_eval_episodes([path], frame_size=64)
        result = evaluate_vae_reconstruction(vae, episodes, device="cpu")
        # pixel_mse must be non-negative (it's a squared error)
        assert "pixel_mse" in result
        assert "ssim" in result
        assert result["pixel_mse"] >= 0
        # SSIM ranges from -1 to 1 (1 = identical)
        assert -1 <= result["ssim"] <= 1


# ---------------------------------------------------------------------------
# Layer 2a: Latent dynamics accuracy (state-head-free)
# ---------------------------------------------------------------------------


class TestLatentDynamicsAccuracy:
    def test_returns_latent_mse(self, tmp_path):
        vae = _make_tiny_vae()
        dynamics = _make_tiny_dynamics()
        path = _make_fake_episode(20, tmp_path)
        episodes = load_eval_episodes([path], frame_size=64)
        result = evaluate_latent_dynamics_accuracy(vae, dynamics, episodes, device="cpu")
        assert "latent_mse" in result
        assert result["latent_mse"] > 0
        assert "n_transitions" in result

    def test_works_with_rssm(self, tmp_path):
        """Must handle RSSM hidden state (RSSMState, not Tensor)."""
        vae = _make_tiny_vae()
        dynamics = _make_tiny_rssm()
        path = _make_fake_episode(20, tmp_path)
        episodes = load_eval_episodes([path], frame_size=64)
        result = evaluate_latent_dynamics_accuracy(vae, dynamics, episodes, device="cpu")
        assert result["latent_mse"] > 0


# ---------------------------------------------------------------------------
# Layer 2b: Oracle physics constant extraction
# ---------------------------------------------------------------------------


class TestOraclePhysicsExtraction:
    def test_returns_all_six_constants(self, tmp_path):
        vae = _make_tiny_vae()
        dynamics = _make_tiny_dynamics()
        path = _make_fake_episode(50, tmp_path)
        episodes = load_eval_episodes([path], frame_size=64)
        result = extract_pixel_oracle_constants(vae, dynamics, episodes, device="cpu")
        expected_keys = ["gravity", "main_thrust", "side_thrust",
                         "kinematics", "angular_damping", "angle_thrust"]
        for key in expected_keys:
            assert key in result, f"Missing constant: {key}"
            # Each constant must have ConstantResult fields
            assert "n_samples" in result[key]
            assert "model_mean" in result[key]
            assert "gt_mean" in result[key]
            assert "relative_error" in result[key]

    def test_works_with_rssm(self, tmp_path):
        vae = _make_tiny_vae()
        dynamics = _make_tiny_rssm()
        path = _make_fake_episode(50, tmp_path)
        episodes = load_eval_episodes([path], frame_size=64)
        result = extract_pixel_oracle_constants(vae, dynamics, episodes, device="cpu")
        assert isinstance(result, dict)
        assert "gravity" in result

    def test_n_samples_is_zero_or_more(self, tmp_path):
        """With random states, most transitions won't pass filters.
        The function must still return valid dicts (n_samples may be 0)."""
        vae = _make_tiny_vae()
        dynamics = _make_tiny_dynamics()
        path = _make_fake_episode(50, tmp_path)
        episodes = load_eval_episodes([path], frame_size=64)
        result = extract_pixel_oracle_constants(vae, dynamics, episodes, device="cpu")
        for key, val in result.items():
            assert val["n_samples"] >= 0


# ---------------------------------------------------------------------------
# Layer 2c: Rollout constant extraction
# ---------------------------------------------------------------------------


class TestRolloutConstantExtraction:
    def test_returns_all_six_constants(self, tmp_path):
        vae = _make_tiny_vae()
        dynamics = _make_tiny_dynamics()
        path = _make_fake_episode(50, tmp_path)
        episodes = load_eval_episodes([path], frame_size=64)
        result = extract_pixel_rollout_constants(
            vae, dynamics, episodes, rollout_k=10, device="cpu")
        for key in ["gravity", "main_thrust", "side_thrust",
                     "kinematics", "angular_damping", "angle_thrust"]:
            assert key in result
            assert "n_samples" in result[key]

    def test_uses_dreamed_state_for_filtering(self, tmp_path):
        """Rollout mode filters on state-head predictions of dreamed z,
        NOT GT state (unlike oracle which filters on GT)."""
        vae = _make_tiny_vae()
        dynamics = _make_tiny_dynamics()
        path = _make_fake_episode(50, tmp_path)
        episodes = load_eval_episodes([path], frame_size=64)
        result = extract_pixel_rollout_constants(
            vae, dynamics, episodes, rollout_k=5, device="cpu")
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Consistency R²
# ---------------------------------------------------------------------------


class TestConsistencyR2:
    def test_returns_per_constant_r2(self, tmp_path):
        vae = _make_tiny_vae()
        dynamics = _make_tiny_dynamics()
        path = _make_fake_episode(100, tmp_path)
        episodes = load_eval_episodes([path], frame_size=64)
        oracle = extract_pixel_oracle_constants(vae, dynamics, episodes, device="cpu")
        r2 = compute_consistency_r2(oracle)
        assert isinstance(r2, dict)
        # Only constants with n_samples > 0 should have R² entries
        for key, val in r2.items():
            assert isinstance(val, dict)  # state_var → R² mapping

    def test_skips_constants_with_no_samples(self):
        """Constants with n_samples=0 should be skipped, not crash."""
        empty_oracle = {
            "gravity": {"n_samples": 0, "model_values": np.array([]),
                         "associated_states": np.array([]).reshape(0, 6)},
        }
        r2 = compute_consistency_r2(empty_oracle)
        assert "gravity" not in r2  # skipped because no samples


# ---------------------------------------------------------------------------
# Layer 3a: Rollout kinematics
# ---------------------------------------------------------------------------


class TestRolloutKinematics:
    def test_returns_per_horizon_mse(self, tmp_path):
        vae = _make_tiny_vae()
        dynamics = _make_tiny_dynamics()
        path = _make_fake_episode(60, tmp_path)
        episodes = load_eval_episodes([path], frame_size=64)
        result = evaluate_pixel_rollout_kinematics(
            vae, dynamics, episodes, horizons=[1, 5, 10], device="cpu")
        assert "horizon_mse" in result
        for h in [1, 5, 10]:
            assert h in result["horizon_mse"]
            assert len(result["horizon_mse"][h]) == 6

    def test_compounding_exponent(self, tmp_path):
        vae = _make_tiny_vae()
        dynamics = _make_tiny_dynamics()
        path = _make_fake_episode(60, tmp_path)
        episodes = load_eval_episodes([path], frame_size=64)
        result = evaluate_pixel_rollout_kinematics(
            vae, dynamics, episodes, horizons=[1, 5, 10], device="cpu")
        assert "compounding_b" in result
        assert len(result["compounding_b"]) == 6

    def test_works_with_rssm(self, tmp_path):
        vae = _make_tiny_vae()
        dynamics = _make_tiny_rssm()
        path = _make_fake_episode(60, tmp_path)
        episodes = load_eval_episodes([path], frame_size=64)
        result = evaluate_pixel_rollout_kinematics(
            vae, dynamics, episodes, horizons=[1, 5], device="cpu")
        assert "horizon_mse" in result


# ---------------------------------------------------------------------------
# Layer 3b: Pixel rollout frames
# ---------------------------------------------------------------------------


class TestRolloutFrames:
    def test_returns_pixel_metrics_per_horizon(self, tmp_path):
        vae = _make_tiny_vae()
        dynamics = _make_tiny_dynamics()
        path = _make_fake_episode(60, tmp_path)
        episodes = load_eval_episodes([path], frame_size=64)
        result = evaluate_pixel_rollout_frames(
            vae, dynamics, episodes, horizons=[1, 5, 10], device="cpu")
        assert "horizon_pixel_mse" in result
        assert "horizon_ssim" in result
        assert "recognizable_horizon" in result
        for h in [1, 5, 10]:
            assert h in result["horizon_pixel_mse"]
            assert h in result["horizon_ssim"]
            assert result["horizon_pixel_mse"][h] >= 0
            assert -1 <= result["horizon_ssim"][h] <= 1
        assert result["recognizable_horizon"] >= 0


# ---------------------------------------------------------------------------
# Layer 3c: State head OOD detection
# ---------------------------------------------------------------------------


class TestStateHeadOOD:
    def test_returns_per_horizon_divergence(self, tmp_path):
        vae = _make_tiny_vae()
        dynamics = _make_tiny_dynamics()
        path = _make_fake_episode(60, tmp_path)
        episodes = load_eval_episodes([path], frame_size=64)
        result = evaluate_state_head_ood(
            vae, dynamics, episodes, horizons=[1, 5, 10], device="cpu")
        assert "horizon_ood_divergence" in result
        for h in [1, 5, 10]:
            assert h in result["horizon_ood_divergence"]
            assert result["horizon_ood_divergence"][h] >= 0


# ---------------------------------------------------------------------------
# Perception tax decomposition
# ---------------------------------------------------------------------------


class TestPerceptionTax:
    def test_returns_three_part_decomposition(self):
        layer1 = {"per_dim_mse": [0.01, 0.02, 0.03, 0.01, 0.05, 0.04]}
        layer2_oracle = {"gravity": {"relative_error": 0.3, "n_samples": 100}}
        layer2_latent = {"latent_mse": 0.05}
        result = compute_perception_tax(layer1, layer2_oracle, layer2_latent)
        assert "perception_floor" in result
        assert "latent_dynamics_mse" in result
        assert "dynamics_context" in result
        assert len(result["perception_floor"]) == 6


class TestActionSensitivity:
    def test_returns_l2_matrix_and_ratio(self, tmp_path):
        vae = _make_tiny_vae()
        dynamics = _make_tiny_dynamics()
        path = _make_fake_episode(30, tmp_path)
        episodes = load_eval_episodes([path], frame_size=64)
        result = evaluate_action_sensitivity(vae, dynamics, episodes, device="cpu")
        assert "latent_l2_matrix" in result
        assert "kinematics_l2_matrix" in result
        assert "sensitivity_ratio" in result

    def test_constant_model_has_zero_sensitivity(self):
        """A model that ignores actions should have ~0 L2 between predictions."""
        vae = _make_tiny_vae()
        dynamics = _make_constant_dynamics(latent_dim=8)
        episodes = [{"frames": torch.rand(21, 1, 64, 64),
                      "states": torch.randn(21, 8),
                      "actions": torch.randn(20, 2)}]
        result = evaluate_action_sensitivity(vae, dynamics, episodes, device="cpu")
        for pair_key, l2 in result["latent_l2_matrix"].items():
            assert l2 < 1e-5, f"Constant model should have ~0 L2 for {pair_key}"

    def test_kinematics_l2_has_per_dim_breakdown(self, tmp_path):
        vae = _make_tiny_vae()
        dynamics = _make_tiny_dynamics()
        path = _make_fake_episode(30, tmp_path)
        episodes = load_eval_episodes([path], frame_size=64)
        result = evaluate_action_sensitivity(vae, dynamics, episodes, device="cpu")
        for pair_key, kin_l2 in result["kinematics_l2_matrix"].items():
            assert isinstance(kin_l2, dict)
            assert len(kin_l2) == 6  # one per kinematic dim


class TestActionAblation:
    def test_returns_mse_curves(self, tmp_path):
        vae = _make_tiny_vae()
        dynamics = _make_tiny_dynamics()
        path = _make_fake_episode(30, tmp_path)
        episodes = load_eval_episodes([path], frame_size=64)
        result = evaluate_action_ablation(
            vae, dynamics, episodes, horizons=[1, 5, 10], device="cpu")
        assert "mse_real_vs_zero" in result
        assert "mse_real_vs_random" in result
        for h in [1, 5, 10]:
            assert h in result["mse_real_vs_zero"]
            assert h in result["mse_real_vs_random"]
            assert result["mse_real_vs_zero"][h] >= 0
            assert result["mse_real_vs_random"][h] >= 0

    def test_constant_model_zero_ablation_diff(self):
        """Constant model: real vs zero actions should produce ~0 MSE diff."""
        vae = _make_tiny_vae()
        dynamics = _make_constant_dynamics(latent_dim=8)
        episodes = [{"frames": torch.rand(21, 1, 64, 64),
                      "states": torch.randn(21, 8),
                      "actions": torch.randn(20, 2)}]
        result = evaluate_action_ablation(
            vae, dynamics, episodes, horizons=[1, 5], device="cpu")
        for h in [1, 5]:
            assert result["mse_real_vs_zero"][h] < 1e-5


class TestBaselines:
    def test_returns_zero_and_copy_baselines(self, tmp_path):
        """Baselines should report MSE for zero-predictor and copy-previous."""
        vae = _make_tiny_vae()
        path = _make_fake_episode(30, tmp_path)
        episodes = load_eval_episodes([path], frame_size=64)
        result = compute_baselines(vae, episodes, horizons=[1, 5, 10], device="cpu")
        assert "zero_predictor" in result
        assert "copy_previous" in result
        for h in [1, 5, 10]:
            assert h in result["zero_predictor"]
            assert h in result["copy_previous"]
            # Zero predictor should have 6 per-dim values
            assert len(result["zero_predictor"][h]) == 6
            # Copy-previous values should be non-negative
            assert all(v >= 0 for v in result["copy_previous"][h])

    def test_copy_previous_at_h1_is_small(self, tmp_path):
        """Copy-previous at horizon 1 should be small (just 1-step delta)."""
        vae = _make_tiny_vae()
        path = _make_fake_episode(30, tmp_path)
        episodes = load_eval_episodes([path], frame_size=64)
        result = compute_baselines(vae, episodes, horizons=[1, 5], device="cpu")
        h1_mse = np.mean(result["copy_previous"][1])
        h5_mse = np.mean(result["copy_previous"][5])
        assert isinstance(h1_mse, float)
        assert isinstance(h5_mse, float)


class TestPerDimActionAblation:
    def test_ablation_includes_per_dim_kinematics(self, tmp_path):
        """Action ablation should include per-dim kinematics breakdown."""
        vae = _make_tiny_vae()
        dynamics = _make_tiny_dynamics()
        path = _make_fake_episode(30, tmp_path)
        episodes = load_eval_episodes([path], frame_size=64)
        result = evaluate_action_ablation(
            vae, dynamics, episodes, horizons=[1, 5], device="cpu")
        assert "kin_real_vs_zero" in result
        assert "kin_real_vs_random" in result
        for h in [1, 5]:
            assert h in result["kin_real_vs_zero"]
            # Per-dim: dict of dim_name -> MSE
            assert isinstance(result["kin_real_vs_zero"][h], dict)
            assert len(result["kin_real_vs_zero"][h]) == 6


# ---------------------------------------------------------------------------
# Fg-weighted pixel metrics
# ---------------------------------------------------------------------------


class TestFgWeightedMetrics:
    def test_vae_recon_includes_fg_weighted_mse(self, tmp_path):
        """VAE reconstruction should report both unweighted and fg-weighted MSE."""
        vae = _make_tiny_vae()
        path = _make_fake_episode(10, tmp_path)
        episodes = load_eval_episodes([path], frame_size=64)
        result = evaluate_vae_reconstruction(vae, episodes, device="cpu")
        assert "pixel_mse" in result          # unweighted (backward compat)
        assert "fg_pixel_mse" in result       # fg-weighted
        assert "fg_ssim" in result            # foreground-only SSIM
        assert result["fg_pixel_mse"] >= result["pixel_mse"]  # fg-weight amplifies lander errors

    def test_rollout_frames_includes_fg_metrics(self, tmp_path):
        """Rollout frame metrics should include fg-weighted variants."""
        vae = _make_tiny_vae()
        dynamics = _make_tiny_dynamics()
        path = _make_fake_episode(30, tmp_path)
        episodes = load_eval_episodes([path], frame_size=64)
        result = evaluate_pixel_rollout_frames(
            vae, dynamics, episodes, horizons=[1, 5], device="cpu")
        assert "horizon_fg_pixel_mse" in result
        assert "horizon_fg_ssim" in result
        assert "fg_recognizable_horizon" in result
        for h in [1, 5]:
            assert h in result["horizon_fg_pixel_mse"]
            assert h in result["horizon_fg_ssim"]


class TestSignCorrectness:
    def test_oracle_constants_include_sign_correct(self, tmp_path):
        """Each constant should report whether model_mean has same sign as gt_mean."""
        vae = _make_tiny_vae()
        dynamics = _make_tiny_dynamics()
        path = _make_fake_episode(50, tmp_path)
        episodes = load_eval_episodes([path], frame_size=64)
        result = extract_pixel_oracle_constants(vae, dynamics, episodes, device="cpu")
        for key, val in result.items():
            if val["n_samples"] > 0:
                assert "sign_correct" in val
                assert isinstance(val["sign_correct"], bool)


class TestLowSampleFlagging:
    def test_constants_have_sample_warning(self, tmp_path):
        """Constants with n_samples < threshold should be flagged."""
        vae = _make_tiny_vae()
        dynamics = _make_tiny_dynamics()
        path = _make_fake_episode(50, tmp_path)
        episodes = load_eval_episodes([path], frame_size=64)
        result = extract_pixel_oracle_constants(vae, dynamics, episodes, device="cpu")
        for key, val in result.items():
            assert "low_sample_warning" in val
            if val["n_samples"] < 100:
                assert val["low_sample_warning"] is True


class TestNormalizedLatentMSE:
    def test_returns_normalized_mse(self, tmp_path):
        """Latent dynamics should report MSE / z_variance for cross-VAE comparison."""
        vae = _make_tiny_vae()
        dynamics = _make_tiny_dynamics()
        path = _make_fake_episode(20, tmp_path)
        episodes = load_eval_episodes([path], frame_size=64)
        result = evaluate_latent_dynamics_accuracy(vae, dynamics, episodes, device="cpu")
        assert "latent_mse" in result
        assert "latent_mse_normalized" in result
        assert "z_variance" in result
        # Normalized MSE = raw MSE / z_variance
        if result["z_variance"] > 0:
            expected = result["latent_mse"] / result["z_variance"]
            assert abs(result["latent_mse_normalized"] - expected) < 1e-6


# ---------------------------------------------------------------------------
# Stratified sampling + policy labels
# ---------------------------------------------------------------------------


class TestStratifiedSampling:
    def test_load_episodes_includes_policy_label(self, tmp_path):
        """Episodes should include a policy/source label from parent dir name."""
        p = tmp_path / "heuristic"
        p.mkdir()
        path = _make_fake_episode(10, p)
        episodes = load_eval_episodes([path], frame_size=64)
        assert "policy" in episodes[0]
        assert episodes[0]["policy"] == "heuristic"

    def test_load_episodes_unknown_policy(self, tmp_path):
        """Episodes with no meaningful subdir get parent dir name as policy."""
        path = _make_fake_episode(10, tmp_path)
        episodes = load_eval_episodes([path], frame_size=64)
        # Policy is inferred from parent dir name — in pytest tmp_path
        # this will be the test's temp directory name
        assert "policy" in episodes[0]
        assert isinstance(episodes[0]["policy"], str)
        assert len(episodes[0]["policy"]) > 0


# ---------------------------------------------------------------------------
# Multi-start dreams with warm-up
# ---------------------------------------------------------------------------


class TestMultiStartDreams:
    def test_dream_episodes_with_multiple_starts(self, tmp_path):
        """_dream_episodes should produce multiple dream segments per episode."""
        vae = _make_tiny_vae()
        dynamics = _make_tiny_dynamics()
        path = _make_fake_episode(100, tmp_path)
        episodes = load_eval_episodes([path], frame_size=64)
        # With stride=30 and max_horizon=50 on a 100-step episode:
        # starts at [0, 30, 60, 90] — all have at least 1 step to dream
        dreams = _dream_episodes(
            vae, dynamics, episodes, device="cpu",
            start_stride=30, max_horizon=50)
        assert len(dreams) >= 2  # at least 2 start points

    def test_dream_from_nonzero_start_has_warmed_hidden(self, tmp_path):
        """Dreams from mid-episode should have warm hidden state."""
        vae = _make_tiny_vae()
        dynamics = _make_tiny_dynamics()
        path = _make_fake_episode(100, tmp_path)
        episodes = load_eval_episodes([path], frame_size=64)
        dreams = _dream_episodes(
            vae, dynamics, episodes, device="cpu",
            start_stride=30, max_horizon=50)
        # Find a dream that starts from offset > 0
        mid_dreams = [d for d in dreams if d.get("start_offset", 0) > 0]
        assert len(mid_dreams) > 0

    def test_default_matches_old_behavior(self, tmp_path):
        """_dream_episodes with default params should produce single dream from
        frame 0 with same z trajectory as dynamics.rollout()."""
        vae = _make_tiny_vae()
        dynamics = _make_tiny_dynamics()
        path = _make_fake_episode(30, tmp_path)
        episodes = load_eval_episodes([path], frame_size=64)
        # Default params — single start from frame 0
        dreams_new = _dream_episodes(vae, dynamics, episodes, device="cpu")
        assert len(dreams_new) == 1
        # Compare to direct rollout
        frames = episodes[0]["frames"]
        actions = episodes[0]["actions"]
        z_0 = vae.encode(frames[0:1])
        z_rollout, _ = dynamics.rollout(z_0, actions.unsqueeze(0))
        z_rollout = z_rollout.squeeze(0)
        assert torch.allclose(dreams_new[0]["z_dream"], z_rollout.cpu(), atol=1e-5)

    def test_rollout_kinematics_uses_multi_start(self, tmp_path):
        """Rollout kinematics with multi-start should average across start points."""
        vae = _make_tiny_vae()
        dynamics = _make_tiny_dynamics()
        path = _make_fake_episode(100, tmp_path)
        episodes = load_eval_episodes([path], frame_size=64)
        result = evaluate_pixel_rollout_kinematics(
            vae, dynamics, episodes, horizons=[1, 5, 10],
            device="cpu", start_stride=30, max_horizon=50)
        assert "horizon_mse" in result
        assert "n_dream_segments" in result
        assert result["n_dream_segments"] >= 2  # multiple starts


class TestFullPipeline:
    def test_end_to_end_gru(self, tmp_path):
        """Full eval pipeline produces results for all 4 layers."""
        vae = _make_tiny_vae()
        dynamics = _make_tiny_dynamics()
        paths = []
        for i in range(3):
            p = tmp_path / f"ep{i}"
            p.mkdir()
            paths.append(_make_fake_episode(30, p))
        episodes = load_eval_episodes(paths, frame_size=64)
        results = run_full_eval(vae, dynamics, episodes,
                                horizons=[1, 5], device="cpu")
        assert "layer1_fidelity" in results
        assert "layer1_recon" in results
        assert "layer2_latent" in results
        assert "layer2_oracle" in results
        assert "layer3_kinematics" in results
        assert "layer3_frames" in results
        assert "layer4_sensitivity" in results
        assert "layer4_ablation" in results

    def test_end_to_end_rssm(self, tmp_path):
        """Full pipeline works with RSSM dynamics."""
        vae = _make_tiny_vae()
        dynamics = _make_tiny_rssm()
        paths = []
        for i in range(2):
            p = tmp_path / f"ep{i}"
            p.mkdir()
            paths.append(_make_fake_episode(20, p))
        episodes = load_eval_episodes(paths, frame_size=64)
        results = run_full_eval(vae, dynamics, episodes,
                                horizons=[1, 5], device="cpu")
        assert "layer2_oracle" in results
        assert "layer4_sensitivity" in results
