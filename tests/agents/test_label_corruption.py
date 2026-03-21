"""Tests for LabelCorruptionWrapper."""
import json

import numpy as np
import pytest
import gymnasium
from gymnasium import spaces


class FakeEnv(gymnasium.Env):
    """Minimal env that returns a fixed 15D observation.

    Simulates the ParameterizedLunarLander observation layout:
      dims 0-7: kinematic state
      dims 8-14: physics params (7 values)
    """

    def __init__(self, physics_values=None):
        super().__init__()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32,
        )
        # Fixed physics values for testing — recognizable pattern
        self._physics = np.array(
            physics_values or [-10.0, 13.0, 0.6, 5.0, 2.0, 15.0, 2.0],
            dtype=np.float32,
        )
        self._kinematic = np.zeros(8, dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        self._kinematic = np.arange(8, dtype=np.float32)  # 0,1,2,...,7
        obs = np.concatenate([self._kinematic, self._physics])
        return obs, {}

    def step(self, action):
        # Slightly change kinematics each step to simulate dynamics
        self._kinematic += 0.1
        obs = np.concatenate([self._kinematic, self._physics])
        return obs, 1.0, False, False, {}


class TestLabelCorruptionZero:
    """Test zero-out corruption mode."""

    def test_zero_sets_physics_dims_to_zero(self):
        """Zero corruption should set dims 8-14 to 0."""
        from lwp.agents.label_corruption import LabelCorruptionWrapper

        env = LabelCorruptionWrapper(FakeEnv(), corruption_type="zero")
        obs, _ = env.reset()
        # Kinematic dims (0-7) should be unchanged
        np.testing.assert_array_equal(obs[:8], np.arange(8, dtype=np.float32))
        # Physics dims (8-14) should be zeroed
        np.testing.assert_array_equal(obs[8:15], np.zeros(7, dtype=np.float32))

    def test_zero_preserves_obs_shape(self):
        """Corruption should not change observation shape."""
        from lwp.agents.label_corruption import LabelCorruptionWrapper

        env = LabelCorruptionWrapper(FakeEnv(), corruption_type="zero")
        obs, _ = env.reset()
        assert obs.shape == (15,)
        obs, _, _, _, _ = env.step(np.zeros(2))
        assert obs.shape == (15,)

    def test_zero_on_step_observations(self):
        """Physics dims should stay zeroed on every step, not just reset."""
        from lwp.agents.label_corruption import LabelCorruptionWrapper

        env = LabelCorruptionWrapper(FakeEnv(), corruption_type="zero")
        env.reset()
        for _ in range(5):
            obs, _, _, _, _ = env.step(np.zeros(2))
            np.testing.assert_array_equal(obs[8:15], np.zeros(7))
            # Kinematics should still change
            assert obs[0] != 0.0

    def test_zero_preserves_obs_space(self):
        """observation_space should match original (same shape, same bounds)."""
        from lwp.agents.label_corruption import LabelCorruptionWrapper

        base = FakeEnv()
        env = LabelCorruptionWrapper(base, corruption_type="zero")
        assert env.observation_space.shape == base.observation_space.shape

    def test_invalid_corruption_type_raises(self):
        """Should raise ValueError for unknown corruption type."""
        from lwp.agents.label_corruption import LabelCorruptionWrapper

        with pytest.raises(ValueError, match="corruption_type"):
            LabelCorruptionWrapper(FakeEnv(), corruption_type="invalid")


class TestLabelCorruptionWithExtendedObs:
    """Test corruption works on 22D obs (after RaycastWrapper adds 7 rays)."""

    def test_zero_on_22d_obs(self):
        """Physics dims (8-14) should be zeroed; ray dims (15-21) untouched."""
        from lwp.agents.label_corruption import LabelCorruptionWrapper

        # Create 22D env (simulating labeled + raycast)
        env = FakeEnv()
        env.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(22,), dtype=np.float32,
        )
        # Override to return 22D obs
        original_reset = env.reset

        def extended_reset(**kwargs):
            obs, info = original_reset(**kwargs)
            rays = np.ones(7, dtype=np.float32) * 0.5  # mock ray values
            return np.concatenate([obs, rays]), info

        env.reset = extended_reset

        original_step = env.step

        def extended_step(action):
            obs, r, term, trunc, info = original_step(action)
            rays = np.ones(7, dtype=np.float32) * 0.5
            return np.concatenate([obs, rays]), r, term, trunc, info

        env.step = extended_step

        wrapped = LabelCorruptionWrapper(env, corruption_type="zero")
        obs, _ = wrapped.reset()
        assert obs.shape == (22,)
        np.testing.assert_array_equal(obs[8:15], np.zeros(7))
        # Rays at 15-21 should be untouched
        np.testing.assert_array_almost_equal(obs[15:22], np.full(7, 0.5))


class TestLabelCorruptionShuffle:
    """Test shuffle corruption mode."""

    def test_shuffle_permutes_physics_dims(self):
        """Shuffled physics dims should contain same values in different order."""
        from lwp.agents.label_corruption import LabelCorruptionWrapper

        env = FakeEnv(physics_values=[-10.0, 13.0, 0.6, 5.0, 2.0, 15.0, 2.0])
        wrapped = LabelCorruptionWrapper(env, corruption_type="shuffle", seed=42)
        obs, _ = wrapped.reset()
        original = np.array([-10.0, 13.0, 0.6, 5.0, 2.0, 15.0, 2.0])
        shuffled = obs[8:15]
        # Same values, potentially different order (use almost_equal for float32 precision)
        np.testing.assert_array_almost_equal(np.sort(shuffled), np.sort(original))

    def test_shuffle_consistent_within_episode(self):
        """All steps in one episode should see the same permutation."""
        from lwp.agents.label_corruption import LabelCorruptionWrapper

        env = FakeEnv(physics_values=[-10.0, 13.0, 0.6, 5.0, 2.0, 15.0, 2.0])
        wrapped = LabelCorruptionWrapper(env, corruption_type="shuffle", seed=42)
        obs, _ = wrapped.reset()
        perm_at_reset = obs[8:15].copy()
        for _ in range(5):
            obs, _, _, _, _ = wrapped.step(np.zeros(2))
            np.testing.assert_array_equal(obs[8:15], perm_at_reset)

    def test_shuffle_changes_across_episodes(self):
        """Different episodes should (usually) get different permutations."""
        from lwp.agents.label_corruption import LabelCorruptionWrapper

        # Use distinct physics values so permutations are distinguishable
        env = FakeEnv(physics_values=[-10.0, 13.0, 0.6, 5.0, 2.0, 15.0, 3.5])
        wrapped = LabelCorruptionWrapper(env, corruption_type="shuffle", seed=42)

        perms = []
        for _ in range(10):
            obs, _ = wrapped.reset()
            perms.append(obs[8:15].copy())

        # At least 2 different permutations in 10 episodes
        unique = len(set(tuple(p) for p in perms))
        assert unique >= 2, f"Expected varied permutations, got {unique} unique out of 10"

    def test_shuffle_deterministic_with_same_seed(self):
        """Same seed should produce same sequence of permutations."""
        from lwp.agents.label_corruption import LabelCorruptionWrapper

        physics = [-10.0, 13.0, 0.6, 5.0, 2.0, 15.0, 3.5]
        env1 = LabelCorruptionWrapper(FakeEnv(physics), corruption_type="shuffle", seed=99)
        env2 = LabelCorruptionWrapper(FakeEnv(physics), corruption_type="shuffle", seed=99)

        for _ in range(5):
            obs1, _ = env1.reset()
            obs2, _ = env2.reset()
            np.testing.assert_array_equal(obs1[8:15], obs2[8:15])


class TestLabelCorruptionMean:
    """Test mean-replace corruption mode."""

    def test_mean_replaces_with_training_means(self):
        """Physics dims should be replaced with the provided training means."""
        from lwp.agents.label_corruption import LabelCorruptionWrapper

        means = np.array([-8.0, 15.0, 0.85, 6.25, 2.5, 15.0, 2.5], dtype=np.float32)
        env = LabelCorruptionWrapper(
            FakeEnv(), corruption_type="mean", training_means=means,
        )
        obs, _ = env.reset()
        np.testing.assert_array_almost_equal(obs[8:15], means)

    def test_mean_constant_across_steps(self):
        """Mean values should be the same every step (no variance)."""
        from lwp.agents.label_corruption import LabelCorruptionWrapper

        means = np.array([-8.0, 15.0, 0.85, 6.25, 2.5, 15.0, 2.5], dtype=np.float32)
        env = LabelCorruptionWrapper(
            FakeEnv(), corruption_type="mean", training_means=means,
        )
        env.reset()
        for _ in range(5):
            obs, _, _, _, _ = env.step(np.zeros(2))
            np.testing.assert_array_almost_equal(obs[8:15], means)

    def test_mean_requires_training_means(self):
        """Should raise ValueError when training_means is not provided."""
        from lwp.agents.label_corruption import LabelCorruptionWrapper

        with pytest.raises(ValueError, match="training_means"):
            LabelCorruptionWrapper(FakeEnv(), corruption_type="mean")


class TestLabelCorruptionNoise:
    """Test noise corruption mode."""

    def test_noise_modifies_physics_dims(self):
        """Noisy physics dims should differ from originals."""
        from lwp.agents.label_corruption import LabelCorruptionWrapper

        env = LabelCorruptionWrapper(
            FakeEnv(), corruption_type="noise", sigma=0.5, seed=42,
        )
        obs, _ = env.reset()
        original = np.array([-10.0, 13.0, 0.6, 5.0, 2.0, 15.0, 2.0])
        # With sigma=0.5 (50% of range), values should differ
        assert not np.allclose(obs[8:15], original, atol=0.01)

    def test_noise_stays_in_valid_range(self):
        """Noisy values should be clipped to valid parameter ranges."""
        from lwp.agents.label_corruption import LabelCorruptionWrapper
        from parametric_lunar_lander.physics_config import LunarLanderPhysicsConfig

        # Use extreme sigma to force clipping
        env = LabelCorruptionWrapper(
            FakeEnv(), corruption_type="noise", sigma=10.0, seed=42,
        )
        for _ in range(20):
            obs, _ = env.reset()
            for i, name in enumerate(LunarLanderPhysicsConfig.PARAM_NAMES):
                lo, hi = LunarLanderPhysicsConfig.RANGES[name]
                assert lo <= obs[8 + i] <= hi, (
                    f"{name}={obs[8 + i]} outside [{lo}, {hi}]"
                )

    def test_noise_varies_per_step(self):
        """Each step should get fresh noise (not the same as reset)."""
        from lwp.agents.label_corruption import LabelCorruptionWrapper

        env = LabelCorruptionWrapper(
            FakeEnv(), corruption_type="noise", sigma=0.3, seed=42,
        )
        obs_reset, _ = env.reset()
        physics_values = [obs_reset[8:15].copy()]
        for _ in range(5):
            obs, _, _, _, _ = env.step(np.zeros(2))
            physics_values.append(obs[8:15].copy())

        # At least some variation across steps
        unique_vals = len(set(tuple(v) for v in physics_values))
        assert unique_vals > 1, "Expected noise to vary across steps"

    def test_noise_sigma_zero_is_identity(self):
        """sigma=0 should return original values (no noise)."""
        from lwp.agents.label_corruption import LabelCorruptionWrapper

        env = LabelCorruptionWrapper(
            FakeEnv(), corruption_type="noise", sigma=0.0, seed=42,
        )
        obs, _ = env.reset()
        original = np.array([-10.0, 13.0, 0.6, 5.0, 2.0, 15.0, 2.0])
        np.testing.assert_array_almost_equal(obs[8:15], original)


class TestComputeTrainingMeans:
    """Test training-set mean computation from trajectory files."""

    def test_computes_means_from_npz_files(self, tmp_path):
        """Should compute per-param means from trajectory metadata."""
        from lwp.agents.label_corruption import compute_training_means

        # Create mock trajectory .npz files with known physics configs
        configs = [
            {"gravity": -10.0, "main_engine_power": 13.0, "side_engine_power": 0.6,
             "lander_density": 5.0, "angular_damping": 2.0, "wind_power": 15.0,
             "turbulence_power": 2.0},
            {"gravity": -8.0, "main_engine_power": 17.0, "side_engine_power": 1.0,
             "lander_density": 7.0, "angular_damping": 4.0, "wind_power": 5.0,
             "turbulence_power": 4.0},
        ]
        for i, cfg in enumerate(configs):
            metadata = json.dumps({"physics_config": cfg, "outcome": "landed"})
            np.savez(
                tmp_path / f"episode_{i:04d}.npz",
                states=np.zeros((10, 15)),
                actions=np.zeros((9, 2)),
                rewards=np.zeros(9),
                dones=np.zeros(9, dtype=bool),
                metadata_json=metadata,
            )

        means = compute_training_means(str(tmp_path))
        assert means.shape == (7,)
        # gravity mean: (-10 + -8) / 2 = -9.0
        assert means[0] == pytest.approx(-9.0)
        # main_engine mean: (13 + 17) / 2 = 15.0
        assert means[1] == pytest.approx(15.0)

    def test_empty_dir_raises(self, tmp_path):
        """Should raise if no .npz files found."""
        from lwp.agents.label_corruption import compute_training_means

        with pytest.raises(FileNotFoundError, match="No .npz"):
            compute_training_means(str(tmp_path))
