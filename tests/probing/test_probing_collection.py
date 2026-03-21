"""Tests for probe data collection."""
import json
import numpy as np
import pytest

from lwp.probing.collection import collect_probe_data, build_layer_specs
from lwp.probing.targets import BEHAVIORAL_TARGET_NAMES, PARAMETRIC_TARGET_NAMES


class _ConstantPolicy:
    """Mock policy that returns a fixed action."""
    def predict(self, obs, deterministic=False):
        action = np.array([[0.3, 0.0]], dtype=np.float32)
        return action, None


class TestCollectProbeData:
    def _make_blind_env_fn(self):
        from parametric_lunar_lander.wrappers import make_lunar_lander_env
        def env_fn(seed):
            return make_lunar_lander_env(variant="blind", seed=seed, profile="easy")
        return env_fn

    def test_returns_dict_with_expected_keys(self):
        result = collect_probe_data(
            model=_ConstantPolicy(),
            env_fn=self._make_blind_env_fn(),
            n_episodes=2,
            seed=42,
            variant="blind",
        )
        # Mock model won't have hooks, so no activation keys.
        # But we always get targets + metadata.
        assert "physics_params" in result
        assert "behavioral" in result
        assert "episode_ids" in result
        assert "layer_names" in result
        assert "metadata_json" in result

    def test_array_lengths_consistent(self):
        result = collect_probe_data(
            model=_ConstantPolicy(),
            env_fn=self._make_blind_env_fn(),
            n_episodes=3,
            seed=42,
            variant="blind",
        )
        n = len(result["episode_ids"])
        assert n > 0
        assert result["physics_params"].shape[0] == n
        assert result["behavioral"].shape[0] == n

    def test_physics_params_shape(self):
        result = collect_probe_data(
            model=_ConstantPolicy(),
            env_fn=self._make_blind_env_fn(),
            n_episodes=2,
            seed=42,
            variant="blind",
        )
        assert result["physics_params"].shape[1] == 7

    def test_behavioral_shape(self):
        result = collect_probe_data(
            model=_ConstantPolicy(),
            env_fn=self._make_blind_env_fn(),
            n_episodes=2,
            seed=42,
            variant="blind",
        )
        assert result["behavioral"].shape[1] == 5

    def test_episode_ids_are_sequential(self):
        result = collect_probe_data(
            model=_ConstantPolicy(),
            env_fn=self._make_blind_env_fn(),
            n_episodes=3,
            seed=42,
            variant="blind",
        )
        ids = result["episode_ids"]
        unique_ids = np.unique(ids)
        assert len(unique_ids) == 3
        np.testing.assert_array_equal(unique_ids, [0, 1, 2])

    def test_physics_params_constant_within_episode(self):
        result = collect_probe_data(
            model=_ConstantPolicy(),
            env_fn=self._make_blind_env_fn(),
            n_episodes=2,
            seed=42,
            variant="blind",
        )
        for ep_id in np.unique(result["episode_ids"]):
            mask = result["episode_ids"] == ep_id
            params = result["physics_params"][mask]
            # All rows within an episode should be identical
            np.testing.assert_array_equal(params, params[0:1].repeat(len(params), axis=0))

    def test_metadata_json_is_valid(self):
        result = collect_probe_data(
            model=_ConstantPolicy(),
            env_fn=self._make_blind_env_fn(),
            n_episodes=2,
            seed=42,
            variant="blind",
        )
        meta = json.loads(result["metadata_json"])
        assert "n_episodes" in meta
        assert "n_timesteps" in meta
        assert meta["n_episodes"] == 2
        assert meta["variant"] == "blind"

    def test_metadata_includes_layer_names(self):
        result = collect_probe_data(
            model=_ConstantPolicy(),
            env_fn=self._make_blind_env_fn(),
            n_episodes=2,
            seed=42,
            variant="blind",
        )
        meta = json.loads(result["metadata_json"])
        assert "layer_names" in meta


class TestCollectProbeDataWithRealModel:
    """Integration test using a real (untrained) SB3 model.

    This verifies that hooks fire correctly on the actual SB3 policy
    network structure and that activations have the right shape.
    """

    @pytest.fixture
    def blind_model_and_env(self):
        import torch.nn as nn
        from stable_baselines3 import PPO
        from parametric_lunar_lander.wrappers import make_lunar_lander_env

        def env_fn(seed):
            return make_lunar_lander_env(variant="blind", seed=seed, profile="easy")

        env = env_fn(0)
        model = PPO("MlpPolicy", env, n_steps=64, device="cpu",
                     policy_kwargs={"net_arch": [128, 128], "activation_fn": nn.ReLU})
        env.close()
        return model, env_fn

    def test_activations_are_128_wide(self, blind_model_and_env):
        model, env_fn = blind_model_and_env
        result = collect_probe_data(
            model=model, env_fn=env_fn, n_episodes=2, seed=42,
            variant="blind",
        )
        assert result["activations_L1"].shape[1] == 128
        assert result["activations_L2"].shape[1] == 128

    def test_activations_are_post_relu(self, blind_model_and_env):
        model, env_fn = blind_model_and_env
        result = collect_probe_data(
            model=model, env_fn=env_fn, n_episodes=2, seed=42,
            variant="blind",
        )
        assert np.all(result["activations_L1"] >= 0)
        assert np.all(result["activations_L2"] >= 0)

    def test_build_layer_specs_blind(self, blind_model_and_env):
        model, _ = blind_model_and_env
        specs = build_layer_specs(model, variant="blind")
        names = [name for name, _ in specs]
        assert "L1" in names
        assert "L2" in names
        assert "CNN" not in names


class TestBuildLayerSpecsVisual:
    """Test build_layer_specs for visual agents."""

    @pytest.fixture
    def visual_model(self):
        import torch.nn as nn
        from stable_baselines3 import PPO
        from parametric_lunar_lander.wrappers import make_lunar_lander_env

        def env_fn(seed):
            return make_lunar_lander_env(variant="visual", seed=seed, frame_size=84)

        from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
        vec_env = DummyVecEnv([lambda: env_fn(0)])
        vec_env = VecFrameStack(vec_env, n_stack=4)

        model = PPO("CnnPolicy", vec_env, n_steps=64, device="cpu",
                     policy_kwargs={"net_arch": [128, 128], "activation_fn": nn.ReLU})
        vec_env.close()
        return model

    def test_includes_cnn_layer(self, visual_model):
        specs = build_layer_specs(visual_model, variant="visual")
        names = [name for name, _ in specs]
        assert "CNN" in names
        assert "L1" in names
        assert "L2" in names

    def test_cnn_is_first(self, visual_model):
        specs = build_layer_specs(visual_model, variant="visual")
        assert specs[0][0] == "CNN"


class TestCollectProbeDataVisual:
    """Integration test: collect probe data from a visual agent."""

    @pytest.fixture
    def visual_model_and_env(self):
        import torch.nn as nn
        from stable_baselines3 import PPO
        from parametric_lunar_lander.wrappers import make_lunar_lander_env

        def env_fn(seed):
            return make_lunar_lander_env(variant="visual", seed=seed, frame_size=84)

        from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
        vec_env = DummyVecEnv([lambda: env_fn(0)])
        vec_env = VecFrameStack(vec_env, n_stack=4)

        model = PPO("CnnPolicy", vec_env, n_steps=64, device="cpu",
                     policy_kwargs={"net_arch": [128, 128], "activation_fn": nn.ReLU})
        vec_env.close()
        return model, env_fn

    def test_visual_collection_has_cnn_activations(self, visual_model_and_env):
        model, env_fn = visual_model_and_env
        result = collect_probe_data(
            model=model, env_fn=env_fn, n_episodes=2, seed=42,
            variant="visual", n_stack=4,
        )
        assert "activations_CNN" in result
        assert "activations_L1" in result
        assert "activations_L2" in result

    def test_visual_cnn_activation_shape(self, visual_model_and_env):
        model, env_fn = visual_model_and_env
        result = collect_probe_data(
            model=model, env_fn=env_fn, n_episodes=2, seed=42,
            variant="visual", n_stack=4,
        )
        # NatureCNN at 84x84 -> 512D output
        assert result["activations_CNN"].shape[1] == 512
        assert result["activations_L1"].shape[1] == 128
        assert result["activations_L2"].shape[1] == 128

    def test_visual_layer_names_in_metadata(self, visual_model_and_env):
        model, env_fn = visual_model_and_env
        result = collect_probe_data(
            model=model, env_fn=env_fn, n_episodes=2, seed=42,
            variant="visual", n_stack=4,
        )
        meta = json.loads(result["metadata_json"])
        assert meta["layer_names"] == ["CNN", "L1", "L2"]
        assert meta["variant"] == "visual"
        assert meta["n_stack"] == 4
