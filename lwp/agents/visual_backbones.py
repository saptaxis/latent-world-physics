"""Visual CNN backbones for RL feature extraction.

Custom feature extractors that plug into SB3 via policy_kwargs:
    policy_kwargs={"features_extractor_class": ImpalaCNN}

Available backbones:
    ImpalaCNN — 3-block residual CNN from Espeholt et al. 2018 (IMPALA paper).
        Standard upgrade from NatureCNN in RL. Uses 3x3 kernels throughout
        (preserves spatial detail better than NatureCNN's 8x8 first layer).
        Proven on Procgen, NetHack, and similar pixel-based RL tasks.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch as th
import torch.nn as nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.preprocessing import is_image_space, get_flattened_obs_dim


class _ResidualBlock(nn.Module):
    """Single residual block: two 3x3 convs with skip connection."""

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        return x + self.block(x)


class _ConvSequence(nn.Module):
    """One IMPALA conv sequence: conv -> maxpool -> 2x residual blocks.

    Each sequence:
    1. 3x3 conv to change channel count
    2. 3x3 maxpool with stride 2 (halves spatial dims)
    3. Two residual blocks (each is 2x 3x3 conv with skip)
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.res1 = _ResidualBlock(out_channels)
        self.res2 = _ResidualBlock(out_channels)

    def forward(self, x: th.Tensor) -> th.Tensor:
        x = self.conv(x)
        x = self.pool(x)
        x = self.res1(x)
        x = self.res2(x)
        return x


class ImpalaCNN(BaseFeaturesExtractor):
    """IMPALA-style residual CNN (Espeholt et al. 2018).

    Architecture: 3 conv sequences with channels [16, 32, 32], each halving
    spatial dims via maxpool. Final ReLU -> flatten -> linear -> ReLU.

    Key differences from NatureCNN:
    - 3x3 kernels only (vs 8x8/4x4/3x3) — preserves spatial detail
    - Residual connections — more stable training despite more depth
    - ~15 conv layers total but narrow channels keep param count reasonable

    An AdaptiveAvgPool2d(4,4) before flatten caps the linear input size
    regardless of input resolution, preventing param blowup at 256px+.

    Spatial dims at each stage (128x128 input):
        Input:    128x128 (4ch)
        Block 1:  64x64   (16ch)
        Block 2:  32x32   (32ch)
        Block 3:  16x16   (32ch)
        Pool:     4x4     (32ch)
        Flatten:  32*4*4 = 512
        Linear:   256 (default features_dim)

    Args:
        observation_space: Image observation space (CxHxW).
        features_dim: Output feature dimension (default 256).
        normalized_image: If True, skip dtype/bounds checks.
        channels: Channel counts per conv sequence (default [16, 32, 32]).
        pool_size: Adaptive avg pool spatial size before flatten (default 4).
    """

    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 256,
        normalized_image: bool = False,
        channels: list[int] | None = None,
        pool_size: int = 4,
    ) -> None:
        assert isinstance(observation_space, spaces.Box), (
            f"ImpalaCNN requires Box observation space, got {observation_space}"
        )
        super().__init__(observation_space, features_dim)
        assert len(observation_space.shape) == 3, (
            f"ImpalaCNN requires 3D image (CxHxW), got shape {observation_space.shape}"
        )
        # check_channels=False: VecFrameStack creates arbitrary channel counts
        # (e.g., n_stack=2 -> 2ch, n_stack=8 -> 8ch). Only validate shape/dtype.
        assert is_image_space(observation_space, check_channels=False, normalized_image=normalized_image), (
            f"ImpalaCNN requires image observations, got {observation_space}"
        )

        n_input_channels = observation_space.shape[0]
        channels = channels or [16, 32, 32]

        # Build the 3 conv sequences
        sequences = []
        in_ch = n_input_channels
        for out_ch in channels:
            sequences.append(_ConvSequence(in_ch, out_ch))
            in_ch = out_ch

        self.conv_sequences = nn.Sequential(*sequences)
        self.relu = nn.ReLU()
        # Adaptive pool caps flatten size regardless of input resolution.
        # 4x4 keeps enough spatial info for the linear layer without
        # blowing up params at 256px+ inputs.
        self.pool = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        self.flatten = nn.Flatten()

        # Compute flattened size with a dummy forward pass
        with th.no_grad():
            dummy = th.zeros(1, *observation_space.shape)
            n_flatten = self.flatten(self.pool(self.relu(self.conv_sequences(dummy)))).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        x = self.conv_sequences(observations)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        return self.linear(x)


class ImpalaCombinedExtractor(BaseFeaturesExtractor):
    """CombinedExtractor using ImpalaCNN for image subspaces.

    SB3's default CombinedExtractor hardcodes NatureCNN for images.
    This variant uses ImpalaCNN instead, supporting Dict observation
    spaces (visual-labeled variant) with the IMPALA backbone.

    For each key in the Dict observation space:
    - Image keys → ImpalaCNN (same architecture as standalone ImpalaCNN)
    - Vector keys → nn.Flatten

    All outputs concatenated → features_dim = cnn_output_dim + vector_dims.

    The CNN sub-extractor lives at self.extractors["image"], which is an
    ImpalaCNN instance with identical state_dict keys to the standalone
    ImpalaCNN. This means pre-trained encoder weights load directly into
    self.extractors["image"] without key remapping.

    Args:
        observation_space: Dict space (e.g. {"image": Box, "physics": Box}).
        cnn_output_dim: ImpalaCNN output dimension (default 256).
        normalized_image: If True, skip image dtype/bounds checks.
        channels: ImpalaCNN channel widths (default [16, 32, 32]).
        pool_size: ImpalaCNN adaptive pool spatial size (default 4).
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        cnn_output_dim: int = 256,
        normalized_image: bool = False,
        channels: list[int] | None = None,
        pool_size: int = 4,
    ) -> None:
        # BaseFeaturesExtractor requires features_dim upfront.
        # We compute it below and update _features_dim manually.
        super().__init__(observation_space, features_dim=1)

        extractors: dict[str, nn.Module] = {}
        total_concat_size = 0

        for key, subspace in observation_space.spaces.items():
            if is_image_space(subspace, check_channels=False, normalized_image=normalized_image):
                extractors[key] = ImpalaCNN(
                    subspace,
                    features_dim=cnn_output_dim,
                    normalized_image=normalized_image,
                    channels=channels,
                    pool_size=pool_size,
                )
                total_concat_size += cnn_output_dim
            else:
                extractors[key] = nn.Flatten()
                total_concat_size += get_flattened_obs_dim(subspace)

        self.extractors = nn.ModuleDict(extractors)
        self._features_dim = total_concat_size

    def forward(self, observations: dict[str, th.Tensor]) -> th.Tensor:
        encoded = []
        for key, extractor in self.extractors.items():
            encoded.append(extractor(observations[key]))
        return th.cat(encoded, dim=1)


class ImpalaBranchCombinedExtractor(BaseFeaturesExtractor):
    """CombinedExtractor with a dedicated physics branch MLP.

    Solves the dimensional mismatch problem in visual-labeled experiments:
    when 7 physics params are raw-concatenated to 256 CNN features, physics
    is only 2.7% of the input and easily ignored by the policy MLP. This
    extractor gives the physics vector a dedicated projection branch
    (normalize → Linear → ReLU) before concatenation, increasing its
    representational footprint to ~11% (32D / 288D) by default.

    Architecture:
        Image (4, 128, 128) → ImpalaCNN → 256D ──────────────────┐
                                                                  ├─ concat → 288D
        Physics (7,) → normalize → Linear(7, 32) → ReLU → 32D ──┘

    Normalization uses registered buffers (mean, std) computed from the
    training profile's physics parameter distribution. Buffers are not
    learned — they checkpoint cleanly, move with .to(device), and don't
    receive gradients. When stats are not provided, identity normalization
    (mean=0, std=1) is used as a fallback — this is a pass-through that
    exists for testing and debugging. In production, train_rl.py always
    computes and provides stats from the training profile. The boolean
    has_physics_norm flag records which mode was used.

    The CNN sub-extractor lives at self.extractors["image"], with identical
    state_dict keys to standalone ImpalaCNN and ImpalaCombinedExtractor.
    This means:
    - Pre-trained encoder weights load via extractors["image"].load_state_dict()
    - Encoder freezing freezes only the CNN, not the physics branch
    - The existing _find_cnn() logic in lwp/rl/training.py works unchanged

    The physics branch lives at self.physics_branch (NOT in self.extractors)
    to keep it architecturally separate from the CNN. This ensures:
    - Encoder weight loading doesn't touch the branch
    - Encoder freezing doesn't freeze the branch
    - The branch is always randomly initialized and trained with the MLP

    Args:
        observation_space: Dict space (e.g. {"image": Box, "physics": Box}).
        cnn_output_dim: ImpalaCNN output dimension (default 256).
        physics_branch_dim: Physics branch output dimension (default 32).
            Controls the representational footprint of physics in the
            concatenated features. 32 → ~11% of 288D total.
        physics_mean: Per-dimension mean for physics normalization, shape (n_physics,).
            Computed from the training profile (e.g., 10K samples from SamplingProfile).
            Registered as a buffer. Must be provided together with physics_std, or
            both omitted (falls back to identity: mean=0, std=1).
        physics_std: Per-dimension std for physics normalization, shape (n_physics,).
            Registered as a buffer. Clamped to min 1e-6 to avoid division by zero
            (e.g., wind_power=0 fixed in easy profile → std=0). Must be provided
            together with physics_mean, or both omitted.
        normalized_image: If True, skip image dtype/bounds checks.
        channels: ImpalaCNN channel widths (default [16, 32, 32]).
        pool_size: ImpalaCNN adaptive pool spatial size (default 4).
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        cnn_output_dim: int = 256,
        physics_branch_dim: int = 32,
        physics_mean: np.ndarray | None = None,
        physics_std: np.ndarray | None = None,
        normalized_image: bool = False,
        channels: list[int] | None = None,
        pool_size: int = 4,
    ) -> None:
        # BaseFeaturesExtractor requires features_dim upfront.
        # We compute it below and update _features_dim manually.
        super().__init__(observation_space, features_dim=1)

        # --- Validate observation space ---
        # Require explicit "image" and "physics" keys. Don't discover by
        # exclusion — that's brittle if the Dict ever gains other keys.
        assert "image" in observation_space.spaces, (
            "ImpalaBranchCombinedExtractor requires an 'image' key in the "
            f"Dict observation space. Got keys: {list(observation_space.spaces.keys())}"
        )
        assert "physics" in observation_space.spaces, (
            "ImpalaBranchCombinedExtractor requires a 'physics' key in the "
            f"Dict observation space. Got keys: {list(observation_space.spaces.keys())}"
        )

        # --- Image extractor (ImpalaCNN) ---
        # Lives in self.extractors["image"] for compatibility with the
        # existing weight-loading path: _find_cnn() in lwp/rl/training.py
        # checks hasattr(fe, "extractors") and "image" in fe.extractors.
        image_subspace = observation_space.spaces["image"]
        self.extractors = nn.ModuleDict({
            "image": ImpalaCNN(
                image_subspace,
                features_dim=cnn_output_dim,
                normalized_image=normalized_image,
                channels=channels,
                pool_size=pool_size,
            ),
        })

        # --- Physics branch ---
        # Separate from self.extractors so encoder weight loading and
        # freezing don't touch it. Always randomly initialized, always
        # trainable (even when the CNN is frozen).
        physics_subspace = observation_space.spaces["physics"]
        physics_input_dim = get_flattened_obs_dim(physics_subspace)
        self.physics_branch = nn.Sequential(
            nn.Linear(physics_input_dim, physics_branch_dim),
            nn.ReLU(),
        )

        self._features_dim = cnn_output_dim + physics_branch_dim

        # --- Normalization buffers ---
        # Always register as real tensors (not None) so state_dict
        # roundtrips work cleanly. When stats are not provided, use
        # identity normalization (mean=0, std=1) which passes through
        # unchanged. A boolean flag records whether real stats were given.
        both_provided = physics_mean is not None and physics_std is not None
        neither_provided = physics_mean is None and physics_std is None
        assert both_provided or neither_provided, (
            "physics_mean and physics_std must be both provided or both None. "
            f"Got physics_mean={'set' if physics_mean is not None else 'None'}, "
            f"physics_std={'set' if physics_std is not None else 'None'}."
        )
        self.has_physics_norm = both_provided
        if both_provided:
            self.register_buffer(
                "physics_mean",
                th.tensor(physics_mean, dtype=th.float32),
            )
            self.register_buffer(
                "physics_std",
                th.tensor(physics_std, dtype=th.float32),
            )
        else:
            # Identity normalization: (x - 0) / 1 = x
            self.register_buffer(
                "physics_mean",
                th.zeros(physics_input_dim, dtype=th.float32),
            )
            self.register_buffer(
                "physics_std",
                th.ones(physics_input_dim, dtype=th.float32),
            )

    def _normalize_physics(self, physics: th.Tensor) -> th.Tensor:
        """Normalize physics input using registered mean/std buffers.

        Always applies (physics - mean) / std.clamp(min=1e-6). When no
        stats were provided at construction, buffers are identity
        (mean=0, std=1) so this is a no-op. The clamp handles fixed
        physics params (e.g., wind_power=0 in easy profile → std=0).
        """
        return (physics - self.physics_mean) / self.physics_std.clamp(min=1e-6)

    def forward(self, observations: dict[str, th.Tensor]) -> th.Tensor:
        # Image → CNN
        image_features = self.extractors["image"](observations["image"])

        # Physics → normalize → branch MLP
        physics = self._normalize_physics(observations["physics"])
        physics_features = self.physics_branch(physics)

        return th.cat([image_features, physics_features], dim=1)
