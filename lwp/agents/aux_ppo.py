"""PPO with auxiliary kinematic state prediction head.

Subclasses SB3's PPO to add an MLP head on the CNN features that predicts
kinematic state [x, y, vx, vy, angle, angular_vel]. The aux MSE loss
backprops through the CNN encoder, forcing it to learn physics-relevant
features — particularly angle, which standard PPO's reward gradient alone
fails to teach (angle R²=0.29 for IMPALA, -0.12 for NatureCNN).

The aux head is a separate small MLP (features_dim -> 64 -> 6) trained
jointly with the PPO losses. The aux loss coefficient (aux_coef) controls
the tradeoff: too high and aux dominates the RL gradient, too low and it
has no effect.

Requires KinematicInfoWrapper on the env to pass ground truth kinematic
state through info["kinematic_state"]. The kinematic data is stored in a
custom rollout buffer alongside the standard PPO data.

Usage:
    from lwp.agents.aux_ppo import AuxPPO
    model = AuxPPO("CnnPolicy", env, aux_coef=0.5, ...)
"""
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.type_aliases import RolloutBufferSamples
from stable_baselines3.common.utils import obs_as_tensor


# Number of kinematic targets: x, y, vx, vy, angle, angular_vel
N_KINEMATIC = 6


class RolloutBufferWithKinematic(RolloutBuffer):
    """Extends RolloutBuffer to store kinematic state for aux loss training."""

    def reset(self) -> None:
        super().reset()
        # Allocate storage for kinematic targets alongside standard PPO data.
        # Shape: (buffer_size, n_envs, N_KINEMATIC)
        self.kinematic_states = np.zeros(
            (self.buffer_size, self.n_envs, N_KINEMATIC), dtype=np.float32
        )

    def add(self, *args, kinematic_state: np.ndarray | None = None, **kwargs) -> None:
        """Add a transition. kinematic_state is (n_envs, 6) from KinematicInfoWrapper."""
        # Store kinematic state BEFORE super().add() increments self.pos.
        if kinematic_state is not None:
            self.kinematic_states[self.pos] = kinematic_state
        super().add(*args, **kwargs)

    def _get_samples(self, batch_inds: np.ndarray, env=None) -> RolloutBufferSamples:
        """Return standard PPO samples. Kinematic data accessed separately."""
        return super()._get_samples(batch_inds, env=env)

    def get_kinematic_batch(self, batch_inds: np.ndarray) -> th.Tensor:
        """Get kinematic targets for a batch of indices (after swap_and_flatten)."""
        return th.as_tensor(self.kinematic_states[batch_inds], device="cpu")

    def get(self, batch_size=None):
        """Override to also flatten kinematic states."""
        # On first call per training step, flatten the (buffer_size, n_envs, ...)
        # arrays into (buffer_size * n_envs, ...) for minibatch sampling.
        if not self.generator_ready:
            # Flatten kinematic states same way as other buffer arrays.
            self.kinematic_states = self.swap_and_flatten(self.kinematic_states)
        # Yield from parent (handles observations, actions, etc.)
        yield from super().get(batch_size)


class AuxKinematicHead(nn.Module):
    """Small MLP that predicts kinematic state from CNN features.

    Architecture: features_dim -> 64 -> N_KINEMATIC (6D).
    Deliberately small — this is a regularizer, not the main model.
    """

    def __init__(self, features_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(features_dim, 64),
            nn.ReLU(),
            nn.Linear(64, N_KINEMATIC),
        )

    def forward(self, features: th.Tensor) -> th.Tensor:
        return self.net(features)


class AuxPPO(PPO):
    """PPO with auxiliary kinematic state prediction loss on CNN features.

    Adds a small MLP head that predicts [x, y, vx, vy, angle, angular_vel]
    from the CNN encoder output. The MSE loss backprops through the encoder,
    directly shaping the learned representation.

    Args:
        aux_coef: Weight of aux loss relative to PPO loss. Start with 0.5.
            Higher = stronger representation shaping, risk of dominating RL.
            Lower = weaker effect on encoder, safer for RL performance.
    """

    def __init__(self, *args, aux_coef: float = 0.5, **kwargs):
        # Force our custom rollout buffer.
        kwargs["rollout_buffer_class"] = RolloutBufferWithKinematic
        super().__init__(*args, **kwargs)
        self.aux_coef = aux_coef

        # Build aux head after policy is created (so we know features_dim).
        features_dim = self.policy.features_extractor.features_dim
        self.aux_head = AuxKinematicHead(features_dim).to(self.device)

        # Add aux head params to the policy optimizer so they're updated
        # in the same step as the rest of the model.
        self.policy.optimizer.add_param_group(
            {"params": self.aux_head.parameters()}
        )

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):
        """Override to store kinematic state from info dict in the buffer."""
        # This follows SB3's OnPolicyAlgorithm.collect_rollouts() but adds
        # kinematic state extraction from infos.
        assert self._last_obs is not None, "No previous observation"
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()

        callback.on_rollout_start()
        # Reset noise for exploration (if using gSDE)
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()

            clipped_actions = actions
            if isinstance(self.action_space, type(env.action_space)):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, type(env.action_space)):
                actions = np.clip(actions, self.action_space.low, self.action_space.high)

            # Extract kinematic state from infos.
            # infos is a list of dicts (one per env). Each has "kinematic_state"
            # if KinematicInfoWrapper is in the stack.
            kinematic_batch = np.zeros((env.num_envs, N_KINEMATIC), dtype=np.float32)
            for i, info in enumerate(infos):
                if "kinematic_state" in info:
                    kinematic_batch[i] = info["kinematic_state"]

            # Handle terminal observations (SB3 convention: new_obs is the
            # reset obs when done=True, but we need the terminal obs for
            # value bootstrapping).
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(
                        infos[idx]["terminal_observation"]
                    )[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value.item()

            rollout_buffer.add(
                self._last_obs,
                actions,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
                kinematic_state=kinematic_batch,
            )
            self._last_obs = new_obs
            self._last_episode_starts = dones

        with th.no_grad():
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()

        return True

    def train(self) -> None:
        """PPO training with auxiliary kinematic prediction loss."""
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        # Get current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        clip_range_vf = None
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        aux_losses = []

        continue_training = True

        for epoch in range(self.n_epochs):
            approx_kl_divs = []

            # Track batch indices to get matching kinematic data.
            # We iterate the buffer generator and track position manually.
            batch_idx = 0
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, actions
                )
                values = values.flatten()

                # Normalize advantage
                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # Ratio: π_new / π_old
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # Clipped PPO loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                # Value loss
                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss
                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)
                entropy_losses.append(entropy_loss.item())

                # --- Auxiliary kinematic prediction loss ---
                # Get CNN features from the current forward pass.
                # policy.extract_features() runs the CNN on observations.
                features = self.policy.extract_features(
                    rollout_data.observations, self.policy.features_extractor
                )
                kinematic_pred = self.aux_head(features)

                # Get matching kinematic targets from the buffer.
                # The buffer's get() flattened everything, so we need the
                # same indices. We use the buffer's internal index tracking.
                start = batch_idx * self.batch_size
                end = start + len(rollout_data.observations)
                batch_inds = np.arange(start, end)
                kinematic_target = self.rollout_buffer.get_kinematic_batch(
                    batch_inds
                ).to(self.device)
                aux_loss = F.mse_loss(kinematic_pred, kinematic_target)
                aux_losses.append(aux_loss.item())
                batch_idx += 1

                # Total loss
                loss = (
                    policy_loss
                    + self.ent_coef * entropy_loss
                    + self.vf_coef * value_loss
                    + self.aux_coef * aux_loss
                )

                # KL early stopping
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                # Also clip aux head grads (not in policy.parameters()).
                th.nn.utils.clip_grad_norm_(self.aux_head.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = self._compute_explained_variance()

        # Log all metrics
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/aux_kinematic_loss", np.mean(aux_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

    def _compute_explained_variance(self):
        """Compute explained variance from the rollout buffer."""
        y_pred = self.rollout_buffer.values.flatten()
        y_true = self.rollout_buffer.returns.flatten()
        var_y = np.var(y_true)
        if var_y == 0:
            return float("nan")
        return float(1 - np.var(y_true - y_pred) / var_y)
