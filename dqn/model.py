import torch as th
import torch.nn.functional as F
import numpy as np

from typing import Any
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import QNetwork, DQNPolicy
from sb3_contrib.common.maskable.utils import get_action_masks
from gymnasium import spaces


class MaskedDQN(DQN):
    """DQN with action masking for JANI environments."""
    def __init__(
        self,
        policy: str | type[DQNPolicy],
        env: GymEnv | str,
        learning_rate: float | Schedule = 1e-4,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 32,
        tau: float = 1.0,
        gamma: float = 0.99,
        train_freq: int | tuple[int, str] = 4,
        gradient_steps: int = 1,
        replay_buffer_class: type[ReplayBuffer] | None = None,
        replay_buffer_kwargs: dict[str, Any] | None = None,
        optimize_memory_usage: bool = False,
        n_steps: int = 1,
        target_update_interval: int = 10000,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        max_grad_norm: float = 10,
        stats_window_size: int = 100,
        tensorboard_log: str | None = None,
        policy_kwargs: dict[str, Any] | None = None,
        verbose: int = 0,
        seed: int | None = None,
        use_mask: bool = False,
        device: th.device | str = "auto",
        _init_setup_model: bool = True,
    ) -> None:
        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            replay_buffer_class,
            replay_buffer_kwargs,
            optimize_memory_usage,
            n_steps,
            target_update_interval,
            exploration_fraction,
            exploration_initial_eps,
            exploration_final_eps,
            max_grad_norm,
            stats_window_size,
            tensorboard_log,
            policy_kwargs,
            verbose,
            seed,
            device,
            _init_setup_model,
        )
        self.use_mask = use_mask

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
            # For n-step replay, discount factor is gamma**n_steps (when no early termination)
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            with th.no_grad():
                # Compute the next Q-values using the target network
                next_q_values = self.q_net_target(replay_data.next_observations)
                # Apply action masking if enabled
                if self.use_mask:
                    action_masks = np.array(self.env.env_method('action_mask_for_obs', replay_data.next_observations, indices=[0]), dtype=bool)
                    action_masks = np.squeeze(action_masks, axis=0)  # shape (batch_size, n_actions)
                    invalid_next = ~action_masks.any(axis=1)
                    assert all(replay_data.dones[invalid_next] == 1)
                    next_q_values[~action_masks] = -1e9  # large negative value
                # Follow greedy policy: use the one with the highest value
                next_q_values, _ = next_q_values.max(dim=1)
                # Avoid potential broadcast issue
                next_q_values = next_q_values.reshape(-1, 1)
                # 1-step TD target
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

            # Get current Q-values estimates
            current_q_values = self.q_net(replay_data.observations)

            # Retrieve the q-values for the actions from the replay buffer
            current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long())

            # Compute Huber loss (less sensitive to outliers)
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        # Increase update counter
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))

    def _sample_action(
        self,
        learning_starts,
        action_noise,
        n_envs = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Sample an action according to the exploration policy.
        This is either done by sampling the probability distribution of the policy,
        or sampling a random action (from a uniform distribution over the action space)
        or by adding noise to the deterministic output.

        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param n_envs:
        :return: action to take in the environment
            and scaled action that will be stored in the replay buffer.
            The two differs when the action space is not normalized (bounds are not [-1, 1]).
        """
        def sample_masked_action(space, mask) -> int:
            """
            action_mask: array-like of shape (space.n,), with True/1 meaning valid.
            """
            if mask.shape != (space.n,):
                raise ValueError(f"mask must have shape ({space.n},), got {mask.shape}")

            valid = np.flatnonzero(mask)
            if valid.size == 0:
                raise RuntimeError("No valid actions (mask is all False).")

            return int(np.random.choice(valid))
        
        # Select action randomly or according to policy
        action_masks = self.env.env_method('action_mask')
        action_masks = np.stack(action_masks, axis=0)  # shape (n_envs, n_actions)
        # action_masks = np.array(action_masks, dtype=bool).reshape(n_envs, -1)
        if self.num_timesteps < learning_starts and not (self.use_sde and self.use_sde_at_warmup):
            # Warmup phase
            unscaled_action = np.array([sample_masked_action(self.action_space, mask) for mask in action_masks])
        else:
            # Note: when using continuous actions,
            # the policy internally uses tanh to bound the action but predict() returns
            # actions unscaled to the original action space [low, high]
            # We use non-deterministic action in the case of SAC, for TD3, it does not matter
            assert self._last_obs is not None, "self._last_obs was not set"
            unscaled_action, _ = self.predict(self._last_obs, action_masks=action_masks, deterministic=False)

        # Rescale the action from [low, high] to [-1, 1]
        if isinstance(self.action_space, spaces.Box):
            scaled_action = self.policy.scale_action(unscaled_action)

            # Add noise to the action (improve exploration)
            if action_noise is not None:
                scaled_action = np.clip(scaled_action + action_noise(), -1, 1)

            # We store the scaled action in the buffer
            buffer_action = scaled_action
            action = self.policy.unscale_action(scaled_action)
        else:
            # Discrete case, no need to normalize or clip
            buffer_action = unscaled_action
            action = buffer_action
        return action, buffer_action

    def predict(self, observation, state = None, episode_start = None, deterministic = False, action_masks = None):
        obs, vectorized_env = self.policy.obs_to_tensor(observation)
        with th.no_grad():
            q_values = self.policy.q_net(obs)
        masks = action_masks.astype(bool) # shape (1, n_actions)
        # print(f"DEBUG: Q-values shape: {q_values.shape}, dtype: {q_values.dtype}")
        # print(f"DEBUG: Action masks shape: {masks.shape}, dtype: {masks.dtype}")
        q_values[~masks] = -np.inf  # large negative value
        actions = q_values.argmax(dim=1).unsqueeze(-1)
        return actions, state
