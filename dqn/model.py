import torch as th
import torch.nn.functional as F
import numpy as np

from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import QNetwork, DQNPolicy
from sb3_contrib.common.maskable.utils import get_action_masks
from gymnasium import spaces


class MaskedDQN(DQN):
    """DQN with action masking for JANI environments."""
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
