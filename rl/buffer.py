import torch
import numpy as np

from classifier import predict
from stable_baselines3.common.buffers import RolloutBuffer


class RolloutBufferWithLB(RolloutBuffer):
    """
    RolloutBuffer with a floored baseline for GAE advantages.
    """
    def __init__(self, *args, classifier, scaler=None, alpha=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.classifier = classifier # classifier for providing lower bound
        self.scaler = scaler
        self.alpha = alpha

    def compute_returns_and_advantage(self, last_values, dones):
        """
        Compute GAE advantages with a floored baseline from the classifier.
        
        Args:
            last_values: Value estimates for the last observations
            dones: Done flags for the last observations
        """
        # Compute classifier-based baseline
        with torch.no_grad():
            obs_tensor = torch.tensor(self.observations, dtype=torch.float32)
            obs_tensor = obs_tensor.view(-1, self.observations.shape[-1])
            _, preds = predict(self.classifier, obs_tensor, self.scaler)
            preds = preds.reshape(-1, 1).astype(np.float32)
            lower_bounds = self.alpha * (1 - preds)  # Scale to [0, alpha]

        last_values = last_values.clone().cpu().numpy().flatten()

        last_gae_lam = 0
        last_gae_lam_floored = 0
        advantages = np.zeros_like(self.advantages, dtype=np.float32)
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones.astype(np.float32)
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            delta_floored = self.rewards[step] + self.gamma * next_values * next_non_terminal - max(lower_bounds[step], self.values[step])
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            last_gae_lam_floored = delta_floored + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam_floored
            # using original advantage for computing returns
            advantages[step] = last_gae_lam
            # use floored advantage for updating the policy
            self.advantages[step] = last_gae_lam_floored

        self.returns = advantages + self.values