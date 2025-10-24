import torch
import torch.nn as nn
import numpy as np

from torchrl.envs import Transform
from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage


class GCSLReplayBuffer:
    """Replay buffer for Goal-Conditioned Supervised Learning (GCSL)."""

    class SampleIndexTransform(Transform):
        """Transform for sampling start and end indices for GCSL."""

        def __init__(self, env):
            super().__init__(in_keys=[], out_keys=['current_obs', 'goal_conditions'])
            self.env = env

        def forward(self, x):
            batch_size = x.batch_size[0] 
            lengths = x["episode_length"]  # Assuming episode_length is provided in x
            # Sample start and end indices for each sampled trajectory
            first_idxs = (np.random.rand(batch_size) * lengths.numpy()).astype(int)
            second_idxs = (np.random.rand(batch_size) * lengths.numpy()).astype(int)
            # Ensure start_idx is less than end_idx
            start_idxs = np.minimum(first_idxs, second_idxs)
            end_idxs = np.maximum(first_idxs, second_idxs) + 1
            # The observation is of shape (batch_size, max_length, obs_dim)
            x["current_obs"] = x["observation"][torch.arange(batch_size), start_idxs, :]
            # Extract the goal condition based on the reached state at end_idx
            reached_state = x["observation"][torch.arange(batch_size), end_idxs, :]
            x['goal_conditions'] = self.env.compute_goal_condition(reached_state)
            return x

    def __init__(self, buffer_size: int, max_horizon: int = 2048):
        self.buffer_size = buffer_size
        self.max_horizon = max_horizon
        self.storage = LazyTensorStorage(max_size=buffer_size)
        self.replay_buffer = TensorDictReplayBuffer(storage=self.storage)