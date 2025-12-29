import torch
import torch.nn as nn
import numpy as np

from tensordict import TensorDict
from torchrl.envs import Transform
from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage


class GCSLReplayBuffer:
    """Replay buffer for Goal-Conditioned Supervised Learning (GCSL)."""

    class SampleIndexTransform(Transform):
        """Transform for sampling start and end indices for GCSL."""

        def __init__(self, env):
            super().__init__(in_keys=[], out_keys=['current_obs', 'goal_conditions'])
            self.env = env

        def forward(self, x: TensorDict) -> TensorDict:
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
            x['reached_conditions'] = self.env.extract_reached_conditions(reached_state)
            return x

    def __init__(self, buffer_size: int, env, max_horizon: int = 2048):
        self.buffer_size = buffer_size
        self.env = env
        self.max_horizon = max_horizon
        # Initialize storage for replay buffer
        self.storage = LazyTensorStorage(max_size=buffer_size)
        self.replay_buffer = TensorDictReplayBuffer(
            storage=self.storage,
            transform=self.SampleIndexTransform(env),
        )

    def sample_batch(self, batch_size: int):
        """Sample a batch of transitions for GCSL training."""
        sampled_data = self.replay_buffer.sample(batch_size)
        return sampled_data
    
    def add_trajectory(self, trajectory: TensorDict):
        """Add a trajectory to the replay buffer."""
        self.replay_buffer.add(trajectory)