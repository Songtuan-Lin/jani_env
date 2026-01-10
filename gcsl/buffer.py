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

        def __init__(self):
            super().__init__(in_keys=["observation", "episode_length", "condition"], out_keys=['current_observation', 'reached_condition', 'selected_action', 'valid_actions'])

        def forward(self, x: TensorDict) -> TensorDict:
            batch_size = x.batch_size[0] 
            lengths = x["episode_length"]  # Assuming episode_length is provided in x
            # Sample start and end indices for each sampled trajectory
            first_idxs = (np.random.rand(batch_size) * lengths.numpy()).astype(int)
            second_idxs = (np.random.rand(batch_size) * lengths.numpy()).astype(int)
            # Ensure start_idx is less than end_idx
            start_idxs = np.minimum(first_idxs, second_idxs)
            end_idxs = np.maximum(first_idxs, second_idxs)
            # The observation is of shape (batch_size, max_length, obs_dim)
            # Expected shape after indexing: (batch_size, obs_dim)
            x["current_observation"] = x["observation"][torch.arange(batch_size), start_idxs, :]
            # Fetch the conditions at the end indices (expected shape: (batch_size, condition_dim))
            x["reached_condition"] = x["condition"][torch.arange(batch_size), end_idxs, :]
            # Action is expected to have shape (batch_size, max_length) (assuming discrete actions)
            x["selected_action"] = x["action"][torch.arange(batch_size), start_idxs]
            # Action mask is expected to have shape (batch_size, max_length, n_actions)
            x["valid_actions"] = x["action_mask"][torch.arange(batch_size), start_idxs, :]
            # # Extract the goal condition based on the reached state at end_idx
            # reached_state = x["observation"][torch.arange(batch_size), end_idxs, :]
            # x['reached_conditions'] = self.env.extract_reached_conditions(reached_state)
            return x

    def __init__(self, buffer_size: int, max_horizon: int = 2048):
        self.buffer_size = buffer_size
        self.max_horizon = max_horizon
        # Initialize storage for replay buffer
        self.storage = LazyTensorStorage(max_size=buffer_size)
        self.replay_buffer = TensorDictReplayBuffer(
            storage=self.storage,
            transform=self.SampleIndexTransform(),
        )

    def sample_batch(self, batch_size: int):
        """Sample a batch of transitions for GCSL training."""
        sampled_data = self.replay_buffer.sample(batch_size)
        return sampled_data
    
    def add_trajectory(self, trajectory: TensorDict):
        """Add a trajectory to the replay buffer."""
        # Padding the trajectory to max_horizon if necessary
        trajectory_len = trajectory["observation"].shape[1] # observation is expected to be of shape (1, trajectory_length, obs_dim)
        if trajectory_len < self.max_horizon:
            target_keys = ["observation", "action", "condition", "next_observation", "action_mask"]
            # Pad each key in the trajectory, all keys are expected to have shape (1, trajectory_length, dim)
            for key in target_keys:
                trajectory[key] = torch.nn.functional.pad(
                    trajectory[key],
                    (0, 0, 0, self.max_horizon - trajectory_len),
                    mode='constant',
                    value=torch.nan
                )
        # Set the episode length
        trajectory["episode_length"] = torch.tensor([trajectory_len])
        self.replay_buffer.add(trajectory)

    # def extend_trajectories(self, trajectories: TensorDict):
    #     """Extend the replay buffer with multiple trajectories."""
    #     self.replay_buffer.extend(trajectories)