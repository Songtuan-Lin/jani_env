import torch
import numpy as np
import pandas as pd

from torchrl.data import TensorDictReplayBuffer, TensorStorage, SliceSamplerWithoutReplacement
from tensordict import TensorDict


def read_trajectories(file_path: str) -> TensorDict:
    df = pd.read_csv(file_path, header=None)
    # next component in the tensordict
    next_observations, rewards, dones, term_signs, trunc_signs, safeties = [], [], [], [], [], []
    # root component in the tensordict
    observations, actions, root_dones, root_term_signs, root_trunc_signs, root_safeties, episodes = [], [], [], [], [], [], []
    episode_counter = 0
    for r in range(df.shape[0] - 1):
        # Get the current and next observations, actions, rewards, and termination signals
        obs = df.iloc[r, :-5]
        next_obs = df.iloc[r + 1, :-5]
        action = df.iloc[r, -5]
        reward = df.iloc[r, -4]
        term_sign = df.iloc[r, -3]
        trunc_sign = df.iloc[r, -2]
        safety = df.iloc[r, -1]
        next_safety = df.iloc[r + 1, -1]
        # Skip transitions where action is -1 (terminal state)
        if action == -1:
            continue
        episodes.append(episode_counter)
        if term_sign == 1 or trunc_sign == 1:
            episode_counter += 1
        observations.append(obs)
        next_observations.append(next_obs)
        if len(term_signs) > 0 and term_signs[-1] == 1:
            # If the previous state was terminal, mark this transition as starting a new episode
            root_term_signs.append(1)
        else:
            root_term_signs.append(0)
        if len(trunc_signs) > 0 and trunc_signs[-1] == 1:
            # If the previous state was truncated, mark this transition as starting a new episode
            root_trunc_signs.append(1)
        else:
            root_trunc_signs.append(0)
        root_dones.append(1 if term_sign == 1 or trunc_sign == 1 else 0)
        actions.append(action)
        rewards.append(reward)
        term_signs.append(term_sign)
        trunc_signs.append(trunc_sign)
        dones.append(1 if term_sign == 1 or trunc_sign == 1 else 0)
        root_safeties.append(safety)
        safeties.append(next_safety)
    # Create a TensorDict from the collected lists
    return TensorDict({
        "observation": torch.tensor(observations, dtype=torch.float32),
        "action": torch.tensor(actions, dtype=torch.int64).view(-1, 1),
        "done": torch.tensor(root_dones, dtype=torch.int64).view(-1, 1),
        "terminated": torch.tensor(root_term_signs, dtype=torch.int64).view(-1, 1),
        "truncated": torch.tensor(root_trunc_signs, dtype=torch.int64).view(-1, 1),
        "safety": torch.tensor(root_safeties, dtype=torch.int64).view(-1, 1),
        "episode": torch.tensor(episodes, dtype=torch.int64).view(-1, 1),
        "next": TensorDict({
            "observation": torch.tensor(next_observations, dtype=torch.float32),
            "reward": torch.tensor(rewards, dtype=torch.float32).view(-1, 1),
            "done": torch.tensor(dones, dtype=torch.int64).view(-1, 1),
            "terminated": torch.tensor(term_signs, dtype=torch.int64).view(-1, 1),
            "truncated": torch.tensor(trunc_signs, dtype=torch.int64).view(-1, 1),
            "safety": torch.tensor(safeties, dtype=torch.int64).view(-1, 1)
        }, batch_size=len(next_observations))
    }, batch_size=len(observations))


def create_replay_buffer(tensordict: TensorDict, buffer_size: int = 100000, batch_size: int = 32) -> TensorDictReplayBuffer:
    storage = TensorStorage(tensordict, max_size=buffer_size, device='cpu')
    sampler = SliceSamplerWithoutReplacement(storage, end_key=("next", "done"), traj_key="episode", batch_size=batch_size)
    replay_buffer = TensorDictReplayBuffer(storage=storage, sampler=sampler, batch_size=batch_size)
    return replay_buffer

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, required=True, help="Path to the CSV file containing trajectories.")
    args = parser.parse_args()

    td = read_trajectories(args.file_path)