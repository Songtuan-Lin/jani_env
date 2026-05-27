import torch
import numpy as np
import pandas as pd

from pathlib import Path

from torchrl.data import TensorDictReplayBuffer, TensorStorage, SliceSampler
from torchrl.data.replay_buffers import LazyTensorStorage
from torchrl.data.replay_buffers.samplers import RandomSampler
from torchrl.envs.transforms import ActionMask, TransformedEnv
from torchrl.collectors import SyncDataCollector
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase

from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn, TimeElapsedColumn

from jani import TorchRLJANIEnv as JANIEnv


def load_replay_buffer(path: str, batch_size: int = 64) -> TensorDictReplayBuffer:
    """
    Load a replay buffer from disk that was saved using storage.dumps().

    This loads replay buffers saved by sample.py's sample_trajectories function.
    The loaded buffer contains transitions with safety information (is_action_safe,
    is_state_safe) required for DiscreteIQLLossActionSafeLB.

    Args:
        path: Path to the saved replay buffer directory (created by storage.dumps()).
        batch_size: Batch size for sampling from the buffer.

    Returns:
        A TensorDictReplayBuffer ready for training.
    """
    import json

    path = Path(path)

    # Read meta.json to get the storage shape (max_size)
    meta_path = path / "meta.json"
    with open(meta_path) as f:
        meta = json.load(f)
    max_size = meta["shape"][0]

    # Create storage with correct size and load data
    storage = LazyTensorStorage(max_size=max_size)
    storage.loads(path)

    replay_buffer = TensorDictReplayBuffer(
        storage=storage,
        sampler=RandomSampler(),
        batch_size=batch_size
    )
    return replay_buffer


def read_trajectories(file_path: str, action_dim: int = None, penalize_unsafe: bool = False, unsafe_reward: float = -0.01) -> TensorDict:
    df = pd.read_csv(file_path, header=None)
    if action_dim is None:
        print("Warning: action_dim not provided, inferring from data. This may lead to incorrect results if the dataset does not contain all possible actions.")
        action_dim = int(df.iloc[:, -5].max()) + 1  # Actions are 0-indexed integers
    # next component in the tensordict
    next_observations, rewards, dones, term_signs, trunc_signs, safeties = [], [], [], [], [], []
    # root component in the tensordict
    observations, actions, root_dones, root_term_signs, root_trunc_signs, root_safeties, episodes, action_masks = [], [], [], [], [], [], [], []
    episode_counter = 0
    with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            transient=False,
        ) as progress:
        task = progress.add_task("Reading Trajectories", total=df.shape[0] - 1)

        for r in range(df.shape[0] - 1):
            # Get the current and next observations, actions, rewards, and termination signals
            obs = df.iloc[r, :-5]
            next_obs = df.iloc[r + 1, :-5]
            action = df.iloc[r, -5]
            reward = df.iloc[r, -4]
            if penalize_unsafe and df.iloc[r + 1, -1] == 0 and reward != -1.0:
                assert reward == 0.0, "Expected reward to be 0 for unsafe transitions"
                reward = unsafe_reward  # Penalize unsafe transitions
            term_sign = df.iloc[r, -3]
            trunc_sign = df.iloc[r, -2]
            safety = df.iloc[r, -1]
            next_safety = df.iloc[r + 1, -1]
            action_mask = [True] * action_dim # In the offline setting, all actions are valid
            # Skip transitions where action is -1 (terminal state)
            if action == -1:
                progress.advance(task, advance=1)
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
            action_masks.append(action_mask)

            progress.advance(task, advance=1)
    # Create a TensorDict from the collected lists
    return TensorDict({
        "observation": torch.tensor(observations, dtype=torch.float32),
        "action": torch.tensor(actions, dtype=torch.int64),
        "done": torch.tensor(root_dones, dtype=torch.int64).view(-1, 1),
        "terminated": torch.tensor(root_term_signs, dtype=torch.int64).view(-1, 1),
        "truncated": torch.tensor(root_trunc_signs, dtype=torch.int64).view(-1, 1),
        "safety": torch.tensor(root_safeties, dtype=torch.int64).view(-1, 1),
        "episode": torch.tensor(episodes, dtype=torch.int64),
        "action_mask": torch.tensor(action_masks, dtype=torch.bool),
        "next": TensorDict({
            "observation": torch.tensor(next_observations, dtype=torch.float32),
            "reward": torch.tensor(rewards, dtype=torch.float32).view(-1, 1),
            "done": torch.tensor(dones, dtype=torch.int64).view(-1, 1),
            "terminated": torch.tensor(term_signs, dtype=torch.int64).view(-1, 1),
            "truncated": torch.tensor(trunc_signs, dtype=torch.int64).view(-1, 1),
            "safety": torch.tensor(safeties, dtype=torch.int64).view(-1, 1)
        }, batch_size=len(next_observations))
    }, batch_size=len(observations))


def create_replay_buffer(tensordict: TensorDict, num_slices: int = 32, batch_size: int = 32) -> TensorDictReplayBuffer:
    storage = TensorStorage(tensordict, device='cpu')
    sampler = SliceSampler(num_slices=num_slices, end_key=("next", "done"), traj_key="episode")
    replay_buffer = TensorDictReplayBuffer(storage=storage, sampler=sampler, batch_size=batch_size)
    return replay_buffer


def collect_trajectories(
        env: JANIEnv,
        policy: TensorDictModuleBase | None,
        num_total_steps: int,
        n_steps: int,
        include_safety: bool = True) -> TensorDictReplayBuffer:
    """Collect trajectories from the environment using the provided policy.

    Args:
        env: The JANI environment to collect from.
        policy: The policy to use for action selection. If None, uses random valid actions.
        num_total_steps: Maximum number of steps to collect.
        n_steps: Maximum steps per episode.
        include_safety: If True, includes is_action_safe and is_state_safe in transitions.
                       Required for DiscreteIQLLossActionSafeLB.

    Returns:
        A replay buffer containing the collected transitions.
    """
    import sys

    # Create the replay buffer
    replay_buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(max_size=num_total_steps),
        sampler=RandomSampler(),
    )

    if not include_safety:
        # Use fast rollout-based collection without safety info
        transformed_env = TransformedEnv(env, ActionMask())
        with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TextColumn("•"),
                TimeElapsedColumn(),
                TextColumn("•"),
                TimeRemainingColumn(),
                transient=False,
                disable=not sys.stdout.isatty()
            ) as progress:
            task = progress.add_task("Collecting trajectories", total=10000)
            for _ in range(10000):
                rollout = transformed_env.rollout(max_steps=n_steps)
                replay_buffer.extend(rollout)
                progress.update(task, advance=1)
        return replay_buffer

    # Collection with safety information
    total_steps = 0

    with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            transient=False,
            disable=not sys.stdout.isatty()
        ) as progress:
        task = progress.add_task("Collecting trajectories with safety", total=num_total_steps)

        while total_steps < num_total_steps:
            # Reset environment
            td = env.reset()
            episode_transitions = []

            for _ in range(n_steps):
                # Store current state info
                current_obs = td.get("observation").clone()
                current_action_mask = td.get("action_mask").clone()

                # Select action
                if policy is not None:
                    td_with_action = policy(td.clone())
                    action = td_with_action.get("action").item()
                else:
                    # Random valid action
                    valid_actions = torch.where(current_action_mask)[0]
                    if len(valid_actions) == 0:
                        break
                    action = valid_actions[torch.randint(len(valid_actions), (1,)).item()].item()

                # Get safety information using oracle
                current_is_state_safe, safe_action = env.current_state_safety_with_action(action)
                is_action_safe = (safe_action == action)

                # Take the step
                td.set("action", torch.tensor(action, dtype=torch.int64))
                next_td = env.step(td)

                # Extract transition data
                reward = next_td.get(("next", "reward")).item()
                done = next_td.get(("next", "done")).item()
                terminated = next_td.get(("next", "terminated")).item()
                truncated = next_td.get(("next", "truncated")).item()
                next_obs = next_td.get(("next", "observation")).clone()
                next_action_mask = next_td.get(("next", "action_mask")).clone()

                # Create transition tensordict with safety info
                transition = TensorDict({
                    "observation": current_obs,
                    "action": torch.tensor(action, dtype=torch.int64),
                    "action_mask": current_action_mask,
                    "is_action_safe": torch.tensor([is_action_safe], dtype=torch.bool),
                    "is_state_safe": torch.tensor([current_is_state_safe], dtype=torch.bool),
                    "next": TensorDict({
                        "observation": next_obs,
                        "action_mask": next_action_mask,
                        "reward": torch.tensor([reward], dtype=torch.float32),
                        "done": torch.tensor([done], dtype=torch.bool),
                        "terminated": torch.tensor([terminated], dtype=torch.bool),
                        "truncated": torch.tensor([truncated], dtype=torch.bool),
                    }, batch_size=[]),
                }, batch_size=[])

                episode_transitions.append(transition)
                total_steps += 1
                progress.update(task, advance=1)

                if done or total_steps >= num_total_steps:
                    break

                # Update for next iteration
                td = next_td.get("next").clone()

            # Add episode transitions to replay buffer
            if episode_transitions:
                episode_td = torch.stack(episode_transitions, dim=0)
                replay_buffer.extend(episode_td)

            if total_steps >= num_total_steps:
                break

    return replay_buffer



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, required=True, help="Path to the CSV file containing trajectories.")
    args = parser.parse_args()

    td = read_trajectories(args.file_path)