"""
Trajectory sampling module for offline reinforcement learning.

This module provides functionality to sample trajectories from JANI environments
and store them in TorchRL replay buffers with safety information for each transition.
"""

import sys
import torch
import numpy as np
import pandas as pd

from pathlib import Path
from typing import Optional

from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase
from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage
from torchrl.data.replay_buffers.samplers import RandomSampler

from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
    TimeRemainingColumn
)

from jani import TorchRLJANIEnv as JANIEnv


class RandomPolicy(TensorDictModuleBase):
    """A random policy that samples valid actions uniformly from the action mask."""

    in_keys = ["action_mask"]
    out_keys = ["action"]

    def __init__(self, n_actions: int):
        super().__init__()
        self.n_actions = n_actions

    def forward(self, td: TensorDict) -> TensorDict:
        action_mask = td.get("action_mask")

        # Handle batched input
        if action_mask.dim() == 1:
            # Single observation
            valid_actions = torch.where(action_mask)[0]
            if len(valid_actions) == 0:
                # No valid actions, select action 0 (will be handled by environment)
                action = torch.tensor(0, dtype=torch.int64)
            else:
                idx = torch.randint(len(valid_actions), (1,)).item()
                action = valid_actions[idx]
            td.set("action", action)
        else:
            # Batched observations
            batch_size = action_mask.shape[0]
            actions = []
            for i in range(batch_size):
                valid_actions = torch.where(action_mask[i])[0]
                if len(valid_actions) == 0:
                    actions.append(0)
                else:
                    idx = torch.randint(len(valid_actions), (1,)).item()
                    actions.append(valid_actions[idx].item())
            td.set("action", torch.tensor(actions, dtype=torch.int64))

        return td


def sample_trajectories(
    env: JANIEnv,
    policy: Optional[TensorDictModuleBase] = None,
    num_episodes: int = 1000,
    max_steps_per_episode: int = 256,
    buffer_size: Optional[int] = None,
    show_progress: bool = True,
    seed: Optional[int] = None
) -> TensorDictReplayBuffer:
    """
    Sample trajectories from a JANI environment and store them in a replay buffer.

    This function collects transitions from the environment, recording for each step:
    - Current observation
    - Action taken
    - Whether the action is safe (is_action_safe)
    - Whether the current state is safe (is_state_safe)
    - Next observation
    - Reward
    - Done flags (terminated, truncated)
    - Whether the next state is safe (next_is_state_safe)

    Args:
        env: The JANI TorchRL environment to sample from.
        policy: The policy to use for action selection. If None, uses a random policy.
        num_episodes: Number of episodes to collect.
        max_steps_per_episode: Maximum steps per episode before truncation.
        buffer_size: Maximum size of the replay buffer. If None, uses num_episodes * max_steps_per_episode.
        show_progress: Whether to show a progress bar.
        seed: Random seed for reproducibility.

    Returns:
        A TensorDictReplayBuffer containing the sampled trajectories with safety information.
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Use random policy if none provided
    if policy is None:
        policy = RandomPolicy(env.n_actions)

    # Determine buffer size
    if buffer_size is None:
        buffer_size = num_episodes * max_steps_per_episode

    # Create replay buffer with lazy storage
    replay_buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(max_size=buffer_size),
        sampler=RandomSampler(),
    )

    # Track episode statistics
    episode_lengths = []
    episode_rewards = []
    safe_action_counts = []

    progress_context = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        transient=False,
        disable=not show_progress or not sys.stdout.isatty()
    )

    with progress_context as progress:
        task = progress.add_task("Sampling trajectories", total=num_episodes)

        for episode_idx in range(num_episodes):
            # Reset environment
            td = env.reset()
            episode_transitions = []
            episode_reward = 0.0
            episode_safe_actions = 0

            for step in range(max_steps_per_episode):
                # Store current state info
                current_obs = td.get("observation").clone()
                current_action_mask = td.get("action_mask").clone()

                # Select action using policy
                td_with_action = policy(td.clone())
                action = td_with_action.get("action").item()

                # Use current_state_safety_with_action(action) to get BOTH:
                # 1. Whether the current state is safe
                # 2. Whether the selected action is safe (if safe_action == action)
                # This avoids redundant oracle calls
                current_is_state_safe, safe_action = env.current_state_safety_with_action(action)
                is_action_safe = (safe_action == action)

                # Take the step
                td.set("action", torch.tensor(action, dtype=torch.int64))
                next_td = env.step(td)

                # Extract reward and done flags
                reward = next_td.get(("next", "reward")).item()
                done = next_td.get(("next", "done")).item()
                terminated = next_td.get(("next", "terminated")).item()
                truncated = next_td.get(("next", "truncated")).item()
                next_obs = next_td.get(("next", "observation")).clone()
                next_action_mask = next_td.get(("next", "action_mask")).clone()

                # Create transition tensordict
                # Note: We don't include next.is_state_safe to avoid an extra oracle call.
                # The next state's safety will be recorded as is_state_safe in the following transition.
                transition = TensorDict({
                    "observation": current_obs,
                    "action": torch.tensor(action, dtype=torch.int64),
                    "action_mask": current_action_mask,
                    "is_action_safe": torch.tensor([is_action_safe], dtype=torch.bool),
                    "is_state_safe": torch.tensor([current_is_state_safe], dtype=torch.bool),
                    "episode": torch.tensor([episode_idx], dtype=torch.int64),
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
                episode_reward += reward
                if is_action_safe:
                    episode_safe_actions += 1

                if done:
                    break

                # Update for next iteration
                td = next_td.get("next").clone()

            # Add episode transitions to replay buffer
            if episode_transitions:
                episode_td = torch.stack(episode_transitions, dim=0)
                replay_buffer.extend(episode_td)
                episode_lengths.append(len(episode_transitions))
                episode_rewards.append(episode_reward)
                safe_action_counts.append(episode_safe_actions)

            progress.update(task, advance=1)

    # Print statistics
    if show_progress:
        print(f"\nSampling complete:")
        print(f"  Total episodes: {num_episodes}")
        print(f"  Total transitions: {len(replay_buffer)}")
        print(f"  Avg episode length: {np.mean(episode_lengths):.2f}")
        print(f"  Avg episode reward: {np.mean(episode_rewards):.4f}")
        print(f"  Avg safe actions per episode: {np.mean(safe_action_counts):.2f}")
        safe_action_rate = sum(safe_action_counts) / sum(episode_lengths) if sum(episode_lengths) > 0 else 0
        print(f"  Overall safe action rate: {safe_action_rate:.2%}")

    return replay_buffer


def sample_trajectories_with_safety_ratio(
    env: JANIEnv,
    policy: Optional[TensorDictModuleBase] = None,
    num_episodes: int = 1000,
    max_steps_per_episode: int = 256,
    target_safe_ratio: float = 0.5,
    buffer_size: Optional[int] = None,
    show_progress: bool = True,
    seed: Optional[int] = None
) -> TensorDictReplayBuffer:
    """
    Sample trajectories with a target ratio of safe vs unsafe actions.

    This is useful for creating balanced datasets for offline RL experiments.
    The function tries to achieve the target ratio by biasing action selection.

    Args:
        env: The JANI TorchRL environment to sample from.
        policy: Base policy for action selection. If None, uses random policy.
        num_episodes: Number of episodes to collect.
        max_steps_per_episode: Maximum steps per episode.
        target_safe_ratio: Target ratio of safe actions (0.0 to 1.0).
        buffer_size: Maximum replay buffer size.
        show_progress: Whether to show progress bar.
        seed: Random seed.

    Returns:
        A TensorDictReplayBuffer with the sampled trajectories.
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    if buffer_size is None:
        buffer_size = num_episodes * max_steps_per_episode

    replay_buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(max_size=buffer_size),
        sampler=RandomSampler(),
    )

    progress_context = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        transient=False,
        disable=not show_progress or not sys.stdout.isatty()
    )

    total_safe = 0
    total_unsafe = 0

    with progress_context as progress:
        task = progress.add_task("Sampling with safety ratio", total=num_episodes)

        for episode_idx in range(num_episodes):
            td = env.reset()

            for step in range(max_steps_per_episode):
                current_obs = td.get("observation").clone()
                current_action_mask = td.get("action_mask").clone()

                # Get valid actions
                action_mask = td.get("action_mask")
                valid_actions = torch.where(action_mask)[0].tolist()

                if not valid_actions:
                    break

                # Use current_state_safety_with_action() to get state safety and one safe action
                # This is more efficient than checking each action individually
                current_is_state_safe, known_safe_action = env.current_state_safety_with_action()

                # Classify actions as safe/unsafe
                # We already know one safe action (if state is safe), use it to speed up
                safe_actions = []
                unsafe_actions = []
                for a in valid_actions:
                    if current_is_state_safe and a == known_safe_action:
                        # We already know this action is safe from the oracle call
                        safe_actions.append(a)
                    elif env.is_state_action_safe(a):
                        safe_actions.append(a)
                    else:
                        unsafe_actions.append(a)

                # Calculate current ratio and decide which type to sample
                current_total = total_safe + total_unsafe
                if current_total == 0:
                    current_ratio = target_safe_ratio
                else:
                    current_ratio = total_safe / current_total

                # Decide whether to pick safe or unsafe action
                if current_ratio < target_safe_ratio and safe_actions:
                    # Need more safe actions
                    action = safe_actions[np.random.randint(len(safe_actions))]
                    is_action_safe = True
                elif current_ratio >= target_safe_ratio and unsafe_actions:
                    # Need more unsafe actions
                    action = unsafe_actions[np.random.randint(len(unsafe_actions))]
                    is_action_safe = False
                elif safe_actions:
                    action = safe_actions[np.random.randint(len(safe_actions))]
                    is_action_safe = True
                elif unsafe_actions:
                    action = unsafe_actions[np.random.randint(len(unsafe_actions))]
                    is_action_safe = False
                else:
                    action = valid_actions[0]
                    is_action_safe = False  # No safe actions available

                if is_action_safe:
                    total_safe += 1
                else:
                    total_unsafe += 1

                # Take the step
                td.set("action", torch.tensor(action, dtype=torch.int64))
                next_td = env.step(td)

                reward = next_td.get(("next", "reward")).item()
                done = next_td.get(("next", "done")).item()
                terminated = next_td.get(("next", "terminated")).item()
                truncated = next_td.get(("next", "truncated")).item()
                next_obs = next_td.get(("next", "observation")).clone()
                next_action_mask = next_td.get(("next", "action_mask")).clone()

                # Note: We don't include next.is_state_safe to avoid an extra oracle call.
                # The next state's safety will be recorded as is_state_safe in the following transition.
                transition = TensorDict({
                    "observation": current_obs,
                    "action": torch.tensor(action, dtype=torch.int64),
                    "action_mask": current_action_mask,
                    "is_action_safe": torch.tensor([is_action_safe], dtype=torch.bool),
                    "is_state_safe": torch.tensor([current_is_state_safe], dtype=torch.bool),
                    "episode": torch.tensor([episode_idx], dtype=torch.int64),
                    "next": TensorDict({
                        "observation": next_obs,
                        "action_mask": next_action_mask,
                        "reward": torch.tensor([reward], dtype=torch.float32),
                        "done": torch.tensor([done], dtype=torch.bool),
                        "terminated": torch.tensor([terminated], dtype=torch.bool),
                        "truncated": torch.tensor([truncated], dtype=torch.bool),
                    }, batch_size=[]),
                }, batch_size=[])

                replay_buffer.add(transition)

                if done:
                    break

                td = next_td.get("next").clone()

            progress.update(task, advance=1)

    if show_progress:
        final_ratio = total_safe / (total_safe + total_unsafe) if (total_safe + total_unsafe) > 0 else 0
        print(f"\nSampling complete:")
        print(f"  Total transitions: {len(replay_buffer)}")
        print(f"  Safe actions: {total_safe}, Unsafe actions: {total_unsafe}")
        print(f"  Final safe action ratio: {final_ratio:.2%} (target: {target_safe_ratio:.2%})")

    return replay_buffer


def save_trajectories_to_csv(
    replay_buffer: TensorDictReplayBuffer,
    output_path: str,
    include_safety: bool = True
) -> None:
    """
    Save trajectories from a replay buffer to a CSV file.

    The CSV format is compatible with the read_trajectories function in load_dataset.py.

    Format: obs_dim columns + [action, reward, terminated, truncated, safety]

    Args:
        replay_buffer: The replay buffer containing trajectories.
        output_path: Path to save the CSV file.
        include_safety: Whether to include safety information in the output.
    """
    # Sample all data from buffer
    all_data = replay_buffer.storage._storage[:len(replay_buffer)]

    rows = []
    for i in range(len(all_data)):
        transition = all_data[i]
        obs = transition.get("observation").numpy().tolist()
        action = transition.get("action").item()
        reward = transition.get(("next", "reward")).item()
        terminated = int(transition.get(("next", "terminated")).item())
        truncated = int(transition.get(("next", "truncated")).item())

        if include_safety:
            # Use is_state_safe from current state for compatibility
            safety = int(transition.get("is_state_safe").item())
            row = obs + [action, reward, terminated, truncated, safety]
        else:
            row = obs + [action, reward, terminated, truncated]

        rows.append(row)

    # Add final states (for transitions that end episodes)
    # The last row should have action=-1 to indicate terminal
    # Note: We don't have next state safety info, so we use 0 as placeholder for terminal states
    for i in range(len(all_data)):
        transition = all_data[i]
        if transition.get(("next", "done")).item():
            next_obs = transition.get(("next", "observation")).numpy().tolist()
            if include_safety:
                # Terminal states don't have safety info computed, use 0 as placeholder
                row = next_obs + [-1, 0, 0, 0, 0]
            else:
                row = next_obs + [-1, 0, 0, 0]
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, header=False)
    print(f"Saved {len(rows)} rows to {output_path}")


def load_replay_buffer_from_disk(
    path: str,
    batch_size: int = 64
) -> TensorDictReplayBuffer:
    """
    Load a replay buffer from disk.

    Args:
        path: Path to the saved replay buffer directory.
        batch_size: Batch size for sampling.

    Returns:
        The loaded TensorDictReplayBuffer.
    """
    from torchrl.data import TensorStorage

    storage = TensorStorage.loads(path)
    replay_buffer = TensorDictReplayBuffer(
        storage=storage,
        sampler=RandomSampler(),
        batch_size=batch_size
    )
    return replay_buffer


def main():
    """Main function demonstrating trajectory sampling."""
    import argparse

    parser = argparse.ArgumentParser(description="Sample trajectories from JANI environment")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the JANI model file."
    )
    parser.add_argument(
        "--property_path",
        type=str,
        required=True,
        help="Path to the JANI property file."
    )
    parser.add_argument(
        "--start_states",
        type=str,
        required=True,
        help="Path to the start states file."
    )
    parser.add_argument(
        "--objective",
        type=str,
        default="",
        help="Path to the objective file."
    )
    parser.add_argument(
        "--failure_property",
        type=str,
        default="",
        help="Path to the failure property file."
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=1000,
        help="Number of episodes to sample."
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=256,
        help="Maximum steps per episode."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save the replay buffer. If not provided, buffer is not saved."
    )
    parser.add_argument(
        "--target_safe_ratio",
        type=float,
        default=None,
        help="Target ratio of safe actions (0-1). If set, uses biased sampling."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed."
    )
    parser.add_argument(
        "--disable_oracle_cache",
        action="store_true",
        help="Disable oracle caching (slower but less memory)."
    )
    parser.add_argument(
        "--reduced_memory_mode",
        action="store_true",
        help="Enable reduced memory mode for oracle."
    )
    args = parser.parse_args()

    # Create environment
    env = JANIEnv(
        jani_model_path=args.model_path,
        jani_property_path=args.property_path,
        start_states_path=args.start_states,
        objective_path=args.objective,
        failure_property_path=args.failure_property,
        seed=args.seed,
        use_oracle=False,  # We handle oracle manually for safety checks
        disable_oracle_cache=args.disable_oracle_cache,
        reduced_memory_mode=args.reduced_memory_mode
    )

    print(f"Environment created:")
    print(f"  Observation dim: {env.obs_dim}")
    print(f"  Number of actions: {env.n_actions}")
    print()

    # Sample trajectories
    if args.target_safe_ratio is not None:
        replay_buffer = sample_trajectories_with_safety_ratio(
            env=env,
            policy=None,  # Random policy
            num_episodes=args.num_episodes,
            max_steps_per_episode=args.max_steps,
            target_safe_ratio=args.target_safe_ratio,
            seed=args.seed
        )
    else:
        replay_buffer = sample_trajectories(
            env=env,
            policy=None,  # Random policy
            num_episodes=args.num_episodes,
            max_steps_per_episode=args.max_steps,
            seed=args.seed
        )

    # Save results if output directory specified
    if args.output_dir is not None:
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save replay buffer
        rb_path = output_path / "replay_buffer"
        replay_buffer.storage.dumps(str(rb_path))
        print(f"Replay buffer saved to {rb_path}")

        # Save as CSV for compatibility
        csv_path = output_path / "trajectories.csv"
        save_trajectories_to_csv(replay_buffer, str(csv_path))

    # Demonstrate sampling from buffer
    print("\nSample batch from replay buffer:")
    batch = replay_buffer.sample(batch_size=4)
    print(f"  Batch keys: {list(batch.keys())}")
    print(f"  Observations shape: {batch['observation'].shape}")
    print(f"  Actions: {batch['action']}")
    print(f"  Is action safe: {batch['is_action_safe'].squeeze()}")
    print(f"  Is state safe: {batch['is_state_safe'].squeeze()}")
    print(f"  Rewards: {batch['next']['reward'].squeeze()}")


if __name__ == "__main__":
    main()
