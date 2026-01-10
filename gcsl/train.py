import argparse

import torch
import torch.nn as nn
import numpy as np

from tensordict.nn import TensorDictModule
from tensordict import TensorDict

from torchrl.modules import ProbabilisticActor
from torchrl.modules.distributions import MaskedCategorical
from torchrl.envs.transforms import ActionMask, TransformedEnv

from .model import GoalConditionedActor
from .buffer import GCSLReplayBuffer
from .utils import collect_trajectory

from jani import TorchRLJANIEnv


def train_one_step(
        model: GoalConditionedActor, 
        rb: GCSLReplayBuffer, 
        criterion: nn.Module, 
        optimizer: torch.optim.Optimizer,
        batch_size: int, 
        device: torch.device):
    """Perform one training step for GCSL."""
    model.train()
    batch = rb.sample_batch(batch_size=batch_size)  # Sample a batch of transitions
    obs = batch["current_observation"].to(device)
    condition = batch["reached_condition"].to(device)
    actions = batch["selected_action"].to(device)
    valid_actions = batch["valid_actions"].to(device)
    # Forward pass through the student actor
    embedding, logits = model.get_student_embedding_and_logits(obs, condition)
    # set logits of invalid actions to large negative value
    logits = logits.masked_fill(~valid_actions.bool(), -1e9)
    # Compute loss
    loss = criterion(logits, actions.long())
    # Backpropagation and optimization step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def train_model(
        env: TorchRLJANIEnv,
        model: GoalConditionedActor,
        rb: GCSLReplayBuffer,
        hyperparams: dict,
        device: torch.device):
    """Train the GCSL model."""
    # Warm up the replay buffer with random trajectories
    warm_up(env, rb, num_trajectories=hyperparams.get("warmup_trajectories", 10))

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams.get("learning_rate", 1e-3))
    
    num_steps = hyperparams.get("num_steps", 1000)
    for step in range(num_steps):
        loss = train_one_step(model, rb, criterion, optimizer, batch_size=hyperparams.get("batch_size", 64), device=device)
        # Collect new trajectory and add to replay buffer
        trajectory = collect_trajectory(env, model, max_horizon=hyperparams.get("max_horizon", 2048))
        rb.add_trajectory(trajectory)

        if (step + 1) % 100 == 0:
            print(f"Step [{step + 1}/{num_steps}], Loss: {loss:.4f}")
        
        if step % 500 == 0:
            avg_reward = evaluate_model(env, model, max_steps=hyperparams.get("max_horizon", 2048), num_episodes=10, device=device)
            print(f"Step [{step}/{num_steps}], Average Reward: {avg_reward:.4f}")


def warm_up(env: TorchRLJANIEnv, rb: GCSLReplayBuffer, num_trajectories: int, max_horizon: int = 2048):
    """Warm up the replay buffer with random trajectories."""
    env_with_mask = TransformedEnv(env, ActionMask()) # Ensure random actions respect action masks
    for _ in range(num_trajectories):
        trajectory = collect_trajectory(env_with_mask, max_horizon=max_horizon)
        rb.add_trajectory(trajectory)


def evaluate_model(env: TorchRLJANIEnv, model: GoalConditionedActor, max_steps: int, num_episodes: int, device: torch.device):
    """Evaluate the GCSL model."""
    model.eval()
    actor_module = TensorDictModule(
        module=model,
        in_keys=["observation"],
        out_keys=["logits"]
    )
    # Construct the actor
    actor = ProbabilisticActor(
        module=actor_module,
        in_keys={"logits": "logits", "mask": "action_mask"},
        distribution_class=MaskedCategorical,
        out_keys=["action"],
        return_log_prob=True
    )

    rewards = []
    with torch.no_grad():
        for _ in range(num_episodes):
            td = env.rollout(max_steps=max_steps, policy=actor)
            rewards.append(td.get("next", "reward").sum().item())
    avg_reward = np.mean(rewards)
    return avg_reward



def main():
    parser = argparse.ArgumentParser(description="Train a GCSL policy.")
    parser.add_argument('--jani_model', type=str, required=True, help="Path to the JANI model file.")
    parser.add_argument('--jani_property', type=str, default="", help="Path to the JANI property file.")
    parser.add_argument('--start_states', type=str, default="", help="Path to the start states file.")
    parser.add_argument('--objective', type=str, default="", help="Path to the objective file.")
    parser.add_argument('--failure_property', type=str, default="", help="Path to the failure property file.")
    parser.add_argument('--goal_reward', type=float, default=1.0, help="Reward for reaching the goal.")
    parser.add_argument('--failure_reward', type=float, default=-1.0, help="Reward for reaching failure state.")
    parser.add_argument('--use_oracle', action='store_true', help="Use Tarjan oracle for unsafe state detection.")
    parser.add_argument('--unsafe_reward', type=float, default=-0.01, help="Reward for unsafe states when using oracle.")
    parser.add_argument('--seed', type=int, default=42, help="Random seed.")
    parser.add_argument('--device', type=str, default='cpu', help="Device to use for training (cpu or cuda).")

    args = parser.parse_args()
    # Create training environment
    file_args = {
        'jani_model_path': args.jani_model,
        'jani_property_path': args.jani_property,
        'start_states_path': args.start_states,
        'objective_path': args.objective,
        'failure_property_path': args.failure_property,
        'goal_reward': args.goal_reward,
        'failure_reward': args.failure_reward,
        'seed': args.seed,
        'use_oracle': args.use_oracle,
        'unsafe_reward': args.unsafe_reward,
    }
    env = TorchRLJANIEnv(**file_args)

    # Define hyperparameters
    hyperparams = {
        "learning_rate": 1e-3,
        "batch_size": 64,
        "num_steps": 10000,
        "max_horizon": 2048,
        "warmup_trajectories": 100,
        "hidden_sizes": [256, 256],
    }
    # Initialize model and replay buffer
    obs_size = env.observation_spec["observation"].shape[0]
    condition_size = env.observation_spec["condition"].shape[0]
    action_size = env.n_actions
    model = GoalConditionedActor(
        obs_size=obs_size,
        condition_size=condition_size,
        action_size=action_size,
        use_teacher=False,
        student_hidden_sizes=hyperparams["hidden_sizes"]
    )
    rb = GCSLReplayBuffer(buffer_size=1000000, max_horizon=hyperparams.get("max_horizon", 2048))
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    # Train the model
    train_model(env, model, rb, hyperparams, device)


if __name__ == "__main__":
    main()