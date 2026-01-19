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
    model.to(device)
    batch = rb.sample_batch(batch_size=batch_size)  # Sample a batch of transitions
    obs = batch["current_observation"].to(device)
    # print(f"Observation shape in training batch: {obs.shape}")
    condition = batch["reached_condition"].to(device)
    # print(f"Condition shape in training batch: {condition.shape}")
    print(f"Observation sampled: {obs}")
    print(f"Condition sampled: {condition}")
    actions = batch["selected_action"].to(device)
    valid_actions = batch["valid_actions"].to(device)
    assert valid_actions.bool().any(dim=-1).all(), "Some samples have no valid actions"
    chosen_valid = valid_actions.gather(-1, actions.long().unsqueeze(-1)).squeeze(-1)
    assert chosen_valid.all(), "Batch contains invalid (action, mask) pairs"
    # print(f"Valid actions shape in training batch: {valid_actions.shape}")
    # Forward pass through the student actor
    embedding, logits = model.get_student_embedding_and_logits(obs, condition)
    # Suggested by ChatGPT to improve stability
    masked_logits = logits.masked_fill(~valid_actions.bool(), float("-inf"))
    logp = torch.log_softmax(masked_logits, dim=-1)

    loss = torch.nn.NLLLoss()(logp, actions.long())

    # set logits of invalid actions to large negative value
    # logits = logits.masked_fill(~valid_actions.bool(), -1e9)
    # Compute loss
    # loss = criterion(logits, actions.long())

    # Backpropagation and optimization step
    optimizer.zero_grad()
    loss.backward()
    # Gradient clipping for stability
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss.item()


def train_model(
        env: TorchRLJANIEnv,
        model: GoalConditionedActor,
        rb: GCSLReplayBuffer,
        hyperparams: dict,
        device: torch.device,
        verbose: bool = True):
    """Train the GCSL model."""
    # Warm up the replay buffer with random trajectories
    if verbose:
        print("Warming up the replay buffer...")
    warm_up(env, rb, num_trajectories=hyperparams.get("warmup_trajectories", 10))
    # for k, v in rb.replay_buffer.storage._storage.items():
    #     print(f"Buffer key: {k}, shape: {v.shape}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams.get("learning_rate", 1e-3))
    
    num_steps = hyperparams.get("num_steps", 1000)
    for step in range(num_steps):
        if verbose and step % 500 == 0:
            avg_reward = evaluate_model(env, model, max_steps=hyperparams.get("max_horizon", 2048), num_episodes=100)
            print(f"Step [{step}/{num_steps}], Average Reward: {avg_reward:.4f}")
        # Perform one training step
        loss = train_one_step(model, rb, criterion, optimizer, batch_size=hyperparams.get("batch_size", 64), device=device)
        # Collect new trajectory and add to replay buffer
        for _ in range(1):  # collect multiple trajectories per step
            trajectory = collect_trajectory(env, model, max_horizon=hyperparams.get("max_horizon", 2048))
            rb.add_trajectory(trajectory)

        if verbose and (step + 1) % 100 == 0:
            print(f"Step [{step + 1}/{num_steps}], Loss: {loss:.4f}")


def objective(trial, env: TorchRLJANIEnv, device: torch.device):
    """Optuna objective function for hyperparameter tuning of GCSL."""
    # Suggest hyperparameters
    n_layers = trial.suggest_int('n_layers', 2, 5)
    hidden_sizes = []
    for i in range(n_layers):
        size = trial.suggest_int(f'hidden_size_{i}', 256, 512)
        hidden_sizes.append(size)
    
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_int('batch_size', 256, 2048)
    # num_steps = trial.suggest_int('num_steps', 10000, 200000)
    # warmup_trajectories = trial.suggest_int('warmup_trajectories', 100, 2000)

    # Initialize model and replay buffer
    obs_size = env.observation_spec["observation"].shape[0]
    condition_size = env.observation_spec["condition"].shape[0]
    action_size = env.n_actions 
    rb = GCSLReplayBuffer(buffer_size=5000, max_horizon=2048)
    model = GoalConditionedActor(
        obs_size=obs_size,
        condition_size=condition_size,
        action_size=action_size,
        use_teacher=False,
        student_hidden_sizes=hidden_sizes
    ).to(device)

    hyperparams = {
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "num_steps": 500,
        "warmup_trajectories": 2 * batch_size,  # ensure enough data for initial training
        "hidden_sizes": hidden_sizes,
    }

    # Train the model
    train_model(env, model, rb, hyperparams, device, verbose=False)

    # Evaluate the model
    avg_reward = evaluate_model(env, model, max_steps=2048, num_episodes=100)
    return avg_reward  # Optuna maximizes the objective


def warm_up(env: TorchRLJANIEnv, rb: GCSLReplayBuffer, num_trajectories: int, max_horizon: int = 2048):
    """Warm up the replay buffer with random trajectories."""
    env_with_mask = TransformedEnv(env, ActionMask()) # Ensure random actions respect action masks
    for _ in range(num_trajectories):
        trajectory = collect_trajectory(env_with_mask, max_horizon=max_horizon)
        rb.add_trajectory(trajectory)


def evaluate_model(env: TorchRLJANIEnv, model: GoalConditionedActor, max_steps: int, num_episodes: int):
    """Evaluate the GCSL model."""
    model.eval()
    model.cpu() # Ensure model is on CPU for evaluation
    actor_module = TensorDictModule(
        module=model,
        in_keys=["observation_with_goal"],
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
            rewards.append(td["next"]["reward"].sum().item())
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
    parser.add_argument('--hyperparams_tuning', action='store_true', help="Enable hyperparameter tuning with Optuna.")
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
    if args.device == 'cuda':
        if torch.cuda.is_available():
            print("Using GPU for training.")
            device = torch.device('cuda')
        elif torch.mps.is_available():
            print("Using MPS for training.")
            device = torch.device('mps')
        else:
            print("CUDA not available, using CPU.")
            device = torch.device('cpu')
    else:
        device = torch.device('cpu')

    # Define hyperparameters
    if args.hyperparams_tuning:
        print("Starting hyperparameter tuning with Optuna...")

        import optuna

        study = optuna.create_study(direction='maximize')
        study.optimize(
            lambda trial: objective(trial, env, device), 
            n_trials=15
        )

        print("Best hyperparameters found:")
        print(study.best_params)

        hyperparams = {
            "learning_rate": study.best_params['learning_rate'],
            "batch_size": study.best_params['batch_size'],
            "num_steps": 50000, # fixed number of steps for final training
            "warmup_trajectories": 2 * study.best_params['batch_size'],
            "hidden_sizes": [study.best_params[f'hidden_size_{i}'] for i in range(study.best_params['n_layers'])],
        }
    else:
        hyperparams = {
            "learning_rate": 3e-4,
            "batch_size": 256, # larger batch size works clearly better
            "num_steps": 200000,
            "max_horizon": 2048,
            "warmup_trajectories": 300,
            "hidden_sizes": [256, 256],
        }
    # Initialize model and replay buffer
    obs_size = env.observation_spec["observation"].shape[0]
    condition_size = env.observation_spec["condition"].shape[0]
    action_size = env.n_actions 
    rb = GCSLReplayBuffer(buffer_size=300, max_horizon=hyperparams.get("max_horizon", 2048))
    model = GoalConditionedActor(
        obs_size=obs_size,
        condition_size=condition_size,
        action_size=action_size,
        use_teacher=False,
        student_hidden_sizes=hyperparams["hidden_sizes"]
    ).to(device)
    # Train the model
    train_model(env, model, rb, hyperparams, device)
    evaluate_model(env, model, max_steps=2048, num_episodes=100, print_actions=True)


if __name__ == "__main__":
    main()