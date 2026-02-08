import torch
import torch.nn as nn
import argparse

from tensordict.nn import TensorDictModule

from torchrl.modules import MLP, ProbabilisticActor, ValueOperator
from torchrl.modules.distributions import MaskedCategorical
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type

from typing import Dict, Any
from collections import defaultdict
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn, TimeElapsedColumn

from jani.torchrl_env import JANIEnv
from utils import create_eval_file_args


def create_actor(hyperparams: Dict[str, Any], env: JANIEnv) -> TensorDictModule:
    """Create the actor network for the policy."""
    n_actions = env.n_actions
    input_size = env.observation_spec["observation"].shape[0]
    hidden_sizes = hyperparams.get("actor_hidden_sizes", [64, 128])
    dropout = hyperparams.get("actor_dropout", 0.2)
    activation_fn = hyperparams.get("activation_fn", nn.Tanh)
    # Build the actor network
    actor_backbone = MLP(
        in_features=input_size,
        out_features=n_actions,
        num_cells=hidden_sizes,
        dropout=dropout,
        activation_class=activation_fn,
    )
    # Wrap in TensorDictModule
    actor_module = TensorDictModule(
        module=actor_backbone,
        in_keys=["observation"],
        out_keys=["logits"],
    )
    # Create the probabilistic actor with masked categorical distribution
    actor = ProbabilisticActor(
        module=actor_module,
        in_keys={"logits": "logits", "mask": "action_mask"},
        out_keys=["action"],
        distribution_class=MaskedCategorical,
        return_log_prob=True,
    )
    return actor

def create_critic(hyperparams: Dict[str, Any], env: JANIEnv) -> TensorDictModule:
    """Create the critic network for value estimation."""
    input_size = env.observation_spec["observation"].shape[0]
    hidden_sizes = hyperparams.get("critic_hidden_sizes", [64, 128])
    dropout = hyperparams.get("critic_dropout", 0.2)
    activation_fn = hyperparams.get("activation_fn", nn.Tanh)
    # Build the critic network
    critic_backbone = MLP(
        in_features=input_size,
        out_features=1,
        num_cells=hidden_sizes,
        dropout=dropout,
        activation_class=activation_fn,
    )
    # Wrap in TensorDictModule
    critic_module = ValueOperator(
        module=critic_backbone,
        in_keys=["observation"],
    )
    return critic_module

def create_data_collector(hyperparams: Dict[str, Any], env: JANIEnv, policy: TensorDictModule) -> SyncDataCollector:
    """Create a data collector for experience gathering."""
    n_steps = hyperparams.get("n_steps", 2048)
    total_timesteps = hyperparams.get("total_timesteps", 1024000)
    collector = SyncDataCollector(
        create_env_fn=env,
        policy=policy,
        total_frames=total_timesteps,
        frames_per_batch=n_steps,
        split_trajs=False,
    )
    return collector

def create_replay_buffer(hyperparams: Dict[str, Any]) -> ReplayBuffer:
    """Create a replay buffer for experience storage."""
    # For PPO, buffer size must equal to n_steps
    buffer_size = hyperparams.get("n_steps", 100000)
    batch_size = hyperparams.get("batch_size", 64)
    # Create the storage
    storage = LazyTensorStorage(
        max_size=buffer_size,
        device=hyperparams.get("device", "cpu"),
    )
    # Create the sampler
    sampler = SamplerWithoutReplacement()
    # Create the replay buffer
    replay_buffer = ReplayBuffer(
        storage=storage,
        sampler=sampler,
    )
    return replay_buffer

def create_advantage_module(hyperparams: Dict[str, Any], value_module: nn.Module) -> nn.Module:
    """Create an advantage estimation module."""
    gae_lambda = hyperparams.get("gae_lambda", 0.95)
    gamma = hyperparams.get("gamma", 0.99)
    advantage_module = GAE(
        gamma=gamma,
        lmbda=gae_lambda,
        value_network=value_module, # This should be the critic module in PPO
        average_gae=True,
    )
    return advantage_module

def create_loss_module(hyperparams: Dict[str, Any], actor_module: TensorDictModule, critic_module: TensorDictModule) -> nn.Module:
    """Create the loss module for PPO."""
    clip_epsilon = hyperparams.get("clip_epsilon", 0.2)
    entropy_coef = hyperparams.get("ent_coef", 1e-4)
    critic_coeff = hyperparams.get("critic_coeff", 1.0)

    loss_module = ClipPPOLoss(
        actor_network=actor_module,
        critic_network=critic_module,
        clip_epsilon=clip_epsilon,
        entropy_bonus=bool(entropy_coef),
        entropy_coef=entropy_coef,
        critic_coeff=critic_coeff,
        loss_critic_type="smooth_l1",
    )
    return loss_module

def train(hyperparams: Dict[str, Any], env: JANIEnv, eval_env: JANIEnv) -> None:
    """Train the PPO agent."""
    logs = defaultdict(list)
    # Some hyperparameters
    total_timesteps = hyperparams.get("total_timesteps", 1024000)
    n_steps = hyperparams.get("n_steps", 2048)
    batch_size = hyperparams.get("batch_size", 64)
    max_grad_norm = hyperparams.get("max_grad_norm", 0.5)
    n_epochs = hyperparams.get("n_epochs", 10)
    lr = hyperparams.get("learning_rate", 3e-4)
    n_eval_episodes = hyperparams.get("n_eval_episodes", 100)

    # Create actor and critic
    print("Creating actor and critic networks...")
    actor = create_actor(hyperparams, env)
    critic = create_critic(hyperparams, env)
    
    # Create loss module
    print("Creating loss module...")
    loss_module = create_loss_module(hyperparams, actor, critic)

    # Create advantage module
    print("Creating advantage module...")
    advantage_module = create_advantage_module(hyperparams, critic)

    # Create data collector
    print("Creating data collector...")
    collector = create_data_collector(hyperparams, env, actor)

    # Create replay buffer
    print("Creating replay buffer...")
    replay_buffer = create_replay_buffer(hyperparams)

    # Create optimizer
    optim = torch.optim.Adam(loss_module.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, total_timesteps // n_steps, 0.0
    )

    # Training loop
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
        task = progress.add_task("Training PPO Agent", total=hyperparams.get("total_timesteps", 1024000))

        for i, td_data in enumerate(collector):
            # Compute advantages
            with torch.no_grad():
                advantage_module(td_data)
            # Store in replay buffer
            date_view = td_data.reshape(-1)
            replay_buffer.empty()
            replay_buffer.extend(date_view)
            for _ in range(n_epochs):
                # # Compute advantages
                # advantage_module(td_data)
                # # Store in replay buffer
                # date_view = td_data.reshape(-1)
                # replay_buffer.extend(date_view)
                # Sample from replay buffer
                for _ in range(n_steps // batch_size):
                    batch_data = replay_buffer.sample(batch_size)
                    loss = loss_module(batch_data)
                    loss_value = (
                        loss["loss_objective"]
                        + loss["loss_critic"]
                        + loss["loss_entropy"]
                    )
                    # Optimizer step
                    optim.zero_grad()
                    loss_value.backward()
                    torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
                    optim.step()
            
            logs["loss"].append(loss_value.item())
            logs["reward"].append(td_data["next", "reward"].mean().item())
            training_reward_str = f"📊 Average training reward={logs['reward'][-1]: 4.4f} (Init={logs['reward'][0]: 4.4f})"
            avg_loss_str = f" | 🧑‍🏫 Loss={logs['loss'][-1]: 4.4f}"
            if i % 100== 0:
                # Evaluation
                eval_rewards = []
                for _ in range(n_eval_episodes):
                    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
                        eval_rollout = eval_env.rollout(max_steps=1000, policy=actor)
                        eval_reward = eval_rollout["next", "reward"].sum().item()
                        eval_rewards.append(eval_reward)
                mean_eval_reward = sum(eval_rewards) / n_eval_episodes
                eval_reward_str = f" | 🧪 Eval reward={mean_eval_reward: 4.4f}"
                progress.console.print(f"{training_reward_str}{avg_loss_str}{eval_reward_str}")

            progress.update(task, advance=n_steps)
            progress.refresh()
            scheduler.step()


def main():
    parser = argparse.ArgumentParser(description="Train Masked PPO on JANI Environments")
    parser.add_argument('--jani_model', type=str, required=True, help="Path to the JANI model file.")
    parser.add_argument('--jani_property', type=str, default="", help="Path to the JANI property file.")
    parser.add_argument('--start_states', type=str, default="", help="Path to the start states file.")
    parser.add_argument('--objective', type=str, default="", help="Path to the objective file.")
    parser.add_argument('--failure_property', type=str, default="", help="Path to the failure property file.")
    parser.add_argument('--goal_reward', type=float, default=1.0, help="Reward for reaching the goal.")
    parser.add_argument('--failure_reward', type=float, default=-1.0, help="Reward for reaching failure state.")
    parser.add_argument('--use_oracle', action='store_true', help="Use Tarjan oracle for unsafe state detection.")
    parser.add_argument('--unsafe_reward', type=float, default=-0.01, help="Reward for unsafe states when using oracle.")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument('--total_timesteps', type=int, default=1_000_000, help="Total timesteps for training.")
    parser.add_argument('--n_envs', type=int, default=1, help="Number of parallel environments.")
    parser.add_argument('--max_steps', type=int, default=1000, help="Max steps per episode.")
    parser.add_argument('--log_dir', type=str, default="./logs", help="Directory for logging.")
    parser.add_argument('--model_save_dir', type=str, default="./models", help="Directory to save models.")
    parser.add_argument('--eval_freq', type=int, default=2048, help="Evaluation frequency in timesteps.")
    parser.add_argument('--n_eval_episodes', type=int, default=50, help="Number of episodes for each evaluation.")
    parser.add_argument('--verbose', type=int, default=1, help="Verbosity level.")
    parser.add_argument('--device', type=str, default='auto', help="Device to use for training (cpu or cuda).")

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
    env = JANIEnv(**file_args)
    # Create evaluation environment
    eval_file_args = create_eval_file_args(file_args)
    eval_env = JANIEnv(**eval_file_args)

    # Define hyperparameters
    hyperparams = {
        'total_timesteps': args.total_timesteps,
        'n_steps': 2048,
        'batch_size': 64,
        'learning_rate': 3e-4,
        'gae_lambda': 0.95,
        'gamma': 0.99,
        'clip_epsilon': 0.2,
        'ent_coef': 1e-4,
        'critic_coeff': 1.0,
        'max_grad_norm': 0.5,
        'n_epochs': 10,
        'device': args.device,
    }
    print(f"Training with hyperparameters: {hyperparams}")
    # Start training
    train(hyperparams, env, eval_env)

if __name__ == "__main__":
    main()