import torch
import torch.nn as nn
import argparse

from tensordict.nn import TensorDictModule, TensorDictModuleBase

from torchrl.modules import MLP, ProbabilisticActor, ValueOperator
from torchrl.modules.distributions import MaskedCategorical
from torchrl.objectives import DiscreteSACLoss
from torchrl.objectives.value import GAE
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement, Sampler
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type

from typing import Dict, Any
from collections import defaultdict
from rich.progress import (
    Progress, 
    SpinnerColumn, 
    TextColumn, 
    BarColumn, 
    TimeRemainingColumn, 
    TimeElapsedColumn
)

from jani.torchrl_env import JANIEnv
from utils import create_eval_file_args

from .utils import (
    create_actor_module, 
    create_critic, 
    create_data_collector, 
    create_replay_buffer,
    load_recovery_policy_module,
    load_q_risk_backbone,
    load_replay_buffer,
)
from .actor import RecoveryActor


def train(hyperparams: Dict[str, Any], args: dict[str, any], env: JANIEnv, eval_env: JANIEnv) -> None:
    """Train the PPO agent."""
    logs = defaultdict(list)
    # Some hyperparameters
    total_timesteps = hyperparams.get("total_timesteps", 1024000)
    n_steps = hyperparams.get("n_steps", 256)
    batch_size = hyperparams.get("batch_size", 64)
    lr = hyperparams.get("learning_rate", 3e-4)
    n_eval_episodes = hyperparams.get("n_eval_episodes", 100)

    # Create actor and critic (i.e., q_value) networks
    print("Creating task policy and critic networks...")
    task_policy_module = create_actor_module(hyperparams, env)
    # Task policy for collecting data
    task_policy = ProbabilisticActor(
        module=task_policy_module,
        in_keys={"logits": "logits", "mask": "action_mask"},
        out_keys=["task_action"],
        distribution_class=MaskedCategorical,
        return_log_prob=True, # Not sure whether this is actually need
    )
    # Training view of the task policy (outputs "action" instead of "task_action")
    task_policy_training = ProbabilisticActor(
        module=task_policy_module,
        in_keys={"logits": "logits", "mask": "action_mask"},
        out_keys=["action"],
        distribution_class=MaskedCategorical,
        return_log_prob=True, # Not sure whether this is actually need
    )
    # Critic for task policy value estimation
    critic = create_critic(hyperparams, env)

    # Load recovery policy and risk module from pretrained models
    print("Loading pretrained recovery policy and risk module...")
    # rec_policy_path = args.get("recover_policy_path", "")
    # assert rec_policy_path != "", "Path to pretrained recovery policy must be provided in args with key 'recover_policy_path'"

    # q_risk_path = args.get("q_risk_path", "")
    # assert q_risk_path != "", "Path to pretrained q_risk module must be provided in args with key 'q_risk_path'"

    # Load the recovery policy backbone and create the recovery policy module
    # recovery_policy_module = load_recovery_policy_module(rec_policy_path)
    recovery_policy_module = create_actor_module(hyperparams, env)
    # recovery policy for collecting data
    recovery_policy = ProbabilisticActor(
        module=recovery_policy_module,
        in_keys={"logits": "logits", "mask": "action_mask"},
        out_keys=["recovery_action"],
        distribution_class=MaskedCategorical,
        return_log_prob=True, # Not sure whether this is actually need
    )
    # recovery policy for training (outputs "action" instead of "recovery_action")
    recovery_policy_training = ProbabilisticActor(
        module=recovery_policy_module,
        in_keys={"logits": "logits", "mask": "action_mask"},
        out_keys=["action"],
        distribution_class=MaskedCategorical,
        return_log_prob=True, # Not sure whether this is actually need
    )

    # Load the q_risk backbone and create the q_risk module
    # q_risk_backbone = load_q_risk_backbone(q_risk_path)
    q_risk_backbone = create_critic(hyperparams, env)
    # q_risk module for collecting data
    q_risk_module = ValueOperator(
        module=q_risk_backbone,
        in_keys=["observation"],
        out_keys=["q_risk_value"],
    )
    # q_risk module for training (outputs "action_value" instead of "q_risk_value")
    q_risk_module_training = ValueOperator(
        module=q_risk_backbone,
        in_keys=["observation"],
        out_keys=["action_value"],
    )

    # Create the combined recovery actor
    recovery_actor = RecoveryActor(
        task_policy=task_policy,
        recovery_policy=recovery_policy,
        q_risk_module=q_risk_module,
        risk_threshold=-0.35
    )

    # Create replay buffer for training task policy and critic
    print("Creating replay buffer...")
    replay_buffer = create_replay_buffer(hyperparams)

    # Load replay buffer for q_risk and recovery policy
    # offline_buffer_path = args.get("offline_buffer_path", "")
    # assert offline_buffer_path != "", "Path to offline replay buffer must be provided in args with key 'offline_buffer_path'"
    # offline_replay_buffer = load_replay_buffer(offline_buffer_path, hyperparams)
    offline_replay_buffer = create_replay_buffer(hyperparams)
    
    # Create loss module
    print("Creating loss modules...")
    task_loss_module = DiscreteSACLoss(
        actor_network = task_policy_training,
        qvalue_network = critic,
        action_space = "categorical",
        num_actions = env.n_actions,
    )
    risk_loss_module = DiscreteSACLoss(
        actor_network = recovery_policy_training,
        qvalue_network = q_risk_module_training,
        action_space = "categorical",
        num_actions = env.n_actions,
    )

    # Create data collector
    print("Creating data collector...")
    collector = create_data_collector(hyperparams, env, recovery_actor)

    # Create optimizer
    optim = torch.optim.Adam(
        list(task_loss_module.parameters()) + list(risk_loss_module.parameters()), 
        lr=lr
    )
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
        task = progress.add_task("Training Recovery RL Agent", total=hyperparams.get("total_timesteps", 1024000))

        for i, td_data in enumerate(collector):
            td_task_data = td_data.clone(recurse=True).detach()
            # Set the task action as the main action for task policy training
            td_task_data = td_task_data.select(
                "observation", 
                "action",
                "task_action", 
                "action_mask", 
                "next"
            )
            td_task_data.set_("action", td_task_data.get("task_action"))
            del td_task_data["task_action"]
            assert td_task_data["action"].shape == (td_data.batch_size, 1), f"Expected action shape to be (batch_size, 1), but got {td_task_data['action'].shape}"
            # Add the data to the replay buffer
            replay_buffer.extend(td_task_data)

            td_risk_data = td_data.clone(recurse=True).detach()
            # The final action is the main action for risk policy training
            td_risk_data = td_risk_data.select(
                "observation", 
                "action", 
                "action_mask",
                "next"
            )
            assert td_risk_data["action"].shape == (td_data.batch_size, 1), f"Expected action shape to be (batch_size, 1), but got {td_risk_data['action'].shape}"
            # Relabel all goal reached rewards to 0 for risk module training
            td_risk_data["next", "reward"] = torch.where(
                td_risk_data["next", "reward"] == env._goal_reward, 
                torch.tensor(0.0, dtype=torch.float32), 
                td_risk_data["next", "reward"]
            )
            # Add the data to the replay buffer for risk module training
            offline_replay_buffer.extend(td_risk_data)

            # Sample batches for task and risk training
            batch_task_data = replay_buffer.sample(batch_size)
            batch_risk_data = offline_replay_buffer.sample(batch_size)

            task_loss = task_loss_module(batch_task_data)
            risk_loss = risk_loss_module(batch_risk_data)
            loss_value = (
                task_loss["loss_actor"]
                + task_loss["loss_qvalue"]
                + task_loss["loss_alpha"]
                + risk_loss["loss_actor"]
                + risk_loss["loss_alpha"]
                + risk_loss["loss_qvalue"]
            )

            # Optimizer step
            optim.zero_grad()
            loss_value.backward()
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
                        eval_rollout = eval_env.rollout(max_steps=1000, policy=recovery_actor)
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
        'n_steps': 256,
        'batch_size': 64,
        'learning_rate': 3e-4,
        'device': args.device,
    }
    print(f"Training with hyperparameters: {hyperparams}")
    # Start training
    train(hyperparams, vars(args), env, eval_env)

if __name__ == "__main__":
    main()