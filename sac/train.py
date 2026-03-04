import sys
import torch
import torch.nn as nn
import argparse

from pathlib import Path
from tensordict.nn import TensorDictModule

from torch.distributions import Categorical
from torchrl.modules import MLP, ProbabilisticActor, ValueOperator
from torchrl.modules.distributions import MaskedCategorical
from torchrl.objectives import DiscreteSACLoss
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import RandomSampler
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs.utils import ExplorationType, set_exploration_type

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
from utils import create_safety_eval_file_args, safety_evaluation


def create_actor_module(hyperparams: Dict[str, Any], env: JANIEnv) -> ProbabilisticActor:
    """Create the actor network for the policy."""
    n_actions = env.n_actions
    input_size = env.observation_spec["observation"].shape[0]
    hidden_sizes = hyperparams.get("actor_hidden_sizes", [64, 64])
    dropout = hyperparams.get("actor_dropout", 0.0)
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
    return actor_module
    # Create the probabilistic actor with masked categorical distribution
    actor = ProbabilisticActor(
        module=actor_module,
        in_keys={"logits": "logits", "mask": "action_mask"},
        out_keys=["action"],
        distribution_class=MaskedCategorical,
        return_log_prob=True,
    )
    return actor


def create_qvalue_network(hyperparams: Dict[str, Any], env: JANIEnv) -> ValueOperator:
    """Create the Q-value network for SAC.

    For discrete SAC, the Q-network outputs Q-values for all actions given a state.
    Output shape: (batch_size, n_actions)
    """
    n_actions = env.n_actions
    input_size = env.observation_spec["observation"].shape[0]
    hidden_sizes = hyperparams.get("critic_hidden_sizes", [64, 128])
    dropout = hyperparams.get("critic_dropout", 0.0)
    activation_fn = hyperparams.get("activation_fn", nn.Tanh)
    # Build the Q-value network (outputs Q-value for each action)
    qvalue_backbone = MLP(
        in_features=input_size,
        out_features=n_actions,
        num_cells=hidden_sizes,
        dropout=dropout,
        activation_class=activation_fn,
    )
    # Wrap in ValueOperator
    qvalue_module = ValueOperator(
        module=qvalue_backbone,
        in_keys=["observation"],
        out_keys=["action_value"],
    )
    return qvalue_module


def create_data_collector(hyperparams: Dict[str, Any], env: JANIEnv, policy: TensorDictModule) -> SyncDataCollector:
    """Create a data collector for experience gathering."""
    n_steps = hyperparams.get("n_steps", 256)
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
    """Create a replay buffer for off-policy SAC training."""
    buffer_size = hyperparams.get("replay_buffer_size", 100000)
    # Create the storage
    storage = LazyTensorStorage(
        max_size=buffer_size,
        device=hyperparams.get("device", "cpu"),
    )
    # Create the sampler (random sampling for off-policy)
    sampler = RandomSampler()
    # Create the replay buffer
    replay_buffer = ReplayBuffer(
        storage=storage,
        sampler=sampler,
    )
    return replay_buffer


def create_loss_module(
    hyperparams: Dict[str, Any],
    actor: ProbabilisticActor,
    qvalue_network: ValueOperator,
    env: JANIEnv
) -> DiscreteSACLoss:
    """Create the loss module for Discrete SAC."""
    gamma = hyperparams.get("gamma", 0.99)
    alpha_init = hyperparams.get("alpha_init", 1.0)
    target_entropy = hyperparams.get("target_entropy", "auto")

    loss_module = DiscreteSACLoss(
        actor_network=actor,
        qvalue_network=qvalue_network,
        action_space="categorical",
        num_actions=env.n_actions,
        delay_qvalue=True,
        alpha_init=alpha_init,
        target_entropy=target_entropy,
    )
    # Set gamma via the value estimator (DiscreteSACLoss doesn't take gamma directly)
    # loss_module.make_value_estimator(gamma=gamma)

    return loss_module


def train(hyperparams: Dict[str, Any], args: Dict[str, Any], env: JANIEnv, eval_env: JANIEnv) -> None:
    """Train the Discrete SAC agent."""
    logs = defaultdict(list)

    # Hyperparameters
    total_timesteps = hyperparams.get("total_timesteps", 1024000)
    n_steps = hyperparams.get("n_steps", 256)
    batch_size = hyperparams.get("batch_size", 64)
    lr = hyperparams.get("learning_rate", 3e-4)
    max_grad_norm = hyperparams.get("max_grad_norm", 1.0)
    n_updates_per_step = hyperparams.get("n_updates_per_step", 1)
    target_update_freq = hyperparams.get("target_update_freq", 5)
    tau = hyperparams.get("tau", 0.005)  # Soft update coefficient

    warmup_steps = hyperparams.get("warmup_steps", 1000)

    # Setup wandb logging
    use_wandb = not args.get("disable_wandb", False)
    if use_wandb:
        import wandb
        wandb.init(
            project=args.get("wandb_project", "discrete-sac"),
            entity=args.get("wandb_entity"),
            name=args.get("experiment_name"),
            config={**hyperparams, **args},
        )

    log_dir = args.get("log_dir", "")
    log_results = False
    if log_dir != "":
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file_path = log_dir / "training_results.txt"
        log_file_path.open("w").close()  # Create an empty log file
        log_results = True

    model_save_dir = args.get("model_save_dir", "")
    save_models = False
    if model_save_dir != "":
        model_save_dir = Path(model_save_dir)
        model_save_dir.mkdir(parents=True, exist_ok=True)
        save_models = True

    # Create actor and Q-value networks
    print("Creating actor and Q-value networks...")
    actor_module = create_actor_module(hyperparams, env)
    actor_rollout = ProbabilisticActor(
        module=actor_module,
        in_keys={"logits": "logits", "mask": "action_mask"},
        out_keys=["action"],
        distribution_class=MaskedCategorical,
        return_log_prob=True,
    )
    actor_train = ProbabilisticActor(
        module=actor_module,
        in_keys=['logits'],
        out_keys=['action'],
        distribution_class=Categorical,
        return_log_prob=True,
    )
    qvalue_network = create_qvalue_network(hyperparams, env)

    # Create loss module
    print("Creating loss module...")
    loss_module = create_loss_module(hyperparams, actor_train, qvalue_network, env)

    # Create data collector
    print("Creating data collector...")
    collector = create_data_collector(hyperparams, env, actor_rollout)

    # Create replay buffer
    print("Creating replay buffer...")
    replay_buffer = create_replay_buffer(hyperparams)

    # Create optimizers (separate for actor, Q-value, and alpha)
    actor_optim = torch.optim.Adam(actor_train.parameters(), lr=lr)
    qvalue_optim = torch.optim.Adam(qvalue_network.parameters(), lr=lr)
    alpha_optim = torch.optim.Adam([loss_module.log_alpha], lr=lr)

    # Learning rate schedulers
    actor_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        actor_optim, total_timesteps // n_steps, 1e-5
    )
    qvalue_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        qvalue_optim, total_timesteps // n_steps, 1e-5
    )

    # Training loop
    total_steps_collected = 0
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
            disable=not sys.stdout.isatty(),
        ) as progress:
        training_task = progress.add_task("Training Discrete SAC Agent", total=total_timesteps)
        eval_task = progress.add_task("Policy Safety Evaluation", total=100, visible=False)

        for i, td_data in enumerate(collector):
            # Add collected data to replay buffer
            data_view = td_data.reshape(-1)
            replay_buffer.extend(data_view)
            total_steps_collected += n_steps

            # Skip training until we have enough data
            if total_steps_collected < warmup_steps:
                progress.update(training_task, advance=n_steps)
                continue

            # Training updates
            for _ in range(n_updates_per_step):
                # Ensure we have enough samples
                if len(replay_buffer) < batch_size:
                    continue

                batch_data = replay_buffer.sample(batch_size)

                # Compute losses
                loss_dict = loss_module(batch_data)

                # Update Q-value network
                qvalue_loss = loss_dict["loss_qvalue"]
                qvalue_optim.zero_grad()
                qvalue_loss.backward()
                torch.nn.utils.clip_grad_norm_(qvalue_network.parameters(), max_grad_norm)
                qvalue_optim.step()

                # Update actor network
                actor_loss = loss_dict["loss_actor"]
                actor_optim.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor_train.parameters(), max_grad_norm)
                actor_optim.step()

                # Update alpha (entropy coefficient)
                alpha_loss = loss_dict["loss_alpha"]
                alpha_optim.zero_grad()
                alpha_loss.backward()
                alpha_optim.step()

                # Soft update target networks
                if (i + 1) % target_update_freq == 0:
                    with torch.no_grad():
                        # Update task Q-value target network
                        for param, target_param in zip(
                            loss_module.qvalue_network_params.values(),
                            loss_module.target_qvalue_network_params.values()
                        ):
                            target_param.data.copy_(
                                tau * param.data + (1 - tau) * target_param.data
                            )
                # loss_module.target_qvalue_network_params.data.lerp_(
                #     loss_module.qvalue_network_params.data, tau
                # )

            # Logging
            logs["loss_actor"].append(loss_dict["loss_actor"].item())
            logs["loss_qvalue"].append(loss_dict["loss_qvalue"].item())
            logs["loss_alpha"].append(loss_dict["loss_alpha"].item())
            logs["alpha"].append(loss_dict["alpha"].item())
            logs["reward"].append(td_data["next", "reward"].mean().item())

            # Periodic evaluation
            if i % 1 == 0:
                eval_result = safety_evaluation(
                    eval_env, 
                    actor_rollout, 
                    max_steps=hyperparams.get("max_steps", 256), 
                    progress=progress, 
                    task_id=eval_task
                )

                # Log to wandb
                if use_wandb:
                    wandb.log({
                        "loss/actor": logs["loss_actor"][-1],
                        "loss/qvalue": logs["loss_qvalue"][-1],
                        "loss/alpha": logs["loss_alpha"][-1],
                        "train/alpha": logs["alpha"][-1],
                        "train/reward": logs["reward"][-1],
                        "eval/safety_rate": eval_result["safety_rate"],
                        "eval/average_reward": eval_result["average_reward"],
                        "train/timesteps": total_steps_collected,
                    })

                if log_results:
                    with log_file_path.open("a") as f:
                        f.write(f"{eval_result['safety_rate'] * 100:.2f}, {eval_result['average_reward']:.4f}\n")
                loss_str = f" | 🧑‍🏫 Actor={logs['loss_actor'][-1]: 4.4f} Q={logs['loss_qvalue'][-1]: 4.4f}"
                alpha_str = f" | α={logs['alpha'][-1]: 4.4f}"
                progress.console.print(f"Step {total_steps_collected}: Eval Safety Rate: {eval_result['safety_rate']:.4f}, Eval Average Reward: {eval_result['average_reward']:.4f}")
                progress.console.print(f"{loss_str}{alpha_str}")

            # Save model checkpoints
            if save_models and i % 1 == 0:
                model_save_path = model_save_dir / f"checkpoint_{i}"
                model_save_path.mkdir(parents=True, exist_ok=True)

                # Save actor
                actor_path = model_save_path / "actor.pth"
                actor_params = {
                    "input_dim": env.observation_spec["observation"].shape[0],
                    "output_dim": env.n_actions,
                    "hidden_dims": hyperparams.get("actor_hidden_sizes", [64, 64]),
                    "dropout": hyperparams.get("actor_dropout", 0.0),
                    "activation_fn": hyperparams.get("activation_fn", nn.Tanh),
                    "state_dict": actor_module.state_dict()
                }
                torch.save(actor_params, actor_path)

                # Save Q-value network
                qvalue_path = model_save_path / "qvalue.pth"
                qvalue_params = {
                    "input_dim": env.observation_spec["observation"].shape[0],
                    "output_dim": env.n_actions,
                    "hidden_dims": hyperparams.get("critic_hidden_sizes", [64, 64]),
                    "dropout": hyperparams.get("critic_dropout", 0.0),
                    "activation_fn": hyperparams.get("activation_fn", nn.Tanh),
                    "state_dict": qvalue_network.state_dict()
                }
                torch.save(qvalue_params, qvalue_path)

            progress.update(training_task, advance=n_steps)
            progress.refresh()
            actor_scheduler.step()
            qvalue_scheduler.step()

    # Finish wandb logging
    if use_wandb:
        wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Train Discrete SAC on JANI Environments")
    parser.add_argument(
        '--jani_model',
        type=str, required=True, help="Path to the JANI model file.")
    parser.add_argument(
        '--jani_property',
        type=str, default="", help="Path to the JANI property file.")
    parser.add_argument(
        '--start_states',
        type=str, default="", help="Path to the start states file.")
    parser.add_argument(
        '--eval_start_states', 
        type=str, required=True, help="Path to the evaluation start states file (optional). If not provided, training start states will be used for evaluation.")
    parser.add_argument(
        '--objective',
        type=str, default="", help="Path to the objective file.")
    parser.add_argument(
        '--failure_property',
        type=str, default="", help="Path to the failure property file.")
    parser.add_argument(
        '--goal_reward',
        type=float, default=1.0, help="Reward for reaching the goal.")
    parser.add_argument(
        '--failure_reward',
        type=float, default=-1.0, help="Reward for reaching failure state.")
    parser.add_argument(
        '--use_oracle',
        action='store_true', help="Use Tarjan oracle for unsafe state detection.")
    parser.add_argument(
        '--unsafe_reward',
        type=float, default=-0.01, help="Reward for unsafe states when using oracle.")
    parser.add_argument(
        '--seed',
        type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        '--total_timesteps',
        type=int, default=1_000_000, help="Total timesteps for training.")
    parser.add_argument(
        '--max_steps',
        type=int, default=1000, help="Max steps per episode.")
    parser.add_argument(
        '--no_memory_reduced_mode',
        action='store_true', help="Disable memory reduced mode in the environment")
    parser.add_argument(
        '--log_dir',
        type=str, default="", help="Directory for logging.")
    parser.add_argument(
        '--model_save_dir',
        type=str, default="", help="Directory to save models.")
    parser.add_argument(
        '--disable_wandb',
        action='store_true', help="Disable logging to Weights & Biases.")
    parser.add_argument(
        '--wandb_project',
        type=str, default="discrete-sac", help="Weights & Biases project name.")
    parser.add_argument(
        '--experiment_name',
        type=str, default=None, help="Experiment name for Weights & Biases logging.")
    parser.add_argument(
        '--wandb_entity',
        type=str, default=None, help="Weights & Biases entity name.")
    parser.add_argument(
        '--eval_freq',
        type=int, default=2048, help="Evaluation frequency in timesteps.")
    parser.add_argument(
        '--n_eval_episodes',
        type=int, default=50, help="Number of episodes for each evaluation.")
    parser.add_argument(
        '--verbose',
        type=int, default=1, help="Verbosity level.")
    parser.add_argument(
        '--device',
        type=str, default='auto', help="Device to use for training (cpu or cuda).")

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
        'reduced_memory_mode': not args.no_memory_reduced_mode,
    }
    env = JANIEnv(**file_args)

    # Create evaluation environment
    eval_file_args = file_args.copy()
    eval_file_args["start_states_path"] = args.eval_start_states
    eval_file_args["use_oracle"] = False  # Disable oracle for evaluation
    eval_env = JANIEnv(**eval_file_args)

    # Define hyperparameters
    hyperparams = {
        'total_timesteps': args.total_timesteps,
        'n_steps': 256,
        'batch_size': 64,
        'learning_rate': 3e-4,
        'gamma': 0.99,
        'tau': 0.005,  # Soft update coefficient
        'target_update_freq': 1,  # Frequency of target network updates
        'alpha_init': 0.1,  # Initial entropy coefficient
        'target_entropy': 'auto',  # Automatic target entropy
        'replay_buffer_size': 100000,
        'warmup_steps': 1000,
        'n_updates_per_step': 16,
        'max_grad_norm': 1.0,
        'n_eval_episodes': args.n_eval_episodes,
        'device': args.device,
        'actor_hidden_sizes': [256, 256],
        'critic_hidden_sizes': [256, 256],
        'actor_dropout': 0.0,
        'critic_dropout': 0.0,
        'activation_fn': nn.ReLU,
    }
    print(f"Training with hyperparameters: {hyperparams}")

    # Start training
    train(hyperparams, vars(args), env, eval_env)


if __name__ == "__main__":
    main()
