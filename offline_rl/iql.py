import argparse
import numpy as np
import torch

from torchrl.objectives import DiscreteIQLLoss, SoftUpdate
from tensordict import TensorDict
from tensordict.nn import TensorDictModule

from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
from pathlib import Path

from jani import TorchRLJANIEnv as JaniEnv

from .load_dataset import collect_trajectories
from .models import create_q_module, create_v_module, create_actor
from .loss import DiscreteIQLLossValueLB, DiscreteIQLLossQValueLB


def evaluate_on_env(
        env: JaniEnv, 
        actor: TensorDictModule, 
        num_episodes: int = 100, 
        max_steps: int = 256):
    """Evaluate the trained policy on the environment."""
    
    total_rewards = []
    for _ in range(num_episodes):
        rollout = env.rollout(max_steps=max_steps, policy=actor)
        total_rewards.append(rollout["next", "reward"].sum().item())
    total_rewards = np.array(total_rewards)
    avg_reward = np.mean(total_rewards)

    return avg_reward
    # success_rate = np.mean(total_rewards == 1.0)
    # failure_rate = np.mean(total_rewards == -1.0)
    # return {"avg_reward": avg_reward, "success_rate": success_rate, "failure_rate": failure_rate}


def create_loss(args, actor_module, q_module, v_module):
    """Create the IQL loss function based on the provided arguments."""
    kwargs = {
        "actor_network": actor_module,
        "qvalue_network": q_module,
        "value_network": v_module,
        "action_space": "categorical"
    }
    if args.use_lower_bound:
        if args.lower_bound_type == "value":
            iql_loss = DiscreteIQLLossValueLB(**kwargs)
        elif args.lower_bound_type == "qvalue":
            iql_loss = DiscreteIQLLossQValueLB(**kwargs)
        else:
            raise ValueError(f"Invalid lower_bound_type: {args.lower_bound_type}")
    else:
        iql_loss = DiscreteIQLLoss(**kwargs)
    return iql_loss


def train(total_timesteps, steps_per_epoch, batch_size, lr, rb, iql_loss, print_info=False):
    """Train the IQL agent."""
    optimizer = torch.optim.Adam(iql_loss.parameters(), lr=lr)
    updater = SoftUpdate(iql_loss, eps=0.95)
    num_epochs = total_timesteps // steps_per_epoch

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None, style="bar.back", complete_style="bar.complete", finished_style="bar.finished"),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        expand=True,
    ) as progress:
        epoch_task = progress.add_task("Training epochs", total=num_epochs)
        step_task = progress.add_task("", total=steps_per_epoch, visible=False)

        for epoch in range(num_epochs):
            progress.update(step_task, completed=0, visible=True, description=f"Epoch {epoch+1}/{num_epochs} steps")

            for _ in range(steps_per_epoch):
                batch = rb.sample(batch_size)
                loss_td = iql_loss(batch)
                total_loss = loss_td.get("loss_actor") + loss_td.get("loss_qvalue") + loss_td.get("loss_value")
                total_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                progress.update(step_task, advance=1)

            updater.step()
            progress.update(epoch_task, advance=1)
            progress.update(step_task, visible=False)

            if print_info:
                if (epoch + 1) % 100 == 0:
                    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss.item():.4f}")

    return iql_loss.actor_network, iql_loss.qvalue_network, iql_loss.value_network


def objective(trial, rb, env, args):
    """Optuna objective function for hyperparameter tuning."""
    # Hyperparameters to tune
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)

    n_layers_q_module = trial.suggest_int("n_layers_q_module", 1, 3)
    hidden_dims_q_module = []
    for i in range(n_layers_q_module):
        size = trial.suggest_int(f"hidden_size_q_module_{i}", 32, 256)
        hidden_dims_q_module.append(size)

    n_layers_v_module = trial.suggest_int("n_layers_v_module", 1, 3)
    hidden_dims_v_module = []
    for i in range(n_layers_v_module):
        size = trial.suggest_int(f"hidden_size_v_module_{i}", 32, 256)
        hidden_dims_v_module.append(size)

    n_layers_actor = trial.suggest_int("n_layers_actor", 1, 3)
    hidden_dims_actor = []
    for i in range(n_layers_actor):
        size = trial.suggest_int(f"hidden_size_actor_{i}", 32, 256)
        hidden_dims_actor.append(size)

    batch_size = trial.suggest_categorical("batch_size", [args.num_slices * i for i in range(1, 3)])
    steps_per_epoch = trial.suggest_int("steps_per_epoch", 50, 2000)

    tuning_timesteps = min(args.total_timesteps // 4, 10000)

    # Create models
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    q_module = create_q_module(state_dim, action_dim, hidden_dims=hidden_dims_q_module)
    v_module = create_v_module(state_dim, hidden_dims=hidden_dims_v_module)
    actor_module = create_actor(state_dim, action_dim, hidden_dims=hidden_dims_actor)

    # Create IQL loss
    iql_loss = create_loss(args, actor_module, q_module, v_module)
    
    # Training loop
    actor, _, _ = train(tuning_timesteps, steps_per_epoch, batch_size, lr, rb, iql_loss)

    results = evaluate_on_env(env, actor, num_episodes=10)

    return results['avg_reward']


def hyperparameter_tuning(rb, env, args, n_trials=20):
    """Perform hyperparameter tuning using Optuna."""
    import optuna

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, rb, env, args), n_trials=n_trials)

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    return trial.params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", 
        type=str, required=True, help="Path to the JANI model file.")
    parser.add_argument(
        "--property_path",
        type=str, required=True, help="Path to the JANI property file.")
    parser.add_argument(
        "--start_states", 
        type=str, required=True, help="Path to the start states file.")
    parser.add_argument(
        "--objective", 
        type=str, default="", help="Path to the objective file.")
    parser.add_argument(
        "--failure_property", 
        type=str, default="", help="Path to the failure property file.")
    parser.add_argument(
        "--goal_reward", 
        type=float, default=1.0, help="Reward for reaching the goal.")
    parser.add_argument(
        "--failure_reward", 
        type=float, default=-1.0, help="Reward for reaching failure state.")
    parser.add_argument(
        "--use_oracle", 
        action="store_true", help="Use Tarjan oracle for unsafe state detection.")
    parser.add_argument(
        "--unsafe_reward", 
        type=float, default=-0.01, help="Reward for unsafe states when using oracle.")
    parser.add_argument(
        "--batch_size", 
        type=int, default=64, help="Batch size for training.")
    parser.add_argument(
        "--use_lower_bound", 
        action="store_true", help="Whether to use lower bound in IQL loss.")
    parser.add_argument(
        "--total_timesteps", 
        type=int, default=100000, help="Total training timesteps.")
    parser.add_argument(
        "--steps_per_epoch", 
        type=int, default=1000, help="Number of steps per epoch.")
    parser.add_argument(
        "--lower_bound_type", 
        type=str, choices=["value", "qvalue"], 
        default="value", help="Type of lower bound to use in IQL loss.")
    parser.add_argument(
        "--expectile",
        type=float, default=0.7, help="Expectile value for IQL loss.")
    parser.add_argument(
        "--tune_hyperparameters",
        action="store_true", help="Whether to perform hyperparameter tuning.")
    parser.add_argument(
        "--n_trials",
        type=int, default=20, help="Number of trials for hyperparameter tuning.")
    parser.add_argument(
        "--seed", 
        type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--model_save_dir", 
        type=str, default=None, help="Directory to save trained models and results.")
    parser.add_argument(
        "--write_eval_results", 
        type=str, default=None, help="Path to write evaluation results.")
    args = parser.parse_args()

    # torch.manual_seed(args.seed)
    # np.random.seed(args.seed)

    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed(args.seed)
    #     torch.cuda.manual_seed_all(args.seed)
    
    # Make PyTorch deterministic (may impact performance)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    # Initialize environment
    file_args = {
        'jani_model_path': args.model_path,
        'jani_property_path': args.property_path,
        'start_states_path': args.start_states,
        'objective_path': args.objective,
        'failure_property_path': args.failure_property,
        'goal_reward': args.goal_reward,
        'failure_reward': args.failure_reward,
        'seed': args.seed,
        'use_oracle': args.use_oracle,
        'unsafe_reward': args.unsafe_reward,
    }
    env = JaniEnv(**file_args)
    action_dim = env.n_actions

    # Collect trajectories and create replay buffer
    print("Collecting trajectories...")
    replay_buffer = collect_trajectories(
        env, policy=None, 
        num_total_steps=500000, 
        n_steps=256)

    # Extract state and action dimensions
    state_dim = env.observation_spec["observation"].shape[0]

    best_params = {
        "lr": 1e-3,
        "batch_size": args.batch_size,
        "steps_per_epoch": args.steps_per_epoch,
        "expectile": args.expectile,
        "n_layers_q_module": 2,
        "hidden_size_q_module_0": 32,
        "hidden_size_q_module_1": 64,
        "n_layers_v_module": 2,
        "hidden_size_v_module_0": 32,
        "hidden_size_v_module_1": 64,
        "n_layers_actor": 2,
        "hidden_size_actor_0": 32,
        "hidden_size_actor_1": 64
    }
    if args.tune_hyperparameters:
        best_params = hyperparameter_tuning(replay_buffer, env, args, n_trials=args.n_trials)

    # Create models
    hidden_dims_q_module = [best_params[f"hidden_size_q_module_{i}"] for i in range(best_params["n_layers_q_module"])]
    q_module = create_q_module(state_dim, action_dim, hidden_dims=hidden_dims_q_module)
    hidden_dims_v_module = [best_params[f"hidden_size_v_module_{i}"] for i in range(best_params["n_layers_v_module"])]
    v_module = create_v_module(state_dim, hidden_dims=hidden_dims_v_module)
    hidden_dims_actor_module = [best_params[f"hidden_size_actor_{i}"] for i in range(best_params["n_layers_actor"])]
    actor_module = create_actor(state_dim, action_dim, hidden_dims=hidden_dims_actor_module)

    # Create IQL loss
    kwargs = {
        "actor_network": actor_module,
        "qvalue_network": q_module,
        "value_network": v_module,
        "expectile": best_params["expectile"],
        "action_space": "categorical"
    }
    if args.use_lower_bound:
        if args.lower_bound_type == "value":
            iql_loss = DiscreteIQLLossValueLB(**kwargs)
        elif args.lower_bound_type == "qvalue":
            iql_loss = DiscreteIQLLossQValueLB(**kwargs)
        else:
            raise ValueError(f"Invalid lower_bound_type: {args.lower_bound_type}")
    else:
        iql_loss = DiscreteIQLLoss(**kwargs)

    actor, q_net, v_net = train(
        args.total_timesteps, 
        best_params["steps_per_epoch"], 
        best_params["batch_size"], 
        best_params["lr"], 
        replay_buffer, 
        iql_loss,
        print_info=True
    )
    avg_reward = evaluate_on_env(env, actor, num_episodes=100)
    print(f"Final average reward over 100 episodes: {avg_reward:.2f}")
    # if args.write_eval_results is not None:
    #     import json
    #     with open(args.write_eval_results, "w") as f:
    #         json.dump(results, f, indent=4)
    # print(f"Final success rate over 100 episodes: {results['success_rate']:.2f}, avg reward: {results['avg_reward']:.2f}, failure rate: {results['failure_rate']:.2f}")

    # Save the trained models and the replay_buffer
    if args.model_save_dir is not None:
        from utils import save_network

        save_dir = Path(args.model_save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save the actor
        save_network(actor, {
            "input_dim": state_dim,
            "output_dim": action_dim,
            "hidden_dims": hidden_dims_actor_module
        }, save_dir, "iql_actor")

        # Save the Q-network
        save_network(q_net, {
            "input_dim": state_dim,
            "output_dim": action_dim,
            "hidden_dims": hidden_dims_q_module
        }, save_dir, "iql_q_net")

        # Save the V-network
        save_network(v_net, {
            "input_dim": state_dim,
            "output_dim": 1,
            "hidden_dims": hidden_dims_v_module
        }, save_dir, "iql_v_net")

        # Save the replay buffer
        rb_path = save_dir / "replay_buffer"
        replay_buffer.dumps(rb_path)

if __name__ == "__main__":
    main()