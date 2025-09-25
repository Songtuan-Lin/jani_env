import argparse
import numpy as np
import torch

from torchrl.objectives import DiscreteIQLLoss, SoftUpdate
from tensordict import TensorDict
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn

from jani import JaniEnv

from .load_dataset import read_trajectories, create_replay_buffer
from .models import create_q_module, create_v_module, create_actor
from .loss import DiscreteIQLLossValueLB, DiscreteIQLLossQValueLB


def evaluate_on_env(env, actor, num_episodes=10, max_steps=2048):
    """Evaluate the trained policy on the environment."""
    total_rewards = []
    for _ in range(num_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        while (not done) and (max_steps > 0):
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            action_mask = torch.tensor(info.get("action_mask"), dtype=torch.bool).unsqueeze(0)
            td = TensorDict({"observation": obs_tensor, "action_mask": action_mask}, batch_size=[1])
            with torch.no_grad():
                action = actor(td).get("action").squeeze().item()
            obs, reward, done, _, info = env.step(action)
            episode_reward += reward
            max_steps -= 1
        total_rewards.append(episode_reward)
    total_rewards = np.array(total_rewards)
    avg_reward = np.mean(total_rewards)
    success_rate = np.mean(total_rewards == 1.0)
    failure_rate = np.mean(total_rewards == -1.0)
    return {"avg_reward": avg_reward, "success_rate": success_rate, "failure_rate": failure_rate}


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
        "--trajectory_path", 
        type=str, required=True, help="Path to the CSV file containing trajectories.")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the JANI model file.")
    parser.add_argument(
        "--property_path",
        type=str, required=True, help="Path to the JANI property file.")
    parser.add_argument(
        "--num_epochs", 
        type=int, default=100, help="Number of training epochs.")
    parser.add_argument(
        "--batch_size", 
        type=int, default=64, help="Batch size for training.")
    parser.add_argument(
        "--num_slices",
        type=int, default=32, help="Number of slices for replay buffer.")
    parser.add_argument(
        "--penalize_unsafe", 
        action="store_true", help="Whether to penalize unsafe states.")
    parser.add_argument(
        "--penalization", 
        type=float, default=-0.01, help="Penalization reward for unsafe states.")
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
        "--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    
    # Make PyTorch deterministic (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Initialize environment
    env = JaniEnv(model_file=args.model_path, property_file=args.property_path, seed=args.seed)
    action_dim = env.action_space.n

    # Load dataset and create replay buffer
    if args.penalize_unsafe:
        td = read_trajectories(args.trajectory_path, action_dim=action_dim, penalize_unsafe=True, unsafe_reward=args.penalization)
    else:
        td = read_trajectories(args.trajectory_path, action_dim=action_dim)
    replay_buffer = create_replay_buffer(td, num_slices=args.num_slices, batch_size=args.batch_size)

    # Extract state and action dimensions
    state_dim = td["observation"].shape[-1]
    assert state_dim == env.observation_space.shape[0], "State dimension mismatch between dataset and environment."

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

    actor, _, _ = train(
        args.total_timesteps, 
        best_params["steps_per_epoch"], 
        best_params["batch_size"], 
        best_params["lr"], 
        replay_buffer, 
        iql_loss,
        print_info=True
    )
    results = evaluate_on_env(env, actor, num_episodes=100)
    print(f"Final success rate over 100 episodes: {results['success_rate']:.2f}, avg reward: {results['avg_reward']:.2f}, failure rate: {results['failure_rate']:.2f}")

if __name__ == "__main__":
    main()