import argparse

import torch

from torchrl.objectives import DiscreteIQLLoss, SoftUpdate

from .load_dataset import read_trajectories, create_replay_buffer
from .models import create_q_module, create_v_module, create_actor
from .loss import DiscreteIQLLossValueLB, DiscreteIQLLossQValueLB


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file_path", 
        type=str, required=True, help="Path to the CSV file containing trajectories.")
    parser.add_argument(
        "--num_epochs", 
        type=int, default=100, help="Number of training epochs.")
    parser.add_argument(
        "--batch_size", 
        type=int, default=64, help="Batch size for training.")
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
    args = parser.parse_args()

    # Load dataset and create replay buffer
    if args.penalize_unsafe:
        td = read_trajectories(args.file_path, penalize_unsafe=True, unsafe_reward=args.penalization)
    else:
        td = read_trajectories(args.file_path)
    replay_buffer = create_replay_buffer(td, batch_size=args.batch_size)

    # Extract state and action dimensions
    state_dim = td["observation"].shape[-1]
    action_dim = int(td["action"].max().item()) + 1  # Actions are 0-indexed integers

    # Create models
    q_module = create_q_module(state_dim, action_dim)
    v_module = create_v_module(state_dim)
    actor_module = create_actor(state_dim, action_dim)

    # Create IQL loss
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

    # Optimizers
    optimizer = torch.optim.Adam(iql_loss.parameters(), lr=args.lr)
    updater = SoftUpdate(iql_loss, eps=0.95)

    # Training loop
    num_epochs = args.total_timesteps // args.steps_per_epoch
    for epoch in range(num_epochs):
        for iter in range(args.steps_per_epoch):
            batch = replay_buffer.sample()
            loss_dict = iql_loss(batch)
            loss_dict["loss"].backward()
            optimizer.step()
            optimizer.zero_grad()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}, Loss: {loss_dict['loss'].item():.4f}")
        updater.step()