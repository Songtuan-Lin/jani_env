import argparse
import torch
import torch.nn as nn

from pathlib import Path
from tensordict import TensorDict
from torchrl.modules import MaskedCategorical

from jani import JANIEnv
from utils import create_env, create_safety_eval_file_args, create_eval_file_args

from .buffer import collect_trajectory, DAggerBuffer
from .policy import Policy


def evaluate_policy(
        env: JANIEnv, 
        policy: nn.Module, 
        num_episodes: int = 100,
        max_steps: int = 1024, 
        device: torch.device = torch.device("cpu")) -> float:
    """Evaluate the policy over a number of episodes and return the average reward."""
    policy.to(device)
    policy.eval()
    total_reward = 0.0

    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0.0
        step_count = 0

        while not done and step_count < max_steps:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
            action_mask = env.unwrapped.action_mask().astype(int)
            action_mask_tensor = torch.tensor(action_mask, dtype=torch.bool).unsqueeze(0)  # Add batch dimension
            with torch.no_grad():
                logits = policy(obs_tensor)
                action_dist = MaskedCategorical(logits=logits, mask=action_mask_tensor)
                action = action_dist.sample().squeeze(0).item()  # Sample action and remove batch dimension
            obs, reward, done, _, _ = env.step(action)
            step_count += 1
        episode_reward += reward # Consider only reward at the end of episode

        total_reward += episode_reward

    average_reward = total_reward / num_episodes
    return average_reward


def train_step(
        rb: DAggerBuffer, 
        policy: nn.Module, 
        optimizer: torch.optim.Optimizer, 
        batch_size: int, 
        device: torch.device = torch.device("cpu")) -> float:
    """Perform a single training step for the policy network."""
    policy.train()
    policy.to(device)
    # Sample a batch of data from the replay buffer
    batch = rb.sample(batch_size)
    observations = batch["observation"].to(device)  # Shape: (batch_size, obs_dim)
    actions = batch["action"].to(device)  # Shape: (batch_size,)
    action_masks = batch["action_mask"].to(device)  # Shape: (batch_size, n_actions)

    # Forward pass through the policy network to get predicted actions
    logits = policy(observations)  # Shape: (batch_size, n_actions)
    masked_logits = logits.masked_fill(~action_masks.bool(), float('-inf'))  # Mask invalid actions
    logp = torch.log_softmax(masked_logits, dim=-1)

    # Compute loss between predicted actions and expert actions
    loss = torch.nn.NLLLoss()(logp, actions.long())

    # Backpropagation and optimization step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()



def load_policy(checkpoint) -> nn.Module:
    """Load a policy network from a checkpoint."""
    input_dim = checkpoint['input_dim']
    output_dim = checkpoint['output_dim']
    hidden_dims = checkpoint['hidden_dims']
    policy = Policy(input_dim, output_dim, hidden_dims)
    # Mapping sb3 state dict to our Policy state dict
    mapped = {
        "model.0.weight": checkpoint['state_dict']["mlp_extractor.policy_net.0.weight"],
        "model.0.bias":   checkpoint['state_dict']["mlp_extractor.policy_net.0.bias"],
        "model.2.weight": checkpoint['state_dict']["mlp_extractor.policy_net.2.weight"],
        "model.2.bias":   checkpoint['state_dict']["mlp_extractor.policy_net.2.bias"],
        "model.4.weight": checkpoint['state_dict']["action_net.weight"],
        "model.4.bias":   checkpoint['state_dict']["action_net.bias"],
    }
    policy.load_state_dict(mapped, strict=True)
    return policy



def train(args: dict, file_args: dict, hyperparams: dict, device: torch.device = torch.device("cpu")):
    """Main training loop for DAgger."""
    assert args["policy_path"] is not None, "Initial policy path must be provided for DAgger."
    policy_path = Path(args["policy_path"])
    if not policy_path.exists():
        raise FileNotFoundError(f"Policy file not found: {policy_path}")
    # Load initial policy
    print(f"Loading initial policy from {policy_path}")
    checkpoint = torch.load(policy_path, map_location=device, weights_only=False)
    policy = load_policy(checkpoint)
    print("Initial policy loaded.")

    # Decide whether to use mult-processors
    RAY_AVAILABLE = True
    try:
        from . import ray_worker
    except Exception:
        RAY_AVAILABLE = False
        if args.get("use_multiprocessors", False):
            print("Ray is not available. Proceeding without multiprocessors.")

    # Create environment for sequential rollout collection
    safety_eval_file_args = create_safety_eval_file_args(file_args, args)
    safety_eval_env = create_env(safety_eval_file_args, n_envs=1, monitor=False, time_limited=True)

    # Create environment for normal policy evaluation
    eval_file_args = create_eval_file_args(file_args)
    env = create_env(eval_file_args, n_envs=1, monitor=False, time_limited=True)

    # Initialize optimizer
    learning_rate = hyperparams.get("learning_rate", 1e-3)
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)

    # Initialize replay buffer
    rb_capacity = hyperparams.get("replay_buffer_capacity", 10000)
    rb = DAggerBuffer(buffer_size=rb_capacity)

    print(f"Average Reward before training: {evaluate_policy(env, policy, num_episodes=100, device=device):.2f}")
    # Main training loop
    num_iterations = hyperparams.get("num_iterations", 10000)
    for iter in range(num_iterations):
        if args.get("empty_buffer", False):
            rb.empty() # Optionally empty the buffer at each iteration
        # Collect new trajectories and add to replay buffer
        rollouts = []
        init_state_size = safety_eval_env.unwrapped.get_init_state_pool_size()
        init_state_size = min(args.get("num_init_states", 10000), init_state_size)
        if args.get("use_multiprocessors", False) and RAY_AVAILABLE:
            print(f"Collecting trajectories using {hyperparams.get('num_workers', 200)} Ray workers...")
            # Use Ray workers to collect trajectories in parallel
            network_paras = {
                'input_dim': checkpoint['input_dim'],
                'output_dim': checkpoint['output_dim'],
                'hidden_dims': checkpoint['hidden_dims']
            }
            rollouts = ray_worker.run_rollouts(
                file_args=safety_eval_file_args, # Use safety eval file args
                network_paras=network_paras, 
                policy=policy,
                num_workers=hyperparams.get("num_workers", 200), 
                init_size=init_state_size
            )
        else:
            print("Collecting trajectories sequentially...")
            # Collect trajectories sequentially
            for idx in range(init_state_size):
                rollout = collect_trajectory(env=safety_eval_env, policy=policy, idx=idx)
                rollouts.append(rollout)
        # Check whether all trajectories are safe
        all_safe = all([rollout[1]["is_safe_trajectory"] for rollout in rollouts])
        percentage_safe = sum([rollout[1]["is_safe_trajectory"] for rollout in rollouts]) / len(rollouts) * 100.0
        avg_reward = sum([rollout[1]["final_reward"] for rollout in rollouts]) / len(rollouts)
        print(f"Before iteration {iter}: {percentage_safe:.2f}% of collected trajectories are safe with average reward {avg_reward:.2f}.")
        if all_safe:
            avg_reward = evaluate_policy(env, policy, num_episodes=100)
            final_eval_info = {
                "iteration": iter,
                "average_reward": avg_reward
            }
            print(f"All trajectories safe at iteration {iter}. Final evaluation on domain: {final_eval_info["average_reward"]:.2f}")
            return
        # Process and add rollouts to replay buffer
        rb.add_rollouts(rollouts)

        # Perform training steps
        for s in range(hyperparams.get("steps_per_iteration", 5)):
            batch_size = hyperparams.get("batch_size", 256)
            loss = train_step(rb, policy, optimizer, batch_size, device)
            print(f"Iteration {iter} step {s}: Loss = {loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train Masked PPO on JANI Environments")
    parser.add_argument('--jani_model', type=str, required=True, help="Path to the JANI model file.")
    parser.add_argument('--jani_property', type=str, default="", help="Path to the JANI property file.")
    parser.add_argument('--start_states', type=str, required=True, help="Path to the start states file.")
    parser.add_argument('--objective', type=str, default="", help="Path to the objective file.")
    parser.add_argument('--failure_property', type=str, default="", help="Path to the failure property file.")
    parser.add_argument('--eval_start_states', type=str, default="", help="Path to the evaluation start states file.")
    parser.add_argument('--policy_path', type=str, required=True, help="Path to the initial policy file for DAgger.")
    parser.add_argument('--goal_reward', type=float, default=1.0, help="Reward for reaching the goal.")
    parser.add_argument('--failure_reward', type=float, default=-1.0, help="Reward for reaching failure state.")
    parser.add_argument('--unsafe_reward', type=float, default=-0.01, help="Reward for unsafe states when using oracle.")
    parser.add_argument('--num_init_states', type=int, default=10000, help="Number of initial states to sample from.")
    parser.add_argument('--use_multiprocessors', action='store_true', help="Use multiprocessors for rollout collection.")
    parser.add_argument('--num_workers', type=int, default=8, help="Number of workers for multiprocessor rollout collection.")
    parser.add_argument('--empty_buffer', action='store_true', help="Empty the replay buffer at each iteration.")
    parser.add_argument('--device', type=str, default="cpu", help="Device to use for training (cpu or cuda).")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument('--max_steps', type=int, default=1000, help="Max steps per episode.")
    args = parser.parse_args()

    file_args = {
        'jani_model': args.jani_model,
        'jani_property': args.jani_property,
        'start_states': args.start_states,
        'objective': args.objective,
        'failure_property': args.failure_property,
        'goal_reward': args.goal_reward,
        'failure_reward': args.failure_reward,
        'unsafe_reward': args.unsafe_reward,
        'seed': args.seed,
        'use_oracle': True, # Always use oracle during DAgger training
        'max_steps': args.max_steps
    }

    import os
    hyperparams = {
        'learning_rate': 1e-3,
        'replay_buffer_capacity': 10000,
        'num_iterations': 10000,
        'batch_size': 256,
        'num_workers': min(args.num_workers, os.cpu_count() - 2) # Ensure not to exceed available CPUs
    }

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    train(vars(args), file_args, hyperparams, device)


if __name__ == "__main__":
    main()