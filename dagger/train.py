import torch
import torch.nn as nn

from pathlib import Path
from tensordict import TensorDict
from torchrl.modules import MaskedCategorical

from jani import JANIEnv

from .buffer import collect_trajectory, DAggerBuffer
from .policy import Policy


def train_step(
        rb: DAggerBuffer, 
        policy: nn.Module, 
        optimizer: torch.optim.Optimizer, 
        batch_size: int, 
        device: torch.device) -> float:
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


def train(args: dict, file_args: dict, device: torch.device):
    """Main training loop for DAgger."""
    assert args["policy_path"] is not None, "Initial policy path must be provided for DAgger."
    policy_path = Path(args["policy_path"])
    if not policy_path.exists():
        raise FileNotFoundError(f"Policy file not found: {policy_path}")
    # Load initial policy
    checkpoint = torch.load(policy_path, map_location=device)
    input_dim = checkpoint['input_dim']
    output_dim = checkpoint['output_dim']
    hidden_dims = checkpoint['hidden_dims']
    policy = Policy(input_dim, output_dim, hidden_dims)
    policy.load_state_dict(checkpoint['policy_state_dict'])
    