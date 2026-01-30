import torch
import torch.nn as nn

from torchrl.modules import MaskedCategorical

from jani.env import JANIEnv



class Policy(nn.Module):
    """A simple feedforward policy network."""
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: list = [64, 64]):
        super(Policy, self).__init__()
        layers = []
        last_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ReLU())
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def evaluate_policy_safety_on_state(env: JANIEnv, policy: nn.Module, state_idx: int, max_steps: int, device: torch.device) -> tuple[bool, float]:
    """Evaluate the safety of a policy starting from a specific state index."""
    policy.eval()
    policy.to(device)
    obs, _ = env.reset(options={"idx": state_idx})
    done = False
    step_count = 0
    episode_reward = 0.0
    keep_using_oracle = True
    is_safe = True
    while not done and step_count < max_steps:
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)  # Add batch dimension
        action_mask = env.unwrapped.action_mask().astype(int)
        action_mask_tensor = torch.tensor(action_mask, dtype=torch.bool).unsqueeze(0).to(device)  # Add batch dimension
        with torch.no_grad():
            logits = policy(obs_tensor)
            action_dist = MaskedCategorical(logits=logits, mask=action_mask_tensor)
            action = action_dist.sample().squeeze(0).item()  # Sample action and remove batch dimension
        if keep_using_oracle:
            is_state_action_safe = env.unwrapped.is_current_state_action_safe(action)
            if not is_state_action_safe:
                is_safe = False
                keep_using_oracle = False
        obs, reward, done, _, info = env.step(action)
        step_count += 1
    episode_reward = reward # Only using the final reward
    
    return is_safe, episode_reward