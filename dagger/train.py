import torch
import torch.nn as nn

from tensordict import TensorDict
from torchrl.modules import MaskedCategorical

from jani import JANIEnv


def collect_trajectory(env: JANIEnv, policy: nn.Module, idx: int, max_horizon: int = 2048) -> TensorDict:
    """Collect trajectories using the given policy."""
    policy.cpu() # Ensure policy is on CPU
    observations, next_observations, actions, rewards = [], [], [], []
    safety, safe_actions = [], []
    obs_to_correct, corrected_actions = [], []
    obs_to_keep, kept_actions = [], []

    obs, reset_info = env.reset(options={"idx": idx}) # Reset environment to specific initial state
    safety.append(reset_info["current_state_safety"])
    safe_actions.append(reset_info["current_safe_action"])
    for _ in range(max_horizon):
        observations.append(obs)
        action_mask = env.action_mask()
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        action_mask_tensor = torch.tensor(action_mask, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            logits = policy(obs_tensor) # Compute logits from policy
            action_dist = MaskedCategorical(logits=logits, mask=action_mask_tensor)
            action = action_dist.sample().squeeze(0).item()  # Sample action and remove batch dimension
        actions.append(action)
        obs, reward, done, _, info = env.step(action)

        assert "next_state_safety" in info, "Info dict missing 'next_state_safety'"
        assert "next_safe_action" in info, "Info dict missing 'next_safe_action'"
        # If entering into an unsafe state
        if not info["next_state_safety"]:
            assert reward == env._unsafe_reward, "Inconsistent safety info"
            if safety[-1]:
                # If the previous state was safe but current state is unsafe, we need to correct the previous action
                obs_to_correct.append(observations[-1])
                safe_action = safe_actions[-1]
                assert safe_action != -1, "Safe action should be valid"
                assert safe_action != action, "Safe action should differ from the taken unsafe action"
                corrected_actions.append(safe_action)
                assert len(obs_to_correct) == len(corrected_actions), "Mismatch in correction lengths"
        else: # If the next state is still safe
            obs_to_keep.append(observations[-1]) # Keep the action taken in the previous safe state unchanged
            kept_actions.append(action) # This is to restrict the policy not deviate too much
            assert len(obs_to_keep) == len(kept_actions), "Mismatch in kept action lengths"

        rewards.append(reward)
        next_observations.append(obs)
        if done:
            break

        # Record safety info, ignore the last step if done
        safety.append(info["next_state_safety"])
        safe_actions.append(info["next_safe_action"])

    assert len(observations) == len(actions) == len(rewards) == len(next_observations) == len(safety) == len(safe_actions), "Mismatch in trajectory lengths"
    trajectory = TensorDict({
        "observation": torch.tensor(observations, dtype=torch.float32).unsqueeze(0),  # Add batch dimension 1
        "action": torch.tensor(actions, dtype=torch.long).unsqueeze(0),  # Add batch dimension 1
        "next_observation": torch.tensor(next_observations, dtype=torch.float32).unsqueeze(0),  # Add batch dimension 1
        "reward": torch.tensor(rewards, dtype=torch.float32).unsqueeze(0),  # Add batch dimension 1
        "safety": torch.tensor(safety, dtype=torch.bool).unsqueeze(0),  # Add batch dimension 1
        "safe_action": torch.tensor(safe_actions, dtype=torch.long).unsqueeze(0),  # Add batch dimension 1
        "obs_to_correct": torch.tensor(obs_to_correct, dtype=torch.float32).unsqueeze(0),  # Add batch dimension 1
        "corrected_action": torch.tensor(corrected_actions, dtype=torch.long).unsqueeze(0),  # Add batch dimension 1
        "obs_to_keep": torch.tensor(obs_to_keep, dtype=torch.float32).unsqueeze(0),  # Add batch dimension 1
        "kept_action": torch.tensor(kept_actions, dtype=torch.long).unsqueeze(0)  # Add batch dimension 1
    }, batch_size=())

    return trajectory