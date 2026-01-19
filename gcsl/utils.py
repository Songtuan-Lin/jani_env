import torch

from tensordict.nn import TensorDictModule
from tensordict import TensorDict

from torchrl.modules import ProbabilisticActor
from torchrl.modules.distributions import MaskedCategorical
from torchrl.envs import EnvBase

from .model import GoalConditionedActor

from jani import TorchRLJANIEnv


def collect_trajectory(env: EnvBase, policy: GoalConditionedActor | None = None, max_horizon: int = 2048) -> TensorDict:
    if policy is not None:
        policy.cpu() # Ensure policy is on CPU for rollout
        # Use the policy as the backbone for the actor
        actor_module = TensorDictModule(
            module=policy,
            in_keys=["observation_with_goal"],
            out_keys=["logits"]
        )
        # Construct the actor
        actor = ProbabilisticActor(
            module=actor_module,
            in_keys={"logits": "logits", "mask": "action_mask"},
            distribution_class=MaskedCategorical,
            out_keys=["action"],
            return_log_prob=True
        )

        # Collect a trajectory
        with torch.no_grad():
            td_test = env.reset()
            for _ in range(max_horizon):
                x = td_test["observation_with_goal"].unsqueeze(0)  # Add batch dimension
                assert len(x.shape) == 2, f"Expected observation_with_goal to have shape (1, obs_dim + condition_size), got {x.shape}"
                print(f"Observation with goal: {x}")
                logits = policy(x)
                mask = td_test["action_mask"].unsqueeze(0)  # Add batch dimension
                print(f"Action mask: {mask}")
                print(f"Logits: {logits}")
                action = MaskedCategorical(logits=logits, mask=mask).sample()
                assert mask[0, action.item()].item(), (action.item(), mask)
                action_td = TensorDict({"action": action.squeeze(0)}, batch_size=())
                td_test = env.step(action_td)
                if td_test.get("done").item():
                    break
            td = env.rollout(max_steps=max_horizon, policy=actor)
    else:
        # Collect a random trajectory
        td = env.rollout(max_steps=max_horizon)
    # Fetch the relevant keys and construct the trajectory tensordict
    trajectory_td = TensorDict({
        "observation": td["observation"],
        "action": td["action"],
        "condition": td["next"]["condition"],
        "next_observation": td["next"]["observation"],
        "action_mask": td["action_mask"]
    }, batch_size=())
    # print(f"Observation shape in collected trajectory: {trajectory_td['observation'].shape}")
    # print(f"Action shape in collected trajectory: {trajectory_td['action'].shape}")
    # print(f"Condition shape in collected trajectory: {trajectory_td['condition'].shape}")
    # print(f"Next observation shape in collected trajectory: {trajectory_td['next_observation'].shape}")
    # print(f"Action mask shape in collected trajectory: {trajectory_td['action_mask'].shape}")
    return trajectory_td