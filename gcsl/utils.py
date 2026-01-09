import torch

from tensordict.nn import TensorDictModule
from tensordict import TensorDict

from torchrl.modules import ProbabilisticActor
from torchrl.modules.distributions import MaskedCategorical

from .model import GoalConditionedActor

from jani import TorchRLJANIEnv


def collect_trajectory(env: TorchRLJANIEnv, policy: GoalConditionedActor, max_horizon: int = 2048) -> TensorDict:
    # Use the policy as the backbone for the actor
    actor_module = TensorDictModule(
        module=policy,
        in_keys=["observation"],
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
        td = env.rollout(max_steps=max_horizon, policy=actor)
    # Fetch the relevant keys and construct the trajectory tensordict
    trajectory_td = TensorDict({
        "observation": td["observation"],
        "action": td["action"],
        "condition": td["next"]["condition"],
        "next_observation": td["next"]["observation"],
        "action_mask": td["action_mask"]
    }, batch_size=td.batch_size)
    return trajectory_td