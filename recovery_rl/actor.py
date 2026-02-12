import torch

from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase, TensorDictModule


class RecoveryActor(TensorDictModuleBase):
    def __init__(
            self,
            task_policy: TensorDictModule,
            recovery_policy: TensorDictModule,
            q_risk_module: TensorDictModule,
            risk_threshold: float):

        super().__init__()
        # Set up input and output keys
        self.in_keys = ["observation", "action_mask"]
        self.out_keys = ["action"]

        # Store the policies and risk module
        self.task_policy = task_policy
        self.recovery_policy = recovery_policy
        self.q_risk_module = q_risk_module
        self.risk_threshold = risk_threshold

    def forward(self, td: TensorDict) -> TensorDict:
        # A new field task_action will be added to the tensordict of size (batch_size, 1)
        td = self.task_policy(td)
        # print("Task action shape:", td.get("task_action").shape)
        # A new field q_risk will be added to the tensordict of size (batch_size, action_dim)
        td = self.q_risk_module(td)
        # print("Q-risk value shape:", td.get("q_risk_value").shape)
        # A new field recovery_action will be added to the tensordict of size (batch_size, 1)
        td = self.recovery_policy(td)

        task_action = td.get("task_action")  # One-hot encoded action
        action_mask = td.get("action_mask")  # Boolean mask

        # Convert one-hot action to index for easier manipulation
        task_action_idx = task_action.argmax(dim=-1, keepdim=True)

        # Assert task action respects action mask
        # Extract the mask value at the chosen action index
        task_action_valid = action_mask.gather(dim=-1, index=task_action_idx)
        if not torch.all(task_action_valid):
            print("Action mask:", action_mask)
            print("Task action (one-hot):", task_action)
            print("Task action (index):", task_action_idx)
            print("Observation:", td.get("observation"))
            raise ValueError("Task action does not respect action mask")

        # Get Q-risk values and extract the value for the chosen task action
        q_risk_values = td.get("q_risk_value")  # Shape: (batch_size, num_actions) or (num_actions,)

        # Handle both batched and unbatched cases
        if q_risk_values.dim() == 1:
            # Unbatched: q_risk_values is (num_actions,), task_action_idx is (1,)
            chosen_value = q_risk_values.gather(dim=0, index=task_action_idx.squeeze(-1))
        else:
            # Batched: q_risk_values is (batch_size, num_actions)
            chosen_value = q_risk_values.gather(dim=-1, index=task_action_idx).squeeze(-1)

        # Decide whether to use recovery policy based on risk threshold
        use_recovery = (chosen_value < self.risk_threshold)

        # Get recovery policy action (also one-hot encoded)
        recovery_action = td.get("recovery_action")
        recovery_action_idx = recovery_action.argmax(dim=-1, keepdim=True)

        # Assert recovery action respects action mask
        recovery_action_valid = action_mask.gather(dim=-1, index=recovery_action_idx)
        if not torch.all(recovery_action_valid):
            print("Action mask:", action_mask)
            print("Recovery action (one-hot):", recovery_action)
            print("Recovery action (index):", recovery_action_idx)
            raise ValueError("Recovery action does not respect action mask")

        # Compute the final action by selecting between task and recovery actions
        # Expand use_recovery to match the one-hot action shape
        if use_recovery.dim() == 0:
            # Scalar case
            final_action_onehot = recovery_action if use_recovery.item() else task_action
        else:
            # Batch case: use_recovery is (batch_size,), actions are (batch_size, num_actions)
            use_recovery_expanded = use_recovery.unsqueeze(-1).expand_as(task_action)
            final_action_onehot = torch.where(use_recovery_expanded, recovery_action, task_action)

        # Convert one-hot action to index for environment compatibility
        # The environment expects integer actions, not one-hot vectors
        final_action_idx = final_action_onehot.argmax(dim=-1)

        # Set BOTH the one-hot action (for training) and the index action (for environment)
        # The environment step will use "action" field
        td.set("action", final_action_idx)
        # Store the one-hot version separately for potential use in training
        td.set("action_onehot", final_action_onehot)
        return td



if __name__ == "__main__":
    import argparse

    from .utils import create_actor_module, create_data_collector

    from jani import TorchRLJANIEnv

    from torchrl.modules import MLP, ValueOperator, ProbabilisticActor
    from torchrl.modules.distributions import MaskedCategorical

    parser = argparse.ArgumentParser(description="Test RecoveryActor")
    parser.add_argument("--jani_model", type=str, required=True, help="Path to the JANI model file")
    parser.add_argument('--jani_property', type=str, default="", help="Path to the JANI property file.")
    parser.add_argument('--start_states', type=str, default="", help="Path to the start states file.")
    parser.add_argument('--objective', type=str, default="", help="Path to the objective file.")
    parser.add_argument('--failure_property', type=str, default="", help="Path to the failure property file.")
    parser.add_argument('--goal_reward', type=float, default=1.0, help="Reward for reaching the goal.")
    parser.add_argument('--failure_reward', type=float, default=-1.0, help="Reward for reaching failure state.")
    parser.add_argument('--use_oracle', action='store_true', help="Use Tarjan oracle for unsafe state detection.")
    parser.add_argument('--unsafe_reward', type=float, default=-0.01, help="Reward for unsafe states when using oracle.")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for environment and policies.")
    args = parser.parse_args()

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
    }
    env = TorchRLJANIEnv(**file_args)

    task_policy_module = create_actor_module({}, env)
    # Task policy for collecting data
    task_policy = ProbabilisticActor(
        module=task_policy_module,
        in_keys={"logits": "logits", "mask": "action_mask"},
        out_keys=["task_action"],
        distribution_class=MaskedCategorical,
        return_log_prob=True, # Not sure whether this is actually need
    )

    recovery_policy_module = create_actor_module({}, env)
    recovery_policy = ProbabilisticActor(
        module=recovery_policy_module,
        in_keys={"logits": "logits", "mask": "action_mask"},
        out_keys=["recovery_action"],
        distribution_class=MaskedCategorical,
        return_log_prob=True, # Not sure whether this is actually need
    )

    n_actions = env.n_actions
    input_size = env.observation_spec["observation"].shape[0]
    hidden_sizes = [64, 128]

    q_risk_backbone = MLP(
        in_features=input_size, 
        out_features=n_actions, 
        num_cells=hidden_sizes
    )
    q_risk_module = ValueOperator(
        module=q_risk_backbone,
        in_keys=["observation"],
        out_keys=["q_risk_value"]
    )

    recovery_actor = RecoveryActor(
        task_policy=task_policy,
        recovery_policy=recovery_policy,
        q_risk_module=q_risk_module,
        risk_threshold=0.5
    )

    data_collector = create_data_collector({"n_steps": 256, "total_timesteps": 1024}, env, recovery_actor)

    for data in data_collector:
        print("Collected data batch:")
        print(data)
        break  # Just collect one batch for testing