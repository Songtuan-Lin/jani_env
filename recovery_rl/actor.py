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
        # A new field q_risk will be added to the tensordict of size (batch_size, action_dim)
        td = self.q_risk_module(td)
        # A new field recovery_action will be added to the tensordict of size (batch_size, 1)
        td = self.recovery_policy(td)

        task_action = td.get("task_action")
        task_action = task_action.view(-1, 1) # Ensure task_action has shape (batch_size, 1)
        q_risk_values = td.get("q_risk")

        chosen_value = q_risk_values.gather(dim=1, index=task_action)

        # Decide whether to use recovery policy based on risk threshold
        use_recovery = (chosen_value > self.risk_threshold).float().unsqueeze(-1)

        # Get recovery policy action
        recovery_action = td.get("recovery_action")
        # Compute the final action by selecting between task and recovery actions based on the risk assessment
        final_action = torch.where(use_recovery.bool(), recovery_action, task_action)
        # Set the final action in the tensordict
        td.set("action", final_action)
        return td