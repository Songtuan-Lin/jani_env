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
        q_risk_values = td.get("q_risk").view(-1, self.q_risk_module.module.out_features) # Ensure q_risk_values has shape (batch_size, action_dim)
       
        chosen_value = q_risk_values.gather(dim=1, index=task_action)

        # Decide whether to use recovery policy based on risk threshold
        use_recovery = (chosen_value > self.risk_threshold).float()

        # Get recovery policy action
        recovery_action = td.get("recovery_action").view(-1, 1) # Ensure recovery_action has shape (batch_size, 1)
        # Compute the final action by selecting between task and recovery actions based on the risk assessment
        final_action = torch.where(use_recovery.bool(), recovery_action, task_action).squeeze(1)
        # Set the final action in the tensordict
        td.set("action", final_action)
        return td



if __name__ == "__main__":
    import argparse

    from .train import create_actor, create_critic, create_data_collector

    from jani import TorchRLJANIEnv
    from torchrl.modules import MLP, ValueOperator

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

    task_policy = create_actor(hyperparams={}, env=env, out_keys=["task_action"])
    recovery_policy = create_actor(hyperparams={}, env=env, out_keys=["recovery_action"])

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
        out_keys=["q_risk"]
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