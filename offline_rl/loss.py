import torch

from .models import create_actor, create_q_module, create_v_module
from .load_dataset import create_replay_buffer, read_trajectories

from torchrl.objectives import DiscreteIQLLoss
from torchrl.objectives.utils import _pseudo_vmap, _reduce


class DiscreteIQLLossLB(DiscreteIQLLoss):
    """
    Discrete IQL loss with lower bound on advantage.
    
    Args:
        expectile: Expectile parameter for value function update
        temperature: Temperature parameter for advantage weighting
        min_advantage: Minimum advantage value to consider
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def value_loss(self, tensordict) -> tuple[torch.Tensor, dict]:
        # Min Q value
        with torch.no_grad():
            # Min Q value
            td_q = tensordict.select(*self.qvalue_network.in_keys, strict=False)
            td_q = self._vmap_qvalue_networkN0(td_q, self.target_qvalue_network_params)
            state_action_value = td_q.get(self.tensor_keys.state_action_value)
            action = tensordict.get(self.tensor_keys.action)
            if self.action_space == "categorical":
                if action.ndim < (
                    state_action_value.ndim - (td_q.ndim - tensordict.ndim)
                ):
                    # unsqueeze the action if it lacks on trailing singleton dim
                    action = action.unsqueeze(-1)
                if self.deactivate_vmap:
                    vmap = _pseudo_vmap
                else:
                    vmap = torch.vmap
                chosen_state_action_value = vmap(
                    lambda state_action_value, action: torch.gather(
                        state_action_value, -1, index=action
                    ).squeeze(-1),
                    (0, None),
                )(state_action_value, action)
            elif self.action_space == "one_hot":
                action = action.to(torch.float)
                chosen_state_action_value = (state_action_value * action).sum(-1)
            else:
                raise RuntimeError(f"Unknown action space {self.action_space}.")
            min_Q, _ = torch.min(chosen_state_action_value, dim=0)
        # state value
        td_copy = tensordict.select(*self.value_network.in_keys, strict=False)
        with self.value_network_params.to_module(self.value_network):
            self.value_network(td_copy)
        value = td_copy.get(self.tensor_keys.value).squeeze(-1)
        value_loss = self.loss_value_diff(min_Q - value, self.expectile)
        print(tensordict)
        print("Min Q:", min_Q.item())
        value_loss = _reduce(value_loss, reduction=self.reduction)
        self._clear_weakrefs(
            tensordict,
            "actor_network_params",
            "qvalue_network_params",
            "value_network_params",
            "target_actor_network_params",
            "target_qvalue_network_params",
            "target_value_network_params",
        )
        return value_loss, {}
    

if __name__ == "__main__":
    test_trajectories_file = "examples/iql/trajectories_test.csv"
    td = read_trajectories(test_trajectories_file)
    rb = create_replay_buffer(td, num_slices=2, batch_size=2)

    state_dim = td["observation"].shape[-1]
    action_dim = 6  # Assuming
    q_module = create_q_module(state_dim, action_dim)
    v_module = create_v_module(state_dim)
    actor_module = create_actor(state_dim, action_dim)

    loss = DiscreteIQLLossLB(
        actor_network=actor_module,
        qvalue_network=q_module,
        value_network=v_module,
        action_space="categorical"
    )

    data = rb.sample()
    loss_value, info = loss(data)
    print("Loss:", loss_value.item())
    print("Info:", info)