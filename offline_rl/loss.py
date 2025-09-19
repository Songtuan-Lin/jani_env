import tensordict
import torch

from .models import create_actor, create_q_module, create_v_module
from .load_dataset import create_replay_buffer, read_trajectories

from torchrl.objectives import DiscreteIQLLoss
from torchrl.objectives.utils import _pseudo_vmap, _reduce, distance_loss


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


    def qvalue_loss(self, tensordict):
        obs_keys = self.actor_network.in_keys
        next_td = tensordict.select(
            "next", *obs_keys, self.tensor_keys.action, strict=False
        )
        with torch.no_grad():
            target_value = self.value_estimator.value_estimate(
                next_td, target_params=self.target_value_network_params
            ).squeeze(-1)

        # predict current Q value
        td_q = tensordict.select(*self.qvalue_network.in_keys, strict=False)
        td_q = self._vmap_qvalue_networkN0(td_q, self.qvalue_network_params)
        state_action_value = td_q.get(self.tensor_keys.state_action_value)
        action = tensordict.get(self.tensor_keys.action)
        if self.action_space == "categorical":
            if action.ndim < (state_action_value.ndim - (td_q.ndim - tensordict.ndim)):
                # unsqueeze the action if it lacks on trailing singleton dim
                action = action.unsqueeze(-1)
            if self.deactivate_vmap:
                vmap = _pseudo_vmap
            else:
                vmap = torch.vmap
            pred_val = vmap(
                lambda state_action_value, action: torch.gather(
                    state_action_value, -1, index=action
                ).squeeze(-1),
                (0, None),
            )(state_action_value, action)
        elif self.action_space == "one_hot":
            action = action.to(torch.float)
            pred_val = (state_action_value * action).sum(-1)
        else:
            raise RuntimeError(f"Unknown action space {self.action_space}.")
        td_error = (pred_val - target_value.expand_as(pred_val)).pow(2)
        loss_qval = distance_loss(
            pred_val,
            target_value.expand_as(pred_val),
            loss_function=self.loss_function,
        ).sum(0)
        loss_qval = _reduce(loss_qval, reduction=self.reduction)
        metadata = {"td_error": td_error.detach()}
        self._clear_weakrefs(
            tensordict,
            "actor_network_params",
            "qvalue_network_params",
            "value_network_params",
            "target_actor_network_params",
            "target_qvalue_network_params",
            "target_value_network_params",
        )
        print("Q-value Loss computed successfully.")
        return loss_qval, metadata


    def actor_loss(self, tensordict):
        # KL loss
        print("Computing Actor Loss...")
        with self.actor_network_params.to_module(self.actor_network):
            dist = self.actor_network.get_dist(tensordict)

        log_prob = dist.log_prob(tensordict[self.tensor_keys.action])

        # Min Q value
        td_q = tensordict.select(*self.qvalue_network.in_keys, strict=False)
        td_q = self._vmap_qvalue_networkN0(td_q, self.target_qvalue_network_params)
        state_action_value = td_q.get(self.tensor_keys.state_action_value)
        action = tensordict.get(self.tensor_keys.action)
        print("Action:", action)
        if self.action_space == "categorical":
            if action.ndim < (state_action_value.ndim - (td_q.ndim - tensordict.ndim)):
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
        print("Min Q:", min_Q)
        if log_prob.shape != min_Q.shape:
            raise RuntimeError(
                f"Losses shape mismatch: {log_prob.shape} and {min_Q.shape}"
            )
        with torch.no_grad():
            # state value
            td_copy = tensordict.select(
                *self.value_network.in_keys, strict=False
            ).detach()
            with self.value_network_params.to_module(self.value_network):
                self.value_network(td_copy)
            value = td_copy.get(self.tensor_keys.value).squeeze(
                -1
            )  # assert has no gradient

        exp_a = torch.exp((min_Q - value) * self.temperature)
        exp_a = exp_a.clamp_max(100)

        # write log_prob in tensordict for alpha loss
        tensordict.set(self.tensor_keys.log_prob, log_prob.detach())
        loss_actor = -(exp_a * log_prob)
        loss_actor = _reduce(loss_actor, reduction=self.reduction)
        self._clear_weakrefs(
            tensordict,
            "actor_network_params",
            "qvalue_network_params",
            "value_network_params",
            "target_actor_network_params",
            "target_qvalue_network_params",
            "target_value_network_params",
        )
        print("Actor Loss computed successfully.")
        return loss_actor, {}
    
    
    def value_loss(self, tensordict) -> tuple[torch.Tensor, dict]:
        print("Computing Value Loss...")
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
        print("TD Copy Keys:", td_copy.keys())
        with self.value_network_params.to_module(self.value_network):
            print("Value Network Inputs:", td_copy['observation'])
            assert td_copy['observation'].equal(tensordict['observation']), "Observation mismatch"
            self.value_network(td_copy)
        value = td_copy.get(self.tensor_keys.value).squeeze(-1)
        value_loss = self.loss_value_diff(min_Q - value, self.expectile)
        print(tensordict)
        print("Value:", value)
        print("Min Q:", min_Q)
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
        print("Value Loss computed successfully.")
        return value_loss, {}
    

if __name__ == "__main__":
    test_trajectories_file = "examples/iql/trajectories_test.csv"
    td = read_trajectories(test_trajectories_file)
    rb = create_replay_buffer(td, num_slices=1, batch_size=2)
    print("Replay Buffer Size:", len(rb))

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
    print("start sampling")
    data = rb.sample()
    print("type(next):", type(data.get("next")))
    print("action dtype/shape:", data["action"].dtype, data["action"].shape)
    loss_value = loss(data)
    print({k: v.item() for k, v in loss_value.items()})
    print("Loss:", loss_value)