from __future__ import annotations

import warnings
from dataclasses import dataclass

import torch
from tensordict import TensorDict, TensorDictBase, TensorDictParams
from tensordict.nn import dispatch, TensorDictModule
from tensordict.utils import NestedKey

from torchrl.objectives.utils import (
    _GAMMA_LMBDA_DEPREC_ERROR,
    _pseudo_vmap,
    _reduce,
    _vmap_func,
    default_value_kwargs,
    distance_loss,
    ValueEstimators,
)
from torchrl.objectives import DiscreteIQLLoss
from torchrl.objectives.value import TD0Estimator, TD1Estimator, TDLambdaEstimator

from .models import create_actor, create_q_module, create_v_module
from .load_dataset import create_replay_buffer, read_trajectories


class DiscreteIQLLossValueLB(DiscreteIQLLoss):
    """
    Discrete IQL loss with lower bound on value function.
    """
    @dataclass
    class _AcceptedKeys:
        value: NestedKey = "state_value"
        action: NestedKey = "action"
        log_prob: NestedKey = "_log_prob"
        priority: NestedKey = "td_error"
        state_action_value: NestedKey = "state_action_value"
        reward: NestedKey = "reward"
        done: NestedKey = "done"
        terminated: NestedKey = "terminated"
        safety: NestedKey = "safety"

    tensor_keys: _AcceptedKeys
    default_keys = _AcceptedKeys
    default_value_estimator = ValueEstimators.TD0
    out_keys = [
        "loss_actor",
        "loss_qvalue",
        "loss_value",
        "entropy",
    ]

    actor_network: TensorDictModule
    actor_network_params: TensorDictParams
    target_actor_network_params: TensorDictParams
    qvalue_network: TensorDictModule
    qvalue_network_params: TensorDictParams
    target_qvalue_network_params: TensorDictParams
    value_network: TensorDictModule | None
    value_network_params: TensorDictParams | None
    target_value_network_params: TensorDictParams | None
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print("Using lower bound on value function")
    
    
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
            assert td_copy['observation'].equal(tensordict['observation']), "Observation mismatch"
            self.value_network(td_copy)
        value = td_copy.get(self.tensor_keys.value).squeeze(-1)
        lower_bound = tensordict.get("safety").squeeze(-1)
        assert lower_bound.shape == value.shape, f"Shape mismatch: {lower_bound.shape} vs {value.shape}"
        lower_bound = lower_bound - 1.0
        value_loss = self.loss_value_diff(torch.maximum(min_Q, lower_bound) - value, self.expectile)
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


class DiscreteIQLLossQValueLB(DiscreteIQLLoss):
    """
    IQL loss with lower bound on Q-value.
    """
    @dataclass
    class _AcceptedKeys:
        value: NestedKey = "state_value"
        action: NestedKey = "action"
        log_prob: NestedKey = "_log_prob"
        priority: NestedKey = "td_error"
        state_action_value: NestedKey = "state_action_value"
        reward: NestedKey = "reward"
        done: NestedKey = "done"
        terminated: NestedKey = "terminated"
        safety: NestedKey = "safety"

    tensor_keys: _AcceptedKeys
    default_keys = _AcceptedKeys
    default_value_estimator = ValueEstimators.TD0
    out_keys = [
        "loss_actor",
        "loss_qvalue",
        "loss_value",
        "entropy",
    ]

    actor_network: TensorDictModule
    actor_network_params: TensorDictParams
    target_actor_network_params: TensorDictParams
    qvalue_network: TensorDictModule
    qvalue_network_params: TensorDictParams
    target_qvalue_network_params: TensorDictParams
    value_network: TensorDictModule | None
    value_network_params: TensorDictParams | None
    target_value_network_params: TensorDictParams | None
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print("Using lower bound on value function")

    def qvalue_loss(self, tensordict: TensorDictBase) -> tuple[Tensor, dict]:
        obs_keys = self.actor_network.in_keys
        next_td = tensordict.select(
            "next", *obs_keys, self.tensor_keys.action, strict=False
        )
        with torch.no_grad():
            target_value = self.value_estimator.value_estimate(
                next_td, target_params=self.target_value_network_params
            ).squeeze(-1)
            lower_bound = tensordict.get("safety").squeeze(-1)
            assert lower_bound.shape == target_value.shape, f"Shape mismatch: {lower_bound.shape} vs {target_value.shape}"
            lower_bound = lower_bound - 1.0
            target_value = torch.maximum(target_value, lower_bound)

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
        return loss_qval, metadata
    

if __name__ == "__main__":
    from torchrl.objectives import SoftUpdate

    test_trajectories_file = "examples/iql/trajectories_test.csv"
    td = read_trajectories(test_trajectories_file)
    td_penalized = read_trajectories(test_trajectories_file, penalize_unsafe=True, unsafe_reward=-0.01)
    rb = create_replay_buffer(td, num_slices=2, batch_size=2)
    rb_penalized = create_replay_buffer(td_penalized, num_slices=2, batch_size=2)

    state_dim = td["observation"].shape[-1]
    action_dim = 6  # Assuming
    q_module = create_q_module(state_dim, action_dim)
    v_module = create_v_module(state_dim)
    actor_module = create_actor(state_dim, action_dim)

    loss = DiscreteIQLLossValueLB(
        actor_network=actor_module,
        qvalue_network=q_module,
        value_network=v_module,
        action_space="categorical"
    )

    updater = SoftUpdate(loss, eps=0.95)
    optim = torch.optim.Adam(loss.parameters(), lr=3e-4)

    print("Training with lower bound on value function...")
    for iter in range(1000):
        data = rb_penalized.sample()
        loss_td = loss(data)
        optim.zero_grad()
        total_loss = loss_td.get("loss_actor") + loss_td.get("loss_qvalue") + loss_td.get("loss_value")
        total_loss.backward()
        optim.step()
        updater.step()
        if iter % 100 == 0:
            print(f"Iter {iter}: loss {total_loss.item():.3f}")

    
    q_module_no_lb = create_q_module(state_dim, action_dim)
    v_module_no_lb = create_v_module(state_dim)
    actor_module_no_lb = create_actor(state_dim, action_dim)

    loss_no_lb = DiscreteIQLLoss(
        actor_network=actor_module_no_lb,
        qvalue_network=q_module_no_lb,
        value_network=v_module_no_lb,
        action_space="categorical"
    )

    updater_no_lb = SoftUpdate(loss_no_lb, eps=0.95)
    optim_no_lb = torch.optim.Adam(loss_no_lb.parameters(), lr=3e-4)

    print("Training without lower bound on value function...")
    for iter in range(1000):
        data = rb.sample()
        loss_td_no_lb = loss_no_lb(data)
        optim_no_lb.zero_grad()
        total_loss_no_lb = loss_td_no_lb.get("loss_actor") + loss_td_no_lb.get("loss_qvalue") + loss_td_no_lb.get("loss_value")
        total_loss_no_lb.backward()
        optim_no_lb.step()
        updater_no_lb.step()
        if iter % 100 == 0:
            print(f"[No LB] Iter {iter}: loss {total_loss_no_lb.item():.3f}")
    print("Done")

    print("Comparing value functions...")
    with torch.no_grad():
        test_states = torch.tensor([
            [3.0, 2.5],
            [2.7, 2.1],
            [8.7, 3.7],
            [8.3, 4.6],
            [7.6, 5.3],
            [6.8, 5.5],
            [6.3, 6.8],
            [4.7, 5.3],
        ])
        test_td = TensorDict({"observation": test_states}, batch_size=[test_states.shape[0]])
        value = loss.value_network(test_td.clone())
        value_no_lb = loss_no_lb.value_network(test_td.clone())
        print("With LB:", value.get("state_value").squeeze(-1))
        print("Without LB:", value_no_lb.get("state_value").squeeze(-1))

        qvalue = loss.qvalue_network(test_td.clone())
        qvalue_no_lb = loss_no_lb.qvalue_network(test_td.clone())
        print("With LB, Q-values:", qvalue.get("state_action_value"))
        print("Without LB, Q-values:", qvalue_no_lb.get("state_action_value"))

        actor_lb = loss.actor_network(test_td.clone())
        actor_no_lb = loss_no_lb.actor_network(test_td.clone())
        print("With LB, action distribution:", torch.distributions.Categorical(logits=actor_lb.get("logits")).probs)
        print("Without LB, action distribution:", torch.distributions.Categorical(logits=actor_no_lb.get("logits")).probs)