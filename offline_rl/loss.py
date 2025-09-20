from __future__ import annotations

import warnings
from dataclasses import dataclass

import torch
from tensordict import TensorDict, TensorDictBase, TensorDictParams
from tensordict.nn import dispatch, TensorDictModule
from tensordict.utils import NestedKey
from torch import Tensor

from torchrl.data.tensor_specs import TensorSpec
from torchrl.data.utils import _find_action_space
from torchrl.modules import ProbabilisticActor
from torchrl.objectives.common import LossModule
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


class DiscreteIQLLossLB(DiscreteIQLLoss):
    """
    Discrete IQL loss with lower bound on advantage.
    
    Args:
        expectile: Expectile parameter for value function update
        temperature: Temperature parameter for advantage weighting
        min_advantage: Minimum advantage value to consider
    """
    @dataclass
    class _AcceptedKeys:
        """Maintains default values for all configurable tensordict keys.

        This class defines which tensordict keys can be set using '.set_keys(key_name=key_value)' and their
        default values

        Attributes:
            value (NestedKey): The input tensordict key where the state value is expected.
                Will be used for the underlying value estimator. Defaults to ``"state_value"``.
            action (NestedKey): The input tensordict key where the action is expected.
                Defaults to ``"action"``.
            log_prob (NestedKey): The input tensordict key where the log probability is expected.
                Defaults to ``"_log_prob"``.
            priority (NestedKey): The input tensordict key where the target priority is written to.
                Defaults to ``"td_error"``.
            state_action_value (NestedKey): The input tensordict key where the
                state action value is expected. Will be used for the underlying
                value estimator as value key. Defaults to ``"state_action_value"``.
            reward (NestedKey): The input tensordict key where the reward is expected.
                Will be used for the underlying value estimator. Defaults to ``"reward"``.
            done (NestedKey): The key in the input TensorDict that indicates
                whether a trajectory is done. Will be used for the underlying value estimator.
                Defaults to ``"done"``.
            terminated (NestedKey): The key in the input TensorDict that indicates
                whether a trajectory is terminated. Will be used for the underlying value estimator.
                Defaults to ``"terminated"``.
        """

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
    

if __name__ == "__main__":
    from torchrl.objectives import SoftUpdate

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

    updater = SoftUpdate(loss, eps=0.95)
    optim = torch.optim.Adam(loss.parameters(), lr=3e-4)

    for iter in range(1000):
        data = rb.sample()
        loss_td = loss(data)
        optim.zero_grad()
        total_loss = loss_td.get("loss_actor") + loss_td.get("loss_qvalue") + loss_td.get("loss_value")
        total_loss.backward()
        optim.step()
        updater.step()
        if iter % 100 == 0:
            print(f"Iter {iter}: loss {total_loss.item():.3f}")
    print("Done")