# Safety-Aware IQL with Action-Based Q-Value Lower Bound

## Context

We are implementing offline RL using IQL (Implicit Q-Learning) for JANI environments that have a safety oracle. The goal is to leverage safety information during training to extract better policies.

## Key Definitions

**Safe Action**: An action `a` in state `s` is safe if, after taking `a` and reaching state `s'`, there exists a policy that can **never** reach a failure state from `s'`.

**Implication**: If an action is safe, the optimal Q-value for that action is guaranteed to be ≥ 0, because:
- You can never reach failure (reward -1)
- At worst, you accumulate 0 reward forever
- At best, you reach the goal (reward +1)

## The Modification

**Standard IQL Q-value update:**
```python
target = r + γ * V(s')
loss = (Q(s, a) - target)^2
```

**Safety-aware Q-value update:**
```python
target = r + γ * V(s')
if is_action_safe:
    target = max(target, 0)  # Lower bound of 0 for safe actions
loss = (Q(s, a) - target)^2
```

## Why This Works

1. **Corrects pessimism from bad offline data**: If the offline dataset contains trajectories where safe actions were followed by poor play (leading to low V(s')), the learned Q would be pessimistically low. The lower bound corrects this.

2. **Preserves action discrimination**:
   - Safe actions: Q ≥ 0
   - Unsafe actions: Q can be < 0
   - Among safe actions, the ordering is preserved (Q = max(r + γV(s'), 0), so better actions still have higher Q when their natural value exceeds 0)

3. **Uses domain knowledge appropriately**: We have oracle knowledge that safe actions guarantee non-negative value under optimal play. We inject this knowledge into learning.

## Important: Only Rectify Q, Not V

Rectifying **both** Q and V could compress advantages and hurt policy extraction. The recommendation is:

- **Do**: Rectify Q-value targets using `is_action_safe`
- **Don't**: Rectify V-value targets using `is_state_safe` (remove `DiscreteIQLLossValueLB`)

Let V learn naturally from Q via expectile regression. The Q-level lower bound is more surgical and doesn't interfere with advantage computation.

### Why Not Rectify V?

IQL extracts the policy using advantage-weighted regression:
```
L_actor = E[ exp(β * (Q(s,a) - V(s))) * log π(a|s) ]
```

If both Q and V are clamped to ≥ 0:
- Advantages `A = Q - V` could be compressed toward 0
- Discrimination between safe actions could be lost

Example:
- True values: Q(s, a1) = -0.1, Q(s, a2) = -0.3, both safe
- With Q rectification only: Q(a1) → 0, Q(a2) → 0, V learns naturally
- With both: Q(a1) = Q(a2) = 0, V = 0, so A(a1) = A(a2) = 0 → **lost discrimination!**

## Data Requirements

The sampled trajectories (from `sample.py`) include:
- `is_action_safe`: Whether the taken action was safe (use this for Q rectification)
- `is_state_safe`: Whether the current state was safe (available but not needed for Q rectification)

## Implementation

Create a new loss class `DiscreteIQLLossActionSafeLB` that modifies `qvalue_loss()`:

```python
class DiscreteIQLLossActionSafeLB(DiscreteIQLLoss):
    """
    Discrete IQL loss with lower bound on Q-value targets for safe actions.

    For transitions where the action taken was safe, the Q-value target
    is lower-bounded by 0, reflecting the fact that safe actions guarantee
    the existence of a policy that avoids failure.
    """

    @dataclass
    class _AcceptedKeys:
        # ... standard keys ...
        is_action_safe: NestedKey = "is_action_safe"  # Add this key

    def qvalue_loss(self, tensordict: TensorDictBase) -> tuple[Tensor, dict]:
        obs_keys = self.actor_network.in_keys
        next_td = tensordict.select(
            "next", *obs_keys, self.tensor_keys.action, strict=False
        )

        with torch.no_grad():
            # Standard target value computation
            target_value = self.value_estimator.value_estimate(
                next_td, target_params=self.target_value_network_params
            ).squeeze(-1)

            # Apply lower bound for safe actions
            is_action_safe = tensordict.get("is_action_safe").squeeze(-1)
            target_value = torch.where(
                is_action_safe,
                torch.maximum(target_value, torch.zeros_like(target_value)),
                target_value
            )

        # Rest of Q-value loss computation (unchanged from standard IQL)
        td_q = tensordict.select(*self.qvalue_network.in_keys, strict=False)
        td_q = self._vmap_qvalue_networkN0(td_q, self.qvalue_network_params)
        state_action_value = td_q.get(self.tensor_keys.state_action_value)
        action = tensordict.get(self.tensor_keys.action)

        if self.action_space == "categorical":
            if action.ndim < (state_action_value.ndim - (td_q.ndim - tensordict.ndim)):
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
        return loss_qval, metadata
```

## Key Points for Implementation

1. Use `is_action_safe` from the tensordict (recorded during sampling in `sample.py`)
2. Only modify the **target**, not the prediction
3. Remove or don't use the V-function lower bound (`DiscreteIQLLossValueLB`)
4. The actor loss and value loss remain unchanged - they use the learned Q and V naturally

## Theoretical Justification

The modification is sound because:

1. **Safe action semantics**: A safe action guarantees that from the next state, there exists a policy that never reaches failure. This means V*(s') ≥ 0 under optimal safe play.

2. **Lower bound validity**: Since r ≥ 0 for safe transitions (no failure penalty) and V*(s') ≥ 0, we have Q*(s, safe_a) = r + γV*(s') ≥ 0.

3. **Offline RL goal**: The goal is to extract the best possible policy, not to accurately model the behavior policy. Using oracle knowledge to correct pessimism on safe actions is a principled way to improve policy quality.

4. **Selective optimism**: We're only optimistic about actions we *know* are safe. Unsafe actions retain standard pessimistic learning, preventing overestimation of risky actions.
