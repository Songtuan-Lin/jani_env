import torchrl

from tensordict.nn import TensorDictModule
from torchrl.modules import MLP, ProbabilisticActor, MaskedCategorical


def create_q_module(state_dim, action_dim, hidden_dims=[32, 64]):
    """
    Create a Q-network for state-action value estimation.
    
    Args:
        state_dim: Dimension of the state space
        action_dim: Dimension of the action space
        hidden_dims: List of hidden layer dimensions
    """
    q_net = MLP(
        in_features=state_dim, 
        out_features=action_dim, 
        num_cells=hidden_dims
    )
    q_module = TensorDictModule(
        module=q_net, 
        in_keys=["observation"], 
        out_keys=["state_action_value"]
    )
    return q_module


def create_v_module(state_dim, hidden_dims=[32, 64]):
    """
    Create a V-network for state value estimation.
    
    Args:
        state_dim: Dimension of the state space
        hidden_dims: List of hidden layer dimensions
    """
    v_net = MLP(
        in_features=state_dim, 
        out_features=1, 
        num_cells=hidden_dims
    )
    v_module = TensorDictModule(
        module=v_net, 
        in_keys=["observation"], 
        out_keys=["state_value"]
    )
    return v_module


def create_actor(state_dim, action_dim, hidden_dims=[32, 64]):
    """
    Create a policy network for action selection.
    """
    policy_net = MLP(
        in_features=state_dim, 
        out_features=action_dim, 
        num_cells=hidden_dims
    )
    policy_module = TensorDictModule(
        module=policy_net, 
        in_keys=["observation"],
        out_keys=["logits"]
    )
    actor = ProbabilisticActor(
        module=policy_module,
        in_keys={"logits": "logits", "mask": "action_mask"},
        out_keys=["action"],
        spec=torchrl.data.Categorical(action_dim),
        distribution_class=MaskedCategorical,
        return_log_prob=True
    )
    return actor