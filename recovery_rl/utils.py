import torch
import torch.nn as nn

from tensordict.nn import TensorDictModule, TensorDictModuleBase

from torchrl.modules import MLP, ProbabilisticActor, ValueOperator
from torchrl.modules.distributions import MaskedCategorical
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement, RandomSampler
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type

from typing import Dict, Any

from jani.torchrl_env import JANIEnv


def create_actor_module(
        hyperparams: Dict[str, Any],
        env: JANIEnv,
    ) -> TensorDictModule:
    """Create the actor network for the policy."""
    n_actions = env.n_actions
    input_size = env.observation_spec["observation"].shape[0]
    hidden_sizes = hyperparams.get("actor_hidden_sizes", [64, 128])
    dropout = hyperparams.get("actor_dropout", 0.0)
    activation_fn = hyperparams.get("activation_fn", nn.Tanh)
    # Build the actor network
    actor_backbone = MLP(
        in_features=input_size,
        out_features=n_actions,
        num_cells=hidden_sizes,
        dropout=dropout,
        activation_class=activation_fn,
    )

    # Initialize weights properly to prevent NaN/Inf in logits
    for module in actor_backbone.modules():
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=1.0)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    # Wrap in TensorDictModule
    actor_module = TensorDictModule(
        module=actor_backbone,
        in_keys=["observation"],
        out_keys=["logits"],
    )
    # # Create the probabilistic actor with masked categorical distribution
    # actor = ProbabilisticActor(
    #     module=actor_module,
    #     in_keys={"logits": "logits", "mask": "action_mask"},
    #     out_keys=["task_action"],
    #     distribution_class=MaskedCategorical,
    #     return_log_prob=True, # Not sure whether this is actually need
    # )

    # actor_training = ProbabilisticActor(
    #     module=actor_module,
    #     in_keys={"logits": "logits", "mask": "action_mask"},
    #     out_keys=["action"],
    #     distribution_class=MaskedCategorical,
    #     return_log_prob=True, # Not sure whether this is actually need
    # )
    return actor_module


def create_critic(hyperparams: Dict[str, Any], env: JANIEnv) -> TensorDictModule:
    """Create the critic network for value estimation."""
    n_actions = env.n_actions
    input_size = env.observation_spec["observation"].shape[0]
    hidden_sizes = hyperparams.get("critic_hidden_sizes", [64, 128])
    dropout = hyperparams.get("critic_dropout", 0.0)
    activation_fn = hyperparams.get("activation_fn", nn.Tanh)
    # Build the critic network
    critic_backbone = MLP(
        in_features=input_size,
        out_features=n_actions,
        num_cells=hidden_sizes,
        dropout=dropout,
        activation_class=activation_fn,
    )
    # Wrap in TensorDictModule
    critic_module = ValueOperator(
        module=critic_backbone,
        in_keys=["observation"],
        out_keys=["action_value"],
    )
    return critic_module


def load_q_risk_backbone(path: str, device: torch.device) -> TensorDictModule:
    """Load a pre-trained Q-risk model from the specified path."""
    # Load the checkpoint
    checkpoint = torch.load(path, map_location=device)
    input_dim= checkpoint['input_dim']
    output_dim= checkpoint['output_dim'] # this should equal to the number of actions
    hidden_dims= checkpoint['hidden_dims']

    # Create the backbone model architecture
    q_risk_backbone = MLP(
        in_features=input_dim,
        out_features=output_dim,
        num_cells=hidden_dims,
    )

    # Load the state dict into the backbone model
    q_risk_backbone.load_state_dict(checkpoint['model_state_dict'])

    # Wrap in TensorDictModule
    # q_risk_model = ValueOperator(
    #     module=q_risk_backbone,
    #     in_keys=["observation"],
    #     out_keys=["q_risk_value"],
    # )
    
    return q_risk_backbone


def load_recovery_policy_module(path: str, device: torch.device) -> TensorDictModule:
    """Load a pre-trained recovery actor policy from the specified path."""
    # Load the checkpoint
    checkpoint = torch.load(path, map_location=device)
    input_dim= checkpoint['input_dim']
    output_dim= checkpoint['output_dim'] # this should equal to the number of actions
    hidden_dims= checkpoint['hidden_dims']

    # Create the backbone model architecture
    recovery_policy_backbone = MLP(
        in_features=input_dim,
        out_features=output_dim,
        num_cells=hidden_dims,
    )

    # Load the state dict into the backbone model
    recovery_policy_backbone.load_state_dict(checkpoint['model_state_dict'])

    # Wrap in TensorDictModule
    recovery_policy_module = TensorDictModule(
        module=recovery_policy_backbone,
        in_keys=["observation"],
        out_keys=["logits"],
    )

    # Create the probabilistic actor with masked categorical distribution
    # recovery_policy = ProbabilisticActor(
    #     module=recovery_policy_module,
    #     in_keys={"logits": "logits", "mask": "action_mask"},
    #     out_keys=["recovery_action"],
    #     distribution_class=MaskedCategorical,
    #     return_log_prob=True, # Not sure whether this is actually need
    # )
    
    return recovery_policy_module


def create_data_collector(hyperparams: Dict[str, Any], env: JANIEnv, policy: TensorDictModuleBase) -> SyncDataCollector:
    """Create a data collector for experience gathering."""
    n_steps = hyperparams.get("n_steps", 2048)
    total_timesteps = hyperparams.get("total_timesteps", 1024000)
    collector = SyncDataCollector(
        create_env_fn=env,
        policy=policy,
        total_frames=total_timesteps,
        frames_per_batch=n_steps,
        split_trajs=False,
    )
    return collector


def create_replay_buffer(hyperparams: dict[str, any]) -> ReplayBuffer:
    """Create a replay buffer for Q-risk training."""
    buffer_size = hyperparams.get("replay_buffer_size", 100000)
    # Create the storage
    storage = LazyTensorStorage(
        max_size=buffer_size,
        device=hyperparams.get("device", "cpu"),
    )
    # Create the sampler
    sampler = RandomSampler()
    # Create the replay buffer
    replay_buffer = ReplayBuffer(
        storage=storage,
        sampler=sampler,
    )
    return replay_buffer


def load_replay_buffer(path: str, hyperparams: dict[str, any]) -> ReplayBuffer:
    """Load a replay buffer from the specified path."""
    replay_buffer = create_replay_buffer(hyperparams)
    # Load the checkpoint
    replay_buffer.loads(path) # TODO: Check whether the parameters like buffer size need to be strictly matched
    return replay_buffer


def create_advantage_module(hyperparams: Dict[str, Any], value_module: nn.Module) -> nn.Module:
    """Create an advantage estimation module."""
    gae_lambda = hyperparams.get("gae_lambda", 0.95)
    gamma = hyperparams.get("gamma", 0.99)
    advantage_module = GAE(
        gamma=gamma,
        lmbda=gae_lambda,
        value_network=value_module, # This should be the critic module in PPO
        average_gae=True,
    )
    return advantage_module


def create_loss_module(hyperparams: Dict[str, Any], actor_module: TensorDictModule, critic_module: TensorDictModule) -> nn.Module:
    """Create the loss module for PPO."""
    clip_epsilon = hyperparams.get("clip_epsilon", 0.2)
    entropy_coef = hyperparams.get("ent_coef", 1e-4)
    critic_coeff = hyperparams.get("critic_coeff", 1.0)

    loss_module = ClipPPOLoss(
        actor_network=actor_module,
        critic_network=critic_module,
        clip_epsilon=clip_epsilon,
        entropy_bonus=bool(entropy_coef),
        entropy_coef=entropy_coef,
        critic_coeff=critic_coeff,
        loss_critic_type="smooth_l1",
    )
    return loss_module