import sys
import torch
import torch.nn as nn

from tensordict import TensorDict
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

from rich.progress import (
    Progress, 
    SpinnerColumn, 
    TextColumn, 
    BarColumn, 
    TimeRemainingColumn, 
    TimeElapsedColumn
)

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
    dropout = hyperparams.get("actor_dropout", 0.2)
    activation_fn = hyperparams.get("activation_fn", nn.Tanh)
    # Build the actor network
    actor_backbone = MLP(
        in_features=input_size,
        out_features=n_actions,
        num_cells=hidden_sizes,
        dropout=dropout,
        activation_class=activation_fn,
    )
    # Wrap in TensorDictModule
    actor_module = TensorDictModule(
        module=actor_backbone,
        in_keys=["observation"],
        out_keys=["logits"],
    )
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
        out_features=1,
        num_cells=hidden_sizes,
        dropout=dropout,
        activation_class=activation_fn,
    )
    # Wrap in TensorDictModule
    critic_module = ValueOperator(
        module=critic_backbone,
        in_keys=["observation"],
    )
    return critic_module


def _strip_state_dict_prefix(state_dict: dict) -> dict:
    """
    Strip wrapper prefixes from state_dict keys to load into bare MLP.

    TorchRL's TensorDictModule and ProbabilisticActor add prefixes like:
    - TensorDictModule wrapping MLP: 'module.0.weight' -> '0.weight'
    - ProbabilisticActor wrapping TensorDictModule wrapping MLP:
      'module.0.module.0.weight' -> '0.weight'

    This function strips these prefixes to match the bare MLP's expected keys.
    """
    import re
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key
        # First strip 'module.' prefix (from TensorDictModule)
        if new_key.startswith('module.'):
            new_key = new_key[7:]
        # Then check for 'N.module.' pattern (from ProbabilisticActor's inner TensorDictModule)
        match = re.match(r'^(\d+)\.module\.(.+)$', new_key)
        if match:
            new_key = match.group(2)
        new_state_dict[new_key] = value
    return new_state_dict


def load_q_risk_backbone(path: str) -> TensorDictModule:
    """Load a pre-trained Q-risk model from the specified path."""
    # Load the checkpoint
    checkpoint = torch.load(path)
    input_dim= checkpoint['input_dim']
    output_dim= checkpoint['output_dim'] # this should equal to the number of actions
    hidden_dims= checkpoint['hidden_dims']

    # Create the backbone model architecture
    q_risk_backbone = MLP(
        in_features=input_dim,
        out_features=output_dim,
        num_cells=hidden_dims,
    )

    # Strip wrapper prefixes and load the state dict into the backbone model
    cleaned_state_dict = _strip_state_dict_prefix(checkpoint['state_dict'])
    q_risk_backbone.load_state_dict(cleaned_state_dict)
    
    return q_risk_backbone


def load_recovery_policy_module(path: str) -> TensorDictModule:
    """Load a pre-trained recovery actor policy from the specified path."""
    # Load the checkpoint
    checkpoint = torch.load(path)
    input_dim= checkpoint['input_dim']
    output_dim= checkpoint['output_dim'] # this should equal to the number of actions
    hidden_dims= checkpoint['hidden_dims']

    # Create the backbone model architecture
    recovery_policy_backbone = MLP(
        in_features=input_dim,
        out_features=output_dim,
        num_cells=hidden_dims,
    )

    # Strip wrapper prefixes and load the state dict into the backbone model
    cleaned_state_dict = _strip_state_dict_prefix(checkpoint['state_dict'])
    recovery_policy_backbone.load_state_dict(cleaned_state_dict)

    # Wrap in TensorDictModule
    recovery_policy_module = TensorDictModule(
        module=recovery_policy_backbone,
        in_keys=["observation"],
        out_keys=["logits"],
    )
    
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

def create_rollout_buffer(hyperparams: dict[str, any]) -> ReplayBuffer:
    """Create a rollout buffer for PPO training."""
    buffer_size = hyperparams.get("n_steps", 256)
    # Create the storage
    storage = LazyTensorStorage(
        max_size=buffer_size,
        device=hyperparams.get("device", "cpu"),
    )
    # Create the sampler
    sampler = SamplerWithoutReplacement()
    # Create the replay buffer
    rollout_buffer = ReplayBuffer(
        storage=storage,
        sampler=sampler,
    )
    return rollout_buffer

def load_replay_buffer(path: str, hyperparams: dict[str, any]) -> ReplayBuffer:
    """Load a replay buffer from the specified path."""
    replay_buffer = create_replay_buffer(
        {"replay_buffer_size": 500000}
    )
    # Load the checkpoint
    replay_buffer.loads(path) # TODO: The buffer size must align with the one used during saving, otherwise it will cause issues when loading. We can either enforce this in the code or implement a more flexible loading that can handle different buffer sizes.
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
        entropy_coeff=entropy_coef,
        critic_coeff=critic_coeff,
        loss_critic_type="smooth_l1",
    )
    return loss_module


def safety_evaluation(
    env: JANIEnv,
    actor: TensorDictModule,
    max_steps: int = 256,
    progress: Progress = None,
    task_id = None
) -> Dict[str, float]:
    """Evaluate the safety of the current policy.

    Args:
        env: The environment to evaluate on.
        actor: The actor module to evaluate.
        max_steps: Maximum steps per episode.
        progress: Optional Progress instance to use for progress bar.
        task_id: Optional task ID within the progress instance.
    """
    num_init_states = env.get_init_state_pool_size()

    num_unsafe = 0
    rewards = []

    def _run_evaluation(prog, tid):
        nonlocal num_unsafe
        for idx in range(num_init_states):
            episode_reward = 0.0
            td_reset = TensorDict({"idx": torch.tensor(idx)}, batch_size=())
            obs_td = env.reset(td_reset)
            done = False
            keep_using_oracle = True
            step_count = 0
            while not done and step_count < max_steps:
                assert "observation" in obs_td, "Observation key missing in reset output"
                # Action selection using the actor
                td_action = actor(obs_td)
                assert "action" in td_action, "Action key missing in actor output"
                action = td_action.get("action").item()

                # Check whether the action is a safe action under the current state
                if keep_using_oracle:
                    is_safe = env.is_state_action_safe(action)
                    if not is_safe:
                        num_unsafe += 1
                        # If an unsafe action is found, no need to keep using the oracle
                        keep_using_oracle = False

                # Step the environment
                next_td = env.step(td_action)
                done = next_td.get(("next", "done")).item()
                episode_reward += next_td.get(("next", "reward")).item()
                obs_td = next_td.get("next")
                step_count += 1
            assert episode_reward == 0 or episode_reward == env._goal_reward or episode_reward == env._failure_reward, "Unexpected episode reward: {}".format(episode_reward)
            rewards.append(episode_reward)
            prog.update(tid, advance=1)

    if progress is not None and task_id is not None:
        # Use the provided progress bar
        progress.reset(task_id, total=num_init_states, visible=True)
        _run_evaluation(progress, task_id)
        progress.update(task_id, visible=False)
    else:
        # Create a standalone progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeRemainingColumn(),
            TimeElapsedColumn(),
            disable=not sys.stdout.isatty(),
        ) as prog:
            tid = prog.add_task("Evaluating safety...", total=num_init_states)
            _run_evaluation(prog, tid)

    # Compute safety rate and average reward
    safety_rate = 1 - num_unsafe / num_init_states
    average_reward = sum(rewards) / len(rewards)
    results = {
        "safety_rate": safety_rate,
        "average_reward": average_reward,
    }
    return results