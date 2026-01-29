import ray
import torch
import torch.nn as nn

from .policy import Policy
from .buffer import collect_trajectory

from jani import JANIEnv
from utils import create_env


def to_cpu_state_dict(policy: torch.nn.Module):
    # Help function to move model state dict to CPU
    return {k: v.detach().cpu() for k, v in policy.state_dict().items()}



@ray.remote(num_cpus=1)
class RolloutWorker:
    def __init__(self, file_args: dict, network_paras: dict, network_state_dict):
        """Initialize the RolloutWorker with environment and policy network."""
        self.env = create_env(file_args, n_envs=1, monitor=False, time_limited=True)
        self.policy = Policy(
            input_dim=network_paras['input_dim'],
            output_dim=network_paras['output_dim'],
            hidden_dims=network_paras['hidden_dims']
        )
        self.policy.load_state_dict(network_state_dict)
        self.policy.eval()

    @torch.no_grad()
    def run_one_rollout(self, idx: int):
        """Run one rollout in the environment using the current policy."""
        trajectory = collect_trajectory(self.env, self.policy, idx)
        return trajectory
    
    def set_weights(self, state_dict):
        self.policy.load_state_dict(state_dict)
        self.policy.eval()
    


class RolloutManager:
    def __init__(self, file_args: dict, network_paras: dict, policy: nn.Module, num_workers: int):
        """Initialize the RolloutManager with multiple RolloutWorker actors."""
        # Initialize Ray
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=False, log_to_driver=False, include_dashboard=False)

        network_state_dict = to_cpu_state_dict(policy)
        # state_dict_ref = ray.put(network_state_dict)

        # Create RolloutWorker actors
        self.workers = [RolloutWorker.remote(file_args, network_paras, network_state_dict) for _ in range(num_workers)]
        self.num_workers = num_workers


    def update_policy(self, policy: nn.Module):
        """Update the policy weights in all RolloutWorker actors."""
        network_state_dict = to_cpu_state_dict(policy)
        state_dict_ref = ray.put(network_state_dict)
        for w in self.workers:
            w.set_weights.remote(state_dict_ref)


    def run_rollouts(self, init_size: int) -> list:
        """Run multiple rollouts in parallel using the RolloutWorker actors."""
        futures = []
        for idx in range(init_size):
            worker = self.workers[idx % self.num_workers]
            futures.append(worker.run_one_rollout.remote(idx))

        # Gather results
        rollouts = ray.get(futures)
        return rollouts


def run_rollouts(
        file_args: dict, 
        network_paras: dict, 
        policy: nn.Module,
        num_workers: int, 
        init_size: int) -> list:
    """Run multiple rollouts in parallel using Ray."""
    # Initialize Ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=False, log_to_driver=False, include_dashboard=False)

    network_state_dict = to_cpu_state_dict(policy)
    # state_dict_ref = ray.put(network_state_dict)

    # Create a RolloutWorker actor
    workers = [RolloutWorker.remote(file_args, network_paras, network_state_dict) for _ in range(num_workers)]

    # Launch rollouts in parallel
    futures = []
    for idx in range(init_size):
        worker = workers[idx % num_workers]
        futures.append(worker.run_one_rollout.remote(idx))

    # Gather results
    rollouts = ray.get(futures)
    return rollouts