import ray
import torch

from .policy import Policy
from .buffer import collect_trajectory

from jani import JANIEnv
from utils import create_env


@ray.remote
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

    @torch.no_grad()
    def run_one_rollout(self, idx: int):
        """Run one rollout in the environment using the current policy."""
        trajectory = collect_trajectory(self.env, self.policy, idx)
        return trajectory
    

def run_rollouts(
        file_args: dict, 
        network_paras: dict, 
        network_state_dict,
        num_workers: int, 
        init_size: int) -> list:
    """Run multiple rollouts in parallel using Ray."""
    # Initialize Ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=False)

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