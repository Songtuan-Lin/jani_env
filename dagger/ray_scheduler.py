import argparse
import ray
import torch

from .train import train

from utils import get_all_configs


@ray.remote(num_gpus=1)
class BenchmarkTrainer:
    def __init__(self, hyperparams):
        self.hyperparams = hyperparams

    def train_on_benchmark(self, args, file_args, device: torch.device):
        """Train the model on the benchmark dataset."""
        train(args, file_args, self.hyperparams, device)


def main():
    parser = argparse.ArgumentParser(description="Benchmark Trainer with Ray")
    parser.add_argument("--root", type=str, required=True, help="Path to the benchmark root directory")
    parser.add_argument("--log_directory", type=str, default="./logs", help="Directory to save logs")
    parser.add_argument("--num_trainers", type=int, default=4, help="Number of policy trainers to run in parallel")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of rollout workers per trainer")
    parser.add_argument("--num_iterations", type=int, default=50, help="Number of training iterations per benchmark")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for training (e.g., 'cuda' or 'cpu')")
    args = parser.parse_args()

    # Load file arguments and hyperparameters
    shared_args = {
        "log_directory": args.log_directory,
    }
    hyperparams = {
        'learning_rate': 1e-3,
        'replay_buffer_capacity': 10000,
        'num_iterations': args.num_iterations,
        'batch_size': 256,
        'num_workers': args.num_workers # Ensure not to exceed available CPUs
    }
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    list_configs = get_all_configs(args.root, shared_args)

    # Initialize Ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=False, log_to_driver=False, include_dashboard=False)

    # Create BenchmarkTrainer actors
    trainers = [BenchmarkTrainer.remote(hyperparams) for _ in range(args.num_trainers)]

    # Launch training in parallel
    futures = []
    for idx, (benc_args, file_args) in enumerate(list_configs):
        trainer = trainers[idx % args.num_trainers]
        futures.append(trainer.train_on_benchmark.remote(benc_args, file_args, device))

    # Wait for all trainings to complete
    ray.get(futures)


if __name__ == "__main__":
    main()