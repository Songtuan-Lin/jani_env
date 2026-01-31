import ray

from ray.exceptions import RayActorError, RayTaskError
from pathlib import Path
from types import SimpleNamespace

from .train import train_model



def get_configs_for_benchmark(variant_dir: str, domain_dir: str, shared_args: dict[str, any]) -> list[dict[str, any]]:
    """Get file arguments for benchmark environment."""
    variant_dir = Path(variant_dir)
    domain_dir = Path(domain_dir)
    variant_name = variant_dir.name
    domain_name = domain_dir.name

    list_configs = []
    for model_file in variant_dir.iterdir():
        # Ignore directories storing policy files
        if not model_file.is_file():
            continue
        jani_name = model_file.name.replace(".jani", "")
        if variant_name == "models":
            property_dir = domain_dir / "additional_properties"
            model_save_dir = domain_dir / "policies"
        else:
            property_dir = domain_dir / "additional_properties" / variant_name
            model_save_dir = domain_dir / "policies" / variant_name

        # Locate the training property file
        training_property_dir = property_dir / "random_starts_20000" / jani_name
        assert training_property_dir.exists(), f"Training property directory {training_property_dir} does not exist."
        all_training_properties_files = list(training_property_dir.iterdir())
        assert len(all_training_properties_files) == 1, f"Expected one property file in {training_property_dir}, found {len(all_training_properties_files)}"
        training_property_file = all_training_properties_files[0]

        # Locate the evaluation property file
        eval_property_dir = property_dir / "random_starts_5000" / jani_name
        assert eval_property_dir.exists(), f"Evaluation property directory {eval_property_dir} does not exist."
        all_eval_properties_files = list(eval_property_dir.iterdir())
        assert len(all_eval_properties_files) == 1, f"Expected one property file in {eval_property_dir}, found {len(all_eval_properties_files)}"
        eval_property_file = all_eval_properties_files[0]

        # set seed
        seed = 42
        # Set up arguments
        args = {
            "jani_model": str(model_file),
            "jani_property": str(training_property_file),
            "start_states": str(training_property_file),
            "objective": "",
            "failure_property": "",
            "eval_start_states": str(eval_property_file),
            "goal_reward": shared_args.get("goal_reward", 1.0),
            "failure_reward": shared_args.get("failure_reward", -1.0),
            "unsafe_reward": shared_args.get("unsafe_reward", -0.01),
            "max_steps": shared_args.get("max_steps", 1024),
            "total_timesteps": shared_args.get("total_timesteps", 35000),
            "n_envs": shared_args.get("n_envs", 1),
            "n_steps": shared_args.get("n_steps", 1024),
            "disable_oracle_cache": shared_args.get("disable_oracle_cache", False),
            "n_eval_episodes": 100,
            "wandb_project": f"{jani_name}",
            "wandb_entity": "",
            "experiment_name": jani_name,
            "log_dir": Path(shared_args.get("log_directory", "./logs")) / domain_name / variant_name / jani_name if variant_name != "models" else Path(shared_args.get("log_directory", "./logs")) / domain_name / jani_name,
            "model_save_dir": model_save_dir,
            "disable_wandb": True,
            "disable_eval": False,
            "use_separate_eval_env": True,
            "enumate_all_init_states": False,
            "log_reward": True,
            "eval_freq": 1025,
            "eval_safety": False,
            "save_all_checkpoints": True,
            "use_oracle": False,
            "verbose": 1,
            "device": "cpu",
            "seed": shared_args.get("seed", 42),
        }
        # Create file arguments
        file_args = {
            "jani_model": args.get("jani_model", str(model_file)),
            "jani_property": args.get("jani_property", str(training_property_file)),
            "start_states": args.get("start_states", str(training_property_file)),
            "eval_start_states": args.get("eval_start_states", str(eval_property_file)),
            "objective": args.get("objective", ""),
            "failure_property": args.get("failure_property", ""),
            "seed": args.get("seed", 42),
            "goal_reward": args.get("goal_reward", 1.0),
            "failure_reward": args.get("failure_reward", -1.0),
            "unsafe_reward": args.get("unsafe_reward", -0.01),
            "disable_oracle_cache": args.get("disable_oracle_cache", False),
            "use_oracle": args.get("use_oracle", False),
            "max_steps": args.get("max_steps", 1024)
        }
        list_configs.append((args, file_args))
    return list_configs


def get_configs_for_domain(domain_dir: str, shared_args: dict[str, any]) -> list[dict[str, any]]:
    """Get file arguments for all models in a domain directory."""
    domain_dir = Path(domain_dir)
    model_dir = domain_dir / "models"
    list_configs = []
    if all(e.is_dir() for e in model_dir.iterdir()):
        for variant_dir in model_dir.iterdir():
            variant_configs = get_configs_for_benchmark(str(variant_dir), str(domain_dir), shared_args)
            list_configs.extend(variant_configs)
    elif all(e.is_file() for e in model_dir.iterdir()):
        variant_configs = get_configs_for_benchmark(str(model_dir), str(domain_dir), shared_args)
        list_configs.extend(variant_configs)
    else:
        raise ValueError(f"Model directory {model_dir} contains a mix of files and directories.")
    return list_configs


def get_all_configs(root_dir: str, shared_args: dict[str, any]) -> list[dict[str, any]]:
    """Get file arguments for all domains in the root directory."""
    root_dir = Path(root_dir)
    list_configs = []
    for domain_dir in root_dir.iterdir():
        if domain_dir.is_dir():
            domain_configs = get_configs_for_domain(str(domain_dir), shared_args)
            list_configs.extend(domain_configs)
    return list_configs


@ray.remote(num_cpus=1, num_gpus=0)
class BenchmarkTrainer:
    def train_on_benchmark(self, args, file_args):
        """Train the model on the benchmark dataset."""
        train_model(args, file_args) # So far only support CPU training


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Distributed Benchmark Trainer")
    parser.add_argument('--root_dir', type=str, required=True, help="Root directory containing benchmark domains.")
    parser.add_argument('--num_trainers', type=int, default=4, help="Number of parallel workers.")
    parser.add_argument('--log_directory', type=str, default="./logs", help="Directory to save logs.")
    parser.add_argument('--goal_reward', type=float, default=1.0, help="Reward for reaching the goal.")
    parser.add_argument('--failure_reward', type=float, default=-1.0, help="Reward for reaching failure state.")
    parser.add_argument('--unsafe_reward', type=float, default=-0.01, help="Reward for unsafe states when using oracle.")
    parser.add_argument('--n_envs', type=int, default=1, help="Number of parallel environments.")
    parser.add_argument('--max_steps', type=int, default=1024, help="Max steps per episode.")
    parser.add_argument('--n_steps', type=int, default=1024, help="Number of steps per update.")
    parser.add_argument('--disable_oracle_cache', action='store_true', help="Disable caching in the oracle.")
    parser.add_argument('--total_timesteps', type=int, default=35000, help="Total timesteps for training.")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for training.")
    args = parser.parse_args()

    shared_args = {
        "log_directory": args.log_directory,
        "goal_reward": args.goal_reward,
        "failure_reward": args.failure_reward,
        "unsafe_reward": args.unsafe_reward,
        "n_envs": args.n_envs,
        "max_steps": args.max_steps,
        "n_steps": args.n_steps,
        "total_timesteps": args.total_timesteps,
        "seed": args.seed,
        "disable_oracle_cache": args.disable_oracle_cache
    }
    list_configs = get_all_configs(args.root_dir, shared_args)

    # Initialize Ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=False)
    
    trainers = [BenchmarkTrainer.remote() for _ in range(args.num_trainers)]
    
    futures = []
    for i, (config_args, config_file_args) in enumerate(list_configs):
        trainer = trainers[i % args.num_trainers]
        future = trainer.train_on_benchmark.remote(SimpleNamespace(**config_args), config_file_args)
        futures.append(future)

    # Wait for all trainers to complete
    # Wait for all trainings to complete
    results = []
    for idx, ref in enumerate(futures):
        try:
            results.append(ray.get(ref))
        except (RayTaskError, RayActorError) as e:
            print(f"BenchmarkTrainer {idx % args.num_trainers} failed during training: {e}")


if __name__ == "__main__":
    main()