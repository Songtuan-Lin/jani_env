import argparse

from pathlib import Path


def get_configs_for_benchmark(variant_dir: str, domain_dir: str, shared_args: dict[str, any]) -> list[dict[str, any]]:
    """Get file arguments for benchmark environment."""
    variant_dir = Path(variant_dir)
    domain_dir = Path(domain_dir)
    variant_name = variant_dir.name
    domain_name = domain_dir.name

    condor_dir_prefix = Path(shared_args.get("condor_dir_prefix", ""))

    list_configs = []
    for model_file in variant_dir.iterdir():
        # Ignore directories storing policy files
        if not model_file.is_file():
            continue
        jani_name = model_file.name.replace(".jani", "")
        if variant_name == "models":
            property_dir = domain_dir / "additional_properties"
            model_save_dir = condor_dir_prefix / domain_dir / "ppo_policies"
            log_dir = condor_dir_prefix / Path(shared_args.get("log_directory", "./logs")) / domain_name / jani_name
        else:
            property_dir = domain_dir / "additional_properties" / variant_name
            model_save_dir = condor_dir_prefix / domain_dir / "ppo_policies" / variant_name
            log_dir = condor_dir_prefix / Path(shared_args.get("log_directory", "./logs")) / domain_name / variant_name / jani_name

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
            "jani_model": condor_dir_prefix / model_file,
            "jani_property": condor_dir_prefix / training_property_file,
            "start_states": condor_dir_prefix / training_property_file,
            "objective": "",
            "failure_property": "",
            "eval_start_states": condor_dir_prefix / eval_property_file,
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
            "experiment_name": f"{jani_name}_{variant_name}" if variant_name != "models" else jani_name,
            "log_dir": log_dir,
            "model_save_dir": model_save_dir,
            "disable_wandb": False,
            "disable_eval": False,
            "use_separate_eval_env": False,
            "enumate_all_init_states": False,
            "log_reward": True,
            "eval_freq": 1025,
            "eval_safety": False,
            "save_all_checkpoints": True,
            "use_oracle": shared_args.get("use_oracle", False),
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


def main():
    parser = argparse.ArgumentParser(description="Benchmark Trainer with Ray")
    parser.add_argument("--root", type=str, required=True, help="Path to the benchmark root directory")
    parser.add_argument("--condor_dir_prefix", type=str, required=True, help="Prefix of the path when running with HTcondor")
    parser.add_argument("--log_directory", type=str, default="./logs", help="Directory to save logs")
    parser.add_argument("--use_oracle", action="store_true", help="Whether to use oracle in training")
    parser.add_argument("--disable_oracle_cache", action="store_true", help="Disable caching in the oracle.")
    parser.add_argument("--total_timesteps", type=int, default=35000, help="Total training steps")
    parser.add_argument("--disable_wandb", action="store_true", help="Disable Weights & Biases logging")
    parser.add_argument('--goal_reward', type=float, default=1.0, help="Reward for reaching the goal")
    parser.add_argument('--failure_reward', type=float, default=-1.0, help="Reward for reaching failure state.")
    parser.add_argument('--unsafe_reward', type=float, default=-0.01, help="Reward for unsafe states when using oracle")
    parser.add_argument('--n_envs', type=int, default=1, help="Number of parallel environments.")
    parser.add_argument('--max_steps', type=int, default=1024, help="Max steps per episode")
    parser.add_argument('--n_steps', type=int, default=1024, help="Number of steps per update")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for training (e.g., 'cuda' or 'cpu')")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for training")
    parser.add_argument("--output_file", type=Path, required=True, help="File to save generated configurations")
    args = parser.parse_args()

    shared_args = {
        "condor_dir_prefix": args.condor_dir_prefix,
        "log_directory": args.log_directory,
        "goal_reward": args.goal_reward,
        "failure_reward": args.failure_reward,
        "unsafe_reward": args.unsafe_reward,
        "n_envs": args.n_envs,
        "max_steps": args.max_steps,
        "n_steps": args.n_steps,
        "total_timesteps": args.total_timesteps,
        "seed": args.seed,
        "disable_oracle_cache": args.disable_oracle_cache,
        "disable_wandb": args.disable_wandb,
        "use_oracle": args.use_oracle
    }

    all_configs = get_all_configs(args.root, shared_args)

    lines = []
    for benchmark_args, _ in all_configs:

        python_prefix = [
            "-m",
            "mask_ppo.train",
        ]

        line = python_prefix.copy()
        for k, v in benchmark_args.items():
            arg_key = "--" + k
            if type(v) == bool:
                if v:
                    line.append(arg_key)
            else:
                if v:
                    line.append(arg_key)
                    line.append(str(v))
        line = " ".join(line)

        lines.append(line)

    with args.output_file.open("w") as f:
        for idx, line in enumerate(lines):
            if idx == len(lines) - 1:
                f.write(line)
            else:
                f.write(line + "\n")


if __name__ == "__main__":
    main()