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
            model_save_dir = condor_dir_prefix / domain_dir / "iql_policies" / jani_name
        else:
            property_dir = domain_dir / "additional_properties" / variant_name
            model_save_dir = condor_dir_prefix / domain_dir / "iql_policies" / variant_name / jani_name

        # Locate the training property file
        training_property_dir = property_dir / "random_starts_20000" / jani_name
        assert training_property_dir.exists(), f"Training property directory {training_property_dir} does not exist."
        all_training_properties_files = list(training_property_dir.iterdir())
        assert len(all_training_properties_files) == 1, f"Expected one property file in {training_property_dir}, found {len(all_training_properties_files)}"
        training_property_file = all_training_properties_files[0]

        # Set up arguments
        args = {
            "model_path": condor_dir_prefix / model_file,
            "property_path": condor_dir_prefix / training_property_file,
            "start_states": condor_dir_prefix / training_property_file,
            "objective": "",
            "failure_property": "",
            "goal_reward": 0.0, # No goal reward for training risk recovery policy with IQL
            "failure_reward": -1.0,
            "unsafe_reward": -0.01,
            "batch_size": shared_args.get("batch_size", 64),
            "total_timesteps": shared_args.get("total_timesteps", 512000),
            "steps_per_epoch": shared_args.get("steps_per_epoch", 1000),
            "expectile": shared_args.get("expectile", 0.7),
            "model_save_dir": model_save_dir,
            "seed": shared_args.get("seed", 42),
        }
        # Create file arguments
        file_args = {
            "jani_model": args.get("jani_model", str(model_file)),
            "jani_property": args.get("jani_property", str(training_property_file)),
            "start_states": args.get("start_states", str(training_property_file)),
            "objective": args.get("objective", ""),
            "failure_property": args.get("failure_property", ""),
            "seed": args.get("seed", 42),
            "goal_reward": args.get("goal_reward", 0.0),
            "failure_reward": args.get("failure_reward", -1.0),
            "unsafe_reward": args.get("unsafe_reward", -0.01),
            "use_oracle": True,
            "disable_oracle_cache": args.get("disable_oracle_cache", False),
            "max_steps": args.get("max_steps", 256)
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
    parser.add_argument("--log_directory", type=str, default="./logs", help="Directory to save logs")
    parser.add_argument("--condor_dir_prefix", required=True, type=str, default="", help="Prefix path for HTCondor environment")
    parser.add_argument("--total_timesteps", type=int, default=512000, help="Total timesteps for training")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--steps_per_epoch", type=int, default=1000, help="Number of steps per epoch")
    parser.add_argument("--expectile", type=float, default=0.7, help="Expectile value for IQL")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for training")
    parser.add_argument("--output_file", type=Path, required=True, help="File to save generated configurations")
    args = parser.parse_args()

    shared_args = {
        "total_timesteps": args.total_timesteps,
        "batch_size": args.batch_size,
        "steps_per_epoch": args.steps_per_epoch,
        "expectile": args.expectile,
        "condor_dir_prefix": args.condor_dir_prefix,
        "seed": args.seed
    }

    all_configs = get_all_configs(args.root, shared_args)

    lines = []
    for benchmark_args, _ in all_configs:

        python_prefix = [
            "-m",
            "dagger.train",
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