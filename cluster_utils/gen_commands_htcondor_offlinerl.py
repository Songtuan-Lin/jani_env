import argparse

from pathlib import Path


def get_config_for_benchmark(benchmark_dir: Path, shared_args: dict) -> dict:
    all_jani_files = list(benchmark_dir.glob("*.jani"))

    assert len(all_jani_files) == 2, f"Expected exactly 2 .jani files in {benchmark_dir}, found {len(all_jani_files)}"
    model_file = benchmark_dir / "model.jani"
    property_file = benchmark_dir / "pa_model_random_starts_100000.jani"
    assert model_file.exists(), f"Expected model.jani in {benchmark_dir} but it does not exist."
    assert property_file.exists(), f"Expected pa_model_random_starts_100000.jani

    log_dir = shared_args.get("log_dir", None)
    assert log_dir is not None, "log_dir must be specified in shared_args"
    benchmark_suffix = Path(*benchmark_dir.parts[1:])
    benchmark_log_dir = Path(log_dir) / benchmark_suffix
    benchmark_log_dir.mkdir(parents=True, exist_ok=True) 

    safe_ratio = shared_args.get("safe_ratio", None)
    if safe_ratio is not None:
        dataset_path = benchmark_dir / f"sampled_trajectories_{safe_ratio}"
    else:
        dataset_path = benchmark_dir / "sampled_trajectories"
    assert dataset_path.exists(), f"Expected sampled trajectories at {dataset_path} but it does not exist."

    # Number of experiments to run for each benchmark
    num_exps = shared_args.get("num_exps", 20)

    prefix = Path(shared_args.get("condor_dir_prefix", ""))

    args_list = []
    for exp_idx in range(num_exps):
        seed = shared_args.get("seed", 42) + exp_idx # Vary seed across experiments
        exp_log_dir = benchmark_log_dir / f"seed_{seed}"
        exp_log_dir.mkdir(parents=True, exist_ok=True)

        args_vanilla = {
            "model_path": prefix / model_file,
            "property_path": prefix / property_file,
            "start_states": prefix / property_file,
            "unsafe_reward": 0.0,
            "dataset_path": prefix / dataset_path,
            "max_steps_per_episode": shared_args.get("max_steps", 256),
            "total_timesteps": shared_args.get("total_timesteps", 50000),
            "steps_per_epoch": shared_args.get("steps_per_epoch", 50),
            "batch_size": shared_args.get("batch_size", 128),
            "write_eval_results": prefix / exp_log_dir / "eval_results_vanilla.json",
            "seed": seed
        }

        args_lower_bound = {
            "model_path": prefix / model_file,
            "property_path": prefix / property_file,
            "start_states": prefix / property_file,
            "unsafe_reward": 0.0,
            "dataset_path": prefix / dataset_path,
            "max_steps_per_episode": shared_args.get("max_steps", 256),
            "total_timesteps": shared_args.get("total_timesteps", 50000),
            "steps_per_epoch": shared_args.get("steps_per_epoch", 50),
            "batch_size": shared_args.get("batch_size", 128),
            "use_lower_bound": True,
            "lower_bound_type": "action_safe",
            "write_eval_results": prefix / exp_log_dir / "eval_results_lower_bound.json",
            "seed": seed, # Vary seed across experiments
        }

        args_list.append(args_vanilla)
        args_list.append(args_lower_bound)

    return args_list


def get_all_configs(root: Path, shared_args: dict) -> list[dict]:
    list_configs = []
    for domain_dir in root.iterdir():
        if domain_dir.is_dir():
            for benchmark_dir in domain_dir.iterdir():
                if benchmark_dir.is_dir():
                    configs = get_config_for_benchmark(benchmark_dir, shared_args)
                    list_configs.extend(configs)
    return list_configs


def main():
    parser = argparse.ArgumentParser(description="Generate configuration for sampling trajectories with HTCondor.")
    parser.add_argument("--root_dir", type=str, required=True, help="Root directory containing domain subdirectories.")
    parser.add_argument("--condor_dir_prefix", required=True, type=str, default="", help="Prefix path for HTCondor environment")
    parser.add_argument("--safe_ratio", type=float, default=None, help="Safe ratio for selecting the dataset (optional).")
    parser.add_argument("--max_steps", type=int, default=256, help="Maximum steps per episode.")
    parser.add_argument("--total_timesteps", type=int, default=50000, help="Total timesteps for training")
    parser.add_argument("--steps_per_epoch", type=int, default=50, help="Steps per epoch for training.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training.")
    parser.add_argument("--num_exps", type=int, default=20, help="Number of experiments to run for each benchmark.")
    parser.add_argument("--log_dir", type=str, required=True, help="Directory to save logs.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the generated configurations as a JSON file.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    args = parser.parse_args()

    shared_args = {
        "num_episodes": args.num_episodes,
        "max_steps": args.max_steps,
        "safe_ratio": args.safe_ratio,
        "total_timesteps": args.total_timesteps,
        "condor_dir_prefix": args.condor_dir_prefix,
        "log_dir": args.log_dir,
        "steps_per_epoch": args.steps_per_epoch,
        "batch_size": args.batch_size,
        "num_exps": args.num_exps,
        "seed": args.seed
    }

    root_path = Path(args.root_dir)
    all_configs = get_all_configs(root_path, shared_args)

    # Write the configurations to the output file
    lines = []
    for config in all_configs:
        python_prefix = [
            "-m",
            "offline_rl.iql",
        ]

        line = python_prefix.copy()
        for k, v in config.items():
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