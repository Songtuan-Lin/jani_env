import argparse

from pathlib import Path


def get_config_for_benchmark(benchmark_dir: Path, shared_args: dict) -> dict:
    all_jani_files = list(benchmark_dir.glob("*.jani"))

    assert len(all_jani_files) == 2, f"Expected exactly 2 .jani files in {benchmark_dir}, found {len(all_jani_files)}"
    model_file = benchmark_dir / "model.jani"
    property_file = benchmark_dir / "pa_model_random_starts_100000.jani"
    assert model_file.exists(), f"Expected model.jani in {benchmark_dir} but it does not exist."
    assert property_file.exists(), f"Expected pa_model_random_starts_100000.jani

    prefix = Path(shared_args.get("condor_dir_prefix", ""))
    save_trajs_dir = prefix / benchmark_dir / "sampled_trajectories"

    args = {
        "model_path": prefix / model_file,
        "property_path": prefix / property_file,
        "start_states": prefix / property_file,
        "num_episodes": shared_args.get("num_episodes", 10000),
        "max_steps": shared_args.get("max_steps", 256),
        "output_dir": save_trajs_dir,
        "target_safe_ratio": shared_args.get("target_safe_ratio", None),
        "seed": shared_args.get("seed", 42),
        "reduced_memory_mode": True
    }

    return args


def get_all_configs(root: Path, shared_args: dict) -> list[dict]:
    list_configs = []
    for domain_dir in root.iterdir():
        if domain_dir.is_dir():
            for benchmark_dir in domain_dir.iterdir():
                if benchmark_dir.is_dir():
                    config = get_config_for_benchmark(benchmark_dir, shared_args)
                    list_configs.append(config)
    return list_configs


def main():
    parser = argparse.ArgumentParser(description="Generate configuration for sampling trajectories with HTCondor.")
    parser.add_argument("--root_dir", type=str, required=True, help="Root directory containing domain subdirectories.")
    parser.add_argument("--condor_dir_prefix", required=True, type=str, default="", help="Prefix path for HTCondor environment")
    parser.add_argument("--num_episodes", type=int, default=10000, help="Number of episodes to sample per benchmark.")
    parser.add_argument("--max_steps", type=int, default=256, help="Maximum steps per episode.")
    parser.add_argument("--target_safe_ratio", type=float, default=None, help="Target safe ratio for sampling (optional).")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the generated configurations as a JSON file.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    args = parser.parse_args()

    shared_args = {
        "num_episodes": args.num_episodes,
        "max_steps": args.max_steps,
        "target_safe_ratio": args.target_safe_ratio,
        "condor_dir_prefix": args.condor_dir_prefix,
        "seed": args.seed
    }

    root_path = Path(args.root_dir)
    all_configs = get_all_configs(root_path, shared_args)

    # Write the configurations to the output file
    lines = []
    for config in all_configs:
        python_prefix = [
            "-m",
            "offline_rl.sample",
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