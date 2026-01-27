"""
Utility functions for training policies with stable-baselines3.
"""

import numpy as np

from typing import Dict, Any, Optional, Tuple
from pathlib import Path

from jani.env import JANIEnv

from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecMonitor

from sb3_contrib.common.wrappers import ActionMasker


def create_env(file_args: dict, n_envs: int = 1, monitor: bool = False, time_limited: bool = True) -> JANIEnv:
    """Create JANI environment with specified parameters."""
    def make_env():
        env = JANIEnv(
            jani_model_path=file_args["jani_model"],
            jani_property_path=file_args["jani_property"],
            start_states_path=file_args["start_states"],
            objective_path=file_args["objective"],
            failure_property_path=file_args["failure_property"],
            seed=file_args["seed"],
            goal_reward=file_args["goal_reward"],
            use_oracle=file_args.get("use_oracle", False),
            failure_reward=file_args["failure_reward"],
            unsafe_reward=file_args.get("unsafe_reward", -0.01)
        ) 

        if time_limited:
            env = TimeLimit(env, max_episode_steps=file_args["max_steps"])
        # Apply action masking
        env = ActionMasker(env, mask_fn)
        if monitor:
            env = Monitor(env)
        return env
    
    env = None
    if n_envs == 1:
        env = make_env()
    else:
        env = make_vec_env(make_env, n_envs=n_envs)
        if monitor:
            env = VecMonitor(env)

    return env

def create_eval_file_args(file_args: Dict[str, Any]) -> Dict[str, Any]:
    """Create file arguments for evaluation environment."""
    eval_file_args = file_args.copy()
    # Modify any parameters specific to evaluation if needed
    eval_file_args["seed"] += 1000  # offset seed for evaluation
    eval_file_args["use_oracle"] = False  # disable oracle during evaluation
    return eval_file_args

def create_safety_eval_file_args(file_args: Dict[str, Any], args: Dict[str, Any], use_oracle: bool = True) -> Dict[str, Any]:
    """Create file arguments for safety evaluation environment."""
    safety_eval_file_args = file_args.copy()
    # Modify any parameters specific to safety evaluation if needed
    eval_start_states = args.get("eval_start_states", "")
    if eval_start_states == "":
        eval_start_states = file_args.get("start_states", "")
    safety_eval_file_args["start_states"] = eval_start_states # use different start states for safety evaluation
    safety_eval_file_args["seed"] += 2000  # offset seed for safety evaluation
    safety_eval_file_args["use_oracle"] = use_oracle  # enable or disable oracle during safety evaluation
    return safety_eval_file_args

def mask_fn(env) -> np.ndarray:
    """Action masking function for the environment."""
    return env.unwrapped.action_mask()


def get_configs_for_benchmark(variant_dir: str, domain_dir: str, shared_args: dict[str, Any]) -> list[dict[str, Any]]:
    """Get file arguments for benchmark environment."""
    variant_dir = Path(variant_dir)
    domain_dir = Path(domain_dir)
    variant_name = variant_dir.name
    domain_name = domain_dir.name

    list_configs = []
    for model_file in variant_dir.iterdir():
        jani_name = model_file.name.replace(".jani", "")
        if variant_name == "models":
            property_dir = domain_dir / "additional_properties"
        else:
            property_dir = domain_dir / "additional_properties" / variant_name

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
            "policy_path": variant_dir / jani_name / "policy.pth",
            "goal_reward": 1.0,
            "failure_reward": -1.0,
            "unsafe_reward": -0.01,
            "num_init_states": 20000,
            "num_iterations": 50,
            "max_steps": 1024,
            "use_multiprocessors": True,
            "empty_buffer": True,
            "wandb_project": f"{jani_name}",
            "experiment_name": f"dagger_{variant_name}" if variant_name != "models" else "dagger",
            "log_directory": Path(shared_args.get("log_directory", "./logs")) / domain_name / variant_name / jani_name if variant_name != "models" else Path(shared_args.get("log_directory", "./logs")) / domain_name / jani_name,
            "model_save_dir": Path(shared_args.get("log_directory", "./logs")) / domain_name / variant_name / jani_name / "models" if variant_name != "models" else Path(shared_args.get("log_directory", "./logs")) / domain_name / jani_name / "models",
            "disable_wandb": False,
            "seed": seed,
        }
        # Create file arguments
        file_args = {
            "jani_model": args.get("jani_model", str(model_file)),
            "jani_property": args.get("jani_property", str(training_property_file)),
            "start_states": args.get("start_states", str(training_property_file)),
            "objective": args.get("objective", ""),
            "failure_property": args.get("failure_property", ""),
            "seed": args.get("seed", 42),
            "goal_reward": args.get("goal_reward", 1.0),
            "failure_reward": args.get("failure_reward", -1.0),
            "unsafe_reward": args.get("unsafe_reward", -0.01),
            "use_oracle": True,
            "max_steps": args.get("max_steps", 1024)
        }
        list_configs.append((args, file_args))
    return list_configs


def get_configs_for_domain(domain_dir: str, shared_args: dict[str, Any]) -> list[dict[str, Any]]:
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


def get_all_configs(root_dir: str, shared_args: dict[str, Any]) -> list[dict[str, Any]]:
    """Get file arguments for all domains in the root directory."""
    root_dir = Path(root_dir)
    list_configs = []
    for domain_dir in root_dir.iterdir():
        if domain_dir.is_dir():
            domain_configs = get_configs_for_domain(str(domain_dir), shared_args)
            list_configs.extend(domain_configs)
    return list_configs