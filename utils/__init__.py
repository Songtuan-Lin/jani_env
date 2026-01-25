"""
Utility functions for training policies with stable-baselines3.
"""

import numpy as np

from typing import Dict, Any, Optional, Tuple

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