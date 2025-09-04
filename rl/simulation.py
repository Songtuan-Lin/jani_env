import sys
sys.setrecursionlimit(10**6)

import argparse
import numpy as np
from pathlib import Path

from jani.environment import JaniEnv
from jani.oracle import TarjanOracle
from jani.core import State
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib import MaskablePPO


class Simulator:
    def __init__(self, env_args: dict, max_episodes: int, output_path: str = None, policy = None, n_steps: int = 2048, run_oracle: bool = True, cache_states: bool = True):
        def mask_fn(env) -> np.ndarray:
            return env.unwrapped.action_mask()

        self._env = ActionMasker(JaniEnv(**env_args), mask_fn)
        self._oracle = TarjanOracle(self._env.unwrapped.get_model())
        self._policy = policy
        self._max_episodes = max_episodes
        self._output_path = output_path
        self._n_steps = n_steps
        self._run_oracle = run_oracle
        self._cache_states = cache_states

    def run(self) -> dict:
        cached_states = []
        num_unsafe = 0 if self._run_oracle else None

        def cache_state(state: State) -> None:
            if not self._cache_states:
                return
            nonlocal cached_states, num_unsafe
            state_vec = state.to_vector()
            if self._run_oracle:
                state_safe = self._oracle.is_safe(state)
                if not state_safe:
                    num_unsafe += 1
                state_vec.append(int(state_safe))
            cached_states.append(state_vec)

        rewards = []
        for _ in range(self._max_episodes):
            obs, _ = self._env.reset()
            cache_state(self._env.unwrapped.get_state_repr())
            episode_reward = 0.0
            done = False
            remaining_steps = self._n_steps
            while not done and remaining_steps > 0:
                action_masks = get_action_masks(self._env)
                if self._policy is not None:
                    action, _state = self._policy.predict(obs, action_masks=action_masks)
                else:
                    valid_actions = np.flatnonzero(action_masks)
                    # sample one
                    action = np.random.choice(valid_actions)
                obs, reward, done, truncated, info = self._env.step(action)
                episode_reward += reward
                cache_state(self._env.unwrapped.get_state_repr())
                remaining_steps -= 1
            rewards.append(episode_reward)

        if self._output_path:
            cached_states = np.array(cached_states, dtype=np.float32)
            num_columns = cached_states.shape[-1]
            fmt = ["%.3f"] * (num_columns - 1)
            fmt.append("%d")
            np.savetxt(self._output_path, cached_states, delimiter=",", fmt=fmt)
        
        mean_reward = np.mean(rewards)
        results = {"mean_reward": mean_reward, "num_episodes": self._max_episodes}
        if self._run_oracle:
            results["num_unsafe"] = num_unsafe
        return results

def load_policy(policy_file_path: str):
    """Load a trained MaskablePPO policy from file."""
    if not Path(policy_file_path).exists():
        raise FileNotFoundError(f"Policy file not found: {policy_file_path}")
    
    try:
        policy = MaskablePPO.load(policy_file_path)
        print(f"Successfully loaded policy from: {policy_file_path}")
        return policy
    except Exception as e:
        raise RuntimeError(f"Failed to load policy from {policy_file_path}: {e}")


def parse_args():
    parser = argparse.ArgumentParser(description="Simulate policy in JANI environment and record safety results")
    parser.add_argument("--model_file", required=True, help="Path to the JANI model file")
    parser.add_argument("--start_file", help="Path to the start condition file")
    parser.add_argument("--goal_file", help="Path to the goal file")
    parser.add_argument("--safe_file", help="Path to the safe file")
    parser.add_argument("--property_file", help="Path to the property file")
    parser.add_argument("--policy_file", help="Path to the policy file")
    parser.add_argument("--random_init", action="store_true", help="Use random initial state generator")
    parser.add_argument("--output_path", default="simulation_results.csv", help="Path to save the CSV results file")
    parser.add_argument("--max_episodes", type=int, default=50, help="Maximum number of episodes to simulate")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Validation logic: if property_file is not provided, then start_file, goal_file, and safe_file must all be provided
    if args.property_file is None:
        if args.start_file is None or args.goal_file is None or args.safe_file is None:
            parser = argparse.ArgumentParser()
            parser.error("When --property_file is not provided, --start_file, --goal_file, and --safe_file must all be specified")
    
    env_args = {
        "model_file": args.model_file,
        "start_file": args.start_file,
        "goal_file": args.goal_file,
        "safe_file": args.safe_file,
        "property_file": args.property_file,
        "random_init": args.random_init
    }
    
    # Load policy if specified
    policy = None
    if args.policy_file:
        try:
            policy = load_policy(args.policy_file)
        except (FileNotFoundError, RuntimeError) as e:
            print(f"Error: {e}")
            sys.exit(1)
    else:
        print("No policy file specified. Using random policy.")
    
    simulator = Simulator(env_args, max_episodes=args.max_episodes, output_path=args.output_path, policy=policy)
    simulator.run()