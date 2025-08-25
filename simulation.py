import sys
sys.setrecursionlimit(10**6)

import numpy as np

from jani_env import *
from oracle import *
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.utils import get_action_masks


class Simulator:
    def __init__(self, env_args: dict, max_episodes: int, policy = None):
        def mask_fn(env) -> np.ndarray:
            return env.unwrapped.action_mask()

        self._env = ActionMasker(JaniEnv(**env_args), mask_fn)
        self._oracle = TarjanOracle(self._env.unwrapped.get_model())
        self._policy = policy
        self._max_episodes = max_episodes

    def run(self):
        cached = []
        
        def cache_state(state: State) -> None:
            safe_state = self._oracle.is_safe(state)
            state_vec = state.to_vector()
            state_vec.append(int(safe_state))
            cached.append(state_vec)

        for _ in range(self._max_episodes):
            obs = self._env.reset()
            cache_state(self._env.unwrapped.get_state_repr())
            done = False
            max_step = 2048
            while not done and max_step > 0:
                action_masks = get_action_masks(self._env)
                if self._policy is not None:
                    action, _state = self._policy.predict(obs, action_masks=action_masks)
                else:
                    valid_actions = np.flatnonzero(action_masks)
                    # sample one
                    action = np.random.choice(valid_actions)
                obs, reward, done, truncated, info = self._env.step(action)
                cache_state(self._env.unwrapped.get_state_repr())
                max_step -= 1

        results = np.array(cached, dtype=np.float32)
        num_columns = results.shape[-1]
        fmt = ["%.3f"] * (num_columns - 1)
        fmt.append("%d")
        np.savetxt("simulation_results.csv", results, delimiter=",", fmt=fmt)

if __name__ == "__main__":
    env_args = {
        "model_file": "examples/inverted_pendulum/inverted_pendulum.jani",
        "start_file": "examples/inverted_pendulum/start.jani",
        "goal_file": "examples/inverted_pendulum/objective.jani",
        "safe_file": "examples/inverted_pendulum/safe.jani"
    }
    simulator = Simulator(env_args, max_episodes=50)
    simulator.run()