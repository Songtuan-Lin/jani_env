import numpy as np
import gymnasium as gym

from jani import *
from typing import Optional


class JaniEnv(gym.Env):
    def __init__(self, model_file, start_file):
        super().__init__()
        self._jani = JANI(model_file, start_file)
        # Define action and observation space
        self.action_space = gym.spaces.Discrete(self._jani.get_action_count())
        observation_dict = {}
        for var in self._jani.get_constants_variables():
            lower_bound = var.lower_bound if var.lower_bound is not None else -np.inf
            upper_bound = var.upper_bound if var.upper_bound is not None else np.inf
            observation_dict[var.name] = gym.spaces.Box(low=lower_bound, high=upper_bound, dtype=np.float32)
        self.observation_space = gym.spaces.Dict(observation_dict)
        # Initialize current state to None
        self._current_state = None

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[dict, dict]:
        super().reset(seed=seed)
        self._current_state = self._jani.reset()
        return self._current_state, {}

    def step(self, action: int) -> tuple[dict, float, bool, bool, dict]:
        if self._current_state is None:
            raise RuntimeError("Environment has not been reset. Call reset() before step().")
        next_state = self._jani.get_transition(self._current_state, action)
        reward = 0.0
        done = False
        if next_state is None:
            reward = -1.0
            done = True
        elif self._jani.goal_reached(next_state):
            reward = 1.0
            done = True
        elif self._jani.failure_reached(next_state):
            reward = -1.0
            done = True
        self._current_state = next_state
        return self._current_state, reward, done, False, {}
