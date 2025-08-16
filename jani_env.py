import numpy as np
import gymnasium as gym

from jani import *
from typing import Optional


class JaniEnv(gym.Env):
    def __init__(self, model_file, start_file, goal_file, safe_file):
        super().__init__()
        self._jani = JANI(model_file, start_file, goal_file, safe_file)
        # Define action and observation space
        self.action_space = gym.spaces.Discrete(self._jani.get_action_count())
        lower_bounds = []
        upper_bounds = []
        for var in self._jani.get_constants_variables():
            lower_bounds.append(var.lower_bound if var.lower_bound is not None else -np.inf)
            upper_bounds.append(var.upper_bound if var.upper_bound is not None else np.inf)
        self.observation_space = gym.spaces.Box(low=np.array(lower_bounds), high=np.array(upper_bounds), dtype=np.float32)
        # Initialize current state to None
        self._current_state = None

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[dict, dict]:
        super().reset(seed=seed)
        self._current_state = self._jani.reset()
        return np.array(self._current_state.to_vector(), dtype=np.float32), {}

    def step(self, action: int) -> tuple[dict, float, bool, bool, dict]:
        if self._current_state is None:
            raise RuntimeError("Environment has not been reset. Call reset() before step().")
        
        # Convert action index to Action object
        if action < 0 or action >= len(self._jani._actions):
            raise ValueError(f"Invalid action index {action}. Must be between 0 and {len(self._jani._actions)-1}")

        action_obj = self._jani.get_action(action)
        next_state = self._jani.get_transition(self._current_state, action_obj)
        reward = 0.0
        done = False
        if next_state is None:
            reward = 0.0
            done = True
        elif self._jani.goal_reached(next_state):
            reward = 1.0
            done = True
        elif self._jani.failure_reached(next_state):
            reward = -1.0
            done = True
        self._current_state = next_state
        return np.array(self._current_state.to_vector(), dtype=np.float32) if self._current_state is not None else None, reward, done, False, {}
