import numpy as np
import gymnasium as gym

from jani import *
from typing import Optional


class JaniEnv(gym.Env):
    def __init__(self, model_file, start_file: str = None , goal_file: str = None, safe_file: str = None, property_file: str = None):
        super().__init__()
        self._jani = JANI(model_file, start_file, goal_file, safe_file, property_file)
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
        action_mask = self.action_mask()
        return np.array(self._current_state.to_vector(), dtype=np.float32), {"action_mask": action_mask}

    def step(self, action: int) -> tuple[dict, float, bool, bool, dict]:
        if self._current_state is None:
            raise RuntimeError("Environment has not been reset. Call reset() before step().")
        
        # Convert action index to Action object
        if action < 0 or action >= len(self._jani._actions):
            raise ValueError(f"Invalid action index {action}. Must be between 0 and {len(self._jani._actions)-1}")

        # Get action masks
        action_mask = self.action_mask()

        action_obj = self._jani.get_action(action)
        next_state = self._jani.get_transition(self._current_state, action_obj)
        # Non-sparse reward works clearly better
        # we can use this as an argument for why we
        # would like to predicate whether a state
        # is safe or not
        reward = 0.0
        done = False
        if next_state is None:
            # reward = -1.0
            # done = True
            raise ValueError(f"Action {action} is not valid in the current state {self._current_state}.")
        elif self._jani.goal_reached(next_state):
            reward = 1.0
            done = True
        elif self._jani.failure_reached(next_state):
            reward = -1.0
            done = True
        self._current_state = next_state
        return np.array(self._current_state.to_vector(), dtype=np.float32) if self._current_state is not None else None, reward, done, False, {"action_mask": action_mask}

    def action_mask(self) -> np.ndarray:
        if self._current_state is None:
            return np.zeros(self.action_space.n, dtype=np.float32)

        mask = np.zeros(self.action_space.n, dtype=np.float32)
        for action in range(self.action_space.n):
            action_obj = self._jani.get_action(action)
            if self._jani.get_transition(self._current_state, action_obj) is not None:
                mask[action] = 1.0
        return mask

    def _print_obs(self):
        '''Print the current state, for debug only'''
        if self._current_state is None:
            print("Environment has not been reset.")
        else:
            print("Current state:", self._current_state.variable_info())
