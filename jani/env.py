import sys
import numpy as np
import gymnasium as gym

from pathlib import Path
from typing import Optional

# Dynamically add the JANI engine binding directory to sys.path
current_dir = Path(__file__).resolve().parent
binding_dir = current_dir / "engine" / "build"
sys.path.append(str(binding_dir))

from backend import JANIEngine


class JANIEnv(gym.Env):
    def __init__(self, 
                 jani_model_path: str, 
                 jani_property_path: str = "",
                 start_states_path: str = "",
                 objective_path: str = "",
                 failure_property_path: str = "",
                 seed: int = 42):
        super().__init__()
        self._engine = JANIEngine(jani_model_path, 
                                  jani_property_path, 
                                  start_states_path, 
                                  objective_path, 
                                  failure_property_path, 
                                  seed)
        # Define action and observation space
        self.action_space = gym.spaces.Discrete(self._engine.get_action_count())
        lower_bounds = self._engine.get_lower_bounds()
        upper_bounds = self._engine.get_upper_bounds()
        self.observation_space = gym.spaces.Box(low=np.array(lower_bounds), 
                                                high=np.array(upper_bounds), 
                                                dtype=np.float32)
        # Initialize current state to None
        self._current_state = None

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[dict, dict]:
        super().reset(seed=seed)
        state_vec = self._engine.reset()
        self._current_state = np.array(state_vec, dtype=np.float32)
        return np.array(state_vec, dtype=np.float32), {}