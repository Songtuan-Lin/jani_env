import sys
import numpy as np
import gymnasium as gym

from pathlib import Path
from typing import Optional

# Dynamically add the JANI engine binding directory to sys.path
current_dir = Path(__file__).resolve().parent
binding_dir = current_dir / "engine" / "build"
sys.path.append(str(binding_dir))

from backend import JANIEngine, TarjanOracle


class JANIEnv(gym.Env):
    def __init__(self, 
                 jani_model_path: str, 
                 jani_property_path: str = "",
                 start_states_path: str = "",
                 objective_path: str = "",
                 failure_property_path: str = "",
                 seed: int = 42,
                 goal_reward: float = 1.0,
                 failure_reward: float = -1.0,
                 use_oracle: bool = False,
                 unsafe_reward: float = -0.01) -> None:
        super().__init__()
        # print(f"DEBUG: Initializing JANIEnv with model: {jani_model_path}, property: {jani_property_path}, start states: {start_states_path}, objective: {objective_path}, failure property: {failure_property_path}, seed: {seed}")
        self._engine = JANIEngine(jani_model_path, 
                                  jani_property_path, 
                                  start_states_path, 
                                  objective_path, 
                                  failure_property_path, 
                                  seed)
        self._goal_reward: float = goal_reward
        self._failure_reward: float = failure_reward
        self._oracle: Optional[TarjanOracle] = None
        if use_oracle:
            self._oracle = TarjanOracle(self._engine)
        self._unsafe_reward: Optional[float] = None
        if self._oracle is not None:
            self._unsafe_reward = unsafe_reward
        # Define action and observation space
        self.action_space = gym.spaces.Discrete(self._engine.get_num_actions())
        lower_bounds = self._engine.get_lower_bounds()
        upper_bounds = self._engine.get_upper_bounds()
        self.observation_space = gym.spaces.Box(low=np.array(lower_bounds), 
                                                high=np.array(upper_bounds), 
                                                dtype=np.float32)
        # Initialize reset flag
        self._reseted = False

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[dict, dict]:
        super().reset(seed=seed)
        if "idx" in options:
            state_vec = self._engine.reset_with_index(options["idx"])
        else:
            state_vec = self._engine.reset()
        self._reseted = True
        assert not self._engine.reach_goal_current(), "Initial state should not be a goal state."
        return np.array(state_vec, dtype=np.float32), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        if not self._reseted:
            raise RuntimeError("Environment must be reset before stepping.")
        next_state_vec = self._engine.step(action) # The current state should be automatically updated in the engine
        if self._oracle is not None:
            assert self._unsafe_reward is not None
            is_next_state_safe = self._oracle.is_engine_state_safe()
        
        # Compute reward and done flag
        reward = None
        done = None
        if self._engine.reach_goal_current():
            reward = self._goal_reward
            done = True
        elif self._engine.reach_failure_current():
            reward = self._failure_reward
            done = True
        elif np.sum(self.action_mask()) == 0.0:
            reward = 0.0
            done = True
        else:
            if self._oracle is not None and not is_next_state_safe:
                reward = self._unsafe_reward
            else:
                reward = 0.0
            done = False

        return np.array(next_state_vec, dtype=np.float32), reward, done, False, {}

    def action_mask(self) -> np.ndarray:
        if not self._reseted:
            raise RuntimeError("Environment must be reset before getting action mask.")
        mask = self._engine.get_current_action_mask()
        return np.array(mask, dtype=np.float32)

    def action_mask_for_obs(self, obs: np.ndarray):
        # print(f"DEBUG: Getting action mask for obs: {obs}")
        # print(f"DEBUG: Obs shape: {obs.shape}, Obs dtype: {obs.dtype}")
        return [self._engine.get_action_mask_for_obs(single_obs.tolist()) for single_obs in obs]
    
    def get_init_state_pool_size(self) -> int:
        return self._engine.get_init_state_pool_size()
    
    def get_unsafe_reward(self) -> Optional[float]:
        assert self._unsafe_reward is not None, "Unsafe reward is not defined."
        return self._unsafe_reward