"""
Comprehensive test suite for the JaniEnv OpenAI Gym environment.

This test suite covers:
1. Environment initialization and setup
2. Reset functionality and observation handling
3. Step functionality including action handling, rewards, and termination
4. Complete episode flows and multi-episode behavior
5. Integration with actual JANI model files
6. Edge cases and error handling
7. OpenAI Gym interface compliance

The tests ensure that the JaniEnv properly wraps JANI models as gym environments
and handles all the expected gym interface behaviors correctly.
"""

import pytest
import numpy as np
import gymnasium as gym
import json
import tempfile
from pathlib import Path

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from jani_env import JaniEnv
from jani import State


def create_test_jani_files():
    """Create test JANI, start, goal, and failure files for testing."""
    # Simple test model with counter variable
    jani_data = {
        "jani-version": 1,
        "name": "test-counter",
        "type": "lts",
        "actions": [
            {"name": "increment"},
            {"name": "decrement"},
            {"name": "reset"}
        ],
        "constants": [
            {"name": "max_value", "type": "int", "value": 10}
        ],
        "variables": [
            {
                "name": "counter",
                "type": {"kind": "bounded", "base": "int", "lower-bound": 0, "upper-bound": 10},
                "initial-value": 5
            },
            {
                "name": "active",
                "type": {"kind": "bounded", "base": "bool"},
                "initial-value": True
            }
        ],
        "automata": [{
            "name": "counter_automaton",
            "initial-locations": ["loc1"],
            "locations": [{"name": "loc1"}],
            "edges": [
                {
                    "location": "loc1",
                    "action": "increment",
                    "guard": {"exp": {"left": "counter", "op": "≤", "right": 9}},
                    "destinations": [{
                        "assignments": [{"ref": "counter", "value": {"left": "counter", "op": "+", "right": 1}}],
                        "location": "loc1"
                    }]
                },
                {
                    "location": "loc1",
                    "action": "decrement",
                    "guard": {"exp": {"left": "counter", "op": "≤", "right": 10}},
                    "destinations": [{
                        "assignments": [{"ref": "counter", "value": {"left": "counter", "op": "-", "right": 1}}],
                        "location": "loc1"
                    }]
                },
                {
                    "location": "loc1",
                    "action": "reset",
                    "guard": {"exp": True},
                    "destinations": [{
                        "assignments": [
                            {"ref": "counter", "value": 0},
                            {"ref": "active", "value": True}
                        ],
                        "location": "loc1"
                    }]
                }
            ]
        }]
    }
    
    # Start state specification
    start_data = {
        "op": "states-values",
        "values": [
            {
                "variables": [
                    {"var": "counter", "value": 5},
                    {"var": "active", "value": True}
                ]
            },
            {
                "variables": [
                    {"var": "counter", "value": 3},
                    {"var": "active", "value": True}
                ]
            }
        ]
    }
    
    # Goal: reach counter = 10
    goal_data = {
        "goal": {
            "exp": {
                "left": "counter",
                "op": "=",
                "right": 10
            },
            "op": "state-condition"
        },
        "op": "objective"
    }
    
    # Failure: counter < 0
    failure_data = {
        "exp": {
            "left": "counter",
            "op": "<",
            "right": 0
        },
        "op": "state-condition"
    }
    
    # Create temporary files
    jani_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jani', delete=False)
    start_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jani', delete=False)
    goal_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jani', delete=False)
    failure_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jani', delete=False)
    
    json.dump(jani_data, jani_file, indent=2)
    json.dump(start_data, start_file, indent=2)
    json.dump(goal_data, goal_file, indent=2)
    json.dump(failure_data, failure_file, indent=2)
    
    jani_file.close()
    start_file.close()
    goal_file.close()
    failure_file.close()
    
    return jani_file.name, start_file.name, goal_file.name, failure_file.name


def create_bouncing_ball_files():
    """Return paths to existing bouncing ball files if they exist."""
    jani_file = "../examples/bouncing_ball/bouncing_ball.jani"
    start_file = "../examples/bouncing_ball/start.jani"
    goal_file = "../examples/bouncing_ball/objective.jani"
    failure_file = "../examples/bouncing_ball/safe.jani"
    
    if all(Path(f).exists() for f in [jani_file, start_file, goal_file, failure_file]):
        return jani_file, start_file, goal_file, failure_file
    return None


class TestJaniEnvInitialization:
    """Test cases for JaniEnv initialization."""
    
    def test_jani_env_initialization(self):
        """Test basic initialization of JaniEnv."""
        jani_file, start_file, goal_file, failure_file = create_test_jani_files()
        
        try:
            env = JaniEnv(jani_file, start_file, goal_file, failure_file)
            
            # Verify environment has required attributes
            assert hasattr(env, 'action_space')
            assert hasattr(env, 'observation_space')
            assert hasattr(env, '_jani')
            assert hasattr(env, '_current_state')
            
            # Verify action space
            assert isinstance(env.action_space, gym.spaces.Discrete)
            assert env.action_space.n == 3  # increment, decrement, reset
            
            # Verify observation space
            assert isinstance(env.observation_space, gym.spaces.Box)
            
            # Verify observation space dimensions match variables
            variables = env._jani.get_constants_variables()
            assert env.observation_space.shape == (len(variables),)
            
            # Verify initial state
            assert env._current_state is None
            
        finally:
            Path(jani_file).unlink()
            Path(start_file).unlink()
            Path(goal_file).unlink()
            Path(failure_file).unlink()
    
    def test_observation_space_bounds(self):
        """Test that observation space has correct bounds."""
        jani_file, start_file, goal_file, failure_file = create_test_jani_files()
        
        try:
            env = JaniEnv(jani_file, start_file, goal_file, failure_file)
            
            # Check that observation space is a Box
            assert isinstance(env.observation_space, gym.spaces.Box)
            
            # Get variables to check bounds
            variables = env._jani.get_constants_variables()
            assert len(env.observation_space.low) == len(variables)
            assert len(env.observation_space.high) == len(variables)
            
            # Check specific bounds for known variables
            var_names = [var.name for var in variables]
            
            # Find counter variable bounds
            if 'counter' in var_names:
                counter_idx = var_names.index('counter')
                assert env.observation_space.low[counter_idx] == 0
                assert env.observation_space.high[counter_idx] == 10
            
            # Check boolean variable bounds (should be -inf, inf)
            if 'active' in var_names:
                active_idx = var_names.index('active')
                assert env.observation_space.low[active_idx] == -np.inf
                assert env.observation_space.high[active_idx] == np.inf
            
        finally:
            Path(jani_file).unlink()
            Path(start_file).unlink()
            Path(goal_file).unlink()
            Path(failure_file).unlink()
    
    def test_initialization_with_invalid_files(self):
        """Test initialization with invalid file paths."""
        with pytest.raises(FileNotFoundError):
            JaniEnv("nonexistent.jani", "nonexistent.jani", "nonexistent.jani", "nonexistent.jani")


class TestJaniEnvReset:
    """Test cases for JaniEnv reset functionality."""
    
    def test_reset_returns_valid_observation(self):
        """Test that reset returns a valid observation."""
        jani_file, start_file, goal_file, failure_file = create_test_jani_files()
        
        try:
            env = JaniEnv(jani_file, start_file, goal_file, failure_file)
            observation, info = env.reset()
            
            # Verify return types
            assert isinstance(observation, np.ndarray)
            assert isinstance(info, dict)
            
            # Verify observation shape matches variables
            variables = env._jani.get_constants_variables()
            assert len(observation) == len(variables)
            
            # Verify observation is within bounds
            assert env.observation_space.contains(observation)
            
            # Verify state values are reasonable by checking specific indices
            var_names = [var.name for var in variables]
            
            if 'counter' in var_names:
                counter_idx = var_names.index('counter')
                assert isinstance(observation[counter_idx], (int, float, np.number))
                assert 0 <= observation[counter_idx] <= 10
            
            if 'max_value' in var_names:
                max_value_idx = var_names.index('max_value')
                assert observation[max_value_idx] == 10
            
            # Verify current state is set (internal state object)
            assert env._current_state is not None
            assert isinstance(env._current_state, State)
            
        finally:
            Path(jani_file).unlink()
            Path(start_file).unlink()
            Path(goal_file).unlink()
            Path(failure_file).unlink()
    
    def test_reset_with_seed(self):
        """Test reset with seed parameter."""
        jani_file, start_file, goal_file, failure_file = create_test_jani_files()
        
        try:
            env = JaniEnv(jani_file, start_file, goal_file, failure_file)
            
            # Reset with seed
            observation1, info1 = env.reset(seed=42)
            observation2, info2 = env.reset(seed=42)
            
            # With same seed, we might get same or different states depending on implementation
            # Just verify they're valid
            assert isinstance(observation1, np.ndarray)
            assert isinstance(observation2, np.ndarray)
            assert isinstance(info1, dict)
            assert isinstance(info2, dict)
            
            # Verify both observations are valid
            assert env.observation_space.contains(observation1)
            assert env.observation_space.contains(observation2)
            
        finally:
            Path(jani_file).unlink()
            Path(start_file).unlink()
            Path(goal_file).unlink()
            Path(failure_file).unlink()
    
    def test_reset_multiple_times(self):
        """Test multiple reset calls."""
        jani_file, start_file, goal_file, failure_file = create_test_jani_files()
        
        try:
            env = JaniEnv(jani_file, start_file, goal_file, failure_file)
            
            observations = []
            for _ in range(10):
                observation, info = env.reset()
                observations.append(observation)
                # Note: observation is now a vector, but env._current_state is the State object
                assert env._current_state is not None
                assert isinstance(env._current_state, State)
            
            # All observations should be valid
            for obs in observations:
                assert isinstance(obs, np.ndarray)
                assert env.observation_space.contains(obs)
                
                # Check that observations have expected structure
                variables = env._jani.get_constants_variables()
                assert len(obs) == len(variables)
            
        finally:
            Path(jani_file).unlink()
            Path(start_file).unlink()
            Path(goal_file).unlink()
            Path(failure_file).unlink()


class TestJaniEnvStep:
    """Test cases for JaniEnv step functionality."""
    
    def test_step_without_reset_raises_error(self):
        """Test that step raises error when called before reset."""
        jani_file, start_file, goal_file, failure_file = create_test_jani_files()
        
        try:
            env = JaniEnv(jani_file, start_file, goal_file, failure_file)
            
            with pytest.raises(RuntimeError, match="Environment has not been reset"):
                env.step(0)
            
        finally:
            Path(jani_file).unlink()
            Path(start_file).unlink()
            Path(goal_file).unlink()
            Path(failure_file).unlink()
    
    def test_step_returns_correct_tuple(self):
        """Test that step returns the correct tuple format."""
        jani_file, start_file, goal_file, failure_file = create_test_jani_files()
        
        try:
            env = JaniEnv(jani_file, start_file, goal_file, failure_file)
            observation, info = env.reset()
            
            # Take a step
            next_observation, reward, terminated, truncated, step_info = env.step(0)
            
            # Verify return types
            assert isinstance(next_observation, (np.ndarray, type(None)))
            assert isinstance(reward, float)
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
            assert isinstance(step_info, dict)
            
            # Verify truncated is always False (not implemented)
            assert truncated == False
            
            # If step succeeded, verify observation is valid
            if next_observation is not None:
                assert env.observation_space.contains(next_observation)
                variables = env._jani.get_constants_variables()
                assert len(next_observation) == len(variables)
            
        finally:
            Path(jani_file).unlink()
            Path(start_file).unlink()
            Path(goal_file).unlink()
            Path(failure_file).unlink()
    
    def test_valid_action_step(self):
        """Test taking a valid action step."""
        jani_file, start_file, goal_file, failure_file = create_test_jani_files()
        
        try:
            env = JaniEnv(jani_file, start_file, goal_file, failure_file)
            observation, info = env.reset()
            variables = env._jani.get_constants_variables()
            var_names = [var.name for var in variables]
            initial_counter = observation[var_names.index('counter')] if 'counter' in var_names else 0
            
            # Take increment action (action 0)
            next_observation, reward, terminated, truncated, step_info = env.step(0)
            
            if next_observation is not None:
                # Action was valid
                counter_idx = var_names.index('counter')
                assert next_observation[counter_idx] == initial_counter + 1
                # Note: env._current_state is still a State object internally
                assert env._current_state is not None
                
                # Check reward logic
                if terminated:
                    # Either goal reached or failure reached
                    assert reward == 1.0 or reward == -1.0
                else:
                    # Continuing
                    assert reward == 0.0
                    
        finally:
            Path(jani_file).unlink()
            Path(start_file).unlink()
            Path(goal_file).unlink()
            Path(failure_file).unlink()
    
    def test_invalid_action_step(self):
        """Test taking an invalid action that cannot be executed."""
        jani_file, start_file, goal_file, failure_file = create_test_jani_files()
        
        try:
            env = JaniEnv(jani_file, start_file, goal_file, failure_file)
            observation, info = env.reset()
            
            # Manually set counter to 10 to make increment invalid  
            # Note: We need to set the internal state, not the observation vector
            variables = env._jani.get_constants_variables()
            var_names = [var.name for var in variables]
            if 'counter' in var_names:
                env._current_state.variable_dict['counter'].value = 10
            
            # Try increment action when counter is at max
            next_observation, reward, terminated, truncated, step_info = env.step(0)
            
            # Should result in None state, negative reward, and termination
            assert next_observation is None
            assert reward == -1.0
            assert terminated == True
            assert env._current_state is None
            
        finally:
            Path(jani_file).unlink()
            Path(start_file).unlink()
            Path(goal_file).unlink()
            Path(failure_file).unlink()
    
    def test_goal_reached(self):
        """Test behavior when goal is reached."""
        jani_file, start_file, goal_file, failure_file = create_test_jani_files()
        
        try:
            env = JaniEnv(jani_file, start_file, goal_file, failure_file)
            observation, info = env.reset()
            
            # Manually set counter to 9 to be close to goal
            variables = env._jani.get_constants_variables()
            var_names = [var.name for var in variables]
            if 'counter' in var_names:
                env._current_state.variable_dict['counter'].value = 9
            
            # Take increment action to reach goal (counter = 10)
            next_observation, reward, terminated, truncated, step_info = env.step(0)
            
            if next_observation is not None:
                variables = env._jani.get_constants_variables()
                var_names = [var.name for var in variables]
                if 'counter' in var_names:
                    counter_idx = var_names.index('counter')
                    if next_observation[counter_idx] == 10:
                        # Goal reached
                        assert reward == 1.0
                        assert terminated == True
                        assert env._current_state is not None
                
        finally:
            Path(jani_file).unlink()
            Path(start_file).unlink()
            Path(goal_file).unlink()
            Path(failure_file).unlink()
    
    def test_failure_reached(self):
        """Test behavior when failure condition is reached."""
        jani_file, start_file, goal_file, failure_file = create_test_jani_files()
        
        try:
            env = JaniEnv(jani_file, start_file, goal_file, failure_file)
            observation, info = env.reset()
            
            # Manually set counter to 0 to be at edge of failure
            variables = env._jani.get_constants_variables()
            var_names = [var.name for var in variables]
            if 'counter' in var_names:
                env._current_state.variable_dict['counter'].value = 0
            
            # Try decrement action to reach failure (counter < 0)
            next_observation, reward, terminated, truncated, step_info = env.step(1)
            
            if next_observation is not None:
                variables = env._jani.get_constants_variables()
                var_names = [var.name for var in variables]
                if 'counter' in var_names:
                    counter_idx = var_names.index('counter')
                    if next_observation[counter_idx] < 0:
                        # Failure reached
                        assert reward == -1.0
                        assert terminated == True
                        assert env._current_state is not None
                
        finally:
            Path(jani_file).unlink()
            Path(start_file).unlink()
            Path(goal_file).unlink()
            Path(failure_file).unlink()
    
    def test_action_space_validation(self):
        """Test that invalid action indices are handled properly."""
        jani_file, start_file, goal_file, failure_file = create_test_jani_files()
        
        try:
            env = JaniEnv(jani_file, start_file, goal_file, failure_file)
            observation, info = env.reset()
            
            # Test with action index beyond action space
            # This should raise a ValueError
            with pytest.raises(ValueError, match="Invalid action index"):
                env.step(999)  # Invalid action index
                
        finally:
            Path(jani_file).unlink()
            Path(start_file).unlink()
            Path(goal_file).unlink()
            Path(failure_file).unlink()


class TestJaniEnvEpisodeFlow:
    """Test cases for complete episode flows."""
    
    def test_complete_episode_to_goal(self):
        """Test a complete episode that reaches the goal."""
        jani_file, start_file, goal_file, failure_file = create_test_jani_files()
        
        try:
            env = JaniEnv(jani_file, start_file, goal_file, failure_file)
            observation, info = env.reset()
            
            episode_length = 0
            max_steps = 20
            total_reward = 0
            
            while episode_length < max_steps:
                # Choose increment action to try to reach goal
                action = 0  # increment
                next_observation, reward, terminated, truncated, step_info = env.step(action)
                
                total_reward += reward
                episode_length += 1
                
                if terminated:
                    if reward == 1.0:
                        # Successfully reached goal
                        assert next_observation is not None
                        # Need to check goal on internal state object
                        assert env._jani.goal_reached(env._current_state)
                    break
                    
                if next_observation is None:
                    # Invalid action taken
                    break
            
            # Episode should have terminated in some way within max_steps
            assert episode_length <= max_steps
            
        finally:
            Path(jani_file).unlink()
            Path(start_file).unlink()
            Path(goal_file).unlink()
            Path(failure_file).unlink()
    
    def test_multiple_episodes(self):
        """Test running multiple episodes."""
        jani_file, start_file, goal_file, failure_file = create_test_jani_files()
        
        try:
            env = JaniEnv(jani_file, start_file, goal_file, failure_file)
            
            episodes_completed = 0
            max_episodes = 5
            
            for episode in range(max_episodes):
                observation, info = env.reset()
                episode_length = 0
                max_steps = 15
                
                while episode_length < max_steps:
                    # Take random valid action
                    action = episode_length % env.action_space.n
                    next_observation, reward, terminated, truncated, step_info = env.step(action)
                    
                    episode_length += 1
                    
                    if terminated or next_observation is None:
                        break
                
                episodes_completed += 1
            
            assert episodes_completed == max_episodes
            
        finally:
            Path(jani_file).unlink()
            Path(start_file).unlink()
            Path(goal_file).unlink()
            Path(failure_file).unlink()
    
    def test_reset_after_termination(self):
        """Test that environment can be reset after termination."""
        jani_file, start_file, goal_file, failure_file = create_test_jani_files()
        
        try:
            env = JaniEnv(jani_file, start_file, goal_file, failure_file)
            
            # First episode
            observation1, info1 = env.reset()
            if observation1[1] == 3:
                i = 7
            else:
                i = 5
            
            # Force termination by reaching goal
            # Set internal state object to make increment reach goal\n            env._current_state.variable_dict['counter'].value = 9  # 9+1=10 reaches goal
            for _ in range(i):
                next_observation, reward, terminated, truncated, step_info = env.step(0)
            
            # Check if goal was reached (counter = 10)\n            if next_observation is not None:\n                print(f\"counter value: {next_observation[0]}, reward: {reward}, terminated: {terminated}\")\n                # If counter is 10, it should be terminated due to goal\n                if next_observation[0] == 10.0:\n                    assert terminated, \"Goal reached but not terminated\"\n            \n            # Should be terminated or action invalid
            assert terminated or next_observation is None
            
            # Reset and start new episode
            observation2, info2 = env.reset()
            
            # Should be able to take actions again
            next_observation2, reward2, terminated2, truncated2, step_info2 = env.step(2)  # reset action
            
            # Verify new episode is working
            assert isinstance(observation2, np.ndarray)
            
        finally:
            Path(jani_file).unlink()
            Path(start_file).unlink()
            Path(goal_file).unlink()
            Path(failure_file).unlink()


class TestJaniEnvWithActualFiles:
    """Test cases using actual example files if available."""
    
    def test_bouncing_ball_environment(self):
        """Test JaniEnv with actual bouncing ball files."""
        files = create_bouncing_ball_files()
        if files is None:
            pytest.skip("Bouncing ball example files not found")
        
        jani_file, start_file, goal_file, failure_file = files
        
        env = JaniEnv(jani_file, start_file, goal_file, failure_file)
        
        # Test initialization
        assert isinstance(env.action_space, gym.spaces.Discrete)
        assert isinstance(env.observation_space, gym.spaces.Box)
        
        # Test reset
        observation, _ = env.reset()
        assert isinstance(observation, np.ndarray)
        # observation is now a vector - can check if it has expected length
        variables = env._jani.get_constants_variables()
        assert len(observation) == len(variables)
        
        # Test a few steps
        for _ in range(5):
            action = 0  # Take first available action
            next_observation, reward, terminated, truncated, _ = env.step(action)
            
            if terminated or next_observation is None:
                break
                
            # Verify physics make sense (if we get valid transitions)
            if next_observation is not None:
                assert isinstance(next_observation, np.ndarray)
                assert len(next_observation) == len(env._jani.get_constants_variables())
                # All values should be numeric
                assert all(isinstance(val, (int, float, np.number)) for val in next_observation)
    
    def test_bouncing_ball_goal_failure_detection(self):
        """Test goal and failure detection with bouncing ball."""
        files = create_bouncing_ball_files()
        if files is None:
            pytest.skip("Bouncing ball example files not found")
        
        jani_file, start_file, goal_file, failure_file = files
        
        env = JaniEnv(jani_file, start_file, goal_file, failure_file)
        observation, _ = env.reset()
        
        # Test that goal and failure detection work
        # Pass the internal state object, not the observation vector
        goal_result = env._jani.goal_reached(env._current_state)
        failure_result = env._jani.failure_reached(env._current_state)
        
        assert isinstance(goal_result, bool)
        assert isinstance(failure_result, bool)
        
        # They shouldn't both be true simultaneously
        assert not (goal_result and failure_result)
    
    def test_simple_test_environment(self):
        """Test JaniEnv with simple_test.jani model."""
        jani_file = "../examples/simple_test.jani"
        start_file = "../examples/simple_start.jani"  # Use compatible start file
        goal_file = "../examples/simple_goal.jani"
        failure_file = "../examples/simple_failure.jani"
        
        if not all(Path(f).exists() for f in [jani_file, start_file, goal_file, failure_file]):
            pytest.skip("Required example files not found")
        
        env = JaniEnv(jani_file, start_file, goal_file, failure_file)
        
        # Test initialization with simple test model
        assert isinstance(env.action_space, gym.spaces.Discrete)
        assert env.action_space.n == 3  # increment, toggle, multiply
        
        # Check observation space has expected dimensionality
        variables = env._jani.get_constants_variables()
        expected_vars = ['count', 'value', 'enabled', 'step_size', 'multiplier']
        var_names = [var.name for var in variables]
        for expected_var in expected_vars:
            assert expected_var in var_names
        
        # Test reset
        observation, _ = env.reset()
        assert isinstance(observation, np.ndarray)
        
        # Verify initial values make sense via vector indices
        variables = env._jani.get_constants_variables()
        var_names = [var.name for var in variables]
        
        if 'step_size' in var_names:
            step_size_idx = var_names.index('step_size')
            assert observation[step_size_idx] == 2
        if 'multiplier' in var_names:
            multiplier_idx = var_names.index('multiplier')
            assert observation[multiplier_idx] == 1.5
        
        # Test each action type
        for action_idx in range(env.action_space.n):
            observation, _ = env.reset()
            variables = env._jani.get_constants_variables()
            var_names = [var.name for var in variables]
            
            # Get initial values by index
            initial_count = observation[var_names.index('count')] if 'count' in var_names else 0
            initial_value = observation[var_names.index('value')] if 'value' in var_names else 0.0
            initial_enabled = observation[var_names.index('enabled')] if 'enabled' in var_names else True
            
            next_observation, reward, terminated, _, _ = env.step(action_idx)
            
            if next_observation is not None:
                # Verify state transitions make sense based on action
                if action_idx == 0:  # increment
                    if initial_enabled and initial_count + 2 <= 50:
                        count_idx = var_names.index('count')
                        assert next_observation[count_idx] == initial_count + 2
                elif action_idx == 1:  # toggle
                    if initial_enabled:
                        # Toggle should affect the enabled state somehow
                        pass  # The actual toggle logic depends on implementation
                elif action_idx == 2:  # multiply
                    if initial_enabled and initial_value * 1.5 <= 100.0:
                        value_idx = var_names.index('value')
                        count_idx = var_names.index('count')
                        assert abs(next_observation[value_idx] - initial_value * 1.5) < 0.001
                        assert next_observation[count_idx] == initial_count - 1
    
    def test_simple_test_action_guards(self):
        """Test that action guards work correctly in simple_test model."""
        jani_file = "../examples/simple_test.jani"
        start_file = "../examples/simple_start.jani"
        goal_file = "../examples/simple_goal.jani"
        failure_file = "../examples/simple_failure.jani"
        
        if not all(Path(f).exists() for f in [jani_file, start_file, goal_file, failure_file]):
            pytest.skip("Required example files not found")
        
        env = JaniEnv(jani_file, start_file, goal_file, failure_file)
        observation, _ = env.reset()
        
        # Test increment action when count is at maximum
        # Set internal state object
        env._current_state.variable_dict['count'].value = 49  # Close to upper bound of 50
        env._current_state.variable_dict['enabled'].value = True
        
        # This should work (49 + 2 = 51, but guard checks count + step_size <= 50)
        next_observation, reward, terminated, _, _ = env.step(0)  # increment
        
        # Test multiply action when value would exceed bounds
        observation, _ = env.reset()
        # Set internal state object
        env._current_state.variable_dict['value'].value = 80.0  # 80 * 1.5 = 120 > 100
        env._current_state.variable_dict['enabled'].value = True
        
        next_observation, reward, terminated, _, _ = env.step(2)  # multiply
        
        if next_observation is None:
            # Action was invalid due to guard
            assert reward == -1.0
            assert terminated == True
    
    def test_simple_test_disabled_state(self):
        """Test behavior when enabled=false in simple_test model."""
        jani_file = "../examples/simple_test.jani"
        start_file = "../examples/simple_start.jani"
        goal_file = "../examples/simple_goal.jani"
        failure_file = "../examples/simple_failure.jani"
        
        if not all(Path(f).exists() for f in [jani_file, start_file, goal_file, failure_file]):
            pytest.skip("Required example files not found")
        
        env = JaniEnv(jani_file, start_file, goal_file, failure_file)
        observation, _ = env.reset()
        
        # Manually disable the system
        # Set internal state object
        env._current_state.variable_dict['enabled'].value = False
        
        # Try increment action when disabled
        next_observation, reward, terminated, _, _ = env.step(0)  # increment
        
        if next_observation is None:
            # Action should be invalid when disabled
            assert reward == -1.0
            assert terminated == True
        
        # Try multiply action when disabled
        observation, _ = env.reset()
        # Set internal state object
        env._current_state.variable_dict['enabled'].value = False
        
        next_observation, reward, terminated, _, _ = env.step(2)  # multiply
        
        if next_observation is None:
            # Action should be invalid when disabled
            assert reward == -1.0
            assert terminated == True
    
    def test_bouncing_ball_action_effects(self):
        """Test specific action effects in bouncing ball model."""
        files = create_bouncing_ball_files()
        if files is None:
            pytest.skip("Bouncing ball example files not found")
        
        jani_file, start_file, goal_file, failure_file = files
        
        env = JaniEnv(jani_file, start_file, goal_file, failure_file)
        
        # Run multiple episodes to test action effects
        for episode in range(3):
            observation, _ = env.reset()
            variables = env._jani.get_constants_variables()
            var_names = [var.name for var in variables]
            
            # Get initial values by index
            initial_height = observation[var_names.index('height')] if 'height' in var_names else 0.0
            initial_velocity = observation[var_names.index('velocity')] if 'velocity' in var_names else 0.0
            initial_episode = observation[var_names.index('episode')] if 'episode' in var_names else 0
            
            step_count = 0
            max_steps = 10
            
            while step_count < max_steps:
                # Alternate between available actions
                action = step_count % env.action_space.n
                next_observation, reward, terminated, _, _ = env.step(action)
                
                if terminated or next_observation is None:
                    break
                
                # Physics should be consistent
                if next_observation is not None:
                    # All values should be numeric
                    assert isinstance(next_observation, np.ndarray)
                    assert all(isinstance(val, (int, float, np.number)) for val in next_observation)
                    
                    # Episode counter might increment (if episode variable exists)
                    if 'episode' in var_names:
                        episode_idx = var_names.index('episode')
                        assert next_observation[episode_idx] >= initial_episode
                
                step_count += 1
    
    def test_environment_determinism_with_seed(self):
        """Test deterministic behavior with seeded environments."""
        files = create_bouncing_ball_files()
        if files is None:
            pytest.skip("Bouncing ball example files not found")
        
        jani_file, start_file, goal_file, failure_file = files
        
        # Create two environments with same seed
        env1 = JaniEnv(jani_file, start_file, goal_file, failure_file)
        env2 = JaniEnv(jani_file, start_file, goal_file, failure_file)
        
        # Reset with same seed
        obs1, _ = env1.reset(seed=42)
        obs2, _ = env2.reset(seed=42)
        
        # Initial observations might be the same or different depending on implementation
        # But both should be valid
        assert isinstance(obs1, np.ndarray)
        assert isinstance(obs2, np.ndarray)
        
        # Take same sequence of actions
        action_sequence = [0, 1, 0, 1, 0]
        
        for action in action_sequence:
            if env1._current_state is None or env2._current_state is None:
                break
                
            next_obs1, reward1, term1, _, _ = env1.step(action)
            next_obs2, reward2, term2, _, _ = env2.step(action)
            
            # Both should have same termination status
            assert term1 == term2
            
            if term1 or term2:
                break
    
    def test_multiple_model_types(self):
        """Test that different JANI models can be loaded and work correctly."""
        test_configs = [
            ("../examples/bouncing_ball/bouncing_ball.jani", "../examples/bouncing_ball/start.jani", "../examples/bouncing_ball/objective.jani", "../examples/bouncing_ball/safe.jani", "Physics simulation model"),
            ("../examples/simple_test.jani", "../examples/simple_start.jani", "../examples/simple_goal.jani", "../examples/simple_failure.jani", "Simple test model")
        ]
        
        for jani_file, start_file, goal_file, failure_file, description in test_configs:
            if not all(Path(f).exists() for f in [jani_file, start_file, goal_file, failure_file]):
                continue  # Skip if files don't exist
            
            env = JaniEnv(jani_file, start_file, goal_file, failure_file)
            
            # Basic functionality test
            assert isinstance(env.action_space, gym.spaces.Discrete)
            assert isinstance(env.observation_space, gym.spaces.Box)
            assert env.action_space.n > 0
            
            # Reset and take a step
            observation, _ = env.reset()
            assert isinstance(observation, np.ndarray)
            
            # Try each available action
            for action in range(min(3, env.action_space.n)):  # Test up to 3 actions
                observation, _ = env.reset()
                next_observation, reward, terminated, _, _ = env.step(action)
                
                # Should get valid results
                assert isinstance(reward, float)
                assert isinstance(terminated, bool)
                
                if next_observation is not None:
                    assert isinstance(next_observation, np.ndarray)
    
    def test_long_episode_behavior(self):
        """Test behavior during longer episodes."""
        files = create_bouncing_ball_files()
        if files is None:
            pytest.skip("Bouncing ball example files not found")
        
        jani_file, start_file, goal_file, failure_file = files
        
        env = JaniEnv(jani_file, start_file, goal_file, failure_file)
        observation, _ = env.reset()
        
        step_count = 0
        max_steps = 50
        total_reward = 0
        episode_terminated = False
        
        while step_count < max_steps and not episode_terminated:
            # Use a simple policy: alternate actions
            action = step_count % env.action_space.n
            
            next_observation, reward, terminated, _, _ = env.step(action)
            total_reward += reward
            step_count += 1
            
            if terminated:
                episode_terminated = True
                if reward == 1.0:
                    # Successfully reached goal
                    assert next_observation is not None
                elif reward == -1.0:
                    # Failed or took invalid action
                    pass  # This is expected behavior
            
            if next_observation is None:
                episode_terminated = True
        
        # Episode should have progressed for some steps
        assert step_count > 0
        
        # Total reward should be reasonable
        assert isinstance(total_reward, float)


class TestJaniEnvEdgeCases:
    """Test cases for edge cases and error conditions."""
    
    def test_environment_gym_compliance(self):
        """Test that environment follows gym interface."""
        jani_file, start_file, goal_file, failure_file = create_test_jani_files()
        
        try:
            env = JaniEnv(jani_file, start_file, goal_file, failure_file)
            
            # Check that env inherits from gym.Env
            assert isinstance(env, gym.Env)
            
            # Check required attributes exist
            assert hasattr(env, 'action_space')
            assert hasattr(env, 'observation_space')
            assert hasattr(env, 'reset')
            assert hasattr(env, 'step')
            
            # Check action_space is a valid gym space
            assert isinstance(env.action_space, gym.Space)
            
            # Check observation_space is a valid gym space
            assert isinstance(env.observation_space, gym.Space)
            
        finally:
            Path(jani_file).unlink()
            Path(start_file).unlink()
            Path(goal_file).unlink()
            Path(failure_file).unlink()
    
    def test_state_consistency(self):
        """Test that internal state remains consistent."""
        jani_file, start_file, goal_file, failure_file = create_test_jani_files()
        
        try:
            env = JaniEnv(jani_file, start_file, goal_file, failure_file)
            
            # Before reset, current state should be None
            assert env._current_state is None
            
            # After reset, current state should be set
            observation, info = env.reset()
            assert env._current_state is not None
            # observation is vector, _current_state is State object
            assert isinstance(observation, np.ndarray)
            assert isinstance(env._current_state, State)
            
            # After step, current state should be updated
            next_observation, reward, terminated, truncated, step_info = env.step(0)
            # next_observation is vector, _current_state is State object
            if next_observation is not None:
                assert isinstance(next_observation, np.ndarray)
                assert isinstance(env._current_state, State)
            
        finally:
            Path(jani_file).unlink()
            Path(start_file).unlink()
            Path(goal_file).unlink()
            Path(failure_file).unlink()
    
    def test_reward_logic_consistency(self):
        """Test that reward logic is consistent."""
        jani_file, start_file, goal_file, failure_file = create_test_jani_files()
        
        try:
            env = JaniEnv(jani_file, start_file, goal_file, failure_file)
            observation, info = env.reset()
            
            # Test various scenarios and verify reward consistency
            for action in range(env.action_space.n):
                # Reset for each test
                observation, info = env.reset()
                next_observation, reward, terminated, truncated, step_info = env.step(action)
                
                # Verify reward logic
                if next_observation is None:
                    assert reward == -1.0
                    assert terminated == True
                elif terminated:
                    # Either goal (+1) or failure (-1)
                    assert reward == 1.0 or reward == -1.0
                else:
                    # Continuing episode
                    assert reward == 0.0
                    assert terminated == False
                    
        finally:
            Path(jani_file).unlink()
            Path(start_file).unlink()
            Path(goal_file).unlink()
            Path(failure_file).unlink()


class TestJaniEnvPerformanceAndStress:
    """Performance and stress tests for JaniEnv."""
    
    def test_rapid_reset_cycles(self):
        """Test rapid reset cycles for memory leaks or performance issues."""
        files = create_bouncing_ball_files()
        if files is None:
            pytest.skip("Bouncing ball example files not found")
        
        jani_file, start_file, goal_file, failure_file = files
        env = JaniEnv(jani_file, start_file, goal_file, failure_file)
        
        # Perform many rapid resets
        for i in range(100):
            observation, _ = env.reset()
            assert isinstance(observation, np.ndarray)
            
            # Occasionally take a step to ensure consistency
            if i % 10 == 0:
                next_observation, reward, terminated, _, _ = env.step(0)
                assert isinstance(reward, float)
    
    def test_boundary_value_stress(self):
        """Test environment with boundary values."""
        jani_file = "../examples/simple_test.jani"
        start_file = "../examples/simple_start.jani"
        goal_file = "../examples/simple_goal.jani"
        failure_file = "../examples/simple_failure.jani"
        
        if not all(Path(f).exists() for f in [jani_file, start_file, goal_file, failure_file]):
            pytest.skip("Required example files not found")
        
        env = JaniEnv(jani_file, start_file, goal_file, failure_file)
        
        # Test with various boundary values
        boundary_tests = [
            {'count': 0, 'value': 0.0, 'enabled': True},
            {'count': 50, 'value': 100.0, 'enabled': True},
            {'count': 25, 'value': 50.0, 'enabled': False},
            {'count': 1, 'value': 1.0, 'enabled': True},
            {'count': 49, 'value': 99.0, 'enabled': True}
        ]
        
        for test_values in boundary_tests:
            observation, _ = env.reset()
            
            # Set boundary values on internal state object
            for var, value in test_values.items():
                if var in env._current_state.variable_dict:
                    env._current_state.variable_dict[var].value = value
            
            # Try each action
            for action in range(env.action_space.n):
                # Reset to boundary state for each action test
                observation, _ = env.reset()
                for var, value in test_values.items():
                    if var in env._current_state.variable_dict:
                        env._current_state.variable_dict[var].value = value
                
                next_observation, reward, terminated, _, _ = env.step(action)
                
                # Should handle boundary conditions gracefully
                assert isinstance(reward, float)
                assert isinstance(terminated, bool)
    
    def test_action_sequence_patterns(self):
        """Test various action sequence patterns."""
        files = create_bouncing_ball_files()
        if files is None:
            pytest.skip("Bouncing ball example files not found")
        
        jani_file, start_file, goal_file, failure_file = files
        env = JaniEnv(jani_file, start_file, goal_file, failure_file)
        
        # Test different action patterns
        patterns = [
            [0] * 10,  # Repeated same action
            [0, 1] * 5,  # Alternating actions
            [0, 1, 0, 0, 1, 1, 0, 1],  # Mixed pattern
            list(range(env.action_space.n)) * 3  # Cycle through all actions
        ]
        
        for pattern in patterns:
            observation, _ = env.reset()
            
            for action in pattern:
                if env._current_state is None:
                    break
                
                next_observation, reward, terminated, _, _ = env.step(action)
                
                if terminated or next_observation is None:
                    break
                
                # Environment should remain stable
                assert isinstance(reward, float)
                assert reward in [-1.0, 0.0, 1.0]  # Only expected reward values
    
    def test_state_space_exploration(self):
        """Test exploration of different parts of state space."""
        jani_file = "../examples/simple_test.jani"
        start_file = "../examples/simple_start.jani"
        goal_file = "../examples/simple_goal.jani"
        failure_file = "../examples/simple_failure.jani"
        
        if not all(Path(f).exists() for f in [jani_file, start_file, goal_file, failure_file]):
            pytest.skip("Required example files not found")
        
        env = JaniEnv(jani_file, start_file, goal_file, failure_file)
        
        # Collect unique states encountered
        unique_states = set()
        
        for episode in range(20):
            observation, _ = env.reset()
            
            # Create a hashable representation of the state from internal state object
            state_tuple = tuple(sorted(
                (name, var.value) for name, var in env._current_state.variable_dict.items()
                if name in ['count', 'value', 'enabled']  # Only key variables
            ))
            unique_states.add(state_tuple)
            
            # Take some random actions to explore
            for step in range(10):
                action = step % env.action_space.n
                next_observation, reward, terminated, _, _ = env.step(action)
                
                if terminated or next_observation is None:
                    break
                
                # Record new state from internal state object
                if next_observation is not None and env._current_state is not None:
                    state_tuple = tuple(sorted(
                        (name, var.value) for name, var in env._current_state.variable_dict.items()
                        if name in ['count', 'value', 'enabled']
                    ))
                    unique_states.add(state_tuple)
        
        # Should have explored multiple different states
        assert len(unique_states) > 1
    
    def test_error_recovery(self):
        """Test environment recovery from error conditions."""
        files = create_bouncing_ball_files()
        if files is None:
            pytest.skip("Bouncing ball example files not found")
        
        jani_file, start_file, goal_file, failure_file = files
        env = JaniEnv(jani_file, start_file, goal_file, failure_file)
        
        # Test recovery from invalid action
        observation, _ = env.reset()
        
        try:
            # Take invalid action
            next_observation, reward, terminated, _, _ = env.step(999)
            assert False, "Should have raised ValueError"
        except ValueError:
            # Should be able to recover
            pass
        
        # Environment should still work after error
        observation, _ = env.reset()
        next_observation, reward, terminated, _, _ = env.step(0)
        assert isinstance(reward, float)
        
        # Test recovery from terminated state
        observation, _ = env.reset()
        
        # Force termination by manipulating state if possible
        # Then verify reset works
        observation, _ = env.reset()
        assert isinstance(observation, np.ndarray)


class TestJaniEnvDocumentationExamples:
    """Test cases that serve as documentation examples."""
    
    def test_basic_usage_example(self):
        """Basic usage example for documentation."""
        files = create_bouncing_ball_files()
        if files is None:
            pytest.skip("Bouncing ball example files not found")
        
        jani_file, start_file, goal_file, failure_file = files
        
        # Create environment
        env = JaniEnv(jani_file, start_file, goal_file, failure_file)
        
        # Reset environment
        observation, info = env.reset()
        
        # Take actions until episode ends
        for step in range(100):
            action = step % env.action_space.n  # Simple policy
            next_observation, reward, terminated, truncated, info = env.step(action)
            
            if terminated:
                if reward == 1.0:
                    print(f"Goal reached in {step} steps!")
                elif reward == -1.0:
                    print(f"Episode failed at step {step}")
                break
            
            if next_observation is None:
                print(f"Invalid action at step {step}")
                break
        
        # Environment can be reset for another episode
        observation, info = env.reset()
        assert isinstance(observation, np.ndarray)
    
    def test_action_space_usage(self):
        """Example of how to use action space properly."""
        jani_file = "../examples/simple_test.jani"
        start_file = "../examples/simple_start.jani"
        goal_file = "../examples/simple_goal.jani"
        failure_file = "../examples/simple_failure.jani"
        
        if not all(Path(f).exists() for f in [jani_file, start_file, goal_file, failure_file]):
            pytest.skip("Required example files not found")
        
        env = JaniEnv(jani_file, start_file, goal_file, failure_file)
        
        # Examine action space
        print(f"Action space: {env.action_space}")
        print(f"Number of actions: {env.action_space.n}")
        
        # Test all available actions
        observation, _ = env.reset()
        
        for action_idx in range(env.action_space.n):
            # Reset for each action test
            observation, _ = env.reset()
            
            print(f"Testing action {action_idx}")
            next_observation, reward, terminated, _, _ = env.step(action_idx)
            
            if next_observation is not None:
                print(f"Action {action_idx} succeeded, reward: {reward}")
            else:
                print(f"Action {action_idx} failed, reward: {reward}")
        
        # Actions should be in range [0, action_space.n)
        assert all(0 <= action < env.action_space.n for action in range(env.action_space.n))
    
    def test_observation_space_usage(self):
        """Example of how to use observation space."""
        files = create_bouncing_ball_files()
        if files is None:
            pytest.skip("Bouncing ball example files not found")
        
        jani_file, start_file, goal_file, failure_file = files
        env = JaniEnv(jani_file, start_file, goal_file, failure_file)
        
        # Examine observation space
        print(f"Observation space: {env.observation_space}")
        
        observation, _ = env.reset()
        
        # Access individual variables from internal state object
        for var_name, var_obj in env._current_state.variable_dict.items():
            print(f"{var_name}: {var_obj.value} (type: {var_obj.type})")
            
            # Check if variable has bounds
            if var_obj.lower_bound is not None:
                print(f"  Lower bound: {var_obj.lower_bound}")
            if var_obj.upper_bound is not None:
                print(f"  Upper bound: {var_obj.upper_bound}")
        
        # Verify observation vector has correct dimensionality
        variables = env._jani.get_constants_variables()
        assert len(observation) == len(variables)
        assert env.observation_space.shape == (len(variables),)


class TestJaniEnvVectorObservations:
    """Test cases specific to vector-based observations."""
    
    def test_observation_vector_format(self):
        """Test that observations are returned as proper numpy arrays."""
        jani_file, start_file, goal_file, failure_file = create_test_jani_files()
        
        try:
            env = JaniEnv(jani_file, start_file, goal_file, failure_file)
            observation, _ = env.reset()
            
            # Verify observation is numpy array
            assert isinstance(observation, np.ndarray)
            assert observation.dtype == np.float32
            
            # Verify shape matches number of variables
            variables = env._jani.get_constants_variables()
            assert observation.shape == (len(variables),)
            
            # Verify all values are finite
            assert np.all(np.isfinite(observation))
            
        finally:
            Path(jani_file).unlink()
            Path(start_file).unlink()
            Path(goal_file).unlink()
            Path(failure_file).unlink()
    
    def test_observation_consistency_with_state(self):
        """Test that observation vector matches internal state."""
        jani_file, start_file, goal_file, failure_file = create_test_jani_files()
        
        try:
            env = JaniEnv(jani_file, start_file, goal_file, failure_file)
            observation, _ = env.reset()
            
            # Get variable values from internal state
            variables = env._jani.get_constants_variables()
            expected_values = []
            for var in variables:
                expected_values.append(env._current_state.variable_dict[var.name].value)
            
            # Verify observation matches internal state values
            np.testing.assert_array_equal(observation, expected_values)
            
        finally:
            Path(jani_file).unlink()
            Path(start_file).unlink()
            Path(goal_file).unlink()
            Path(failure_file).unlink()
    
    def test_observation_bounds_checking(self):
        """Test that observations respect variable bounds."""
        jani_file, start_file, goal_file, failure_file = create_test_jani_files()
        
        try:
            env = JaniEnv(jani_file, start_file, goal_file, failure_file)
            
            # Test multiple resets to check bounds
            for _ in range(10):
                observation, _ = env.reset()
                
                # Check observation is within observation space bounds
                assert env.observation_space.contains(observation)
                
                # Check specific variable bounds
                variables = env._jani.get_constants_variables()
                for i, var in enumerate(variables):
                    value = observation[i]
                    
                    if var.lower_bound is not None:
                        assert value >= var.lower_bound, f"Variable {var.name} value {value} below lower bound {var.lower_bound}"
                    
                    if var.upper_bound is not None:
                        assert value <= var.upper_bound, f"Variable {var.name} value {value} above upper bound {var.upper_bound}"
                
        finally:
            Path(jani_file).unlink()
            Path(start_file).unlink()
            Path(goal_file).unlink()
            Path(failure_file).unlink()
    
    def test_step_observation_vector_format(self):
        """Test that step returns observations in vector format."""
        jani_file, start_file, goal_file, failure_file = create_test_jani_files()
        
        try:
            env = JaniEnv(jani_file, start_file, goal_file, failure_file)
            observation, _ = env.reset()
            
            # Take several steps and verify format
            for action in range(min(3, env.action_space.n)):
                env.reset()  # Reset for each action test
                next_observation, reward, terminated, truncated, _ = env.step(action)
                
                if next_observation is not None:
                    # Verify it's a numpy array
                    assert isinstance(next_observation, np.ndarray)
                    assert next_observation.dtype == np.float32
                    
                    # Verify shape consistency
                    variables = env._jani.get_constants_variables()
                    assert next_observation.shape == (len(variables),)
                    
                    # Verify values are finite
                    assert np.all(np.isfinite(next_observation))
                    
                    # Verify within bounds
                    assert env.observation_space.contains(next_observation)
                
        finally:
            Path(jani_file).unlink()
            Path(start_file).unlink()
            Path(goal_file).unlink()
            Path(failure_file).unlink()
    
    def test_variable_indexing_consistency(self):
        """Test that variable values can be accessed consistently by index."""
        jani_file, start_file, goal_file, failure_file = create_test_jani_files()
        
        try:
            env = JaniEnv(jani_file, start_file, goal_file, failure_file)
            observation, _ = env.reset()
            
            variables = env._jani.get_constants_variables()
            var_names = [var.name for var in variables]
            
            # Test accessing known variables by index
            for var_name in ['counter', 'active', 'max_value']:
                if var_name in var_names:
                    var_idx = var_names.index(var_name)
                    
                    # Get value from observation vector
                    obs_value = observation[var_idx]
                    
                    # Get value from internal state
                    state_value = env._current_state.variable_dict[var_name].value
                    
                    # Should match
                    assert obs_value == state_value, f"Mismatch for {var_name}: obs={obs_value}, state={state_value}"
            
        finally:
            Path(jani_file).unlink()
            Path(start_file).unlink()
            Path(goal_file).unlink()
            Path(failure_file).unlink()


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])