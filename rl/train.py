#!/usr/bin/env python3
"""
Comprehensive training script for JANI environment with hyperparameter tuning.

This script provides advanced training capabilities including:
- Command line argument parsing for JANI model files
- Hyperparameter optimization using Optuna
- TensorBoard logging
- Weights & Biases integration
- Model checkpointing and best model saving
- Support for both property files and individual start/goal/safe files
"""

import argparse
import sys
import warnings
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

import numpy as np
import torch

# Optional imports for advanced features
try:
    import optuna
    OPTUNA_AVAILABLE = True
    try:
        from optuna.integration import WeightsAndBiasesCallback
        OPTUNA_WANDB_CALLBACK_AVAILABLE = True
    except ImportError:
        OPTUNA_WANDB_CALLBACK_AVAILABLE = False
except ImportError:
    OPTUNA_AVAILABLE = False
    OPTUNA_WANDB_CALLBACK_AVAILABLE = False
    print("Warning: Optuna not available. Hyperparameter tuning will be disabled.")

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: Weights & Biases not available. Advanced logging will be disabled.")

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecMonitor, DummyVecEnv

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.utils import get_action_masks

from jani.environment import JaniEnv
from jani.oracle import TarjanOracle

from gymnasium.wrappers import TimeLimit


class WandbCallback(BaseCallback):
    """Custom callback for Weights & Biases logging."""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _on_step(self) -> bool:
        # Log metrics to wandb
        if WANDB_AVAILABLE and wandb.run is not None:
            # Log training environment rewards
            if hasattr(self.model, 'ep_info_buffer') and self.model.ep_info_buffer:
                if len(self.model.ep_info_buffer) > 0:
                    ep_info = self.model.ep_info_buffer[-1]
                    self.episode_rewards.append(ep_info['r'])
                    self.episode_lengths.append(ep_info['l'])
                    
                    # Log individual episode metrics
                    wandb.log({
                        'train/episode_reward': ep_info['r'],
                        'train/episode_length': ep_info['l'],
                        'train/episode_time': ep_info['t'],
                        'train/timesteps': self.num_timesteps
                    })
            
            # Log training statistics every 100 steps
            if self.num_timesteps % 100 == 0 and len(self.episode_rewards) > 0:
                recent_rewards = self.episode_rewards[-10:] if len(self.episode_rewards) >= 10 else self.episode_rewards
                wandb.log({
                    'train/mean_reward_last_10': np.mean(recent_rewards),
                    'train/std_reward_last_10': np.std(recent_rewards),
                    'train/max_reward': np.max(self.episode_rewards),
                    'train/min_reward': np.min(self.episode_rewards),
                    'train/total_episodes': len(self.episode_rewards),
                    'train/timesteps': self.num_timesteps
                })
            
            # Log model training metrics
            if hasattr(self.model, 'logger') and self.model.logger.name_to_value:
                log_dict = {}
                for key, value in self.model.logger.name_to_value.items():
                    if isinstance(value, (int, float)):
                        log_dict[f'train/{key}'] = value
                if log_dict:
                    wandb.log(log_dict)
        
        return True


class ClassifierDebugCallback(BaseCallback):
    """Custom callback for debugging classifier predictions against Tarjan oracle."""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.oracle = None
        self.step_predictions = []  # Store (predicted, actual) pairs for each step
        self.total_correct = 0
        self.total_predictions = 0
        self.last_logged_timestep = 0
        
    def _unwrap_to_jani_env(self, env):
        """Helper method to unwrap environment to get JaniEnv."""
        unwrapped_env = env
        while hasattr(unwrapped_env, 'env'):
            unwrapped_env = unwrapped_env.env
        if hasattr(unwrapped_env, 'unwrapped'):
            unwrapped_env = unwrapped_env.unwrapped
        return unwrapped_env
        
    def _on_training_start(self) -> None:
        """Initialize the oracle when training starts."""
        # Get the JANI model from the first environment
        if hasattr(self.training_env, 'envs'):
            first_env = self.training_env.envs[0]
        else:
            first_env = self.training_env
            
        first_env = self._unwrap_to_jani_env(first_env)
        jani_model = first_env.get_model()
        self.oracle = TarjanOracle(jani_model)
        
        # Get classifier from environment to check predictions
        self.classifier = first_env._classifier if hasattr(first_env, '_classifier') else None
        self.scaler = first_env._scaler if hasattr(first_env, '_scaler') else None
        
        if self.verbose >= 1:
            print("🔍 Classifier debug callback initialized with Tarjan oracle")
        
    def _on_step(self) -> bool:
        """Check classifier predictions against oracle at each step."""
        if self.oracle is None or self.classifier is None:
            return True
            
        # Get the current environment states from all vectorized environments
        if hasattr(self.training_env, 'envs'):
            environments = self.training_env.envs
        else:
            environments = [self.training_env]
            
        for env in environments:
            unwrapped_env = self._unwrap_to_jani_env(env)
                
            if hasattr(unwrapped_env, '_current_state') and unwrapped_env._current_state is not None:
                current_state = unwrapped_env._current_state
                
                # Get classifier prediction
                try:
                    from classifier import predict
                    state_vector = np.array(current_state.to_vector(), dtype=np.float32).reshape(1, -1)
                    _, classifier_is_safe = predict(self.classifier, state_vector, self.scaler)
                    
                    # Get oracle prediction (ground truth)
                    oracle_is_safe = self.oracle.is_safe(current_state)
                    
                    # Record prediction
                    self.step_predictions.append((classifier_is_safe, oracle_is_safe))
                    if classifier_is_safe == oracle_is_safe:
                        self.total_correct += 1
                    self.total_predictions += 1
                    
                except Exception as e:
                    if self.verbose >= 1:
                        print(f"⚠️ Error in classifier debug: {e}")
                    continue
        
        # Log to wandb every 100 steps
        if (self.num_timesteps - self.last_logged_timestep) >= 100:
            if self.total_predictions > 0:
                step_accuracy = self.total_correct / self.total_predictions
                
                # Calculate recent accuracy (last 100 steps)
                recent_predictions = self.step_predictions[-100:] if len(self.step_predictions) >= 100 else self.step_predictions
                recent_correct = sum(1 for pred, actual in recent_predictions if pred == actual)
                recent_accuracy = recent_correct / len(recent_predictions) if recent_predictions else 0.0
                
                if WANDB_AVAILABLE and wandb.run is not None:
                    wandb.log({
                        'debug/classifier_accuracy_overall': step_accuracy,
                        'debug/classifier_accuracy_recent': recent_accuracy,
                        'debug/total_predictions': self.total_predictions,
                        'debug/timesteps': self.num_timesteps
                    })
                
                if self.verbose >= 1:
                    print(f"🎯 Classifier accuracy: {step_accuracy:.3f} (overall), {recent_accuracy:.3f} (recent 100 steps)")
                    
            self.last_logged_timestep = self.num_timesteps
            
        return True


class ClassifierMonitorCallback(BaseCallback):
    '''Custom callback to monitor classifier's information during training.'''
    def __init__(self, env, monitor_freq: int = 10000, num_episodes: int = 30, tune_classifier: bool = False, data_dir: str = None, csv_output_path: str = None, verbose: int = 0):
        super().__init__()
        self.env = env
        self.monitor_freq = monitor_freq
        self.num_episodes = num_episodes
        self.tune_classifier = tune_classifier
        self.data_dir = data_dir
        self.csv_output_path = csv_output_path
        self.verbose = verbose
        if self.tune_classifier and self.data_dir is None:
            raise ValueError("data_dir must be provided if tune_classifier is True")
        
        # Initialize components
        self.oracle = None
        self.classifier = None
        self.scaler = None
        self.jani_model = None
        
        # Global state storage
        self.global_state_vectors = []  # Store vectorized representations
        self.global_state_objects = []  # Store original state objects
        self.global_actions = []  # Store actions that led to each state
        
    def _unwrap_to_jani_env(self, env):
        """Helper method to unwrap environment to get JaniEnv."""
        unwrapped_env = env
        while hasattr(unwrapped_env, 'env'):
            unwrapped_env = unwrapped_env.env
        if hasattr(unwrapped_env, 'unwrapped'):
            unwrapped_env = unwrapped_env.unwrapped
        return unwrapped_env
        
    def _on_training_start(self) -> None:
        """Initialize oracle and classifier when training starts."""
        # Get the JANI model from the passed env - handle vectorized environments
        if hasattr(self.env, 'envs'):
            # Vectorized environment - get first individual environment
            first_env = self._unwrap_to_jani_env(self.env.envs[0])
        else:
            # Single environment
            first_env = self._unwrap_to_jani_env(self.env)
            
        self.jani_model = first_env.get_model()
        self.oracle = TarjanOracle(self.jani_model)
        
        # Get classifier from environment (may be None)
        self.classifier = first_env._classifier if hasattr(first_env, '_classifier') else None
        self.scaler = first_env._scaler if hasattr(first_env, '_scaler') else None
        
        if self.verbose >= 1:
            if self.classifier is not None:
                print("🔍 ClassifierMonitorCallback initialized with Tarjan oracle and classifier")
            else:
                print("🔍 ClassifierMonitorCallback initialized with Tarjan oracle only (no classifier)")
    
    def _run_episodes_and_collect_states(self):
        """Run episodes and collect both state vectors, state objects, and actions taken."""
        encountered_state_vectors = []
        encountered_state_objects = []
        encountered_actions = []
        failure_states_count = 0  # Track number of failure states reached
        
        # Run episodes using the current policy
        for episode in range(self.num_episodes):
            obs, _ = self.env.reset()
            done = False
            truncated = False
            
            # Store initial state (no action taken yet, use None)
            jani_env = self._unwrap_to_jani_env(self.env)
            current_state_obj = jani_env.get_state_repr()
            current_state_vector = current_state_obj.to_vector()
            
            encountered_state_vectors.append(current_state_vector)
            encountered_state_objects.append(current_state_obj)
            encountered_actions.append(None)  # No action for initial state
            
            # Check if initial state is a failure state
            if self.jani_model.failure_reached(current_state_obj):
                failure_states_count += 1

            while not (done or truncated):
                # Get action masks for MaskablePPO
                action_masks = get_action_masks(self.env)
                
                # Get action from model with action masks
                action, _ = self.model.predict(obs, action_masks=action_masks, deterministic=True)
                obs, reward, done, truncated, info = self.env.step(action)
                # Store state after action
                current_state_obj = jani_env.get_state_repr()
                current_state_vector = current_state_obj.to_vector()
                
                encountered_state_vectors.append(current_state_vector)
                encountered_state_objects.append(current_state_obj)
                encountered_actions.append(int(action))  # Store the action that led to this state
                    
                # Check if this is a failure state
                if self.jani_model.failure_reached(current_state_obj):
                    failure_states_count += 1
        
        # Remove duplicates while keeping first occurrence and maintaining order
        unique_state_vectors = []
        unique_state_objects = []
        unique_actions = []
        seen_states = set()
        
        for state_vector, state_obj, action in zip(encountered_state_vectors, encountered_state_objects, encountered_actions):
            state_tuple = tuple(state_vector)  # Convert to tuple for hashing
            if state_tuple not in seen_states:
                unique_state_vectors.append(state_vector)
                unique_state_objects.append(state_obj)
                unique_actions.append(action)
                seen_states.add(state_tuple)
        
        return unique_state_vectors, unique_state_objects, unique_actions, failure_states_count
    
    def _evaluate_classifier_performance(self, state_vectors, state_objects):
        """Evaluate classifier performance against oracle on given states."""
        if not state_vectors or self.oracle is None:
            return {}
        
        import numpy as np
        oracle_predictions = []
        classifier_predictions = []
        
        # Always get oracle predictions
        for state_obj in state_objects:
            try:
                oracle_is_safe = self.oracle.is_safe(state_obj)
                oracle_predictions.append(oracle_is_safe)
            except Exception as e:
                if self.verbose >= 2:
                    print(f"⚠️ Error evaluating state with oracle: {e}")
                continue
        
        if not oracle_predictions:
            return {}
        
        oracle_predictions = np.array(oracle_predictions)
        unsafe_count = np.sum(~oracle_predictions)
        
        # If no classifier, only return basic oracle metrics
        if self.classifier is None:
            return {
                'total_states': len(oracle_predictions),
                'safe_states': np.sum(oracle_predictions),
                'unsafe_states': unsafe_count,
                'oracle_predictions': oracle_predictions.tolist()
            }
        
        # If classifier exists, also evaluate classifier performance
        from classifier import predict
        
        for state_vector in state_vectors:
            try:
                # Get classifier prediction using state vector
                state_array = np.array(state_vector, dtype=np.float32).reshape(1, -1)
                _, classifier_is_safe = predict(self.classifier, state_array, self.scaler)
                classifier_predictions.append(bool(classifier_is_safe))
                
            except Exception as e:
                if self.verbose >= 2:
                    print(f"⚠️ Error evaluating state with classifier: {e}")
                classifier_predictions.append(False)  # Default to unsafe if error
                continue
        
        if len(classifier_predictions) != len(oracle_predictions):
            # Fallback if lengths don't match
            if self.verbose >= 1:
                print(f"⚠️ Length mismatch: oracle={len(oracle_predictions)}, classifier={len(classifier_predictions)}")
            return {
                'total_states': len(oracle_predictions),
                'safe_states': np.sum(oracle_predictions),
                'unsafe_states': unsafe_count,
                'oracle_predictions': oracle_predictions.tolist()
            }
        
        classifier_predictions = np.array(classifier_predictions)
        
        # Calculate performance metrics
        accuracy = np.mean(oracle_predictions == classifier_predictions)
        
        # Calculate precision, recall, F1 if we have both classes
        true_positives = np.sum((oracle_predictions == True) & (classifier_predictions == True))
        false_positives = np.sum((oracle_predictions == False) & (classifier_predictions == True))
        false_negatives = np.sum((oracle_predictions == True) & (classifier_predictions == False))
        true_negatives = np.sum((oracle_predictions == False) & (classifier_predictions == False))
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'total_states': len(oracle_predictions),
            'safe_states': np.sum(oracle_predictions),
            'unsafe_states': unsafe_count,
            'oracle_predictions': oracle_predictions.tolist(),
            'classifier_predictions': classifier_predictions.tolist()
        }
    
    def _fine_tune_classifier(self, state_vectors, oracle_predictions):
        """Fine-tune the classifier using existing training data plus new data."""
        if not self.tune_classifier or not self.data_dir:
            return
            
        try:
            from classifier.data_loader import load_datasets, create_dataloaders
            from classifier.train import train_model
            from classifier.models import Classifier
            import torch
            import numpy as np
            
            if self.verbose >= 1:
                print("🔧 Fine-tuning classifier with new data...")
            
            # Load existing training data
            existing_datasets = load_datasets(self.data_dir)
            
            # Prepare new data from oracle evaluations
            if not state_vectors or not oracle_predictions:
                if self.verbose >= 1:
                    print("⚠️ No new data to fine-tune with")
                return
            
            new_features = np.array(state_vectors)
            new_labels = np.array([1 if pred else 0 for pred in oracle_predictions])
            
            # Combine existing and new data
            train_features, train_labels = existing_datasets['train']
            combined_features = np.vstack([train_features, new_features])
            combined_labels = np.hstack([train_labels, new_labels])
            
            # Create updated datasets
            updated_datasets = {
                'train': (combined_features, combined_labels),
                'val': existing_datasets['val'],
                'test': existing_datasets['test']
            }
            
            # Create data loaders
            dataloaders, new_scaler = create_dataloaders(updated_datasets, batch_size=32)
            
            # Get current model architecture
            input_size = combined_features.shape[1]
            
            # Load current model to get architecture info
            if hasattr(self.classifier, 'hidden_sizes'):
                hidden_sizes = self.classifier.hidden_sizes
            else:
                hidden_sizes = [128, 64]  # Default architecture
                
            if hasattr(self.classifier, 'dropout'):
                dropout = self.classifier.dropout
            else:
                dropout = 0.2
            
            # Create new model with same architecture
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            fine_tuned_model = Classifier(input_size, hidden_sizes, dropout).to(device)
            
            # Initialize with current weights
            fine_tuned_model.load_state_dict(self.classifier.state_dict())
            
            # Fine-tune with reduced learning rate and fewer epochs
            fine_tuned_model = train_model(fine_tuned_model, dataloaders, device, epochs=10, lr=1e-5)
            
            # Update the classifier and scaler
            self.classifier = fine_tuned_model
            self.scaler = new_scaler
            
            if self.verbose >= 1:
                print(f"✅ Classifier fine-tuned with {len(new_features)} new samples")
                
        except Exception as e:
            if self.verbose >= 1:
                print(f"⚠️ Error during fine-tuning: {e}")
    
    def _log_to_wandb(self, metrics):
        """Log metrics to wandb if available."""
        if WANDB_AVAILABLE and wandb.run is not None:
            log_dict = {}
            for key, value in metrics.items():
                if key not in ['oracle_predictions', 'classifier_predictions']:  # Skip raw predictions
                    log_dict[f'classifier_monitor/{key}'] = value
            
            log_dict['classifier_monitor/timesteps'] = self.num_timesteps
            wandb.log(log_dict)
    
    def _update_global_storage(self, state_vectors, state_objects, actions):
        """Update global storage with new states and actions, avoiding duplicates."""
        existing_tuples = set(tuple(vec) for vec in self.global_state_vectors)
        
        for state_vector, state_obj, action in zip(state_vectors, state_objects, actions):
            state_tuple = tuple(state_vector)
            if state_tuple not in existing_tuples:
                self.global_state_vectors.append(state_vector)
                self.global_state_objects.append(state_obj)
                self.global_actions.append(action)
                existing_tuples.add(state_tuple)
        
        if self.verbose >= 2:
            print(f"📦 Global storage updated: {len(self.global_state_vectors)} total unique states")
    
    def _count_previously_visited_unsafe_states(self, state_vectors, oracle_predictions):
        """Count how many unsafe states from current episode were previously visited.
        
        Args:
            state_vectors: List of current episode state vectors
            oracle_predictions: List of oracle predictions (True=safe, False=unsafe)
        """
        if not self.global_state_vectors or not state_vectors or not oracle_predictions:
            return 0
        
        # Create set of previously visited states for fast lookup
        previous_state_tuples = set(tuple(vec) for vec in self.global_state_vectors)
        
        previously_visited_unsafe_count = 0
        
        for state_vector, is_safe in zip(state_vectors, oracle_predictions):
            if not is_safe:  # State is unsafe
                # Check if this state was previously visited
                state_tuple = tuple(state_vector)
                if state_tuple in previous_state_tuples:
                    previously_visited_unsafe_count += 1
        
        return previously_visited_unsafe_count
    
    def _write_to_csv(self, state_vectors, actions, oracle_predictions):
        """Write collected data to CSV file if csv_output_path is provided."""
        if self.csv_output_path is None:
            return
        
        try:
            import pandas as pd
            from pathlib import Path
            
            # Prepare data for CSV
            csv_data = []
            for state_vector, action, oracle_pred in zip(state_vectors, actions, oracle_predictions):
                row = list(state_vector) + [action, int(oracle_pred)]  # Convert boolean to int
                csv_data.append(row)
            
            # Create DataFrame
            df = pd.DataFrame(csv_data)
            
            # Create directory if it doesn't exist
            csv_path = Path(self.csv_output_path)
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Append to CSV without header
            df.to_csv(csv_path, mode='a', header=False, index=False)
            
            if self.verbose >= 1:
                print(f"📝 Appended {len(csv_data)} records to {self.csv_output_path}")
                
        except Exception as e:
            if self.verbose >= 1:
                print(f"⚠️ Error writing to CSV: {e}")
    
    def _on_step(self) -> bool:
        """Monitor classifier performance at specified frequency."""
        if self.num_timesteps % self.monitor_freq == 0 and self.num_timesteps > 0:
            if self.verbose >= 1:
                print(f"🔍 Monitoring classifier at timestep {self.num_timesteps}")
            
            # Run episodes and collect state vectors, objects, actions, and failure states count
            state_vectors, state_objects, actions, failure_states_count = self._run_episodes_and_collect_states()
            
            if self.verbose >= 1:
                print(f"📊 Collected {len(state_vectors)} unique states from {self.num_episodes} episodes")
                print(f"❌ Failure states reached: {failure_states_count}")
                # Count actions (excluding None for initial states)
                valid_actions = [a for a in actions if a is not None]
                if valid_actions:
                    action_counts = {}
                    for action in valid_actions:
                        action_counts[action] = action_counts.get(action, 0) + 1
                    print(f"🎯 Actions taken: {dict(sorted(action_counts.items()))}")
            
            # Evaluate classifier performance against oracle
            performance_metrics = self._evaluate_classifier_performance(state_vectors, state_objects)
            
            if performance_metrics:
                unsafe_states_count = performance_metrics['unsafe_states']
                total_states = performance_metrics['total_states']
                
                # Add failure states count to performance metrics
                performance_metrics['failure_states_reached'] = failure_states_count
                
                # Calculate ratio of unsafe states to total states
                unsafe_ratio = unsafe_states_count / total_states if total_states > 0 else 0.0
                performance_metrics['unsafe_ratio'] = unsafe_ratio
                
                # Calculate number of unsafe states that were previously visited
                # Do this BEFORE updating global storage to avoid counting current states as "previously visited"
                oracle_predictions = performance_metrics.get('oracle_predictions', [])
                previously_visited_unsafe_count = self._count_previously_visited_unsafe_states(state_vectors, oracle_predictions)
                performance_metrics['previously_visited_unsafe_states'] = previously_visited_unsafe_count
            
            # Update global storage AFTER counting previously visited states
            self._update_global_storage(state_vectors, state_objects, actions)
            
            if performance_metrics:
                if self.verbose >= 1:
                    if self.classifier is not None:
                        # With classifier: show classifier performance metrics
                        print(f"🎯 Classifier accuracy: {performance_metrics['accuracy']:.3f}")
                        print(f"   Precision: {performance_metrics['precision']:.3f}, Recall: {performance_metrics['recall']:.3f}")
                        print(f"   F1-score: {performance_metrics['f1_score']:.3f}")
                    
                    # Always show oracle-based metrics
                    unsafe_ratio = performance_metrics['unsafe_ratio']
                    previously_visited_unsafe = performance_metrics['previously_visited_unsafe_states']
                    
                    print(f"   Total states: {performance_metrics['total_states']}, Safe: {performance_metrics['safe_states']}, Unsafe: {unsafe_states_count}")
                    print(f"⚠️  Unsafe states encountered this callback: {unsafe_states_count}")
                    print(f"📊 Unsafe ratio: {unsafe_ratio:.3f} ({unsafe_states_count}/{total_states})")
                    print(f"🔄 Previously visited unsafe states: {previously_visited_unsafe}")
                
                # Log to wandb (includes failure_states_reached and available metrics)
                self._log_to_wandb(performance_metrics)
                
                # Write to CSV if path is provided
                self._write_to_csv(state_vectors, actions, performance_metrics['oracle_predictions'])
                
                # Fine-tune classifier if requested
                if self.tune_classifier:
                    self._fine_tune_classifier(state_vectors, performance_metrics['oracle_predictions'])
            else:
                if self.verbose >= 1:
                    print("⚠️ No performance metrics calculated")
        
        return True


def mask_fn(env) -> np.ndarray:
    """Action masking function for the environment."""
    return env.unwrapped.action_mask()


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Comprehensive training script for JANI environments',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # JANI model file arguments
    parser.add_argument('--model_file', type=str, required=True,
                       help='Path to the JANI model file (mandatory)')
    parser.add_argument('--start_file', type=str, 
                       help='Path to the start state file')
    parser.add_argument('--goal_file', type=str,
                       help='Path to the goal condition file') 
    parser.add_argument('--safe_file', type=str,
                       help='Path to the safe condition file')
    parser.add_argument('--property_file', type=str,
                       help='Path to the property file (if provided, overrides start/goal/safe files)')

    # overwrite initial state generator to a pure random one
    parser.add_argument('--random_init', action='store_true',
                       help='Use random initial state generator instead of the one defined in the model')
    
    # Training arguments
    parser.add_argument('--total_timesteps', type=int, default=1000000,
                       help='Total number of training timesteps')
    parser.add_argument('--n_envs', type=int, default=4,
                       help='Number of parallel environments')
    parser.add_argument('--eval_freq', type=int, default=10000,
                       help='Evaluation frequency (in timesteps)')
    parser.add_argument('--eval_episodes', type=int, default=10,
                       help='Number of episodes for evaluation')
    
    # Hyperparameter tuning
    parser.add_argument('--tune_hyperparams', action='store_true',
                       help='Enable hyperparameter tuning with Optuna')
    parser.add_argument('--n_trials', type=int, default=50,
                       help='Number of hyperparameter tuning trials')
    parser.add_argument('--study_name', type=str, default=None,
                       help='Optuna study name')
    
    # Logging and saving
    parser.add_argument('--log_dir', type=str, default='./logs',
                       help='Directory for logs and tensorboard')
    parser.add_argument('--model_save_dir', type=str, default='./models',
                       help='Directory to save trained models')
    parser.add_argument('--wandb_project', type=str, default='jani-training',
                       help='Weights & Biases project name')
    parser.add_argument('--wandb_entity', type=str, default=None,
                       help='Weights & Biases entity/team name')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Experiment name for logging')
    
    # Other options
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cpu, cuda, auto)')
    parser.add_argument('--verbose', type=int, default=1,
                       help='Verbosity level')
    
    # Control flags
    parser.add_argument('--disable-wandb', action='store_true',
                       help='Disable Weights & Biases logging')
    parser.add_argument('--disable-eval', action='store_true',
                       help='Disable policy evaluation during training')
    
    # Model loading options
    parser.add_argument('--load_model', type=str, default=None,
                       help='Path to a trained model to load and continue training from')
    parser.add_argument('--no_reset_timesteps', action='store_true',
                       help='Continue from saved timesteps when loading a model (default: reset timesteps)')
    
    # Safety classifier arguments
    parser.add_argument('--use_classifier', action='store_true',
                       help='Enable safety classifier for intermediate rewards')
    parser.add_argument('--classifier_model', type=str, default=None,
                       help='Path to trained safety classifier model (.pth file)')
    parser.add_argument('--safe_reward', type=float, default=0.0,
                       help='Reward for safe states when using classifier (default: 0.0)')
    parser.add_argument('--unsafe_reward', type=float, default=-0.01,
                       help='Reward for unsafe states when using classifier (default: -0.01)')
    parser.add_argument('--debug_classifier', action='store_true',
                       help='Enable classifier debugging mode: check classifier predictions against Tarjan oracle and log accuracy to wandb')
    
    # Classifier monitor callback arguments
    parser.add_argument('--monitor_classifier', action='store_true',
                       help='Enable classifier monitoring callback to evaluate performance against oracle during training')
    parser.add_argument('--monitor_freq', type=int, default=10000,
                       help='Frequency (in timesteps) for classifier monitoring (default: 10000)')
    parser.add_argument('--monitor_episodes', type=int, default=30,
                       help='Number of episodes to run for each monitoring evaluation (default: 30)')
    parser.add_argument('--monitor_csv_output', type=str, default=None,
                       help='Path to CSV file for writing monitoring data (states, actions, safety labels)')
    parser.add_argument('--tune_classifier_online', action='store_true',
                       help='Enable online fine-tuning of classifier using new oracle-labeled data')
    parser.add_argument('--classifier_data_dir', type=str, default=None,
                       help='Directory containing classifier training data (required for online fine-tuning)')
    
    return parser.parse_args()


def validate_file_arguments(args) -> Tuple[Dict[str, str], bool]:
    """Validate and process file arguments."""
    file_args = {}
    
    # Check if model file exists
    if not Path(args.model_file).exists():
        raise FileNotFoundError(f"Model file not found: {args.model_file}")
    file_args['model_file'] = args.model_file
    
    # Validate classifier arguments
    if args.use_classifier:
        if args.classifier_model is None:
            raise ValueError("--classifier_model must be provided when --use_classifier is enabled")
        if not Path(args.classifier_model).exists():
            raise FileNotFoundError(f"Classifier model file not found: {args.classifier_model}")
        print(f"🤖 Safety classifier enabled: {args.classifier_model}")
        print(f"   Safe reward: {args.safe_reward}, Unsafe reward: {args.unsafe_reward}")
    
    # Validate classifier monitoring arguments
    if args.monitor_classifier:
        print(f"📊 State monitoring enabled: freq={args.monitor_freq}, episodes={args.monitor_episodes}")
        if args.monitor_csv_output:
            print(f"   CSV output: {args.monitor_csv_output}")
        
        # Validate fine-tuning arguments (only if classifier is used)
        if args.tune_classifier_online:
            if not args.use_classifier:
                raise ValueError("--tune_classifier_online requires --use_classifier to be enabled")
            if args.classifier_data_dir is None:
                raise ValueError("--classifier_data_dir must be provided when --tune_classifier_online is enabled")
            if not Path(args.classifier_data_dir).exists():
                raise FileNotFoundError(f"Classifier data directory not found: {args.classifier_data_dir}")
            print(f"🔧 Online classifier fine-tuning enabled with data from: {args.classifier_data_dir}")
        
        if args.use_classifier:
            print(f"   Will evaluate classifier performance against oracle")
        else:
            print(f"   Will track unsafe and failure states using oracle only")
    
    # Handle property file vs individual files
    if args.property_file:
        if not Path(args.property_file).exists():
            raise FileNotFoundError(f"Property file not found: {args.property_file}")
        
        # Warn if other files are also provided
        other_files_provided = any([args.start_file, args.goal_file, args.safe_file])
        if other_files_provided:
            warnings.warn(
                "Property file provided along with start/goal/safe files. "
                "Only property file will be used, other files will be ignored.",
                UserWarning
            )
        
        file_args['property_file'] = args.property_file
        if args.random_init:
            file_args['random_init'] = args.random_init
        return file_args, True
    
    else:
        # Check individual files
        required_files = ['start_file', 'goal_file', 'safe_file']
        missing_files = []
        
        for file_type in required_files:
            file_path = getattr(args, file_type)
            if not file_path:
                missing_files.append(file_type)
            elif not Path(file_path).exists():
                raise FileNotFoundError(f"{file_type} not found: {file_path}")
            else:
                file_args[file_type] = file_path
        
        if missing_files:
            raise ValueError(
                f"When property_file is not provided, all of start_file, goal_file, and safe_file must be provided. "
                f"Missing: {', '.join(missing_files)}"
            )
        if args.random_init:
            file_args['random_init'] = args.random_init
        return file_args, False


def add_classifier_args(file_args: Dict[str, str], args) -> Dict[str, str]:
    """Add classifier arguments to file_args."""
    file_args['use_classifier'] = args.use_classifier
    if args.use_classifier:
        file_args['classifier_model'] = args.classifier_model
        file_args['safe_reward'] = args.safe_reward
        file_args['unsafe_reward'] = args.unsafe_reward
    return file_args


def create_eval_file_args(file_args: Dict[str, str]) -> Dict[str, str]:
    """Create file_args for evaluation environment with classifier disabled."""
    eval_args = file_args.copy()
    eval_args['use_classifier'] = False
    # Remove classifier-specific keys to avoid passing them to JaniEnv
    eval_args.pop('classifier_model', None)
    eval_args.pop('safe_reward', None) 
    eval_args.pop('unsafe_reward', None)
    
    # Print message if classifier was originally enabled
    if file_args.get('use_classifier', False):
        print("📊 Evaluation environment: Classifier disabled for true reward assessment")
    
    return eval_args


def create_env(file_args: Dict[str, str], n_envs: int = 1, monitor: bool = True, timelimit: bool = True):
    """Create the training environment."""
    def make_env():
        env = JaniEnv(**file_args)
        if timelimit:
            env = TimeLimit(env, max_episode_steps=2048)
        env = ActionMasker(env, mask_fn)
        if monitor:
            env = Monitor(env)
        return env
    
    if n_envs == 1:
        env = make_env()
    else:
        # Create vectorized environment with ActionMasker wrapper
        env = make_vec_env(make_env, n_envs=n_envs)
        if monitor:
            env = VecMonitor(env)
    
    return env


class EvalCallback(BaseCallback):
    """Custom evaluation callback."""
    def __init__(self, eval_env, eval_freq: int, n_eval_episodes: int, best_model_save_path: Optional[str] = None):
        super().__init__()
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.best_model_save_path = best_model_save_path
        self.best_mean_reward = -float('inf')
        self.last_mean_reward = -float('inf')

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.num_timesteps % self.eval_freq == 0:
            mean_reward, _ = evaluate_policy(self.model, self.eval_env, n_eval_episodes=self.n_eval_episodes)
            self.last_mean_reward = mean_reward
            if self.best_model_save_path is not None:
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    self.model.save(self.best_model_save_path)
            if WANDB_AVAILABLE and wandb.run is not None:
                wandb.log({
                    'eval/mean_reward': mean_reward,
                    'eval/best_mean_reward': self.best_mean_reward,
                    'eval/timesteps': self.num_timesteps
                })
        return True



def suggest_hyperparameters(trial) -> Dict[str, Any]:
    """Suggest hyperparameters for Optuna optimization."""
    return {
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
        'n_steps': trial.suggest_categorical('n_steps', [32, 64, 128, 256, 512, 1024, 2048]),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
        'n_epochs': trial.suggest_int('n_epochs', 3, 30),
        'gamma': trial.suggest_float('gamma', 0.9, 0.9999),
        'gae_lambda': trial.suggest_float('gae_lambda', 0.8, 1.0),
        'clip_range': trial.suggest_float('clip_range', 0.1, 0.4),
        'ent_coef': trial.suggest_float('ent_coef', 1e-8, 1e-1, log=True),
        'vf_coef': trial.suggest_float('vf_coef', 0.1, 1.0),
        'max_grad_norm': trial.suggest_float('max_grad_norm', 0.3, 5.0),
    }


def objective(trial, args, file_args: Dict[str, str]) -> float:
    """Objective function for hyperparameter optimization."""
    # Get suggested hyperparameters
    params = suggest_hyperparameters(trial)
    
    # Create environments
    train_env = create_env(file_args, args.n_envs, monitor=False)
    # For hyperparameter tuning, we always need evaluation regardless of disable_eval flag
    eval_env = create_env(create_eval_file_args(file_args), 1, timelimit=True)  # Disable classifier for evaluation
    
    try:
        # Create model with suggested hyperparameters
        model = MaskablePPO(
            MaskableActorCriticPolicy,
            train_env,
            learning_rate=params['learning_rate'],
            n_steps=params['n_steps'],
            batch_size=params['batch_size'],
            n_epochs=params['n_epochs'],
            gamma=params['gamma'],
            gae_lambda=params['gae_lambda'],
            clip_range=params['clip_range'],
            ent_coef=params['ent_coef'],
            vf_coef=params['vf_coef'],
            max_grad_norm=params['max_grad_norm'],
            verbose=0,
            device=args.device,
            seed=args.seed
        )
        
        # Train for a subset of timesteps for hyperparameter tuning
        tuning_timesteps = min(args.total_timesteps // 4, 100000)
        
        # Evaluation callback
        eval_callback = EvalCallback(
            eval_env,
            eval_freq=max(tuning_timesteps // 10, 1000),
            n_eval_episodes=args.eval_episodes,
            best_model_save_path=None,  # Don't save during tuning
        )
        
        # Train model
        model.learn(total_timesteps=tuning_timesteps, callback=eval_callback)
        
        # Return mean reward from evaluation
        mean_reward = eval_callback.last_mean_reward
        
    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        mean_reward = -float('inf')
    
    finally:
        # Clean up environments
        train_env.close()
        eval_env.close()
    
    return mean_reward


def train_model(args, file_args: Dict[str, str], hyperparams: Optional[Dict[str, Any]] = None):
    """Train the model with given hyperparameters."""
    # Set up logging directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = args.experiment_name or f"jani_training_{timestamp}"
    
    log_dir = Path(args.log_dir) / experiment_name
    model_save_dir = Path(args.model_save_dir) / experiment_name
    log_dir.mkdir(parents=True, exist_ok=True)
    model_save_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize wandb if available and not disabled
    if WANDB_AVAILABLE and not args.disable_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=experiment_name,
            config={
                **vars(args),
                **(hyperparams or {}),
                'file_args': file_args
            }
        )
    
    # Create environments
    print("Creating training environment...")
    train_env = create_env(file_args, args.n_envs, monitor=False, timelimit=True)
    
    # Create evaluation environment only if evaluation is not disabled
    eval_env = None
    if not args.disable_eval:
        eval_env = create_env(create_eval_file_args(file_args), 1, timelimit=True)  # Disable classifier for evaluation
    
    # Default hyperparameters if not provided
    if hyperparams is None:
        hyperparams = {
            'learning_rate': 3e-4,
            'n_steps': 2048,
            'batch_size': 64,
            'n_epochs': 10,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'ent_coef': 0.0,
            'vf_coef': 0.5,
            'max_grad_norm': 0.5,
        }
    
    print(f"Training with hyperparameters: {hyperparams}")
    
    # Create or load model
    if args.load_model:
        if not Path(args.load_model).exists():
            raise FileNotFoundError(f"Model file not found: {args.load_model}")
        print(f"Loading existing model from: {args.load_model}")
        model = MaskablePPO.load(
            args.load_model,
            env=train_env,
            device=args.device,
            **hyperparams
        )
        # Reset environment after loading model
        model.set_env(train_env)
        reset_timesteps = not args.no_reset_timesteps
        print(f"Model loaded. Reset timesteps: {reset_timesteps}")
    else:
        print("Creating new model from scratch")
        model = MaskablePPO(
            MaskableActorCriticPolicy,
            train_env,
            tensorboard_log=str(log_dir),
            verbose=args.verbose,
            device=args.device,
            seed=args.seed,
            **hyperparams
        )
        reset_timesteps = True
    
    # Set up callbacks
    callbacks = []
    eval_callback = None
    
    # Evaluation callback (only if evaluation is not disabled)
    if not args.disable_eval and eval_env is not None:
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(model_save_dir / "best_model"),
            eval_freq=args.eval_freq,
            n_eval_episodes=args.eval_episodes,
        )
        callbacks.append(eval_callback)
    
    # Wandb callback (only if available and not disabled)
    if WANDB_AVAILABLE and not args.disable_wandb:
        wandb_callback = WandbCallback(verbose=args.verbose)
        callbacks.append(wandb_callback)
    
    # Classifier debug callback (only if debug_classifier is enabled and classifier is used)
    if args.debug_classifier and args.use_classifier:
        debug_callback = ClassifierDebugCallback(verbose=args.verbose)
        callbacks.append(debug_callback)
        if args.verbose >= 1:
            print("🔍 Classifier debugging enabled - will compare predictions with Tarjan oracle")
    elif args.debug_classifier and not args.use_classifier:
        print("⚠️ Warning: --debug_classifier enabled but --use_classifier is not. Debug callback will not be added.")
    
    # Classifier monitor callback (if monitor_classifier is enabled)
    if args.monitor_classifier:
        # Create an independent monitoring environment identical to training environment
        print("Creating independent monitoring environment...")
        monitor_env = create_env(file_args, n_envs=1, monitor=False, timelimit=True)
        
        monitor_callback = ClassifierMonitorCallback(
            env=monitor_env,
            monitor_freq=args.monitor_freq,
            num_episodes=args.monitor_episodes,
            tune_classifier=args.tune_classifier_online,
            data_dir=args.classifier_data_dir,
            csv_output_path=args.monitor_csv_output,
            verbose=args.verbose
        )
        callbacks.append(monitor_callback)
        if args.verbose >= 1:
            if args.use_classifier:
                print("📊 Classifier monitoring enabled - will evaluate classifier performance against Tarjan oracle")
                if args.tune_classifier_online:
                    print("🔧 Online classifier fine-tuning enabled")
            else:
                print("📊 State monitoring enabled - will track unsafe and failure states using Tarjan oracle only")
    
    # Early stopping on reward threshold (optional)
    # stop_callback = StopTrainingOnRewardThreshold(reward_threshold=200, verbose=1)
    # callbacks.append(stop_callback)
    
    print(f"Starting training for {args.total_timesteps} timesteps...")
    
    # Train the model
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        tb_log_name="PPO",
        reset_num_timesteps=reset_timesteps
    )
    
    # Save final model
    final_model_path = model_save_dir / "final_model"
    model.save(str(final_model_path))
    print(f"Final model saved to: {final_model_path}")
    
    # Save hyperparameters
    hyperparams_path = model_save_dir / "hyperparameters.json"
    with open(hyperparams_path, 'w') as f:
        json.dump(hyperparams, f, indent=2)
    
    # Clean up
    train_env.close()
    if eval_env is not None:
        eval_env.close()
    if WANDB_AVAILABLE and not args.disable_wandb:
        wandb.finish()
    
    # Return best reward if evaluation was performed, otherwise return 0
    best_reward = eval_callback.best_mean_reward if eval_callback is not None else 0.0
    return model, best_reward


def main():
    """Main training function."""
    # Parse arguments
    args = parse_arguments()
    
    # Set seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Validate file arguments
    try:
        file_args, using_property_file = validate_file_arguments(args)
        file_args = add_classifier_args(file_args, args)
        print(f"Using {'property file' if using_property_file else 'individual files'}: {file_args}")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Set device
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {args.device}")
    
    best_hyperparams = None
    best_reward = -float('inf')
    
    # Hyperparameter tuning
    if args.tune_hyperparams:
        if not OPTUNA_AVAILABLE:
            print("Error: Optuna not available. Cannot perform hyperparameter tuning.")
            print("Install optuna with: pip install optuna")
            print("Proceeding with default hyperparameters...")
        else:
            print(f"Starting hyperparameter tuning with {args.n_trials} trials...")
            
            study_name = args.study_name or f"jani_study_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Initialize wandb for hyperparameter tuning if not disabled
            if WANDB_AVAILABLE and not args.disable_wandb:
                wandb.init(
                    project=f"{args.wandb_project}_tuning",
                    entity=args.wandb_entity,
                    name=f"{study_name}_tuning"
                )
            
            # Create Optuna study
            study = optuna.create_study(
                direction='maximize',
                study_name=study_name,
                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
            )
            
            # Run optimization
            try:
                study.optimize(
                    lambda trial: objective(trial, args, file_args),
                    n_trials=args.n_trials,
                    timeout=None
                )
                
                print("Hyperparameter tuning completed!")
                print(f"Best trial: {study.best_trial.number}")
                print(f"Best reward: {study.best_value}")
                print(f"Best params: {study.best_params}")
                
                best_hyperparams = study.best_params
                best_reward = study.best_value
                
                # Log best hyperparameters to wandb if not disabled
                if WANDB_AVAILABLE and not args.disable_wandb:
                    wandb.log({"best_reward": best_reward})
                    wandb.config.update(best_hyperparams)
                
            except Exception as e:
                print(f"Hyperparameter tuning failed: {e}")
                print("Proceeding with default hyperparameters...")
            
            if WANDB_AVAILABLE and not args.disable_wandb:
                wandb.finish()
    
    # Train final model with best hyperparameters
    print("Training final model...")
    final_model, final_reward = train_model(args, file_args, best_hyperparams)
    
    print(f"Training completed!")
    print(f"Final model reward: {final_reward}")
    if best_hyperparams:
        print(f"Best hyperparameters used: {best_hyperparams}")


if __name__ == "__main__":
    main()