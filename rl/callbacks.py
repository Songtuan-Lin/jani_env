"""
Custom callback classes for RL training in JANI environment.

This module contains various callback classes used during training:
- WandbCallback: Weights & Biases logging
- ClassifierDebugCallback: Debug classifier predictions against oracle
- ClassifierMonitorCallback: Monitor classifier performance during training
- EvalCallback: Custom evaluation callback
"""

from dataclasses import dataclass, field
import numpy as np
import torch
from pathlib import Path
from typing import Optional

# Optional imports for advanced features
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from stable_baselines3.common.callbacks import BaseCallback
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.utils import get_action_masks

from jani.oracle import TarjanOracle


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
            if self.n_calls % 100 == 0 and len(self.episode_rewards) > 0:
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
        if (self.n_calls - self.last_logged_timestep) >= 100:
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
                        'debug/timesteps': self.n_calls
                    })
                
                if self.verbose >= 1:
                    print(f"🎯 Classifier accuracy: {step_accuracy:.3f} (overall), {recent_accuracy:.3f} (recent 100 steps)")
                    
            self.last_logged_timestep = self.num_timesteps
            
        return True


class ClassifierMonitorCallback(BaseCallback):

    @dataclass
    class EpisodeData:
        state_vectors: list = field(default_factory=list)
        state_objects: list = field(default_factory=list)
        actions: list = field(default_factory=list)
        safety_labels: list = field(default_factory=list)  # True if safe, False if unsafe
        failure_states_count: int = 0

    '''Custom callback to monitor classifier's information during training.'''
    def __init__(self, env, monitor_freq: int = 10000, num_episodes: int = 30, tune_classifier: bool = False, data_dir: str = None, csv_output_path: str = None, model_save_dir: str = None, verbose: int = 0):
        super().__init__()
        self.env = env
        self.monitor_freq = monitor_freq
        self.num_episodes = num_episodes
        self.tune_classifier = tune_classifier
        self.data_dir = data_dir
        self.csv_output_path = csv_output_path
        self.model_save_dir = model_save_dir
        self.verbose = verbose
        if self.tune_classifier and self.data_dir is None:
            raise ValueError("data_dir must be provided if tune_classifier is True")
        
        # Initialize components
        self.oracle = None
        self.classifier = None
        self.scaler = None
        self.jani_model = None

        self.episodes = []  # Store data for each episode
        self.cached_oracle_results = {}  # Cache oracle results to avoid redundant calls
        
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
        episodes = []
        failure_states_count = 0  # Track number of failure states reached
        
        # Run episodes using the current policy
        for episode in range(self.num_episodes):
            # Initialize episode data
            episode_data = self.EpisodeData()

            obs, _ = self.env.reset()
            done = False
            truncated = False
            
            # Store initial state (no action taken yet, use None)
            jani_env = self._unwrap_to_jani_env(self.env)
            current_state_obj = jani_env.get_state_repr()
            current_state_vector = current_state_obj.to_vector()

            # Store initial state to episode data
            episode_data.state_vectors.append(current_state_vector)
            episode_data.state_objects.append(current_state_obj)
            
            # Check if initial state is a failure state
            if self.jani_model.failure_reached(current_state_obj):
                failure_states_count += 1

            while True:
                # Get action masks for MaskablePPO
                action_masks = get_action_masks(self.env)
                
                # Get action from model with action masks
                action, _ = self.model.predict(obs, action_masks=action_masks, deterministic=True)
                # (state, action taken in the state)
                episode_data.actions.append(int(action))  # Store action taken in episode data
                obs, reward, done, truncated, info = self.env.step(action)

                current_state_obj = jani_env.get_state_repr()
                current_state_vector = current_state_obj.to_vector()

                if done or truncated:
                    if self.jani_model.failure_reached(current_state_obj):
                        failure_states_count += 1
                    # If it is the last state, we do not need to store it
                    break

                episode_data.state_vectors.append(current_state_vector)
                episode_data.state_objects.append(current_state_obj)

            episodes.append(episode_data)

        return episodes, failure_states_count
    
    def _evaluate_classifier_performance(self, episodes):
        """Evaluate classifier performance against oracle on given states."""
        if not episodes or self.oracle is None:
            return {}
        
        import numpy as np
        state_vectors = []
        oracle_predictions = []
        classifier_predictions = []

        seen_states = set()  # To avoid duplicate evaluations
        
        # Always get oracle predictions
        for episode_data in episodes:
            for (state_obj, state_vec) in zip(episode_data.state_objects, episode_data.state_vectors):
                try:
                    if state_obj in seen_states:
                        assert state_obj in self.cached_oracle_results, "State in seen_states but not in cached results"
                    if state_obj in self.cached_oracle_results:
                        oracle_is_safe = self.cached_oracle_results[state_obj]
                    else:
                        oracle_is_safe = self.oracle.is_safe(state_obj)
                        self.cached_oracle_results[state_obj] = oracle_is_safe
                    if state_obj not in seen_states:
                        oracle_predictions.append(oracle_is_safe)
                    episode_data.safety_labels.append(oracle_is_safe)
                    # if classifier exists, get its prediction too
                    if self.classifier is not None:
                        state_array = np.array(state_vec, dtype=np.float32).reshape(1, -1)
                        from classifier import predict
                        _, classifier_is_safe = predict(self.classifier, state_array, self.scaler)
                        if state_obj not in seen_states:
                            classifier_predictions.append(bool(classifier_is_safe))
                            state_vectors.append(state_vec)
                    seen_states.add(state_obj)
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
        
        assert len(classifier_predictions) == len(oracle_predictions), "Mismatch in oracle and classifier predictions lengths"
        
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
            'state_vectors': state_vectors,
            'oracle_predictions': oracle_predictions.tolist(),
            'classifier_predictions': classifier_predictions.tolist()
        }
    
    def _update_env_classifier(self, fine_tuned_model, new_scaler):
        """Update the classifier in the environment with the fine-tuned model."""
        try:
            # Update classifier in the monitoring environment
            if hasattr(self.env, 'envs'):
                # Vectorized environment - update all environments
                for env in self.env.envs:
                    unwrapped_env = self._unwrap_to_jani_env(env)
                    if hasattr(unwrapped_env, '_classifier'):
                        unwrapped_env._classifier = fine_tuned_model
                    if hasattr(unwrapped_env, '_scaler'):
                        unwrapped_env._scaler = new_scaler
            else:
                # Single environment
                unwrapped_env = self._unwrap_to_jani_env(self.env)
                if hasattr(unwrapped_env, '_classifier'):
                    unwrapped_env._classifier = fine_tuned_model
                if hasattr(unwrapped_env, '_scaler'):
                    unwrapped_env._scaler = new_scaler
            
            # Also update the training environment (should be available via BaseCallback)
            if not hasattr(self, 'training_env') or self.training_env is None:
                raise ValueError("training_env not available in callback - this should be set by BaseCallback")
            
            # Update the actual training environment that the agent uses
            if hasattr(self.training_env, 'envs'):
                # Vectorized training environment
                for env in self.training_env.envs:
                    unwrapped_env = self._unwrap_to_jani_env(env)
                    if hasattr(unwrapped_env, '_classifier'):
                        unwrapped_env._classifier = fine_tuned_model
                    if hasattr(unwrapped_env, '_scaler'):
                        unwrapped_env._scaler = new_scaler
            else:
                # Single training environment
                unwrapped_env = self._unwrap_to_jani_env(self.training_env)
                if hasattr(unwrapped_env, '_classifier'):
                    unwrapped_env._classifier = fine_tuned_model
                if hasattr(unwrapped_env, '_scaler'):
                    unwrapped_env._scaler = new_scaler
            
            if self.verbose >= 2:
                print("🔄 Environment classifier updated with fine-tuned model")
                
        except Exception as e:
            if self.verbose >= 1:
                print(f"⚠️ Error updating environment classifier: {e}")
    
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
                dropout = self.classifier.dropout.p
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
            
            # Update the classifier in the environment as well
            self._update_env_classifier(fine_tuned_model, new_scaler)
            
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
                if key not in ['oracle_predictions', 'classifier_predictions', 'state_vectors']:  # Skip raw predictions
                    log_dict[f'classifier_monitor/{key}'] = value
            
            log_dict['classifier_monitor/timesteps'] = self.n_calls
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
    
    def _count_repeated_unsafe_actions(self):
        if hasattr(self.env, 'envs'):
            # Vectorized environment - get first individual environment
            first_env = self._unwrap_to_jani_env(self.env.envs[0])
        else:
            # Single environment
            first_env = self._unwrap_to_jani_env(self.env)
        visited_pairs = set()
        num_previously_unsafe_actions = 0
        num_repeated_unsafe_actions = 0
        num_actions_to_safe = 0
        num_actions_to_unsafe = 0
        for episode_data in self.episodes:
            for idx in range(len(episode_data.state_vectors) - 1):
                if episode_data.safety_labels[idx + 1]:
                    continue  # Next state is safe, skip
                state_vec = episode_data.state_vectors[idx]
                state_obj = episode_data.state_objects[idx]
                original_action = episode_data.actions[idx]
                if (state_obj, original_action) in visited_pairs:
                    continue  # Already evaluated this state-action pair
                visited_pairs.add((state_obj, original_action))
                num_previously_unsafe_actions += 1
                # Get model's predicted action for this state
                action_mask = first_env.action_mask_under_state(state_obj)
                predicted_action, _ = self.model.predict(np.array(state_vec), action_masks=action_mask)

                if predicted_action == original_action:
                    num_repeated_unsafe_actions += 1
                else:
                    predicted_action_obj = self.jani_model.get_action(predicted_action)
                    successors = self.jani_model.get_successors(state_obj, predicted_action_obj)
                    is_safe_action = True
                    for next_state_obj in successors:
                        if next_state_obj in self.cached_oracle_results:
                            is_safe = self.cached_oracle_results[next_state_obj]
                        else:
                            is_safe = self.oracle.is_safe(next_state_obj)
                            self.cached_oracle_results[next_state_obj] = is_safe
                        if not is_safe:
                            is_safe_action = False
                            break
                    if is_safe_action:
                        num_actions_to_safe += 1
                    else:
                        num_actions_to_unsafe += 1

        return {
            "num_repeated_unsafe_actions": num_repeated_unsafe_actions,
            "num_actions_to_safe": num_actions_to_safe,
            "num_actions_to_unsafe": num_actions_to_unsafe,
            "ratio_repeated_unsafe_actions": num_repeated_unsafe_actions / num_previously_unsafe_actions if num_previously_unsafe_actions > 0 else 0.0,
            "ratio_actions_to_safe": num_actions_to_safe / num_previously_unsafe_actions if num_previously_unsafe_actions > 0 else 0.0,
            "ratio_actions_to_unsafe": num_actions_to_unsafe / num_previously_unsafe_actions if num_previously_unsafe_actions > 0 else 0.0
        }

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
    
    def _save_policy_model(self):
        """Save the current policy model with timestep information."""
        if self.model_save_dir is None:
            return
        
        try:
            from pathlib import Path
            
            # Create save directory if it doesn't exist
            save_dir = Path(self.model_save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model with timestep in filename
            model_path = save_dir / f"policy_model_timestep_{self.num_timesteps}"
            self.model.save(str(model_path))
            
            if self.verbose >= 1:
                print(f"💾 Policy model saved: {model_path}")
                
        except Exception as e:
            if self.verbose >= 1:
                print(f"⚠️ Error saving policy model: {e}")
    
    def _on_step(self) -> bool:
        """Monitor classifier performance at specified frequency."""
        if self.n_calls % self.monitor_freq == 0 or self.n_calls == 1:
            if self.verbose >= 1:
                print(f"🔍 Monitoring classifier at timestep {self.n_calls}")

            # Save the current policy model
            # self._save_policy_model()
            
            # Run episodes and collect state vectors, objects, actions, and failure states count
            episodes, failure_states_count = self._run_episodes_and_collect_states()
            
            if self.verbose >= 1:
                # print(f"📊 Collected {len(state_vectors)} unique states from {self.num_episodes} episodes")
                print(f"❌ Failure states reached: {failure_states_count}")
                # Count actions (excluding None for initial states)
            
            # Evaluate classifier performance against oracle
            performance_metrics = self._evaluate_classifier_performance(episodes)
            
            if performance_metrics:
                unsafe_states_count = performance_metrics['unsafe_states']
                total_states = performance_metrics['total_states']
                
                # Add failure states count to performance metrics
                # performance_metrics['failure_states_reached'] = failure_states_count
                
                # Calculate ratio of unsafe states to total states
                unsafe_ratio = unsafe_states_count / total_states if total_states > 0 else 0.0
                performance_metrics['unsafe_ratio'] = unsafe_ratio
                
                # Calculate number of unsafe states that were previously visited
                # Do this BEFORE updating global storage to avoid counting current states as "previously visited"
                comparision_against_history = self._count_repeated_unsafe_actions()
                performance_metrics.update(comparision_against_history)
                # previously_visited_unsafe_count = self._count_previously_visited_unsafe_states(state_vectors, oracle_predictions)
                # performance_metrics['previously_visited_unsafe_states'] = previously_visited_unsafe_count
            
            # Update global storage AFTER counting previously visited states
            self.episodes += episodes  # Store episodes for future reference
            # self._update_global_storage(state_vectors, state_objects, actions)
            
            if performance_metrics:
                if self.verbose >= 1:
                    if self.classifier is not None:
                        # With classifier: show classifier performance metrics
                        print(f"🎯 Classifier accuracy: {performance_metrics['accuracy']:.3f}")
                        print(f"   Precision: {performance_metrics['precision']:.3f}, Recall: {performance_metrics['recall']:.3f}")
                        print(f"   F1-score: {performance_metrics['f1_score']:.3f}")
                    
                    # Always show oracle-based metrics
                    unsafe_ratio = performance_metrics['unsafe_ratio']
                    
                    print(f"   Total states: {performance_metrics['total_states']}, Safe: {performance_metrics['safe_states']}, Unsafe: {unsafe_states_count}")
                    print(f"⚠️  Unsafe states encountered this callback: {unsafe_states_count}")
                    print(f"📊 Unsafe ratio: {unsafe_ratio:.3f} ({unsafe_states_count}/{total_states})")
                
                # Log to wandb (includes failure_states_reached and available metrics)
                self._log_to_wandb(performance_metrics)
                
                # Write to CSV if path is provided
                # self._write_to_csv(state_vectors, actions, performance_metrics['oracle_predictions'])
                
                # Fine-tune classifier if requested
                if self.tune_classifier:
                    self._fine_tune_classifier(performance_metrics['state_vectors'], performance_metrics['oracle_predictions'])
            else:
                if self.verbose >= 1:
                    print("⚠️ No performance metrics calculated")
        
        return True
    

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
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
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
                    'eval/timesteps': self.n_calls
                })
        return True