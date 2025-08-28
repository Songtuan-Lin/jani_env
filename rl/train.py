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
    
    # Safety classifier arguments
    parser.add_argument('--use_classifier', action='store_true',
                       help='Enable safety classifier for intermediate rewards')
    parser.add_argument('--classifier_model', type=str, default=None,
                       help='Path to trained safety classifier model (.pth file)')
    parser.add_argument('--safe_reward', type=float, default=0.0,
                       help='Reward for safe states when using classifier (default: 0.0)')
    parser.add_argument('--unsafe_reward', type=float, default=-0.01,
                       help='Reward for unsafe states when using classifier (default: -0.01)')
    
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


def create_env(file_args: Dict[str, str], n_envs: int = 1, monitor: bool = True):
    """Create the training environment."""
    def make_env():
        env = JaniEnv(**file_args)
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
    eval_env = create_env(create_eval_file_args(file_args), 1)  # Disable classifier for evaluation
    
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
    train_env = create_env(file_args, args.n_envs, monitor=False)
    
    # Create evaluation environment only if evaluation is not disabled
    eval_env = None
    if not args.disable_eval:
        eval_env = create_env(create_eval_file_args(file_args), 1)  # Disable classifier for evaluation
    
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
    
    # Create model
    model = MaskablePPO(
        MaskableActorCriticPolicy,
        train_env,
        tensorboard_log=str(log_dir),
        verbose=args.verbose,
        device=args.device,
        seed=args.seed,
        **hyperparams
    )
    
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
    
    # Early stopping on reward threshold (optional)
    # stop_callback = StopTrainingOnRewardThreshold(reward_threshold=200, verbose=1)
    # callbacks.append(stop_callback)
    
    print(f"Starting training for {args.total_timesteps} timesteps...")
    
    # Train the model
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        tb_log_name="PPO"
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