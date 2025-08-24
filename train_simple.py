#!/usr/bin/env python3
"""
Simple training script for JANI environments.

This script provides basic training functionality without requiring
optional dependencies like Optuna or Weights & Biases.
"""

import argparse
import sys
import warnings
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

import numpy as np
import torch

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecMonitor

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker

from jani_env import JaniEnv


class TrainingCallback(BaseCallback):
    """Simple callback for training metrics logging."""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _on_step(self) -> bool:
        # Track training environment rewards
        if hasattr(self.model, 'ep_info_buffer') and self.model.ep_info_buffer:
            if len(self.model.ep_info_buffer) > 0:
                ep_info = self.model.ep_info_buffer[-1]
                self.episode_rewards.append(ep_info['r'])
                self.episode_lengths.append(ep_info['l'])
                
                # Print training progress every 100 episodes
                if len(self.episode_rewards) % 100 == 0:
                    recent_rewards = self.episode_rewards[-100:]
                    print(f"Episodes: {len(self.episode_rewards)}, "
                          f"Mean reward (last 100): {np.mean(recent_rewards):.2f}, "
                          f"Timesteps: {self.num_timesteps}")
        
        return True


def mask_fn(env) -> np.ndarray:
    """Action masking function for the environment."""
    return env.unwrapped.action_mask()


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Simple training script for JANI environments',
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
    
    # Training arguments
    parser.add_argument('--total_timesteps', type=int, default=100000,
                       help='Total number of training timesteps')
    parser.add_argument('--n_envs', type=int, default=4,
                       help='Number of parallel environments')
    parser.add_argument('--eval_freq', type=int, default=10000,
                       help='Evaluation frequency (in timesteps)')
    parser.add_argument('--eval_episodes', type=int, default=5,
                       help='Number of episodes for evaluation')
    
    # Hyperparameters
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--n_steps', type=int, default=2048,
                       help='Number of steps per update')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--n_epochs', type=int, default=10,
                       help='Number of epochs per update')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor')
    
    # Logging and saving
    parser.add_argument('--log_dir', type=str, default='./logs',
                       help='Directory for logs and tensorboard')
    parser.add_argument('--model_save_dir', type=str, default='./models',
                       help='Directory to save trained models')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Experiment name for logging')
    
    # Other options
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cpu, cuda, auto)')
    parser.add_argument('--verbose', type=int, default=1,
                       help='Verbosity level')
    
    return parser.parse_args()


def validate_file_arguments(args) -> Dict[str, str]:
    """Validate and process file arguments."""
    file_args = {}
    
    # Check if model file exists
    if not Path(args.model_file).exists():
        raise FileNotFoundError(f"Model file not found: {args.model_file}")
    file_args['model_file'] = args.model_file
    
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
    
    return file_args


def create_env(file_args: Dict[str, str], n_envs: int = 1):
    """Create the training environment."""
    def make_env():
        env = JaniEnv(**file_args)
        env = Monitor(env)
        env = ActionMasker(env, mask_fn)
        return env
    
    if n_envs == 1:
        env = make_env()
    else:
        env = make_vec_env(make_env, n_envs=n_envs)
        env = VecMonitor(env)
    
    return env


def train_model(args, file_args: Dict[str, str]):
    """Train the model."""
    # Set up logging directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = args.experiment_name or f"jani_training_{timestamp}"
    
    log_dir = Path(args.log_dir) / experiment_name
    model_save_dir = Path(args.model_save_dir) / experiment_name
    log_dir.mkdir(parents=True, exist_ok=True)
    model_save_dir.mkdir(parents=True, exist_ok=True)
    
    print("Creating training environment...")
    train_env = create_env(file_args, args.n_envs)
    eval_env = create_env(file_args, 1)
    
    # Hyperparameters
    hyperparams = {
        'learning_rate': args.learning_rate,
        'n_steps': args.n_steps,
        'batch_size': args.batch_size,
        'n_epochs': args.n_epochs,
        'gamma': args.gamma,
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
    
    # Training callback for progress logging
    training_callback = TrainingCallback(verbose=args.verbose)
    
    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(model_save_dir / "best_model"),
        log_path=str(log_dir / "eval"),
        eval_freq=args.eval_freq,
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        render=False,
        verbose=args.verbose
    )
    
    print(f"Starting training for {args.total_timesteps} timesteps...")
    
    # Train the model
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[training_callback, eval_callback],
        tb_log_name="MaskablePPO"
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
    eval_env.close()
    
    return model, eval_callback.best_mean_reward


def main():
    """Main training function."""
    # Parse arguments
    args = parse_arguments()
    
    # Set seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Validate file arguments
    try:
        file_args = validate_file_arguments(args)
        print(f"Using JANI files: {file_args}")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Set device
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {args.device}")
    
    # Train model
    print("Training model...")
    model, final_reward = train_model(args, file_args)
    
    print(f"Training completed!")
    print(f"Final model reward: {final_reward}")


if __name__ == "__main__":
    main()