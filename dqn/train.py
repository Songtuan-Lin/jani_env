import argparse
import numpy as np

from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

from callbacks import EvalCallback
from jani.env import JANIEnv
from utils import create_env, create_eval_file_args, mask_fn

from .model import MaskedDQN

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
    print(f"🤖 Oracle enabled: {file_args.get('use_oracle', False)}")
    train_env = create_env(file_args, args.n_envs, monitor=False, time_limited=True)
    model = MaskedDQN(
        policy="MlpPolicy",
        env=train_env,
        gradient_steps=10,
        use_mask=args.use_mask,
        verbose=1
    )

    callbacks = []
    # Create evaluation environment
    eval_file_args = create_eval_file_args(file_args)
    print("Creating evaluation environment...")
    eval_env = create_env(eval_file_args, n_envs=1, monitor=True, time_limited=True)
    eval_callback = EvalCallback(
        eval_env=eval_env,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        best_model_save_path=str(model_save_dir / "best_model")
    )
    callbacks.append(eval_callback)

    # Start training
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        tb_log_name="DQN"
    )


def main():
    parser = argparse.ArgumentParser(description="Train Masked PPO on JANI Environments")
    parser.add_argument('--jani_model', type=str, required=True, help="Path to the JANI model file.")
    parser.add_argument('--jani_property', type=str, default="", help="Path to the JANI property file.")
    parser.add_argument('--start_states', type=str, default="", help="Path to the start states file.")
    parser.add_argument('--objective', type=str, default="", help="Path to the objective file.")
    parser.add_argument('--failure_property', type=str, default="", help="Path to the failure property file.")
    parser.add_argument('--goal_reward', type=float, default=1.0, help="Reward for reaching the goal.")
    parser.add_argument('--failure_reward', type=float, default=-1.0, help="Reward for reaching failure state.")
    parser.add_argument('--use_oracle', action='store_true', help="Use Tarjan oracle for unsafe state detection.")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument('--total_timesteps', type=int, default=1_000_000, help="Total timesteps for training.")
    parser.add_argument('--n_envs', type=int, default=1, help="Number of parallel environments.")
    parser.add_argument('--max_steps', type=int, default=1000, help="Max steps per episode.")
    parser.add_argument('--log_dir', type=str, default="./logs", help="Directory for logging.")
    parser.add_argument('--model_save_dir', type=str, default="./models", help="Directory to save models.")
    parser.add_argument('--eval_freq', type=int, default=2048, help="Evaluation frequency in timesteps.")
    parser.add_argument('--n_eval_episodes', type=int, default=50, help="Number of episodes for each evaluation.")
    parser.add_argument('--experiment_name', type=str, default="", help="Name of the experiment.")
    parser.add_argument('--verbose', type=int, default=1, help="Verbosity level.")
    parser.add_argument('--use_mask', action='store_true', help="Use action masking during training.")
    parser.add_argument('--device', type=str, default='auto', help="Device to use for training (cpu or cuda).")
    parser.add_argument('--disable_wandb', action='store_true', help="Disable Weights & Biases logging.")
    parser.add_argument('--wandb_project', type=str, default="jani_rl", help="Weights & Biases project name.")
    parser.add_argument('--wandb_entity', type=str, default=None, help="Weights & Biases entity name.")

    args = parser.parse_args()

    file_args = {
        'jani_model': args.jani_model,
        'jani_property': args.jani_property,
        'start_states': args.start_states,
        'objective': args.objective,
        'failure_property': args.failure_property,
        'goal_reward': args.goal_reward,
        'failure_reward': args.failure_reward,
        'seed': args.seed,
        'use_oracle': args.use_oracle,
        'max_steps': args.max_steps
    }

    train_model(args, file_args)

if __name__ == "__main__":
    main()