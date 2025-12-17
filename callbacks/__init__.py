""" 
Callback functions for training and evaluating policies using stable-baselines3. 
"""

from dataclasses import dataclass, field
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from typing import Optional
from pathlib import Path

# Optional imports for advanced features
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from stable_baselines3.common.callbacks import BaseCallback
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.utils import get_action_masks


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
            # mean_reward, _ = evaluate_policy(self.model, self.eval_env, n_eval_episodes=self.n_eval_episodes)
            rewards = []
            for i in range(self.n_eval_episodes):
                obs, _ = self.eval_env.reset()
                done = False
                truncated = False
                episode_rewards = 0.0
                while not done and not truncated:
                    action_masks = get_action_masks(self.eval_env)
                    action_masks = np.expand_dims(action_masks, axis=0)  # shape (1, n_actions)
                    action, _ = self.model.predict(obs, action_masks=action_masks)
                    obs, reward, done, truncated, _ = self.eval_env.step(action)
                    episode_rewards += reward
                rewards.append(episode_rewards)
            mean_reward = np.mean(rewards)
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
            self.logger.record('eval/mean_reward', mean_reward)
            self.logger.record('eval/best_mean_reward', self.best_mean_reward)
            self.logger.record('eval/timesteps', self.n_calls)
        return True


class SafetyEvalCallback(BaseCallback):
    """Custom safety evaluation callback."""

    def __init__(self, safety_eval_env, eval_freq: int):
        super().__init__()
        self.safety_eval_env = safety_eval_env
        self.eval_freq = eval_freq

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            init_pool_size = self.safety_eval_env.get_init_state_pool_size()
            safety_rates = []
            for idx in range(init_pool_size):
                obs, _ = self.safety_eval_env.reset(options={"idx": idx})
                done = False
                truncated = False
                unsafe_steps = 0
                total_steps = 0
                while not done and not truncated:
                    action_masks = get_action_masks(self.safety_eval_env)
                    action_masks = np.expand_dims(action_masks, axis=0)  # shape (1, n_actions)
                    action, _ = self.model.predict(obs, action_masks=action_masks)
                    obs, reward, done, truncated, info = self.safety_eval_env.step(action)
                    if reward == -0.01:
                        # reward -0.01 indicates an unsafe step
                        unsafe_steps += 1
                    total_steps += 1
                safety_rate = 1.0 - (unsafe_steps / total_steps) if total_steps > 0 else 1.0
                safety_rates.append(safety_rate)
            safety_rate = np.mean(safety_rates)
            if WANDB_AVAILABLE and wandb.run is not None:
                wandb.log({
                    'safety_eval/safety_rate': safety_rate,
                    'safety_eval/timesteps': self.n_calls
                })
            self.logger.record('safety_eval/safety_rate', safety_rate)
            self.logger.record('safety_eval/timesteps', self.n_calls)
        return True


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