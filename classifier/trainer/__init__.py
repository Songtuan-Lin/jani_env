"""
Training submodule for classifier models.

This submodule provides:
- Training utilities and pipelines
- Data loading and preprocessing for training
- Hyperparameter optimization with Optuna
- Multi-GPU training scheduling
- Results comparison and analysis

For training: from classifier.trainer import train_model, create_data_loaders
For optimization: from classifier.trainer import run_hyperparameter_tuning
"""

# Core training functionality
from .trainer import train_model, evaluate_model, EarlyStopping

# Data handling for training
from .data_loader import (
    SafetyDataset, 
    load_dataset, 
    create_data_loaders, 
    get_available_benchmarks,
    get_input_size
)

# Advanced training features
from .hyperparameter_tuning import (
    run_hyperparameter_tuning,
    save_tuning_results,
    OptunaTuner
)

# Multi-GPU support
from .gpu_scheduler import GPUScheduler, TrainingJob

# Analysis and comparison
from .comparison import (
    load_results,
    run_full_comparison, 
    plot_performance_comparison,
    calculate_statistical_significance
)

__all__ = [
    # Training core
    'train_model',
    'evaluate_model', 
    'EarlyStopping',
    
    # Data handling
    'SafetyDataset',
    'load_dataset',
    'create_data_loaders',
    'get_available_benchmarks',
    'get_input_size',
    
    # Hyperparameter tuning
    'run_hyperparameter_tuning',
    'save_tuning_results',
    'OptunaTuner',
    
    # Multi-GPU
    'GPUScheduler',
    'TrainingJob',
    
    # Analysis
    'load_results',
    'run_full_comparison',
    'plot_performance_comparison',
    'calculate_statistical_significance'
]