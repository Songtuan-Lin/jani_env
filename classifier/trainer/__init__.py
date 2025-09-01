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
    create_comparison_plots,
    create_comparison_dataframe,
    calculate_improvement_statistics,
    generate_comparison_report
)

# Test summary generation
from .test_summary import (
    generate_test_summary,
    generate_all_benchmarks_summary,
    save_test_summaries,
    create_test_summaries_after_training
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
    'create_comparison_plots',
    'create_comparison_dataframe',
    'calculate_improvement_statistics',
    'generate_comparison_report',
    
    # Test summaries
    'generate_test_summary',
    'generate_all_benchmarks_summary',
    'save_test_summaries',
    'create_test_summaries_after_training'
]