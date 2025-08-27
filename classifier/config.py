"""
Configuration file for classifier training and hyperparameter tuning.
"""

import torch

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0

# Data paths
DATA_WITH_NEXT_STATE = "dataset_with_next_state"
DATA_WITHOUT_NEXT_STATE = "dataset_without_next_state"

# Training configuration
RANDOM_SEED = 42
BATCH_SIZE = 64
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10
LEARNING_RATE = 0.001

# Hyperparameter tuning configuration
N_TRIALS = 50
OPTUNA_STUDY_NAME = "classifier_comparison"

# Model architecture search space
HIDDEN_LAYERS_RANGE = (1, 4)
HIDDEN_SIZE_RANGE = (32, 512)
DROPOUT_RANGE = (0.1, 0.5)
LEARNING_RATE_RANGE = (1e-5, 1e-2)

# Results directory
RESULTS_DIR = "training_results"
MODELS_DIR = "trained_models"
PLOTS_DIR = "plots"

# Comparison metrics
CLASSIFICATION_METRICS = ["accuracy", "precision", "recall", "f1_score", "auc_roc"]

# Task types
TASK_CLASSIFICATION = "classification"

# Multi-GPU training configuration
MAX_CONCURRENT_JOBS = NUM_GPUS if NUM_GPUS > 0 else 1
GPU_MEMORY_FRACTION = 0.9  # Use 90% of GPU memory per job
DISTRIBUTED_BACKEND = "nccl"  # For multi-GPU training

