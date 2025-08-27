"""
Configuration file for classifier training and hyperparameter tuning.
"""

import torch

# Device configuration with Apple Silicon MPS support
def get_device():
    """Get the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def get_device_info():
    """Get information about available devices."""
    device = get_device()
    info = {'device': device, 'type': device.type}
    
    if device.type == 'cuda':
        info['count'] = torch.cuda.device_count()
        info['name'] = torch.cuda.get_device_name()
        info['memory'] = torch.cuda.get_device_properties(0).total_memory
    elif device.type == 'mps':
        info['count'] = 1  # MPS typically has one device
        info['name'] = 'Apple Silicon GPU'
        info['memory'] = None  # Memory info not directly available for MPS
    else:
        info['count'] = 0
        info['name'] = 'CPU'
        info['memory'] = None
    
    return info

DEVICE = get_device()
DEVICE_INFO = get_device_info()
NUM_GPUS = DEVICE_INFO['count'] if DEVICE.type in ['cuda', 'mps'] else 0

# Data paths (legacy - no longer used)
# DATA_WITH_NEXT_STATE = "dataset_with_next_state"  
# DATA_WITHOUT_NEXT_STATE = "dataset_without_next_state"

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

# Multi-device training configuration
MAX_CONCURRENT_JOBS = NUM_GPUS if NUM_GPUS > 0 else 1
GPU_MEMORY_FRACTION = 0.9  # Use 90% of GPU memory per job
DISTRIBUTED_BACKEND = "nccl" if DEVICE.type == 'cuda' else "gloo"  # NCCL for CUDA, Gloo for others

