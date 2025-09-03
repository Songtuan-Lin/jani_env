"""
Simplified classifier module for state safety prediction.

This module provides:
- Simple neural network model for binary safety classification  
- Data loading utilities for CSV datasets
- Training script with Optuna hyperparameter tuning

Usage: python -m classifier.train --data-dir /path/to/data
"""

# Core model
from .models import Classifier

# Data loading utilities  
from .data_loader import load_datasets, create_dataloaders, get_input_size

# Model loading utilities
from .utils import load_trained_model, predict

__all__ = [
    'Classifier',
    'load_datasets', 
    'create_dataloaders',
    'get_input_size',
    'load_trained_model',
    'predict'
]