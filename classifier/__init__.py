"""
Classifier module for JANI state safety prediction.

This module provides:
- Neural network models for binary safety classification  
- Model loading and inference utilities for integration
- Training pipeline in classifier.trainer submodule

For standalone training: python -m classifier.train
For integration: from classifier import BasicClassifier, load_trained_model, predict_safety
For training utilities: from classifier.trainer import train_model, create_data_loaders
"""

# Core models for inference and integration
from .models import BasicClassifier, EnhancedClassifier, DynamicClassifier

# Integration utilities
from .utils import load_trained_model, save_model, predict_safety

# Configuration (shared across main and trainer modules)
from . import config

# Training submodule (available as classifier.trainer)
from . import trainer

__all__ = [
    # Models (primary interface for integration)
    'BasicClassifier', 
    'EnhancedClassifier', 
    'DynamicClassifier',
    
    # Integration utilities
    'load_trained_model',
    'save_model', 
    'predict_safety',
    
    # Configuration
    'config',
    
    # Training submodule
    'trainer'
]