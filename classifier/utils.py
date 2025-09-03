"""
Simple utilities for loading trained models.
"""

import torch
import numpy as np
from pathlib import Path
from .models import Classifier


def load_trained_model(model_path, device=None):
    """
    Load a trained classifier model.
    
    Args:
        model_path: Path to the saved model file (.pth)
        device: Device to load the model on (auto-detected if None)
    
    Returns:
        tuple: (model, scaler) where model is ready for inference and scaler is the fitted StandardScaler
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Extract model parameters
    input_size = checkpoint['input_size']
    hidden_sizes = checkpoint['hidden_sizes']
    dropout = checkpoint['dropout']
    scaler = checkpoint.get('scaler', None)  # Get scaler if available
    
    # Create model
    model = Classifier(input_size, hidden_sizes, dropout)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, scaler


def predict(model, features, scaler=None, device=None):
    """
    Predict safety for given features.
    
    Args:
        model: Trained classifier model
        features: Feature vector(s) - can be:
                 - Single sample: 1D array/tensor of shape (n_features,)
                 - Multiple samples: 2D array/tensor of shape (n_samples, n_features)
        scaler: StandardScaler used during training (None if no normalization)
        device: Device for computation
    
    Returns:
        If single sample: tuple (probability, is_safe)
        If multiple samples: tuple (probabilities_array, is_safe_array)
    """
    if device is None:
        device = next(model.parameters()).device
    
    # Convert to numpy for scaler if needed
    if isinstance(features, torch.Tensor):
        features_np = features.cpu().numpy()
    else:
        features_np = features.copy()
    
    # Remember original shape to determine return format
    single_sample = features_np.ndim == 1
    
    # Ensure 2D for processing
    if single_sample:
        features_np = features_np.reshape(1, -1)
    
    # Apply normalization if scaler is provided
    if scaler is not None:
        features_np = scaler.transform(features_np)
    
    # Convert to tensor
    features_tensor = torch.from_numpy(features_np).float().to(device)
    
    # Make predictions
    model.eval()
    with torch.no_grad():
        outputs = model(features_tensor)
        probabilities = outputs.cpu().numpy()
        is_safe = probabilities > 0.5
    
    # Return format based on input
    if single_sample:
        return probabilities[0], is_safe[0]
    else:
        return probabilities, is_safe


