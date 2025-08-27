import pandas as pd
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Union
from .models import BasicClassifier, EnhancedClassifier, DynamicClassifier
from . import config


def get_available_gpus():
    """Get list of available GPU device IDs."""
    if torch.cuda.is_available():
        return list(range(torch.cuda.device_count()))
    return []


def get_gpu_memory_info(device_id=None):
    """Get GPU memory information."""
    if not torch.cuda.is_available():
        return None

    if device_id is None:
        device_id = torch.cuda.current_device()

    # Validate device_id exists
    if device_id >= torch.cuda.device_count():
        return None

    try:
        total_memory = torch.cuda.get_device_properties(device_id).total_memory
        allocated_memory = torch.cuda.memory_allocated(device_id)
        free_memory = total_memory - allocated_memory

        return {
            'total': total_memory,
            'allocated': allocated_memory,
            'free': free_memory,
            'device_id': device_id
        }
    except Exception:
        return None


def validate_gpu_device(device_id):
    """Validate that a GPU device ID is accessible."""
    if not torch.cuda.is_available():
        return False

    if device_id >= torch.cuda.device_count():
        return False

    try:
        # Try to access the device
        current_device = torch.cuda.current_device()
        torch.cuda.set_device(device_id)
        torch.cuda.current_device()
        # Restore original device
        torch.cuda.set_device(current_device)
        return True
    except Exception:
        return False


def load_trained_model(model_path: Union[str, Path], 
                      model_type: str = 'basic',
                      input_size: Optional[int] = None,
                      device: Optional[torch.device] = None) -> torch.nn.Module:
    """
    Load a trained classifier model.
    
    Args:
        model_path: Path to the saved model file
        model_type: Type of model ('basic', 'enhanced', 'dynamic')
        input_size: Input size for the model (required for basic/enhanced models)
        device: Device to load the model on
    
    Returns:
        Loaded model ready for inference
    """
    if device is None:
        device = config.DEVICE
    
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load model state and metadata
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract model parameters from checkpoint if available
    if 'model_params' in checkpoint:
        model_params = checkpoint['model_params']
        input_size = model_params.get('input_size', input_size)
    
    if input_size is None:
        raise ValueError("input_size must be provided or stored in checkpoint")
    
    # Initialize model based on type
    if model_type == 'basic':
        model = BasicClassifier(input_size=input_size)
    elif model_type == 'enhanced':
        model = EnhancedClassifier(input_size=input_size)
    elif model_type == 'dynamic':
        # Dynamic model requires architecture parameters
        if 'model_params' not in checkpoint:
            raise ValueError("Dynamic model requires architecture parameters in checkpoint")
        model = DynamicClassifier(**checkpoint['model_params'])
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load model state
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    return model


def save_model(model: torch.nn.Module, 
               model_path: Union[str, Path],
               model_params: Optional[dict] = None,
               training_info: Optional[dict] = None):
    """
    Save a trained model with metadata.
    
    Args:
        model: The trained model to save
        model_path: Path to save the model
        model_params: Model architecture parameters
        training_info: Training metadata (loss, metrics, etc.)
    """
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
    }
    
    if model_params:
        checkpoint['model_params'] = model_params
    
    if training_info:
        checkpoint['training_info'] = training_info
    
    torch.save(checkpoint, model_path)


def predict_safety(model: torch.nn.Module, 
                  state_features: Union[torch.Tensor, np.ndarray],
                  device: Optional[torch.device] = None) -> float:
    """
    Predict safety probability for a single state.
    
    Args:
        model: Trained classifier model
        state_features: State feature vector
        device: Device for computation
    
    Returns:
        Safety probability (0-1, where 1 = safe)
    """
    if device is None:
        device = config.DEVICE
    
    model.eval()
    
    # Convert to tensor if needed
    if isinstance(state_features, np.ndarray):
        state_features = torch.from_numpy(state_features).float()
    
    # Ensure proper shape (add batch dimension if needed)
    if state_features.dim() == 1:
        state_features = state_features.unsqueeze(0)
    
    state_features = state_features.to(device)
    
    with torch.no_grad():
        output = model(state_features)
        # Apply sigmoid to get probability
        probability = torch.sigmoid(output).item()
    
    return probability