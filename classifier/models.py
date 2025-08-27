"""
Neural network models for binary classification.
Classification models:
1. BasicClassifier: Uses only current state features
2. EnhancedClassifier: Uses current state + next state + next state classification
3. DynamicClassifier: Configurable architecture for hyperparameter optimization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicClassifier(nn.Module):
    """
    Binary classifier using only current state features.
    Input: current state vector (v1)
    Output: binary classification (safe=1, unsafe=0)
    """
    
    def __init__(self, input_size, hidden_sizes=[128, 64], dropout=0.2):
        super(BasicClassifier, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        
        # Input layer
        prev_size = input_size
        
        # Hidden layers
        for hidden_size in hidden_sizes:
            self.layers.append(nn.Linear(prev_size, hidden_size))
            prev_size = hidden_size
        
        # Output layer
        self.output_layer = nn.Linear(prev_size, 1)
        
    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
            x = self.dropout(x)
        
        x = torch.sigmoid(self.output_layer(x))
        return x.squeeze()


class EnhancedClassifier(nn.Module):
    """
    Binary classifier using current state + next state + next state classification.
    Input: concatenated vector (v1 + v2 + c) where:
    - v1: current state vector
    - v2: next state vector  
    - c: classification result for next state
    Output: binary classification (safe=1, unsafe=0)
    """
    
    def __init__(self, input_size, hidden_sizes=[128, 64], dropout=0.2):
        super(EnhancedClassifier, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        
        # Input layer
        prev_size = input_size
        
        # Hidden layers
        for hidden_size in hidden_sizes:
            self.layers.append(nn.Linear(prev_size, hidden_size))
            prev_size = hidden_size
        
        # Output layer
        self.output_layer = nn.Linear(prev_size, 1)
        
    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
            x = self.dropout(x)
        
        x = torch.sigmoid(self.output_layer(x))
        return x.squeeze()


class DynamicClassifier(nn.Module):
    """
    Dynamic classifier that can be configured for different architectures.
    Used for hyperparameter optimization.
    """
    
    def __init__(self, input_size, hidden_sizes, dropout=0.2, activation='relu'):
        super(DynamicClassifier, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.activation = getattr(F, activation)
        
        # Build network
        prev_size = input_size
        for hidden_size in hidden_sizes:
            self.layers.append(nn.Linear(prev_size, hidden_size))
            prev_size = hidden_size
        
        # Output layer
        self.output_layer = nn.Linear(prev_size, 1)
        
    def forward(self, x):
        for layer in self.layers:
            x = self.activation(layer(x))
            x = self.dropout(x)
        
        x = torch.sigmoid(self.output_layer(x))
        return x.squeeze()




def create_model(model_type, input_size, **kwargs):
    """Factory function to create classification models."""
    if model_type == 'basic':
        return BasicClassifier(input_size, **kwargs)
    elif model_type == 'enhanced':
        return EnhancedClassifier(input_size, **kwargs)
    elif model_type == 'dynamic':
        return DynamicClassifier(input_size, **kwargs)
    else:
        raise ValueError(f"Unknown classification model type: {model_type}. Supported types: 'basic', 'enhanced', 'dynamic'")


def count_parameters(model):
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_info(model, model_name="Model"):
    """Print model architecture and parameter count."""
    print(f"\n{model_name} Architecture:")
    print(f"  Parameters: {count_parameters(model):,}")
    print(f"  Layers: {len(list(model.modules())) - 1}")  # -1 to exclude the model itself
    print(f"  Model: {model}")