"""
Simple neural network classifier for binary classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Classifier(nn.Module):
    """
    Simple binary classifier with configurable architecture.
    
    Args:
        input_size: Size of input features
        hidden_sizes: List of hidden layer sizes (e.g., [128, 64])
        dropout: Dropout rate (default: 0.2)
    """
    
    def __init__(self, input_size, hidden_sizes=[128, 64], dropout=0.2):
        super(Classifier, self).__init__()
        
        self.hidden_sizes = hidden_sizes
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        
        # Build network layers
        prev_size = input_size
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