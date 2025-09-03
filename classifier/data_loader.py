"""
Simple data loading utilities for CSV datasets.
"""

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import numpy as np
from pathlib import Path


class StateDataset(Dataset):
    """Dataset for state safety classification."""
    
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


def load_datasets(data_dir):
    """
    Load train/val/test datasets from CSV files.
    
    Args:
        data_dir: Directory containing train.csv, val.csv, test.csv
    
    Returns:
        dict: Dictionary with 'train', 'val', 'test' keys containing (features, targets)
    """
    data_path = Path(data_dir)
    datasets = {}
    
    for split in ['train', 'val', 'test']:
        csv_path = data_path / f"{split}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {csv_path}")
        
        # Load CSV data
        df = pd.read_csv(csv_path, header=None)
        features = df.iloc[:, :-1].values  # All columns except last
        targets = df.iloc[:, -1].values    # Last column (labels)
        
        datasets[split] = (features, targets)
    
    return datasets


def create_dataloaders(datasets, batch_size=32, normalize=True):
    """
    Create PyTorch DataLoaders from datasets.
    
    Args:
        datasets: Dictionary with train/val/test datasets
        batch_size: Batch size for DataLoaders
        normalize: Whether to normalize features using StandardScaler
    
    Returns:
        tuple: (dataloaders_dict, scaler)
    """
    scaler = None
    if normalize:
        scaler = StandardScaler()
        train_features, _ = datasets['train']
        scaler.fit(train_features)
    
    dataloaders = {}
    
    for split, (features, targets) in datasets.items():
        # Normalize features if requested
        if normalize:
            features = scaler.transform(features)
        
        # Create dataset
        dataset = StateDataset(features, targets)
        
        # Create dataloader
        shuffle = (split == 'train')  # Only shuffle training data
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0
        )
        
        dataloaders[split] = dataloader
    
    return dataloaders, scaler


def get_input_size(data_dir):
    """Get the input feature size from training data."""
    train_path = Path(data_dir) / "train.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"Training file not found: {train_path}")
    
    df = pd.read_csv(train_path, header=None)
    return df.shape[1] - 1  # Subtract 1 for the label column