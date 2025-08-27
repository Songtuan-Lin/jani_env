"""
Data loading utilities for binary classification datasets.

Example usage with weighted sampling for imbalanced datasets:

    # Load your datasets
    datasets = load_dataset(".", "your_benchmark", use_next_state=True)
    
    # Create DataLoaders with weighted sampling for imbalanced classification
    data_loaders, scaler = create_data_loaders(
        datasets,
        batch_size=32,
        normalize=True,
        use_weighted_sampling=True  # Enable weighted sampling
    )
"""

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List
from .. import config


class SafetyDataset(Dataset):
    """Custom dataset for safety classification."""

    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


def load_csv_data(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load features and targets from CSV file."""
    df = pd.read_csv(csv_path, header=None)
    features = df.iloc[:, :-1].values  # All columns except last
    targets = df.iloc[:, -1].values    # Last column
    return features, targets


def calculate_sample_weights(targets: np.ndarray) -> torch.Tensor:
    """
    Calculate sample weights for balanced sampling in classification tasks.
    
    Args:
        targets: Array of target labels (0s and 1s for binary classification)
    
    Returns:
        Tensor of sample weights for each sample
    """
    # Count samples per class
    unique_classes, class_counts = np.unique(targets, return_counts=True)
    
    # Calculate weight for each class (inverse frequency)
    total_samples = len(targets)
    class_weights = total_samples / (len(unique_classes) * class_counts)
    
    # Create weight dictionary
    class_weight_dict = dict(zip(unique_classes, class_weights))
    
    # Assign weight to each sample based on its class
    sample_weights = np.array([class_weight_dict[int(target)] for target in targets])
    
    return torch.FloatTensor(sample_weights)


def print_class_distribution_and_weights(targets: np.ndarray, dataset_name: str = "Dataset") -> None:
    """
    Print class distribution and calculated sample weights.
    
    Args:
        targets: Array of target labels
        dataset_name: Name of the dataset for display
    """
    unique_classes, class_counts = np.unique(targets, return_counts=True)
    total_samples = len(targets)
    
    print(f"\n{dataset_name} Class Distribution:")
    for cls, count in zip(unique_classes, class_counts):
        percentage = count / total_samples * 100
        print(f"  Class {int(cls)}: {count} samples ({percentage:.1f}%)")
    
    # Calculate and show class weights
    class_weights = total_samples / (len(unique_classes) * class_counts)
    print(f"\nCalculated Class Weights for Balanced Sampling:")
    for cls, weight in zip(unique_classes, class_weights):
        print(f"  Class {int(cls)}: {weight:.3f}")
    
    # Show imbalance ratio
    if len(unique_classes) == 2:
        max_count = max(class_counts)
        min_count = min(class_counts)
        imbalance_ratio = max_count / min_count
        print(f"\nClass Imbalance Ratio: {imbalance_ratio:.2f}:1")


def load_dataset(data_dir: str, benchmark: str, use_next_state: bool = True) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Load train/val/test datasets for a specific benchmark.
    
    Args:
        data_dir: Root directory containing the datasets
        benchmark: Benchmark name (e.g., 'bouncing_ball_16_16')
        use_next_state: Whether to use enhanced features (with next state)
    
    Returns:
        Dictionary with 'train', 'val', 'test' keys containing (features, targets) tuples
    """
    if use_next_state:
        base_path = Path(data_dir) / config.DATA_WITH_NEXT_STATE / benchmark
    else:
        base_path = Path(data_dir) / config.DATA_WITHOUT_NEXT_STATE / benchmark
    
    datasets = {}
    for split in ['train', 'val', 'test']:
        csv_path = base_path / split / f"{split}.csv"
        if csv_path.exists():
            features, targets = load_csv_data(str(csv_path))
            datasets[split] = (features, targets)
        else:
            raise FileNotFoundError(f"Dataset file not found: {csv_path}")
    
    return datasets




def create_data_loaders(datasets: Dict[str, Tuple[np.ndarray, np.ndarray]],
                       batch_size: int = config.BATCH_SIZE,
                       normalize: bool = True,
                       use_weighted_sampling: bool = False) -> Tuple[Dict[str, DataLoader], StandardScaler]:
    """
    Create PyTorch DataLoaders from datasets with optional normalization and weighted sampling.
    
    Args:
        datasets: Dictionary with train/val/test datasets
        batch_size: Batch size for DataLoaders
        normalize: Whether to normalize features
        use_weighted_sampling: Whether to use weighted sampling for training data
    
    Returns:
        Tuple of (data_loaders_dict, scaler)
    """
    scaler = StandardScaler() if normalize else None
    data_loaders = {}
    
    # Fit scaler on training data
    if normalize and 'train' in datasets:
        train_features, _ = datasets['train']
        scaler.fit(train_features)
    
    for split, (features, targets) in datasets.items():
        # Normalize features if requested
        if normalize and scaler is not None:
            features = scaler.transform(features)
        
        # Create dataset
        dataset = SafetyDataset(features, targets)
        
        # Determine sampling strategy
        sampler = None
        shuffle = False
        
        if split == 'train':
            if use_weighted_sampling:
                # Use weighted sampling for training data
                sample_weights = calculate_sample_weights(targets)
                sampler = WeightedRandomSampler(
                    weights=sample_weights,
                    num_samples=len(dataset),
                    replacement=True
                )
                shuffle = False  # Can't use shuffle with custom sampler
            else:
                # Use regular shuffling
                shuffle = True
        
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=0,  # Set to 0 to avoid multiprocessing issues
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        data_loaders[split] = data_loader
    
    return data_loaders, scaler


def get_input_size(benchmark: str, use_next_state: bool = True, data_dir: str = ".") -> int:
    """Get the input feature size for a specific benchmark."""
    datasets = load_dataset(data_dir, benchmark, use_next_state)
    train_features, _ = datasets['train']
    return train_features.shape[1]


def get_available_benchmarks(data_dir: str = ".") -> List[str]:
    """Get list of available benchmark datasets."""
    with_next_state_path = Path(data_dir) / config.DATA_WITH_NEXT_STATE
    if not with_next_state_path.exists():
        return []
    
    benchmarks = []
    for item in with_next_state_path.iterdir():
        if item.is_dir():
            # Check if corresponding directory exists in without_next_state
            without_path = Path(data_dir) / config.DATA_WITHOUT_NEXT_STATE / item.name
            if without_path.exists():
                benchmarks.append(item.name)
    
    return sorted(benchmarks)


def print_dataset_info(datasets: Dict[str, Tuple[np.ndarray, np.ndarray]], benchmark: str):
    """Print information about loaded classification datasets."""
    print(f"\nDataset Information for {benchmark}:")
    for split, (features, targets) in datasets.items():
        n_samples = features.shape[0]
        n_features = features.shape[1]

        print(f"  {split.capitalize()} set:")
        print(f"    Samples: {n_samples}")
        print(f"    Features: {n_features}")

        n_positive = np.sum(targets == 1)
        n_negative = np.sum(targets == 0)
        print(f"    Positive: {n_positive} ({n_positive/n_samples:.1%})")
        print(f"    Negative: {n_negative} ({n_negative/n_samples:.1%})")




if __name__ == "__main__":
    # Test data loading
    print("=== Testing Classification Data Loading ===")
    benchmarks = get_available_benchmarks(".")
    print("Available classification benchmarks:", benchmarks[:5])  # Show first 5

    if benchmarks:
        benchmark = benchmarks[0]
        print(f"\nTesting with classification benchmark: {benchmark}")

        # Test both types
        for use_next_state in [False, True]:
            datasets = load_dataset(".", benchmark, use_next_state)
            print_dataset_info(datasets, f"{benchmark} ({'with' if use_next_state else 'without'} next state)")

            # Test regular DataLoader
            data_loaders, scaler = create_data_loaders(datasets)
            print(f"  Input size: {get_input_size(benchmark, use_next_state)}")
            
            # Test weighted sampling
            train_targets = datasets['train'][1]
            print_class_distribution_and_weights(train_targets, f"Training set for {benchmark}")
            
            print(f"\n  Testing weighted DataLoader...")
            weighted_data_loaders, _ = create_data_loaders(
                datasets, 
                use_weighted_sampling=True
            )