"""
Training utilities for classification models with multi-GPU support.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from typing import Dict, List, Tuple, Optional
import time
from pathlib import Path
import json
from .. import config


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None) -> Dict[str, float]:
    """Calculate metrics for classification tasks."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
    }

    # AUC-ROC only if we have both classes and probabilities
    if y_prob is not None and len(np.unique(y_true)) > 1:
        metrics['auc_roc'] = roc_auc_score(y_true, y_prob)
    else:
        metrics['auc_roc'] = 0.0

    return metrics


def evaluate_model(model: nn.Module, data_loader: DataLoader, criterion: nn.Module,
                  device: torch.device) -> Tuple[float, Dict[str, float]]:
    """Evaluate classification model on given dataset."""
    model.eval()
    total_loss = 0
    all_targets = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():
        for features, targets in data_loader:
            features, targets = features.to(device), targets.to(device)

            outputs = model(features)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            # Convert to numpy for metrics calculation
            outputs_np = outputs.cpu().numpy()
            targets_np = targets.cpu().numpy()

            all_targets.extend(targets_np)

            probabilities = outputs_np
            predictions = (probabilities > 0.5).astype(int)
            all_predictions.extend(predictions)
            all_probabilities.extend(probabilities)

    avg_loss = total_loss / len(data_loader)

    metrics = calculate_metrics(
        np.array(all_targets),
        np.array(all_predictions),
        np.array(all_probabilities)
    )

    return avg_loss, metrics


def train_epoch(model: nn.Module, train_loader: DataLoader, criterion: nn.Module, 
                optimizer: optim.Optimizer, device: torch.device) -> float:
    """Train model for one epoch."""
    model.train()
    total_loss = 0
    
    for features, targets in train_loader:
        features, targets = features.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def train_model(model: nn.Module,
                train_loader: DataLoader,
                val_loader: DataLoader,
                test_loader: Optional[DataLoader] = None,
                epochs: int = config.MAX_EPOCHS,
                learning_rate: float = config.LEARNING_RATE,
                device: torch.device = config.DEVICE,
                early_stopping_patience: int = config.EARLY_STOPPING_PATIENCE,
                verbose: bool = True) -> Dict:
    """
    Train a model with early stopping and return training history.

    Returns:
        Dictionary containing training history and final metrics
    """
    model = model.to(device)

    # Loss function for classification
    criterion = nn.BCELoss()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    early_stopping = EarlyStopping(patience=early_stopping_patience)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_metrics': [],
        'epoch_times': []
    }
    
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # Training
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validation
        val_loss, val_metrics = evaluate_model(model, val_loader, criterion, device)
        
        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_metrics'].append(val_metrics)
        history['epoch_times'].append(time.time() - epoch_start)
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, "
                  f"Val Accuracy: {val_metrics['accuracy']:.4f}")
        
        # Early stopping
        if early_stopping(val_loss, model):
            if verbose:
                print(f"Early stopping at epoch {epoch+1}")
            break
    
    total_time = time.time() - start_time
    
    # Final evaluation
    final_val_loss, final_val_metrics = evaluate_model(model, val_loader, criterion, device)
    
    results = {
        'training_time': total_time,
        'epochs_trained': len(history['train_loss']),
        'final_val_loss': final_val_loss,
        'final_val_metrics': final_val_metrics,
        'history': history
    }
    
    # Test evaluation if test_loader provided
    if test_loader is not None:
        test_loss, test_metrics = evaluate_model(model, test_loader, criterion, device)
        results['test_loss'] = test_loss
        results['test_metrics'] = test_metrics
    
    return results


def save_model_and_results(model: nn.Module, results: Dict, model_path: str, results_path: str):
    """Save trained model and results."""
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    Path(results_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_class': model.__class__.__name__,
    }, model_path)
    
    # Save results (convert numpy types to Python types for JSON serialization)
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        return obj
    
    results_serializable = convert_numpy(results)
    
    with open(results_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)


def load_model_and_results(model: nn.Module, model_path: str, results_path: str) -> Tuple[nn.Module, Dict]:
    """Load trained model and results."""
    # Load model
    checkpoint = torch.load(model_path, map_location=config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load results
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    return model, results


if __name__ == "__main__":
    # Test training functionality
    from .data_loader import get_available_benchmarks, load_dataset, create_data_loaders
    from ..models import BasicClassifier, EnhancedClassifier
    
    benchmarks = get_available_benchmarks(".")
    if benchmarks:
        benchmark = benchmarks[0]
        print(f"Testing training with benchmark: {benchmark}")
        
        # Test basic classifier
        datasets = load_dataset(".", benchmark, use_next_state=False)
        data_loaders, scaler = create_data_loaders(datasets, batch_size=32)
        
        input_size = datasets['train'][0].shape[1]
        model = BasicClassifier(input_size, hidden_sizes=[64, 32])
        
        print("Training basic classifier...")
        results = train_model(
            model, 
            data_loaders['train'], 
            data_loaders['val'], 
            data_loaders['test'],
            epochs=5,  # Short test
            verbose=True
        )
        
        print(f"Training completed in {results['training_time']:.2f}s")
        print(f"Final validation accuracy: {results['final_val_metrics']['accuracy']:.4f}")
        if 'test_metrics' in results:
            print(f"Test accuracy: {results['test_metrics']['accuracy']:.4f}")