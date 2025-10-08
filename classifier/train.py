"""
Training script with Optuna hyperparameter tuning for state safety classifier.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import optuna
import argparse
from pathlib import Path
import json

from .models import Classifier
from .data_loader import load_datasets, create_dataloaders, get_input_size


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train model for one epoch."""
    model.train()
    total_loss = 0
    
    for features, targets in dataloader:
        features, targets = features.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate_model(model, dataloader, device):
    """Evaluate model and return metrics."""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for features, targets in dataloader:
            features = features.to(device)
            outputs = model(features)
            predictions = (outputs > 0.5).float()
            
            all_preds.extend(predictions.cpu().numpy().flatten())
            all_targets.extend(targets.numpy().flatten())
    
    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds)
    recall = recall_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


def train_model(model, dataloaders, device, epochs=50, lr=0.001):
    """Train model with given hyperparameters."""
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_acc = 0
    best_model_state = None
    
    for epoch in range(epochs):
        # Train
        train_loss = train_epoch(model, dataloaders['train'], criterion, optimizer, device)
        
        # Validate
        val_metrics = evaluate_model(model, dataloaders['val'], device)
        
        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_model_state = model.state_dict().copy()
    
    # Load best model
    model.load_state_dict(best_model_state)
    return model


def objective(trial, input_size, dataloaders, device):
    """Optuna objective function for hyperparameter tuning."""
    # Suggest hyperparameters
    n_layers = trial.suggest_int('n_layers', 1, 3)
    hidden_sizes = []
    for i in range(n_layers):
        size = trial.suggest_int(f'hidden_size_{i}', 32, 256)
        hidden_sizes.append(size)
    
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    epochs = trial.suggest_int('epochs', 20, 100)
    
    # Create and train model
    model = Classifier(input_size, hidden_sizes, dropout).to(device)
    model = train_model(model, dataloaders, device, epochs, lr)
    
    # Return validation accuracy
    val_metrics = evaluate_model(model, dataloaders['val'], device)
    return val_metrics['accuracy']


def main():
    parser = argparse.ArgumentParser(description="Train state safety classifier with hyperparameter tuning")
    parser.add_argument('--data-dir', type=str, required=True, help='Directory containing train.csv, val.csv, test.csv')
    parser.add_argument('--n-trials', type=int, default=100, help='Number of Optuna trials')
    parser.add_argument('--output-dir', type=str, default='results', help='Output directory for results')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading datasets...")
    datasets = load_datasets(args.data_dir)
    dataloaders, scaler = create_dataloaders(datasets, batch_size=args.batch_size)
    input_size = get_input_size(args.data_dir)
    
    print(f"Dataset loaded:")
    for split, (features, targets) in datasets.items():
        print(f"  {split}: {len(features)} samples, {features.shape[1]} features")
    
    # Hyperparameter tuning
    print(f"\nStarting hyperparameter tuning with {args.n_trials} trials...")
    study = optuna.create_study(direction='maximize')
    study.optimize(
        lambda trial: objective(trial, input_size, dataloaders, device),
        n_trials=args.n_trials
    )
    
    print("Hyperparameter tuning completed!")
    print(f"Best validation accuracy: {study.best_value:.4f}")
    print("Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Train final model with best hyperparameters
    print("\nTraining final model...")
    best_params = study.best_params
    
    # Extract best hyperparameters
    n_layers = best_params['n_layers']
    hidden_sizes = [best_params[f'hidden_size_{i}'] for i in range(n_layers)]
    dropout = best_params['dropout']
    lr = best_params['lr']
    epochs = best_params['epochs']
    
    # Create and train final model
    final_model = Classifier(input_size, hidden_sizes, dropout).to(device)
    final_model = train_model(final_model, dataloaders, device, epochs, lr)
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_metrics = evaluate_model(final_model, dataloaders['test'], device)
    
    print("Final test results:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Save results
    results = {
        'best_params': study.best_params,
        'best_val_accuracy': study.best_value,
        'test_metrics': test_metrics,
        'input_size': input_size
    }
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save model with scaler
    torch.save({
        'model_state_dict': final_model.state_dict(),
        'input_size': input_size,
        'hidden_sizes': hidden_sizes,
        'dropout': dropout,
        'scaler': scaler  # Save the fitted scaler
    }, output_dir / 'best_model.pth')
    
    print(f"\nResults saved to {output_dir}")
    print("Training completed successfully!")


if __name__ == "__main__":
    main()