"""
Hyperparameter tuning using Optuna for binary classification models.
"""

import optuna
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
import json
from rich.console import Console

# Create console for rich output
console = Console()
from .. import config
from .data_loader import load_dataset, create_data_loaders, get_input_size
from ..models import DynamicClassifier
from .trainer import train_model


class OptunaTuner:
    """Hyperparameter tuning using Optuna."""
    
    def __init__(self, benchmark: str, use_next_state: bool, data_dir: str = "."):
        self.benchmark = benchmark
        self.use_next_state = use_next_state
        self.data_dir = data_dir
        
        # Load data once
        self.datasets = load_dataset(data_dir, benchmark, use_next_state)
        self.input_size = get_input_size(benchmark, use_next_state, data_dir)
        
        # Create data loaders
        self.data_loaders, self.scaler = create_data_loaders(
            self.datasets, 
            batch_size=config.BATCH_SIZE
        )
        
    def objective(self, trial):
        """Objective function for Optuna optimization."""
        # Suggest hyperparameters
        n_layers = trial.suggest_int('n_layers', *config.HIDDEN_LAYERS_RANGE)
        hidden_sizes = []
        
        for i in range(n_layers):
            size = trial.suggest_int(f'hidden_size_{i}', *config.HIDDEN_SIZE_RANGE)
            hidden_sizes.append(size)
        
        dropout = trial.suggest_float('dropout', *config.DROPOUT_RANGE)
        learning_rate = trial.suggest_float('learning_rate', *config.LEARNING_RATE_RANGE, log=True)
        
        # Create model
        model = DynamicClassifier(
            input_size=self.input_size,
            hidden_sizes=hidden_sizes,
            dropout=dropout
        )
        
        # Train model
        try:
            results = train_model(
                model=model,
                train_loader=self.data_loaders['train'],
                val_loader=self.data_loaders['val'],
                test_loader=None,  # Don't use test during tuning
                epochs=config.MAX_EPOCHS,
                learning_rate=learning_rate,
                device=config.DEVICE,
                early_stopping_patience=config.EARLY_STOPPING_PATIENCE,
                verbose=False
            )
            
            # Return validation accuracy as optimization target
            return results['final_val_metrics']['accuracy']
            
        except Exception as e:
            print(f"Trial failed: {e}")
            return 0.0
    
    def tune(self, n_trials: int = config.N_TRIALS, study_name: str = None) -> optuna.Study:
        """Run hyperparameter tuning."""
        if study_name is None:
            classifier_type = "enhanced" if self.use_next_state else "basic"
            study_name = f"{config.OPTUNA_STUDY_NAME}_{classifier_type}_{self.benchmark}"
        
        study = optuna.create_study(
            direction='maximize',
            study_name=study_name,
            sampler=optuna.samplers.TPESampler(seed=config.RANDOM_SEED)
        )
        
        study.optimize(self.objective, n_trials=n_trials)
        
        return study
    
    def get_best_model(self, study: optuna.Study) -> Tuple[DynamicClassifier, Dict]:
        """Create and train the best model found by tuning."""
        best_params = study.best_params
        
        # Extract hyperparameters
        n_layers = best_params['n_layers']
        hidden_sizes = [best_params[f'hidden_size_{i}'] for i in range(n_layers)]
        dropout = best_params['dropout']
        learning_rate = best_params['learning_rate']
        
        # Create best model
        best_model = DynamicClassifier(
            input_size=self.input_size,
            hidden_sizes=hidden_sizes,
            dropout=dropout
        )
        
        # Train with full monitoring
        results = train_model(
            model=best_model,
            train_loader=self.data_loaders['train'],
            val_loader=self.data_loaders['val'],
            test_loader=self.data_loaders['test'],
            epochs=config.MAX_EPOCHS,
            learning_rate=learning_rate,
            device=config.DEVICE,
            early_stopping_patience=config.EARLY_STOPPING_PATIENCE,
            verbose=True
        )
        
        # Add hyperparameters to results
        results['best_hyperparameters'] = best_params
        results['tuning_trials'] = len(study.trials)
        results['best_trial_value'] = study.best_value
        
        return best_model, results


def run_hyperparameter_tuning(benchmark: str, 
                              classifier_type: str = "both",
                              n_trials: int = config.N_TRIALS,
                              data_dir: str = ".") -> Dict[str, Any]:
    """
    Run hyperparameter tuning for specified classifier type(s).
    
    Args:
        classifier_type: "basic", "enhanced", or "both"
    
    Returns:
        Dictionary containing results for specified classifier type(s)
    """
    results = {}
    
    # Determine which classifiers to train
    if classifier_type == "both":
        use_next_state_options = [False, True]
    elif classifier_type == "basic":
        use_next_state_options = [False]
    elif classifier_type == "enhanced":
        use_next_state_options = [True]
    else:
        raise ValueError(f"Invalid classifier_type: {classifier_type}. Must be 'basic', 'enhanced', or 'both'")
    
    for use_next_state in use_next_state_options:
        current_classifier_type = "enhanced" if use_next_state else "basic"
        print(f"\n🔧 Tuning {current_classifier_type} classifier for {benchmark}")
        
        # Create tuner
        tuner = OptunaTuner(benchmark, use_next_state, data_dir)
        
        # Run tuning
        study = tuner.tune(n_trials=n_trials)
        
        console.print(f"Best trial value: {study.best_value:.4f}")
        console.print(f"Best parameters: {study.best_params}")
        
        # Train best model
        best_model, train_results = tuner.get_best_model(study)
        
        # Store results
        results[current_classifier_type] = {
            'study': study,
            'model': best_model,
            'results': train_results,
            'scaler': tuner.scaler
        }
        
        print(f"Best {current_classifier_type} model trained:")
        print(f"  Validation accuracy: {train_results['final_val_metrics']['accuracy']:.4f}")
        if 'test_metrics' in train_results:
            print(f"  Test accuracy: {train_results['test_metrics']['accuracy']:.4f}")
    
    return results


def save_tuning_results(results: Dict[str, Any], benchmark: str, output_dir: str = config.RESULTS_DIR):
    """Save hyperparameter tuning results with complete model reconstruction info."""
    output_path = Path(output_dir) / benchmark
    output_path.mkdir(parents=True, exist_ok=True)
    
    for classifier_type, data in results.items():
        model = data['model']
        best_params = data['results']['best_hyperparameters']
        
        # Save complete model with reconstruction information
        model_path = output_path / f"{classifier_type}_best_model.pth"
        
        # Get input size from model
        if hasattr(model, 'layers') and len(model.layers) > 0:
            input_size = model.layers[0].in_features
        else:
            input_size = model.output_layer.in_features
        
        # Extract architecture details
        hidden_sizes = []
        if hasattr(model, 'layers'):
            for layer in model.layers:
                if hasattr(layer, 'out_features'):
                    hidden_sizes.append(layer.out_features)
        
        # Get dropout rate
        dropout_rate = 0.2  # default
        if hasattr(model, 'dropout'):
            dropout_rate = model.dropout.p
        
        model_info = {
            'model_state_dict': model.state_dict(),
            'model_class': model.__class__.__name__,
            'model_architecture': {
                'input_size': input_size,
                'hidden_sizes': hidden_sizes,
                'dropout': dropout_rate
            },
            'hyperparameters': best_params,
            'scaler': data['scaler'],
            'benchmark': benchmark,
            'classifier_type': classifier_type,
            'training_metrics': data['results'].get('test_metrics', {}),
            'creation_timestamp': str(pd.Timestamp.now())
        }
        
        torch.save(model_info, model_path)
        
        # Save results (excluding non-serializable objects)
        results_data = data['results'].copy()
        results_path = output_path / f"{classifier_type}_tuning_results.json"
        
        # Convert numpy types for JSON serialization
        def convert_for_json(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            return obj
        
        results_serializable = convert_for_json(results_data)
        
        # Ensure we have a dictionary and add model reconstruction info
        if not isinstance(results_serializable, dict):
            results_serializable = {}
            
        results_serializable['model_reconstruction_info'] = {
            'model_class': model.__class__.__name__,
            'input_size': input_size,
            'hidden_sizes': hidden_sizes,
            'dropout': dropout_rate,
            'benchmark': benchmark,
            'classifier_type': classifier_type
        }
        
        with open(results_path, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        # Save study
        study_path = output_path / f"{classifier_type}_study.pkl"
        with open(study_path, 'wb') as f:
            import pickle
            pickle.dump(data['study'], f)
        
        console.print(f"  ✅ Saved {classifier_type} model and results to {output_path}")
        console.print(f"     Model: {input_size} → {hidden_sizes} → 1 (dropout={dropout_rate})")
        
        # Print test metrics if available
        if 'test_metrics' in data['results']:
            test_acc = data['results']['test_metrics'].get('accuracy', 'N/A')
            test_f1 = data['results']['test_metrics'].get('f1_score', 'N/A')
            console.print(f"     Test Accuracy: {test_acc:.4f}, F1: {test_f1:.4f}" if test_acc != 'N/A' else f"     Test Accuracy: {test_acc}")


def load_trained_model(model_path: str, device: torch.device = config.DEVICE):
    """
    Load a trained model with complete reconstruction.
    
    Args:
        model_path: Path to the saved model file
        device: Device to load the model on
        
    Returns:
        tuple: (model, model_info) where model_info contains all saved metadata
    """
    model_info = torch.load(model_path, map_location=device)
    
    # Reconstruct the model
    model_class_name = model_info['model_class']
    arch_info = model_info['model_architecture']
    
    if model_class_name == 'DynamicClassifier':
        from ..models import DynamicClassifier
        model = DynamicClassifier(
            input_size=arch_info['input_size'],
            hidden_sizes=arch_info['hidden_sizes'],
            dropout=arch_info['dropout']
        )
    elif model_class_name == 'BasicClassifier':
        from ..models import BasicClassifier
        model = BasicClassifier(
            input_size=arch_info['input_size'],
            hidden_sizes=arch_info['hidden_sizes'],
            dropout=arch_info['dropout']
        )
    elif model_class_name == 'EnhancedClassifier':
        from ..models import EnhancedClassifier
        model = EnhancedClassifier(
            input_size=arch_info['input_size'],
            hidden_sizes=arch_info['hidden_sizes'],
            dropout=arch_info['dropout']
        )
    else:
        raise ValueError(f"Unknown model class: {model_class_name}")
    
    # Load the trained weights
    model.load_state_dict(model_info['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, model_info


def reconstruct_model_from_json(results_path: str, device: torch.device = config.DEVICE):
    """
    Reconstruct model architecture from JSON results file.
    Note: This creates the model architecture but doesn't load trained weights.
    Use load_trained_model() for complete model reconstruction.
    """
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    model_info = results['model_reconstruction_info']
    model_class_name = model_info['model_class']
    
    if model_class_name == 'DynamicClassifier':
        from ..models import DynamicClassifier
        model = DynamicClassifier(
            input_size=model_info['input_size'],
            hidden_sizes=model_info['hidden_sizes'],
            dropout=model_info['dropout']
        )
    elif model_class_name == 'BasicClassifier':
        from ..models import BasicClassifier
        model = BasicClassifier(
            input_size=model_info['input_size'],
            hidden_sizes=model_info['hidden_sizes'],
            dropout=model_info['dropout']
        )
    elif model_class_name == 'EnhancedClassifier':
        from ..models import EnhancedClassifier
        model = EnhancedClassifier(
            input_size=model_info['input_size'],
            hidden_sizes=model_info['hidden_sizes'],
            dropout=model_info['dropout']
        )
    else:
        raise ValueError(f"Unknown model class: {model_class_name}")
    
    model.to(device)
    return model, model_info


if __name__ == "__main__":
    # Test hyperparameter tuning
    from data_loader import get_available_benchmarks
    
    benchmarks = get_available_benchmarks(".")
    if benchmarks:
        # Use first benchmark for testing with reduced trials
        benchmark = benchmarks[0]
        print(f"Testing hyperparameter tuning with {benchmark}")
        
        results = run_hyperparameter_tuning(
            benchmark=benchmark, 
            n_trials=5,  # Reduced for testing
            data_dir="."
        )
        
        save_tuning_results(results, benchmark)
        print("Hyperparameter tuning test completed!")