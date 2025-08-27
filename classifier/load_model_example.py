"""
Example script showing how to load and use trained models.
"""

import torch
import numpy as np
from pathlib import Path
from .trainer import load_dataset, create_data_loaders
from .utils import load_trained_model
from . import config


def load_and_test_model(benchmark: str, classifier_type: str, results_dir: str = config.RESULTS_DIR):
    """
    Load a trained model and test it on data.
    
    Args:
        benchmark: Name of the benchmark (e.g., 'bouncing_ball_16_16')
        classifier_type: 'basic' or 'enhanced'
        results_dir: Directory containing saved models
    """
    
    # Construct model path
    model_path = Path(results_dir) / benchmark / f"{classifier_type}_best_model.pth"
    
    if not model_path.exists():
        print(f"❌ Model file not found: {model_path}")
        return None, None
    
    # Load the trained model
    print(f"📥 Loading {classifier_type} model for {benchmark}...")
    model, model_info = load_trained_model(str(model_path))
    
    # Print model information
    print(f"✅ Model loaded successfully!")
    print(f"   Model class: {model_info['model_class']}")
    print(f"   Architecture: {model_info['model_architecture']}")
    print(f"   Training timestamp: {model_info.get('creation_timestamp', 'Unknown')}")
    
    if 'training_metrics' in model_info:
        metrics = model_info['training_metrics']
        print(f"   Training metrics:")
        for metric, value in metrics.items():
            print(f"     {metric}: {value:.4f}")
    
    # Load test data to verify the model works
    use_next_state = (classifier_type == 'enhanced')
    try:
        datasets = load_dataset(".", benchmark, use_next_state)
        data_loaders, scaler = create_data_loaders(datasets, batch_size=32)
        
        # Test prediction on a small batch
        test_loader = data_loaders['test']
        model.eval()
        
        with torch.no_grad():
            for features, targets in test_loader:
                features = features.to(config.DEVICE)
                predictions = model(features)
                
                print(f"   Test batch predictions:")
                print(f"     Input shape: {features.shape}")
                print(f"     Predictions (first 5): {predictions[:5].cpu().numpy()}")
                print(f"     Actual labels (first 5): {targets[:5].numpy()}")
                break
                
    except Exception as e:
        print(f"   ⚠️ Could not test with data: {e}")
    
    return model, model_info


def compare_model_architectures(benchmark: str, results_dir: str = config.RESULTS_DIR):
    """Compare the architectures of basic and enhanced classifiers."""
    
    print(f"\n🔍 Comparing model architectures for {benchmark}")
    print("=" * 60)
    
    for classifier_type in ['basic', 'enhanced']:
        model_path = Path(results_dir) / benchmark / f"{classifier_type}_best_model.pth"
        
        if model_path.exists():
            try:
                model, model_info = load_trained_model(str(model_path))
                arch = model_info['model_architecture']
                
                print(f"\n{classifier_type.upper()} Classifier:")
                print(f"  Input size: {arch['input_size']}")
                print(f"  Hidden layers: {arch['hidden_sizes']}")
                print(f"  Dropout: {arch['dropout']}")
                print(f"  Total parameters: {sum(p.numel() for p in model.parameters())}")
                
                if 'hyperparameters' in model_info:
                    hp = model_info['hyperparameters']
                    print(f"  Best hyperparameters:")
                    for key, value in hp.items():
                        print(f"    {key}: {value}")
                        
            except Exception as e:
                print(f"  ❌ Error loading {classifier_type} model: {e}")
        else:
            print(f"  ❌ {classifier_type.upper()} model not found")


def list_available_models(results_dir: str = config.RESULTS_DIR):
    """List all available trained models."""
    
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"❌ Results directory not found: {results_dir}")
        return
    
    print("📋 Available trained models:")
    print("=" * 40)
    
    benchmarks = [d for d in results_path.iterdir() if d.is_dir()]
    
    if not benchmarks:
        print("  No trained models found.")
        return
    
    for benchmark_dir in sorted(benchmarks):
        benchmark_name = benchmark_dir.name
        print(f"\n📁 {benchmark_name}:")
        
        for classifier_type in ['basic', 'enhanced']:
            model_file = benchmark_dir / f"{classifier_type}_best_model.pth"
            if model_file.exists():
                try:
                    model_info = torch.load(model_file, map_location='cpu')
                    metrics = model_info.get('training_metrics', {})
                    accuracy = metrics.get('accuracy', 'N/A')
                    f1_score = metrics.get('f1_score', 'N/A')
                    
                    print(f"  ✅ {classifier_type}: Acc={accuracy:.4f}, F1={f1_score:.4f}" 
                          if accuracy != 'N/A' else f"  ✅ {classifier_type}: Available")
                except:
                    print(f"  ⚠️ {classifier_type}: File exists but corrupted")
            else:
                print(f"  ❌ {classifier_type}: Not found")


def predict_with_model(benchmark: str, classifier_type: str, input_data: np.ndarray, 
                      results_dir: str = config.RESULTS_DIR):
    """
    Make predictions with a trained model.
    
    Args:
        benchmark: Benchmark name
        classifier_type: 'basic' or 'enhanced'
        input_data: Input features as numpy array
        results_dir: Directory containing saved models
        
    Returns:
        predictions: Probability predictions
    """
    model_path = Path(results_dir) / benchmark / f"{classifier_type}_best_model.pth"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Load model
    model, model_info = load_trained_model(str(model_path))
    
    # Apply same preprocessing as during training
    scaler = model_info.get('scaler')
    if scaler is not None:
        input_data = scaler.transform(input_data)
    
    # Make predictions
    model.eval()
    with torch.no_grad():
        input_tensor = torch.FloatTensor(input_data).to(config.DEVICE)
        predictions = model(input_tensor).cpu().numpy()
    
    return predictions


if __name__ == "__main__":
    # Example usage
    print("🧪 Model Loading Example")
    print("=" * 30)
    
    # List available models
    list_available_models()
    
    # Try to load a specific model (adjust benchmark name as needed)
    available_benchmarks = []
    results_path = Path(config.RESULTS_DIR)
    if results_path.exists():
        available_benchmarks = [d.name for d in results_path.iterdir() if d.is_dir()]
    
    if available_benchmarks:
        benchmark = available_benchmarks[0]
        print(f"\n🔬 Testing with benchmark: {benchmark}")
        
        # Load and test both classifiers
        for classifier_type in ['basic', 'enhanced']:
            model, model_info = load_and_test_model(benchmark, classifier_type)
        
        # Compare architectures
        compare_model_architectures(benchmark)
        
    else:
        print("\n❌ No trained models found. Run main.py first to train models.")
        
        # Show example of how to use prediction function
        print("\n📝 Example prediction usage:")
        print("""
        # After training models, you can make predictions like this:
        
        import numpy as np
        from load_model_example import predict_with_model
        
        # Example input (adjust size based on your data)
        sample_input = np.random.randn(1, 4)  # 1 sample, 4 features for basic classifier
        
        predictions = predict_with_model('bouncing_ball_16_16', 'basic', sample_input)
        print(f"Predictions: {predictions}")
        """)