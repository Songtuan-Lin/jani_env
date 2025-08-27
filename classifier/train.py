"""
Main training script for comparing basic and enhanced classifiers.
Runs hyperparameter tuning, training, and comparison analysis.
"""

import argparse
import sys
from pathlib import Path
from rich.console import Console
import torch
import numpy as np
import random

from . import config
from .trainer import get_available_benchmarks, run_hyperparameter_tuning, save_tuning_results, run_full_comparison
from .utils import print_device_info

# Create console for rich output
console = Console()

def set_random_seeds(seed=config.RANDOM_SEED):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Make CuDNN deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_directories():
    """Create necessary directories."""
    for directory in [config.RESULTS_DIR, config.PLOTS_DIR]:
        Path(directory).mkdir(parents=True, exist_ok=True)


def check_existing_results(benchmarks: list, results_dir: str = config.RESULTS_DIR) -> dict:
    """Check which classifier types have existing results for given benchmarks."""
    existing_results = {}
    results_path = Path(results_dir)
    
    for benchmark in benchmarks:
        benchmark_path = results_path / benchmark
        existing_results[benchmark] = {}
        
        for classifier_type in ['basic', 'enhanced']:
            results_file = benchmark_path / f"{classifier_type}_tuning_results.json"
            existing_results[benchmark][classifier_type] = results_file.exists()
    
    return existing_results


def train_single_benchmark(benchmark: str, classifier_type: str = "both", 
                          n_trials: int = config.N_TRIALS, data_dir: str = ".") -> bool:
    """Train specified classifier(s) for a single benchmark."""
    try:
        # Determine number of classifiers to train  
        classifiers_to_train = ['basic', 'enhanced'] if classifier_type == 'both' else [classifier_type]
        
        console.print(f"🎯 Running hyperparameter tuning for {len(classifiers_to_train)} classifier(s) with {n_trials} trials each...")
        
        # Show progress with spinner
        from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,
            transient=True
        ) as progress:
            task = progress.add_task(f"Optimizing {benchmark}...", total=None)
            
            # Run hyperparameter tuning
            results = run_hyperparameter_tuning(
                benchmark=benchmark,
                classifier_type=classifier_type,
                n_trials=n_trials,
                data_dir=data_dir
            )
        
        # Save results
        save_tuning_results(results, benchmark)
        
        console.print(f"✅ Completed training for {benchmark}")
        
        # Print summary
        for classifier_type in ['basic', 'enhanced']:
            if classifier_type in results:
                test_acc = results[classifier_type]['results'].get('test_metrics', {}).get('accuracy', 'N/A')
                console.print(f"  {classifier_type.capitalize()} classifier test accuracy: {test_acc}")
        
        return True
        
    except Exception as e:
        console.print(f"❌ Failed to train {benchmark}: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Train and compare safety classifiers")
    parser.add_argument("--benchmarks", nargs="+", help="Specific benchmarks to train (default: all)")
    parser.add_argument("--n-trials", type=int, default=config.N_TRIALS, 
                       help=f"Number of hyperparameter tuning trials (default: {config.N_TRIALS})")
    parser.add_argument("--data-dir", type=str, default=".", 
                       help="Directory containing the datasets (default: current directory)")
    parser.add_argument("--skip-training", action="store_true", 
                       help="Skip training and only run comparison")
    parser.add_argument("--comparison-only", action="store_true",
                       help="Only run comparison analysis on existing results")
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED,
                       help=f"Random seed for reproducibility (default: {config.RANDOM_SEED})")
    parser.add_argument("--classifier-type", type=str, choices=["basic", "enhanced", "both"],
                       default="both", help="Which classifier to train: basic, enhanced, or both (default: both)")
    parser.add_argument("--enable-comparison", action="store_true",
                       help="Enable comparison analysis between basic and enhanced classifiers")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.enable_comparison and args.classifier_type in ['basic', 'enhanced']:
        # Check if we need both classifier types for comparison
        print(f"⚠️  Warning: Comparison requested but only training {args.classifier_type} classifier.")
        print("   Comparison requires both basic and enhanced classifier results.")
    
    # Set random seeds
    set_random_seeds(args.seed)
    
    # Setup directories
    setup_directories()
    
    # Print configuration
    print("🚀 Safety Classifier Training and Comparison")
    print("=" * 50)
    
    # Show device information
    print_device_info()
    
    print(f"Random seed: {args.seed}")
    print(f"Data directory: {args.data_dir}")
    print(f"Results directory: {config.RESULTS_DIR}")
    print(f"Classifier type: {args.classifier_type}")
    
    # Get available benchmarks
    if args.benchmarks:
        benchmarks = args.benchmarks
        print(f"Selected benchmarks: {benchmarks}")
    else:
        benchmarks = get_available_benchmarks(args.data_dir)
        print(f"Found {len(benchmarks)} available benchmarks")
    
    if not benchmarks:
        print("❌ No benchmarks found! Please check your data directory.")
        sys.exit(1)
    
    # Additional validation for comparison
    if args.enable_comparison:
        if args.classifier_type in ['basic', 'enhanced']:
            # Check if the other classifier type already has results
            existing_results = check_existing_results(benchmarks)
            other_type = 'enhanced' if args.classifier_type == 'basic' else 'basic'
            
            missing_results = []
            for benchmark in benchmarks:
                if benchmark in existing_results and not existing_results[benchmark].get(other_type, False):
                    missing_results.append(benchmark)
            
            if missing_results:
                print(f"❌ Comparison requested but {other_type} classifier results are missing for: {missing_results}")
                print(f"   Either train both classifiers (--classifier-type both) or ensure {other_type} results exist.")
                sys.exit(1)
            else:
                print(f"✅ Found existing {other_type} classifier results for comparison.")
    
    # Training phase
    if not args.comparison_only and not args.skip_training:
        print(f"\n🔧 Starting hyperparameter tuning and training...")
        print(f"Trials per classifier: {args.n_trials}")
        
        # Simple benchmark counter
        successful_benchmarks = []
        failed_benchmarks = []
        
        classifier_desc = f"{args.classifier_type} classifier{'s' if args.classifier_type == 'both' else ''}"
        
        for i, benchmark in enumerate(benchmarks, 1):
            console.print(f"\n📊 Training {classifier_desc} - Benchmark {i}/{len(benchmarks)}: [cyan]{benchmark}[/cyan]")
            
            success = train_single_benchmark(
                benchmark=benchmark,
                classifier_type=args.classifier_type,
                n_trials=args.n_trials,
                data_dir=args.data_dir
            )
            
            if success:
                successful_benchmarks.append(benchmark)
                console.print(f"✅ Completed benchmark: [green]{benchmark}[/green]")
            else:
                failed_benchmarks.append(benchmark)
                console.print(f"❌ Failed benchmark: [red]{benchmark}[/red]")
        
        # Training summary
        print(f"\n📊 Training Summary:")
        print(f"✅ Successful: {len(successful_benchmarks)}")
        print(f"❌ Failed: {len(failed_benchmarks)}")
        
        if failed_benchmarks:
            print(f"Failed benchmarks: {failed_benchmarks}")
        
        if not successful_benchmarks:
            print("❌ No successful training runs! Cannot proceed with comparison.")
            sys.exit(1)
    
    # Comparison phase
    if args.enable_comparison:
        print(f"\n📈 Starting comparison analysis...")
        
        try:
            comparison_results = run_full_comparison(
                benchmarks=benchmarks if not args.skip_training else None,
                results_dir=config.RESULTS_DIR
            )
            
            if comparison_results and 'improvements' in comparison_results:
                improvements = comparison_results['improvements']
                
                print(f"\n🎉 Comparison Results:")
                print("=" * 30)
                
                if 'n_comparisons' in improvements:
                    print(f"Benchmarks compared: {improvements['n_comparisons']}")
                
                # Print key results
                for metric in ['accuracy', 'f1_score', 'precision', 'recall']:
                    if metric in improvements:
                        imp = improvements[metric]
                        significance = "✓" if imp.get('statistically_significant', False) else "✗"
                        print(f"{metric.capitalize()}:")
                        print(f"  Mean improvement: {imp['mean_relative_improvement_percent']:.2f}%")
                        print(f"  Win rate: {imp['win_rate']:.1%}")
                        print(f"  Statistically significant: {significance}")
                        print()
                
                # Overall conclusion
                if 'accuracy' in improvements:
                    acc_improvement = improvements['accuracy']
                    if acc_improvement.get('statistically_significant', False):
                        if acc_improvement['mean_relative_improvement_percent'] > 0:
                            print("🎯 CONCLUSION: Enhanced classifier shows significant improvement!")
                        else:
                            print("⚠️  CONCLUSION: Enhanced classifier shows significant degradation.")
                    else:
                        print("⚪ CONCLUSION: No significant difference between classifiers.")
                
                print(f"\n📁 Results saved to: {config.RESULTS_DIR}")
                print(f"📊 Plots saved to: {config.PLOTS_DIR}")
                print("📄 Check comparison_report.md for detailed analysis")
            
            else:
                print("❌ Comparison failed - no results found")
        
        except Exception as e:
            print(f"❌ Comparison failed: {str(e)}")
            sys.exit(1)
    else:
        print("\n📈 Comparison skipped (use --enable-comparison to enable)")
    
    print("\n🏁 Training and comparison completed successfully!")


if __name__ == "__main__":
    main()