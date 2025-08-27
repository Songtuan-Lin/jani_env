"""
Comparison utilities for analyzing results between basic and enhanced classifiers.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import Dict, List, Any
from scipy import stats
from .. import config


def load_results(benchmark: str, results_dir: str = config.RESULTS_DIR) -> Dict[str, Dict]:
    """Load training results for both classifiers."""
    results_path = Path(results_dir) / benchmark
    
    results = {}
    for classifier_type in ['basic', 'enhanced']:
        results_file = results_path / f"{classifier_type}_tuning_results.json"
        if results_file.exists():
            with open(results_file, 'r') as f:
                results[classifier_type] = json.load(f)
        else:
            print(f"Warning: Results file not found: {results_file}")
    
    return results


def create_comparison_dataframe(all_results: Dict[str, Dict[str, Dict]]) -> pd.DataFrame:
    """Create a DataFrame comparing results across benchmarks and classifiers."""
    data = []
    
    for benchmark, benchmark_results in all_results.items():
        for classifier_type, results in benchmark_results.items():
            if 'test_metrics' in results:
                metrics = results['test_metrics']
                row = {
                    'benchmark': benchmark,
                    'classifier': classifier_type,
                    'accuracy': metrics['accuracy'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'f1_score': metrics['f1_score'],
                    'auc_roc': metrics['auc_roc'],
                    'training_time': results['training_time'],
                    'epochs_trained': results['epochs_trained']
                }
                
                # Add hyperparameters if available
                if 'best_hyperparameters' in results:
                    hp = results['best_hyperparameters']
                    row.update({
                        'n_layers': hp.get('n_layers'),
                        'dropout': hp.get('dropout'),
                        'learning_rate': hp.get('learning_rate')
                    })
                
                data.append(row)
    
    return pd.DataFrame(data)


def calculate_improvement_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate improvement statistics when using enhanced vs basic classifier."""
    improvements = {}
    
    # Get paired data (same benchmark, different classifier)
    basic_df = df[df['classifier'] == 'basic'].set_index('benchmark')
    enhanced_df = df[df['classifier'] == 'enhanced'].set_index('benchmark')
    
    # Find common benchmarks
    common_benchmarks = basic_df.index.intersection(enhanced_df.index)
    
    if len(common_benchmarks) == 0:
        return {'error': 'No common benchmarks found for comparison'}
    
    for metric in config.METRICS:
        if metric in basic_df.columns and metric in enhanced_df.columns:
            basic_values = basic_df.loc[common_benchmarks, metric]
            enhanced_values = enhanced_df.loc[common_benchmarks, metric]
            
            # Calculate improvements
            absolute_improvements = enhanced_values - basic_values
            relative_improvements = (enhanced_values - basic_values) / basic_values * 100
            
            # Statistical test
            stat, p_value = stats.wilcoxon(basic_values, enhanced_values, 
                                         alternative='two-sided', zero_method='zsplit')
            
            improvements[metric] = {
                'mean_absolute_improvement': float(absolute_improvements.mean()),
                'median_absolute_improvement': float(absolute_improvements.median()),
                'std_absolute_improvement': float(absolute_improvements.std()),
                'mean_relative_improvement_percent': float(relative_improvements.mean()),
                'median_relative_improvement_percent': float(relative_improvements.median()),
                'wins': int((absolute_improvements > 0).sum()),
                'losses': int((absolute_improvements < 0).sum()),
                'ties': int((absolute_improvements == 0).sum()),
                'win_rate': float((absolute_improvements > 0).mean()),
                'wilcoxon_statistic': float(stat),
                'wilcoxon_p_value': float(p_value),
                'statistically_significant': p_value < 0.05
            }
    
    improvements['n_comparisons'] = len(common_benchmarks)
    improvements['common_benchmarks'] = list(common_benchmarks)
    
    return improvements


def create_comparison_plots(df: pd.DataFrame, output_dir: str = config.PLOTS_DIR):
    """Create visualization plots comparing classifiers."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Bar plot comparing metrics
    metrics_to_plot = [m for m in config.METRICS if m in df.columns]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics_to_plot):
        if i < len(axes):
            ax = axes[i]
            
            # Create grouped bar plot
            metric_data = df.groupby('classifier')[metric].agg(['mean', 'std']).reset_index()
            
            x = np.arange(len(metric_data))
            width = 0.35
            
            bars = ax.bar(x, metric_data['mean'], width, 
                         yerr=metric_data['std'], capsize=5, alpha=0.8)
            
            ax.set_xlabel('Classifier Type')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f'{metric.replace("_", " ").title()} Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(metric_data['classifier'])
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom')
    
    # Remove empty subplots
    for i in range(len(metrics_to_plot), len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.savefig(output_path / 'metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Scatter plot: Basic vs Enhanced performance
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    # Get paired data
    basic_df = df[df['classifier'] == 'basic'].set_index('benchmark')
    enhanced_df = df[df['classifier'] == 'enhanced'].set_index('benchmark')
    common_benchmarks = basic_df.index.intersection(enhanced_df.index)
    
    main_metrics = ['accuracy', 'f1_score', 'precision', 'recall']
    
    for i, metric in enumerate(main_metrics):
        if i < len(axes) and metric in basic_df.columns:
            ax = axes[i]
            
            basic_values = basic_df.loc[common_benchmarks, metric]
            enhanced_values = enhanced_df.loc[common_benchmarks, metric]
            
            # Scatter plot
            ax.scatter(basic_values, enhanced_values, alpha=0.7, s=50)
            
            # Diagonal line (y=x)
            min_val = min(basic_values.min(), enhanced_values.min())
            max_val = max(basic_values.max(), enhanced_values.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
            
            ax.set_xlabel(f'Basic Classifier {metric}')
            ax.set_ylabel(f'Enhanced Classifier {metric}')
            ax.set_title(f'{metric.replace("_", " ").title()}: Basic vs Enhanced')
            ax.grid(True, alpha=0.3)
            
            # Add benchmark labels
            for benchmark in common_benchmarks:
                ax.annotate(benchmark.split('_')[0], 
                           (basic_values[benchmark], enhanced_values[benchmark]),
                           xytext=(5, 5), textcoords='offset points', 
                           fontsize=8, alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_path / 'basic_vs_enhanced_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Training time comparison
    plt.figure(figsize=(10, 6))
    
    # Box plot for training times
    training_time_data = df[['classifier', 'training_time']].dropna()
    if not training_time_data.empty:
        sns.boxplot(data=training_time_data, x='classifier', y='training_time')
        plt.title('Training Time Comparison')
        plt.ylabel('Training Time (seconds)')
        plt.xlabel('Classifier Type')
        plt.grid(True, alpha=0.3)
        
        plt.savefig(output_path / 'training_time_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison plots saved to {output_path}")


def generate_comparison_report(all_results: Dict[str, Dict[str, Dict]], 
                             output_file: str = "comparison_report.md") -> str:
    """Generate a comprehensive comparison report in Markdown format."""
    df = create_comparison_dataframe(all_results)
    improvements = calculate_improvement_statistics(df)
    
    report = []
    report.append("# Classifier Comparison Report\n")
    report.append(f"**Generated for {len(df['benchmark'].unique())} benchmarks**\n")
    
    # Overview statistics
    report.append("## Overview\n")
    
    basic_count = len(df[df['classifier'] == 'basic'])
    enhanced_count = len(df[df['classifier'] == 'enhanced'])
    
    report.append(f"- Basic classifiers trained: {basic_count}")
    report.append(f"- Enhanced classifiers trained: {enhanced_count}")
    report.append(f"- Common benchmarks for comparison: {improvements.get('n_comparisons', 0)}\n")
    
    # Performance comparison
    report.append("## Performance Comparison\n")
    
    if 'accuracy' in improvements:
        acc_improvement = improvements['accuracy']
        report.append(f"### Accuracy")
        report.append(f"- Mean improvement: {acc_improvement['mean_relative_improvement_percent']:.2f}%")
        report.append(f"- Enhanced classifier wins: {acc_improvement['wins']}/{improvements['n_comparisons']} ({acc_improvement['win_rate']:.1%})")
        
        if acc_improvement['statistically_significant']:
            report.append(f"- **Statistically significant improvement** (p={acc_improvement['wilcoxon_p_value']:.4f})")
        else:
            report.append(f"- No statistically significant difference (p={acc_improvement['wilcoxon_p_value']:.4f})")
        report.append("")
    
    # Detailed metrics table
    report.append("## Detailed Results by Metric\n")
    report.append("| Metric | Mean Improvement (%) | Win Rate | Statistical Significance |")
    report.append("|--------|---------------------|----------|------------------------|")
    
    for metric in config.METRICS:
        if metric in improvements:
            imp = improvements[metric]
            significance = "✓" if imp['statistically_significant'] else "✗"
            report.append(f"| {metric.replace('_', ' ').title()} | {imp['mean_relative_improvement_percent']:.2f}% | {imp['win_rate']:.1%} | {significance} |")
    
    report.append("")
    
    # Summary by benchmark
    report.append("## Results by Benchmark\n")
    
    # Create summary table
    summary_df = df.pivot(index='benchmark', columns='classifier', values='accuracy')
    if 'basic' in summary_df.columns and 'enhanced' in summary_df.columns:
        summary_df['improvement'] = summary_df['enhanced'] - summary_df['basic']
        summary_df['improvement_pct'] = (summary_df['improvement'] / summary_df['basic'] * 100)
        
        report.append("| Benchmark | Basic Accuracy | Enhanced Accuracy | Improvement |")
        report.append("|-----------|----------------|-------------------|-------------|")
        
        for benchmark in summary_df.index:
            basic_acc = summary_df.loc[benchmark, 'basic']
            enhanced_acc = summary_df.loc[benchmark, 'enhanced']
            improvement_pct = summary_df.loc[benchmark, 'improvement_pct']
            
            if pd.notna(basic_acc) and pd.notna(enhanced_acc):
                report.append(f"| {benchmark} | {basic_acc:.4f} | {enhanced_acc:.4f} | {improvement_pct:+.2f}% |")
    
    report.append("")
    
    # Conclusions
    report.append("## Conclusions\n")
    
    if 'accuracy' in improvements:
        acc_improvement = improvements['accuracy']
        if acc_improvement['statistically_significant']:
            if acc_improvement['mean_relative_improvement_percent'] > 0:
                report.append("✓ **The enhanced classifier (with next state information) shows statistically significant improvement over the basic classifier.**")
            else:
                report.append("⚠️ **The enhanced classifier shows statistically significant but negative performance compared to the basic classifier.**")
        else:
            report.append("⚪ **No statistically significant difference found between the two classifiers.**")
    
    report.append(f"\nThis suggests that including next state information {'does' if improvements.get('accuracy', {}).get('mean_relative_improvement_percent', 0) > 0 else 'does not'} improve classification performance on average.")
    
    # Save report
    output_path = Path(config.RESULTS_DIR) / output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    report_text = "\n".join(report)
    with open(output_path, 'w') as f:
        f.write(report_text)
    
    print(f"Comparison report saved to {output_path}")
    return report_text


def run_full_comparison(benchmarks: List[str] = None, 
                       results_dir: str = config.RESULTS_DIR) -> Dict:
    """Run complete comparison analysis for all available benchmarks."""
    if benchmarks is None:
        # Auto-detect benchmarks with results
        results_path = Path(results_dir)
        if results_path.exists():
            benchmarks = [d.name for d in results_path.iterdir() if d.is_dir()]
        else:
            benchmarks = []
    
    print(f"Running comparison for {len(benchmarks)} benchmarks...")
    
    # Load all results
    all_results = {}
    for benchmark in benchmarks:
        results = load_results(benchmark, results_dir)
        if results:
            all_results[benchmark] = results
    
    if not all_results:
        print("No results found for comparison!")
        return {}
    
    # Create comparison DataFrame
    df = create_comparison_dataframe(all_results)
    
    # Calculate statistics
    improvements = calculate_improvement_statistics(df)
    
    # Create plots
    create_comparison_plots(df)
    
    # Generate report
    report = generate_comparison_report(all_results)
    
    # Save comparison data
    comparison_data = {
        'dataframe': df.to_dict(),
        'improvements': improvements,
        'benchmarks_analyzed': list(all_results.keys())
    }
    
    comparison_file = Path(results_dir) / 'comparison_data.json'
    with open(comparison_file, 'w') as f:
        json.dump(comparison_data, f, indent=2, default=str)
    
    print(f"Full comparison completed! Results saved to {results_dir}")
    
    return {
        'dataframe': df,
        'improvements': improvements,
        'all_results': all_results,
        'report': report
    }


if __name__ == "__main__":
    # Test comparison functionality
    print("Testing comparison functionality...")
    comparison_results = run_full_comparison()
    
    if comparison_results:
        print("\nComparison Summary:")
        df = comparison_results['dataframe']
        print(f"Analyzed {len(df)} classifier results")
        print(f"Benchmarks: {df['benchmark'].unique()}")
        
        improvements = comparison_results['improvements']
        if 'accuracy' in improvements:
            acc_imp = improvements['accuracy']
            print(f"Average accuracy improvement: {acc_imp['mean_relative_improvement_percent']:.2f}%")
            print(f"Enhanced classifier wins: {acc_imp['win_rate']:.1%} of comparisons")