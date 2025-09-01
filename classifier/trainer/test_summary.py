"""
Test set summary generation for classifier training results.
Creates comprehensive summaries of test set performance metrics for each benchmark.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import numpy as np

from .. import config


def generate_test_summary(benchmark: str, results_dir: str = config.RESULTS_DIR) -> Dict[str, Any]:
    """
    Generate a comprehensive test set summary for a benchmark.
    
    Args:
        benchmark: Name of the benchmark
        results_dir: Directory containing training results
        
    Returns:
        Dictionary containing test set summary metrics
    """
    results_path = Path(results_dir) / benchmark
    summary = {
        'benchmark': benchmark,
        'generated_at': datetime.now().isoformat(),
        'classifiers': {}
    }
    
    # Check for both basic and enhanced classifier results
    for classifier_type in ['basic', 'enhanced']:
        results_file = results_path / f"{classifier_type}_tuning_results.json"
        
        if results_file.exists():
            try:
                with open(results_file, 'r') as f:
                    results = json.load(f)
                
                # Extract test metrics if available
                test_metrics = results.get('test_metrics', {})
                
                if test_metrics:
                    summary['classifiers'][classifier_type] = {
                        'test_metrics': test_metrics,
                        'hyperparameters': results.get('best_hyperparameters', {}),
                        'training_info': {
                            'epochs_trained': results.get('epochs_trained'),
                            'training_time': results.get('training_time'),
                            'tuning_trials': results.get('tuning_trials')
                        }
                    }
                else:
                    summary['classifiers'][classifier_type] = {
                        'status': 'no_test_metrics',
                        'note': 'Test metrics not found in results'
                    }
                    
            except Exception as e:
                summary['classifiers'][classifier_type] = {
                    'status': 'error',
                    'error': str(e)
                }
        else:
            summary['classifiers'][classifier_type] = {
                'status': 'not_found',
                'note': f'Results file {results_file} does not exist'
            }
    
    # Add comparison if both classifiers exist
    if ('basic' in summary['classifiers'] and 
        'enhanced' in summary['classifiers'] and
        'test_metrics' in summary['classifiers']['basic'] and
        'test_metrics' in summary['classifiers']['enhanced']):
        
        basic_metrics = summary['classifiers']['basic']['test_metrics']
        enhanced_metrics = summary['classifiers']['enhanced']['test_metrics']
        
        comparison = {}
        for metric in config.CLASSIFICATION_METRICS:
            if metric in basic_metrics and metric in enhanced_metrics:
                basic_val = basic_metrics[metric]
                enhanced_val = enhanced_metrics[metric]
                
                improvement = enhanced_val - basic_val
                improvement_pct = (improvement / basic_val * 100) if basic_val != 0 else 0
                
                comparison[metric] = {
                    'basic': basic_val,
                    'enhanced': enhanced_val,
                    'improvement': improvement,
                    'improvement_percent': improvement_pct,
                    'enhanced_better': enhanced_val > basic_val
                }
        
        summary['comparison'] = comparison
        
        # Overall performance assessment
        accuracy_improvement = comparison.get('accuracy', {}).get('improvement_percent', 0)
        summary['overall_assessment'] = {
            'enhanced_better': accuracy_improvement > 0,
            'accuracy_improvement_percent': accuracy_improvement,
            'significant_improvement': abs(accuracy_improvement) > 1.0  # >1% considered significant
        }
    
    return summary


def generate_all_benchmarks_summary(benchmarks: Optional[List[str]] = None, 
                                   results_dir: str = config.RESULTS_DIR) -> Dict[str, Any]:
    """
    Generate test set summaries for all available benchmarks.
    
    Args:
        benchmarks: List of specific benchmarks to summarize (default: all available)
        results_dir: Directory containing training results
        
    Returns:
        Dictionary containing summaries for all benchmarks
    """
    results_path = Path(results_dir)
    
    if benchmarks is None:
        # Find all benchmark directories
        if results_path.exists():
            benchmarks = [d.name for d in results_path.iterdir() if d.is_dir()]
        else:
            benchmarks = []
    
    all_summaries = {
        'generated_at': datetime.now().isoformat(),
        'results_directory': str(results_dir),
        'benchmarks_found': len(benchmarks),
        'benchmarks': {}
    }
    
    successful_summaries = 0
    
    for benchmark in benchmarks:
        try:
            benchmark_summary = generate_test_summary(benchmark, results_dir)
            all_summaries['benchmarks'][benchmark] = benchmark_summary
            
            # Check if summary contains actual test results
            has_results = any(
                'test_metrics' in classifier_data 
                for classifier_data in benchmark_summary['classifiers'].values()
                if isinstance(classifier_data, dict) and 'test_metrics' in classifier_data
            )
            
            if has_results:
                successful_summaries += 1
                
        except Exception as e:
            all_summaries['benchmarks'][benchmark] = {
                'status': 'error',
                'error': str(e)
            }
    
    all_summaries['successful_summaries'] = successful_summaries
    
    return all_summaries


def save_test_summaries(summaries: Dict[str, Any], results_dir: str = config.RESULTS_DIR):
    """
    Save test summaries to files in the training_results directory.
    
    Args:
        summaries: Summary data from generate_all_benchmarks_summary
        results_dir: Directory to save summaries
    """
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    
    # Save overall summary
    overall_summary_path = results_path / "test_set_summary.json"
    with open(overall_summary_path, 'w') as f:
        json.dump(summaries, f, indent=2)
    
    # Save individual benchmark summaries
    for benchmark, benchmark_summary in summaries['benchmarks'].items():
        benchmark_path = results_path / benchmark
        benchmark_path.mkdir(parents=True, exist_ok=True)
        
        individual_summary_path = benchmark_path / "test_summary.json"
        with open(individual_summary_path, 'w') as f:
            json.dump(benchmark_summary, f, indent=2)
    
    # Create a readable markdown report
    create_markdown_report(summaries, results_path / "test_set_report.md")
    
    return overall_summary_path


def create_markdown_report(summaries: Dict[str, Any], output_path: Path):
    """
    Create a human-readable markdown report of test results.
    
    Args:
        summaries: Summary data from generate_all_benchmarks_summary
        output_path: Path to save the markdown report
    """
    report_lines = []
    
    # Header
    report_lines.extend([
        "# Test Set Performance Summary",
        "",
        f"**Generated at:** {summaries['generated_at']}",
        f"**Results directory:** {summaries['results_directory']}",
        f"**Benchmarks processed:** {summaries['benchmarks_found']}",
        f"**Successful summaries:** {summaries['successful_summaries']}",
        ""
    ])
    
    # Individual benchmark results
    report_lines.append("## Benchmark Results\n")
    
    for benchmark, benchmark_data in summaries['benchmarks'].items():
        if 'status' in benchmark_data:
            # Handle error cases
            report_lines.extend([
                f"### {benchmark}",
                f"**Status:** {benchmark_data['status']}",
                ""
            ])
            if 'error' in benchmark_data:
                report_lines.extend([
                    f"**Error:** {benchmark_data['error']}",
                    ""
                ])
            continue
        
        report_lines.append(f"### {benchmark}\n")
        
        # Test metrics table
        has_test_data = False
        table_rows = []
        
        for classifier_type in ['basic', 'enhanced']:
            if (classifier_type in benchmark_data['classifiers'] and 
                'test_metrics' in benchmark_data['classifiers'][classifier_type]):
                
                has_test_data = True
                metrics = benchmark_data['classifiers'][classifier_type]['test_metrics']
                
                row = f"| {classifier_type.title()} |"
                for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']:
                    if metric in metrics:
                        row += f" {metrics[metric]:.4f} |"
                    else:
                        row += " N/A |"
                
                table_rows.append(row)
        
        if has_test_data:
            report_lines.extend([
                "| Classifier | Accuracy | Precision | Recall | F1 Score | AUC-ROC |",
                "|------------|----------|-----------|--------|----------|---------|"
            ])
            report_lines.extend(table_rows)
            report_lines.append("")
            
            # Comparison if available
            if 'comparison' in benchmark_data:
                report_lines.append("**Performance Comparison:**\n")
                comparison = benchmark_data['comparison']
                
                for metric, comp_data in comparison.items():
                    improvement = comp_data['improvement_percent']
                    symbol = "📈" if improvement > 0 else "📉" if improvement < 0 else "➡️"
                    
                    report_lines.append(
                        f"- **{metric.title()}:** {symbol} "
                        f"{improvement:+.2f}% "
                        f"({comp_data['basic']:.4f} → {comp_data['enhanced']:.4f})"
                    )
                
                report_lines.append("")
                
                # Overall assessment
                if 'overall_assessment' in benchmark_data:
                    assessment = benchmark_data['overall_assessment']
                    if assessment['enhanced_better']:
                        report_lines.append(
                            f"✅ **Enhanced classifier outperforms basic** "
                            f"(+{assessment['accuracy_improvement_percent']:.2f}% accuracy)"
                        )
                    else:
                        report_lines.append(
                            f"❌ **Basic classifier outperforms enhanced** "
                            f"({assessment['accuracy_improvement_percent']:.2f}% accuracy difference)"
                        )
                    report_lines.append("")
        else:
            report_lines.extend([
                "⚠️ No test metrics available for this benchmark.",
                ""
            ])
    
    # Summary statistics
    if summaries['successful_summaries'] > 0:
        report_lines.extend([
            "## Overall Statistics",
            ""
        ])
        
        # Collect stats across all benchmarks
        total_comparisons = 0
        enhanced_wins = 0
        accuracy_improvements = []
        
        for benchmark_data in summaries['benchmarks'].values():
            if ('overall_assessment' in benchmark_data and 
                isinstance(benchmark_data['overall_assessment'], dict)):
                
                total_comparisons += 1
                assessment = benchmark_data['overall_assessment']
                
                if assessment.get('enhanced_better', False):
                    enhanced_wins += 1
                
                acc_imp = assessment.get('accuracy_improvement_percent', 0)
                if acc_imp is not None and not np.isnan(acc_imp):
                    accuracy_improvements.append(acc_imp)
        
        if total_comparisons > 0:
            win_rate = enhanced_wins / total_comparisons
            mean_improvement = np.mean(accuracy_improvements) if accuracy_improvements else 0
            
            report_lines.extend([
                f"- **Total comparisons:** {total_comparisons}",
                f"- **Enhanced classifier win rate:** {win_rate:.1%} ({enhanced_wins}/{total_comparisons})",
                f"- **Average accuracy improvement:** {mean_improvement:+.2f}%",
                ""
            ])
    
    # Write the report
    with open(output_path, 'w') as f:
        f.write('\n'.join(report_lines))


def create_test_summaries_after_training(benchmarks: List[str], 
                                        results_dir: str = config.RESULTS_DIR) -> Path:
    """
    Convenience function to create test summaries after training completion.
    
    Args:
        benchmarks: List of benchmarks that were trained
        results_dir: Directory containing training results
        
    Returns:
        Path to the saved summary file
    """
    print("\n📊 Generating test set summaries...")
    
    summaries = generate_all_benchmarks_summary(benchmarks, results_dir)
    summary_file = save_test_summaries(summaries, results_dir)
    
    successful = summaries['successful_summaries']
    total = summaries['benchmarks_found']
    
    print(f"✅ Test summaries generated for {successful}/{total} benchmarks")
    print(f"📄 Summary saved to: {summary_file}")
    print(f"📄 Markdown report: {Path(results_dir) / 'test_set_report.md'}")
    
    return summary_file


if __name__ == "__main__":
    # Test summary generation
    from .data_loader import get_available_benchmarks
    
    benchmarks = get_available_benchmarks(".")
    if benchmarks:
        print(f"Testing summary generation with available benchmarks: {benchmarks}")
        
        summaries = generate_all_benchmarks_summary(benchmarks[:1])  # Test with first benchmark
        print("Sample summary:")
        print(json.dumps(summaries, indent=2)[:500] + "...")
        
        # Save summaries
        summary_file = save_test_summaries(summaries)
        print(f"Test summary saved to: {summary_file}")
    else:
        print("No benchmarks found for testing")