#!/usr/bin/env python3
"""
Evaluation module for trained RL policies on benchmark datasets.

This module provides functionality to:
- Load and evaluate trained policies on benchmark directories
- Process benchmark subdirectories containing model.jani, property.jani, and final_model.zip
- Generate evaluation summaries across all benchmarks
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import sys

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from gymnasium.wrappers import TimeLimit

from jani.environment import JaniEnv


def mask_fn(env) -> np.ndarray:
    """Action masking function for the environment."""
    return env.unwrapped.action_mask()


class BenchmarkEvaluator:
    """Evaluates trained policies on benchmark datasets."""
    
    def __init__(self, benchmark_dir: str, n_eval_episodes: int = 10, max_episode_steps: int = 2048, verbose: int = 1):
        """
        Initialize the benchmark evaluator.
        
        Args:
            benchmark_dir: Directory containing benchmark subdirectories
            n_eval_episodes: Number of episodes to evaluate per benchmark
            max_episode_steps: Maximum number of steps per episode to prevent infinite cycles
            verbose: Verbosity level
        """
        self.benchmark_dir = Path(benchmark_dir)
        self.n_eval_episodes = n_eval_episodes
        self.max_episode_steps = max_episode_steps
        self.verbose = verbose
        self.results = []
        
        if not self.benchmark_dir.exists():
            raise FileNotFoundError(f"Benchmark directory not found: {benchmark_dir}")
    
    def _find_benchmarks(self) -> List[Path]:
        """Find all benchmark subdirectories."""
        benchmarks = []
        for subdir in self.benchmark_dir.iterdir():
            if subdir.is_dir():
                required_files = ['model.jani', 'property.jani', 'final_model.zip']
                if all((subdir / file).exists() for file in required_files):
                    benchmarks.append(subdir)
                elif self.verbose >= 1:
                    print(f"⚠️  Skipping {subdir.name}: missing required files")
        
        if self.verbose >= 1:
            print(f"Found {len(benchmarks)} valid benchmarks")
        
        return sorted(benchmarks)
    
    def _create_environment(self, benchmark_path: Path) -> Any:
        """Create evaluation environment for a benchmark."""
        model_file = str(benchmark_path / 'model.jani')
        property_file = str(benchmark_path / 'property.jani')
        
        # Create environment with TimeLimit wrapper to prevent infinite cycles
        env = JaniEnv(model_file=model_file, property_file=property_file)
        env = TimeLimit(env, max_episode_steps=self.max_episode_steps)
        env = ActionMasker(env, mask_fn)
        
        return env
    
    def _evaluate_benchmark(self, benchmark_path: Path) -> Dict[str, Any]:
        """Evaluate a single benchmark."""
        benchmark_name = benchmark_path.name
        
        if self.verbose >= 1:
            print(f"📊 Evaluating benchmark: {benchmark_name}")
        
        try:
            # Create environment
            env = self._create_environment(benchmark_path)
            
            # Load the trained model using sb3's load function
            policy_path = str(benchmark_path / 'final_model.zip')
            model = MaskablePPO.load(policy_path)
            
            # Evaluate policy (TimeLimit wrapper handles max episode length)
            mean_reward, std_reward = evaluate_policy(
                model, env, n_eval_episodes=self.n_eval_episodes, deterministic=True
            )
            
            # Calculate success rate (assuming positive rewards indicate success)
            success_rate = 1.0 if mean_reward > 0 else 0.0
            
            result = {
                'benchmark_name': benchmark_name,
                'mean_reward': float(mean_reward),
                'std_reward': float(std_reward),
                'success_rate': float(success_rate),
                'n_episodes': self.n_eval_episodes,
                'status': 'success'
            }
            
            if self.verbose >= 1:
                print(f"  ✅ Mean reward: {mean_reward:.3f} ± {std_reward:.3f}")
                print(f"  ✅ Success rate: {success_rate:.1%}")
            
            env.close()
            return result
            
        except Exception as e:
            if self.verbose >= 1:
                print(f"  ❌ Error evaluating {benchmark_name}: {e}")
            
            return {
                'benchmark_name': benchmark_name,
                'mean_reward': 0.0,
                'std_reward': 0.0,
                'success_rate': 0.0,
                'n_episodes': 0,
                'status': 'error',
                'error': str(e)
            }
    
    def evaluate_all(self) -> List[Dict[str, Any]]:
        """Evaluate all benchmarks and return results."""
        benchmarks = self._find_benchmarks()
        
        if not benchmarks:
            print("❌ No valid benchmarks found!")
            return []
        
        print(f"🚀 Starting evaluation of {len(benchmarks)} benchmarks...")
        
        self.results = []
        for benchmark in benchmarks:
            result = self._evaluate_benchmark(benchmark)
            self.results.append(result)
        
        return self.results
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics across all benchmarks."""
        if not self.results:
            return {}
        
        successful_results = [r for r in self.results if r['status'] == 'success']
        
        if not successful_results:
            return {
                'total_benchmarks': len(self.results),
                'successful_evaluations': 0,
                'failed_evaluations': len(self.results),
                'overall_success_rate': 0.0,
                'mean_reward_avg': 0.0,
                'mean_reward_std': 0.0,
                'benchmark_success_rate_avg': 0.0,
                'benchmark_success_rate_std': 0.0
            }
        
        # Calculate aggregate statistics
        mean_rewards = [r['mean_reward'] for r in successful_results]
        success_rates = [r['success_rate'] for r in successful_results]
        
        summary = {
            'total_benchmarks': len(self.results),
            'successful_evaluations': len(successful_results),
            'failed_evaluations': len(self.results) - len(successful_results),
            'overall_success_rate': len(successful_results) / len(self.results),
            'mean_reward_avg': float(np.mean(mean_rewards)),
            'mean_reward_std': float(np.std(mean_rewards)),
            'benchmark_success_rate_avg': float(np.mean(success_rates)),
            'benchmark_success_rate_std': float(np.std(success_rates)),
            'best_benchmark': max(successful_results, key=lambda x: x['mean_reward'])['benchmark_name'],
            'worst_benchmark': min(successful_results, key=lambda x: x['mean_reward'])['benchmark_name'],
        }
        
        return summary
    
    def save_results(self, output_path: Optional[str] = None) -> str:
        """Save evaluation results and summary to JSON file."""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.benchmark_dir / f"evaluation_results_{timestamp}.json"
        else:
            output_path = Path(output_path)
        
        # Prepare output data
        output_data = {
            'evaluation_metadata': {
                'benchmark_directory': str(self.benchmark_dir),
                'n_eval_episodes': self.n_eval_episodes,
                'timestamp': datetime.now().isoformat(),
                'total_benchmarks': len(self.results)
            },
            'summary': self.generate_summary(),
            'detailed_results': self.results
        }
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        if self.verbose >= 1:
            print(f"📄 Results saved to: {output_path}")
        
        return str(output_path)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate trained RL policies on benchmark datasets',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('benchmark_dir', type=str,
                       help='Directory containing benchmark subdirectories')
    parser.add_argument('--n_episodes', type=int, default=10,
                       help='Number of episodes to evaluate per benchmark')
    parser.add_argument('--max_steps', type=int, default=2048,
                       help='Maximum number of steps per episode to prevent infinite cycles')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file path for results (default: auto-generated)')
    parser.add_argument('--verbose', type=int, default=1,
                       help='Verbosity level (0=quiet, 1=normal, 2=verbose)')
    
    return parser.parse_args()


def main():
    """Main evaluation function."""
    args = parse_arguments()
    
    try:
        # Create evaluator
        evaluator = BenchmarkEvaluator(
            benchmark_dir=args.benchmark_dir,
            n_eval_episodes=args.n_episodes,
            max_episode_steps=args.max_steps,
            verbose=args.verbose
        )
        
        # Run evaluation
        results = evaluator.evaluate_all()
        
        if not results:
            print("❌ No benchmarks were evaluated successfully.")
            sys.exit(1)
        
        # Generate and display summary
        summary = evaluator.generate_summary()
        
        print("\n" + "="*50)
        print("📊 EVALUATION SUMMARY")
        print("="*50)
        print(f"Total benchmarks: {summary['total_benchmarks']}")
        print(f"Successful evaluations: {summary['successful_evaluations']}")
        print(f"Failed evaluations: {summary['failed_evaluations']}")
        print(f"Overall success rate: {summary['overall_success_rate']:.1%}")
        
        if summary['successful_evaluations'] > 0:
            print(f"Average mean reward: {summary['mean_reward_avg']:.3f} ± {summary['mean_reward_std']:.3f}")
            print(f"Average benchmark success rate: {summary['benchmark_success_rate_avg']:.1%} ± {summary['benchmark_success_rate_std']:.1%}")
            print(f"Best performing benchmark: {summary['best_benchmark']}")
            print(f"Worst performing benchmark: {summary['worst_benchmark']}")
        
        # Save results
        output_file = evaluator.save_results(args.output)
        
        print(f"\n✅ Evaluation completed! Results saved to: {output_file}")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()