#!/usr/bin/env python3
"""
Example script demonstrating how to use the comprehensive training script.

This script shows various usage patterns and can be used as a template.
"""

import subprocess
import sys
from pathlib import Path


def run_training_example():
    """Run example training commands."""
    
    # Example 1: Basic training with individual files
    print("Example 1: Basic training with individual files")
    cmd1 = [
        sys.executable, "train_comprehensive.py",
        "--model_file", "examples/inverted_pendulum/inverted_pendulum.jani",
        "--start_file", "examples/inverted_pendulum/start.jani", 
        "--goal_file", "examples/inverted_pendulum/objective.jani",
        "--safe_file", "examples/inverted_pendulum/safe.jani",
        "--total_timesteps", "100000",
        "--n_envs", "4",
        "--experiment_name", "inverted_pendulum_basic",
        "--wandb_project", "jani-examples"
    ]
    print("Command:", " ".join(cmd1))
    print()
    
    # Example 2: Training with property file
    print("Example 2: Training with property file (if available)")
    cmd2 = [
        sys.executable, "train_comprehensive.py", 
        "--model_file", "examples/mod_linetrack/mod_linetrack.jani",
        "--property_file", "examples/mod_linetrack/property.jani",
        "--total_timesteps", "500000",
        "--experiment_name", "linetrack_property",
        "--wandb_project", "jani-examples"
    ]
    print("Command:", " ".join(cmd2))
    print()
    
    # Example 3: Training with hyperparameter tuning
    print("Example 3: Training with hyperparameter tuning")
    cmd3 = [
        sys.executable, "train_comprehensive.py",
        "--model_file", "examples/bouncing_ball/bouncing_ball.jani",
        "--start_file", "examples/bouncing_ball/start.jani",
        "--goal_file", "examples/bouncing_ball/objective.jani", 
        "--safe_file", "examples/bouncing_ball/safe.jani",
        "--tune_hyperparams",
        "--n_trials", "20",
        "--total_timesteps", "200000",
        "--experiment_name", "bouncing_ball_tuned",
        "--wandb_project", "jani-tuning",
        "--study_name", "bouncing_ball_study"
    ]
    print("Command:", " ".join(cmd3))
    print()
    
    # Example 4: Simple test run
    print("Example 4: Quick test run")
    cmd4 = [
        sys.executable, "train_comprehensive.py",
        "--model_file", "examples/simple_test.jani",
        "--start_file", "examples/simple_start.jani",
        "--goal_file", "examples/simple_goal.jani",
        "--safe_file", "examples/simple_failure.jani", 
        "--total_timesteps", "10000",
        "--n_envs", "2",
        "--eval_freq", "2000",
        "--experiment_name", "simple_test_quick",
        "--verbose", "2"
    ]
    print("Command:", " ".join(cmd4))
    print()
    
    print("To run any of these examples, copy the command and run it in your terminal.")
    print("Make sure you have installed the required dependencies:")
    print("pip install -r requirements_training.txt")


def check_files_exist():
    """Check if example files exist."""
    examples = [
        "examples/inverted_pendulum/inverted_pendulum.jani",
        "examples/inverted_pendulum/start.jani",
        "examples/inverted_pendulum/objective.jani", 
        "examples/inverted_pendulum/safe.jani",
        "examples/bouncing_ball/bouncing_ball.jani",
        "examples/bouncing_ball/start.jani",
        "examples/bouncing_ball/objective.jani",
        "examples/bouncing_ball/safe.jani",
        "examples/simple_test.jani",
        "examples/simple_start.jani", 
        "examples/simple_goal.jani",
        "examples/simple_failure.jani"
    ]
    
    print("Checking example files...")
    missing = []
    for file_path in examples:
        if not Path(file_path).exists():
            missing.append(file_path)
        else:
            print(f"✓ {file_path}")
    
    if missing:
        print(f"\n❌ Missing files: {missing}")
        print("Some examples may not work without these files.")
    else:
        print("\n✅ All example files found!")


if __name__ == "__main__":
    print("JANI Environment Training Examples")
    print("=" * 40)
    print()
    
    check_files_exist()
    print()
    run_training_example()