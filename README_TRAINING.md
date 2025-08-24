# JANI Environment Training Scripts

This directory contains comprehensive training scripts for JANI environments using reinforcement learning.

## Files

- `train_comprehensive.py` - Advanced training with hyperparameter tuning and W&B integration
- `train_simple.py` - Basic training script without optional dependencies
- `example_training.py` - Examples showing how to use the training scripts
- `requirements_training.txt` - Additional dependencies needed for advanced features

## Quick Start

### Basic Training (No Additional Dependencies)

Use the simple training script for basic functionality:

```bash
python train_simple.py \
    --model_file examples/simple_test.jani \
    --start_file examples/simple_start.jani \
    --goal_file examples/simple_goal.jani \
    --safe_file examples/simple_failure.jani \
    --total_timesteps 100000
```

### Advanced Training (With Optional Dependencies)

First install additional dependencies:

```bash
pip install -r requirements_training.txt
```

Then use the comprehensive training script:

```bash
python train_comprehensive.py \
    --model_file examples/inverted_pendulum/inverted_pendulum.jani \
    --start_file examples/inverted_pendulum/start.jani \
    --goal_file examples/inverted_pendulum/objective.jani \
    --safe_file examples/inverted_pendulum/safe.jani \
    --tune_hyperparams \
    --n_trials 20 \
    --total_timesteps 500000 \
    --wandb_project jani-training
```

## Command Line Arguments

### Required Arguments

- `--model_file` - Path to the JANI model file (mandatory)

### File Specification (Choose One)

**Option 1: Individual Files**
- `--start_file` - Path to start state file
- `--goal_file` - Path to goal condition file  
- `--safe_file` - Path to safe condition file

**Option 2: Property File**
- `--property_file` - Path to property file (overrides individual files)

### Training Parameters

- `--total_timesteps` - Total training timesteps (default: 100000)
- `--n_envs` - Number of parallel environments (default: 4)
- `--eval_freq` - Evaluation frequency (default: 10000)
- `--eval_episodes` - Episodes per evaluation (default: 5)

### Hyperparameter Tuning (Comprehensive Script Only)

- `--tune_hyperparams` - Enable hyperparameter optimization
- `--n_trials` - Number of tuning trials (default: 50)
- `--study_name` - Name for optimization study

### Logging and Saving

- `--log_dir` - Directory for logs (default: ./logs)
- `--model_save_dir` - Directory for saved models (default: ./models)
- `--experiment_name` - Custom experiment name
- `--wandb_project` - Weights & Biases project name
- `--wandb_entity` - W&B team/entity name

### Other Options

- `--seed` - Random seed (default: 42)
- `--device` - Training device (auto, cpu, cuda)
- `--verbose` - Verbosity level (0-2)

## Examples

### 1. Quick Test Run

```bash
python train_simple.py \
    --model_file examples/simple_test.jani \
    --start_file examples/simple_start.jani \
    --goal_file examples/simple_goal.jani \
    --safe_file examples/simple_failure.jani \
    --total_timesteps 10000 \
    --experiment_name quick_test
```

### 2. Training with Property File

```bash
python train_simple.py \
    --model_file examples/mod_linetrack/mod_linetrack.jani \
    --property_file examples/mod_linetrack/property.jani \
    --total_timesteps 200000
```

### 3. Hyperparameter Tuning

```bash
python train_comprehensive.py \
    --model_file examples/bouncing_ball/bouncing_ball.jani \
    --start_file examples/bouncing_ball/start.jani \
    --goal_file examples/bouncing_ball/objective.jani \
    --safe_file examples/bouncing_ball/safe.jani \
    --tune_hyperparams \
    --n_trials 30 \
    --total_timesteps 300000 \
    --wandb_project bouncing-ball-optimization
```

### 4. Custom Hyperparameters (Simple Script)

```bash
python train_simple.py \
    --model_file examples/inverted_pendulum/inverted_pendulum.jani \
    --start_file examples/inverted_pendulum/start.jani \
    --goal_file examples/inverted_pendulum/objective.jani \
    --safe_file examples/inverted_pendulum/safe.jani \
    --learning_rate 0.001 \
    --n_steps 1024 \
    --batch_size 128 \
    --total_timesteps 1000000
```

## Output

Both scripts will create:

- **Model files**: Best and final trained models
- **Logs**: TensorBoard logs for monitoring training
- **Hyperparameters**: JSON file with used hyperparameters
- **Evaluation metrics**: Performance over time

### Directory Structure

```
logs/
└── experiment_name_YYYYMMDD_HHMMSS/
    ├── MaskablePPO_1/
    │   └── events.out.tfevents.*
    └── eval/
        └── evaluations.npz

models/
└── experiment_name_YYYYMMDD_HHMMSS/
    ├── best_model.zip
    ├── final_model.zip
    └── hyperparameters.json
```

## Monitoring Training

### TensorBoard

View training progress:

```bash
tensorboard --logdir=logs/
```

### Weights & Biases (Comprehensive Script)

Training metrics are automatically logged to W&B if available. View at:
https://wandb.ai/your-entity/your-project

## Troubleshooting

### Common Issues

1. **File not found errors**: Ensure JANI files exist at specified paths
2. **Import errors**: Install required dependencies with pip
3. **CUDA errors**: Set `--device cpu` if GPU issues occur
4. **Memory issues**: Reduce `--n_envs` or `--batch_size`

### File Validation

The scripts will automatically:
- Check that the model file exists
- Validate start/goal/safe files OR property file
- Warn if property file overrides individual files
- Create necessary output directories

### Performance Tips

1. Use multiple environments (`--n_envs`) for faster training
2. Adjust `--eval_freq` based on training length
3. Use hyperparameter tuning for optimal performance
4. Monitor training with TensorBoard or W&B

## Dependencies

### Required (Always)
- stable-baselines3
- sb3-contrib 
- torch
- numpy

### Optional (For Advanced Features)
- optuna (hyperparameter tuning)
- wandb (advanced logging)
- tensorboard (visualization)

Install all with: `pip install -r requirements_training.txt`