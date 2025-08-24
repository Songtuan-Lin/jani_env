# Makefile for JANI Environment Training

# Default parameters
PYTHON := python
VENV_ACTIVATE := source ~/miniconda3/bin/activate python3.12 &&

# Common file paths
SIMPLE_MODEL := examples/simple_test.jani
SIMPLE_START := examples/simple_start.jani
SIMPLE_GOAL := examples/simple_goal.jani
SIMPLE_SAFE := examples/simple_failure.jani

PENDULUM_MODEL := examples/inverted_pendulum/inverted_pendulum.jani
PENDULUM_START := examples/inverted_pendulum/start.jani
PENDULUM_GOAL := examples/inverted_pendulum/objective.jani
PENDULUM_SAFE := examples/inverted_pendulum/safe.jani

BALL_MODEL := examples/bouncing_ball/bouncing_ball.jani
BALL_START := examples/bouncing_ball/start.jani
BALL_GOAL := examples/bouncing_ball/objective.jani
BALL_SAFE := examples/bouncing_ball/safe.jani

.PHONY: help install-deps test-quick test-pendulum test-ball train-simple train-comprehensive clean

help:
	@echo "JANI Environment Training Makefile"
	@echo "=================================="
	@echo ""
	@echo "Available targets:"
	@echo "  help              Show this help message"
	@echo "  install-deps      Install training dependencies"
	@echo "  test-quick        Quick test run (1000 timesteps)"
	@echo "  test-pendulum     Test with inverted pendulum (10k timesteps)"
	@echo "  test-ball         Test with bouncing ball (10k timesteps)"
	@echo "  train-simple      Basic training with simple model"
	@echo "  train-comprehensive Full training with hyperparameter tuning"
	@echo "  clean             Clean up log and model directories"
	@echo ""
	@echo "Examples:"
	@echo "  make test-quick"
	@echo "  make train-pendulum"
	@echo "  make install-deps && make train-comprehensive"

install-deps:
	@echo "Installing training dependencies..."
	$(VENV_ACTIVATE) pip install -r requirements_training.txt

test-quick:
	@echo "Running quick test (1000 timesteps)..."
	$(VENV_ACTIVATE) $(PYTHON) train_simple.py \
		--model_file $(SIMPLE_MODEL) \
		--start_file $(SIMPLE_START) \
		--goal_file $(SIMPLE_GOAL) \
		--safe_file $(SIMPLE_SAFE) \
		--total_timesteps 1000 \
		--n_envs 2 \
		--eval_freq 500 \
		--experiment_name quick_test \
		--verbose 2

test-pendulum:
	@echo "Running pendulum test (10k timesteps)..."
	$(VENV_ACTIVATE) $(PYTHON) train_simple.py \
		--model_file $(PENDULUM_MODEL) \
		--start_file $(PENDULUM_START) \
		--goal_file $(PENDULUM_GOAL) \
		--safe_file $(PENDULUM_SAFE) \
		--total_timesteps 10000 \
		--experiment_name pendulum_test

test-ball:
	@echo "Running bouncing ball test (10k timesteps)..."
	$(VENV_ACTIVATE) $(PYTHON) train_simple.py \
		--model_file $(BALL_MODEL) \
		--start_file $(BALL_START) \
		--goal_file $(BALL_GOAL) \
		--safe_file $(BALL_SAFE) \
		--total_timesteps 10000 \
		--experiment_name ball_test

train-simple:
	@echo "Training simple model (100k timesteps)..."
	$(VENV_ACTIVATE) $(PYTHON) train_simple.py \
		--model_file $(SIMPLE_MODEL) \
		--start_file $(SIMPLE_START) \
		--goal_file $(SIMPLE_GOAL) \
		--safe_file $(SIMPLE_SAFE) \
		--total_timesteps 100000 \
		--experiment_name simple_training

train-pendulum:
	@echo "Training inverted pendulum (500k timesteps)..."
	$(VENV_ACTIVATE) $(PYTHON) train_simple.py \
		--model_file $(PENDULUM_MODEL) \
		--start_file $(PENDULUM_START) \
		--goal_file $(PENDULUM_GOAL) \
		--safe_file $(PENDULUM_SAFE) \
		--total_timesteps 500000 \
		--learning_rate 0.0003 \
		--n_steps 1024 \
		--experiment_name pendulum_training

train-comprehensive:
	@echo "Comprehensive training with hyperparameter tuning..."
	$(VENV_ACTIVATE) $(PYTHON) train_comprehensive.py \
		--model_file $(PENDULUM_MODEL) \
		--start_file $(PENDULUM_START) \
		--goal_file $(PENDULUM_GOAL) \
		--safe_file $(PENDULUM_SAFE) \
		--tune_hyperparams \
		--n_trials 10 \
		--total_timesteps 200000 \
		--experiment_name comprehensive_training \
		--wandb_project jani-training

tensorboard:
	@echo "Starting TensorBoard..."
	$(VENV_ACTIVATE) tensorboard --logdir=logs/

clean:
	@echo "Cleaning up logs and models..."
	rm -rf logs/ models/
	@echo "Cleanup completed."

# Individual model training targets
train-all: train-simple train-pendulum train-ball

train-ball:
	@echo "Training bouncing ball (300k timesteps)..."
	$(VENV_ACTIVATE) $(PYTHON) train_simple.py \
		--model_file $(BALL_MODEL) \
		--start_file $(BALL_START) \
		--goal_file $(BALL_GOAL) \
		--safe_file $(BALL_SAFE) \
		--total_timesteps 300000 \
		--experiment_name ball_training