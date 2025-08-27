"""
GPU job scheduler for multi-GPU training of classification models.
Efficiently distributes training jobs across available GPUs.
"""

import os
import time
import queue
import threading
import subprocess
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import torch
from .. import config
from .data_loader import get_available_benchmarks


@dataclass
class TrainingJob:
    """Represents a training job for a specific benchmark."""
    benchmark: str
    gpu_id: int
    model_type: str = 'basic'
    task_type: str = 'classification'
    n_trials: int = config.N_TRIALS
    data_dir: str = "."
    status: str = 'pending'  # pending, running, completed, failed
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    process: Optional[subprocess.Popen] = None
    log_file: Optional[str] = None


class GPUScheduler:
    """Manages training jobs across multiple GPUs."""
    
    def __init__(self, max_concurrent_jobs: Optional[int] = None, log_dir: str = "logs"):
        # Setup logging first
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_dir / 'scheduler.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

        # Now initialize other attributes
        self.available_gpus = self._get_available_gpus()
        self.max_concurrent_jobs = max_concurrent_jobs or min(len(self.available_gpus), config.MAX_CONCURRENT_JOBS)
        self.job_queue = queue.Queue()
        self.running_jobs: Dict[int, TrainingJob] = {}  # gpu_id -> job
        self.completed_jobs: List[TrainingJob] = []
        self.failed_jobs: List[TrainingJob] = []
        
    def _get_available_gpus(self) -> List[int]:
        """Get list of available GPU IDs."""
        if not torch.cuda.is_available():
            self.logger.warning("CUDA not available. Running on CPU.")
            return [0]  # Use CPU as device 0

        gpu_count = torch.cuda.device_count()
        self.logger.info(f"Found {gpu_count} available GPUs")

        # Validate each GPU is actually accessible
        valid_gpus = []
        for gpu_id in range(gpu_count):
            try:
                # Test if we can access this GPU
                torch.cuda.set_device(gpu_id)
                torch.cuda.current_device()
                valid_gpus.append(gpu_id)
                self.logger.info(f"GPU {gpu_id} validated successfully")
            except Exception as e:
                self.logger.warning(f"GPU {gpu_id} not accessible: {str(e)}")

        if not valid_gpus:
            self.logger.warning("No valid GPUs found. Falling back to CPU.")
            return [0]

        return valid_gpus
    
    def add_benchmark_jobs(self, benchmarks: List[str], model_types: List[str] = ['basic'], 
                          task_type: str = 'classification', n_trials: int = config.N_TRIALS,
                          data_dir: str = "."):
        """Add training jobs for multiple benchmarks and model types."""
        for benchmark in benchmarks:
            for model_type in model_types:
                job = TrainingJob(
                    benchmark=benchmark,
                    gpu_id=-1,  # Will be assigned when job starts
                    model_type=model_type,
                    task_type=task_type,
                    n_trials=n_trials,
                    data_dir=data_dir
                )
                self.job_queue.put(job)
                self.logger.info(f"Added job: {benchmark} - {model_type}")
    
    def _get_next_available_gpu(self) -> Optional[int]:
        """Get the next available GPU ID."""
        for gpu_id in self.available_gpus:
            if gpu_id not in self.running_jobs:
                return gpu_id
        return None
    
    def _create_training_script(self, job: TrainingJob) -> str:
        """Create a training script for the job."""
        # For GPU scheduler, we use CUDA_VISIBLE_DEVICES to isolate GPUs
        # So the device_id in the script should always be 0 (the only visible GPU)
        script_device_id = 0 if job.gpu_id > 0 else None

        script_content = f"""
import os
import sys
import torch

# Set CUDA_VISIBLE_DEVICES to isolate this GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '{job.gpu_id}'

# Add current directory to path
sys.path.append('.')

from regression_trainer import train_regression_model

if __name__ == "__main__":
    try:
        results = train_regression_model(
            benchmark="{job.benchmark}",
            model_type="{job.model_type}",
            n_trials={job.n_trials},
            data_dir="{job.data_dir}",
            device_id={script_device_id}
        )
        print(f"Training completed successfully for {job.benchmark}")
        print(f"Best validation RMSE: {{results.get('best_val_rmse', 'N/A')}}")
    except Exception as e:
        print(f"Training failed for {job.benchmark}: {{str(e)}}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
"""
        
        script_path = self.log_dir / f"train_{job.benchmark}_{job.model_type}_gpu{job.gpu_id}.py"
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        return str(script_path)
    
    def _start_job(self, job: TrainingJob) -> bool:
        """Start a training job on an available GPU."""
        gpu_id = self._get_next_available_gpu()
        if gpu_id is None:
            return False
        
        job.gpu_id = gpu_id
        job.status = 'running'
        job.start_time = time.time()
        
        # Create log file
        job.log_file = str(self.log_dir / f"{job.benchmark}_{job.model_type}_gpu{gpu_id}.log")
        
        # Create training script
        script_path = self._create_training_script(job)
        
        # Start the training process
        try:
            # Validate GPU is still available
            if gpu_id >= len(self.available_gpus) and gpu_id != 0:
                raise ValueError(f"GPU {gpu_id} is not in available GPUs list: {self.available_gpus}")

            # Create environment with proper GPU isolation
            env = dict(os.environ)
            if gpu_id == 0 and not torch.cuda.is_available():
                # CPU mode
                env['CUDA_VISIBLE_DEVICES'] = ''
            else:
                # GPU mode - map to actual GPU ID
                actual_gpu_id = self.available_gpus[min(gpu_id, len(self.available_gpus) - 1)]
                env['CUDA_VISIBLE_DEVICES'] = str(actual_gpu_id)

            with open(job.log_file, 'w') as log_file:
                job.process = subprocess.Popen(
                    ['python', script_path],
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    env=env
                )

            self.running_jobs[gpu_id] = job
            self.logger.info(f"Started job {job.benchmark} - {job.model_type} on GPU {gpu_id} (actual GPU: {env.get('CUDA_VISIBLE_DEVICES', 'CPU')})")
            return True

        except Exception as e:
            self.logger.error(f"Failed to start job {job.benchmark}: {str(e)}")
            job.status = 'failed'
            return False
    
    def _check_running_jobs(self):
        """Check status of running jobs and clean up completed ones."""
        completed_gpus = []
        
        for gpu_id, job in self.running_jobs.items():
            if job.process.poll() is not None:  # Process has finished
                job.end_time = time.time()
                runtime = job.end_time - job.start_time
                
                if job.process.returncode == 0:
                    job.status = 'completed'
                    self.completed_jobs.append(job)
                    self.logger.info(f"Job {job.benchmark} - {job.model_type} completed successfully in {runtime:.1f}s")
                else:
                    job.status = 'failed'
                    self.failed_jobs.append(job)
                    self.logger.error(f"Job {job.benchmark} - {job.model_type} failed after {runtime:.1f}s")
                
                completed_gpus.append(gpu_id)
        
        # Remove completed jobs from running jobs
        for gpu_id in completed_gpus:
            del self.running_jobs[gpu_id]
    
    def run_all_jobs(self, check_interval: float = 10.0):
        """Run all queued jobs, managing GPU resources."""
        self.logger.info(f"Starting scheduler with {self.max_concurrent_jobs} max concurrent jobs")
        self.logger.info(f"Total jobs in queue: {self.job_queue.qsize()}")
        
        while not self.job_queue.empty() or self.running_jobs:
            # Check and clean up completed jobs
            self._check_running_jobs()
            
            # Start new jobs if GPUs are available
            while (len(self.running_jobs) < self.max_concurrent_jobs and 
                   not self.job_queue.empty()):
                try:
                    job = self.job_queue.get_nowait()
                    if self._start_job(job):
                        pass  # Job started successfully
                    else:
                        # No GPU available, put job back in queue
                        self.job_queue.put(job)
                        break
                except queue.Empty:
                    break
            
            # Wait before next check
            time.sleep(check_interval)
        
        # Final summary
        self.print_summary()
    
    def print_summary(self):
        """Print summary of all jobs."""
        total_jobs = len(self.completed_jobs) + len(self.failed_jobs)
        
        self.logger.info("=" * 50)
        self.logger.info("TRAINING SUMMARY")
        self.logger.info("=" * 50)
        self.logger.info(f"Total jobs: {total_jobs}")
        self.logger.info(f"Completed: {len(self.completed_jobs)}")
        self.logger.info(f"Failed: {len(self.failed_jobs)}")
        
        if self.failed_jobs:
            self.logger.info("\nFailed jobs:")
            for job in self.failed_jobs:
                self.logger.info(f"  - {job.benchmark} - {job.model_type}")
        
        if self.completed_jobs:
            total_time = sum(job.end_time - job.start_time for job in self.completed_jobs)
            avg_time = total_time / len(self.completed_jobs)
            self.logger.info(f"\nAverage training time: {avg_time:.1f}s")


def main():
    """Main function to run multi-GPU training."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-GPU regression model training")
    parser.add_argument("--benchmarks", nargs="+", help="Specific benchmarks to train")
    parser.add_argument("--model-types", nargs="+", default=['basic'],
                       choices=['basic'], help="Model types to train")
    parser.add_argument("--n-trials", type=int, default=config.N_TRIALS,
                       help="Number of hyperparameter tuning trials")
    parser.add_argument("--max-jobs", type=int, help="Maximum concurrent jobs")
    parser.add_argument("--data-dir", type=str, default=".", help="Data directory")
    parser.add_argument("--log-dir", type=str, default="logs", help="Log directory")
    
    args = parser.parse_args()
    
    # Get benchmarks
    if args.benchmarks:
        benchmarks = args.benchmarks
    else:
        benchmarks = get_available_benchmarks(args.data_dir)
        if not benchmarks:
            print("No offline benchmarks found!")
            return
    
    print(f"Found {len(benchmarks)} benchmarks: {benchmarks}")
    
    # Create scheduler
    scheduler = GPUScheduler(max_concurrent_jobs=args.max_jobs, log_dir=args.log_dir)
    
    # Add jobs
    scheduler.add_benchmark_jobs(
        benchmarks=benchmarks,
        model_types=args.model_types,
        n_trials=args.n_trials,
        data_dir=args.data_dir
    )
    
    # Run all jobs
    scheduler.run_all_jobs()


if __name__ == "__main__":
    main()
