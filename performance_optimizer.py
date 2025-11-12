# File: performance_optimizer.py
# Description: Comprehensive training performance monitoring and optimization
# =============================================================================
import psutil
import torch
import gc
import time
import logging
import os
from typing import Dict, List
import pandas as pd
import numpy as np
from pathlib import Path

class TrainingOptimizer:
    """Comprehensive training performance optimizer"""
    
    def __init__(self):
        self.start_time = time.time()
        self.step_times = []
        self.memory_usage = []
        self.gpu_usage = []
        self.last_cleanup = time.time()
        self.cleanup_interval = 60  # seconds
        
        # Performance thresholds
        self.max_memory_usage = 0.85  # 85% of available memory
        self.target_steps_per_second = 100
        self.memory_cleanup_threshold = 0.8
        
        self.setup_logging()
        
    def setup_logging(self):
        """Setup performance logging"""
        perf_dir = Path("performance_logs")
        perf_dir.mkdir(exist_ok=True)
        
        self.perf_logger = logging.getLogger('performance')
        if not self.perf_logger.handlers:
            handler = logging.FileHandler(perf_dir / "training_performance.log")
            formatter = logging.Formatter('%(asctime)s - %(message)s')
            handler.setFormatter(formatter)
            self.perf_logger.addHandler(handler)
            self.perf_logger.setLevel(logging.INFO)
    
    def optimize_system_settings(self):
        """Apply system-level optimizations"""
        logging.info("Applying system optimizations...")
        
        # PyTorch optimizations
        if torch.cuda.is_available():
            # Enable GPU optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logging.info("GPU optimizations enabled")
            
            # Clear GPU cache
            torch.cuda.empty_cache()
            
        # CPU optimizations
        torch.set_num_threads(min(psutil.cpu_count(), 8))
        torch.set_num_interop_threads(2)
        
        # Memory management
        if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
            torch.cuda.set_per_process_memory_fraction(0.8)
            
        # Process priority (Unix systems)
        try:
            os.nice(-5)  # Higher priority for training process
            logging.info("Process priority increased")
        except (OSError, AttributeError):
            pass
    
    def monitor_performance(self, step_count: int, model=None) -> Dict:
        """Monitor and log performance metrics"""
        current_time = time.time()
        
        # Calculate performance metrics
        elapsed_time = current_time - self.start_time
        steps_per_second = step_count / elapsed_time if elapsed_time > 0 else 0
        
        # Memory usage
        memory_info = psutil.virtual_memory()
        memory_usage_pct = memory_info.percent / 100.0
        
        # GPU usage
        gpu_usage_pct = 0
        gpu_memory_pct = 0
        if torch.cuda.is_available():
            gpu_memory_pct = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_usage_pct = gpu_util.gpu / 100.0
            except:
                pass
        
        metrics = {
            'step_count': step_count,
            'steps_per_second': steps_per_second,
            'elapsed_hours': elapsed_time / 3600,
            'memory_usage_pct': memory_usage_pct,
            'gpu_usage_pct': gpu_usage_pct,
            'gpu_memory_pct': gpu_memory_pct,
            'cpu_usage_pct': psutil.cpu_percent() / 100.0
        }
        
        # Store for trend analysis
        self.step_times.append(steps_per_second)
        self.memory_usage.append(memory_usage_pct)
        self.gpu_usage.append(gpu_usage_pct)
        
        # Keep only recent history
        max_history = 100
        if len(self.step_times) > max_history:
            self.step_times = self.step_times[-max_history:]
            self.memory_usage = self.memory_usage[-max_history:]
            self.gpu_usage = self.gpu_usage[-max_history:]
        
        # Log metrics
        self.perf_logger.info(
            f"Step: {step_count:,} | "
            f"Speed: {steps_per_second:.1f} steps/s | "
            f"Memory: {memory_usage_pct:.1%} | "
            f"GPU: {gpu_usage_pct:.1%} | "
            f"GPU_Mem: {gpu_memory_pct:.1%}"
        )
        
        return metrics
    
    def cleanup_memory(self, force=False):
        """Perform memory cleanup"""
        current_time = time.time()
        
        if force or (current_time - self.last_cleanup > self.cleanup_interval):
            # Python garbage collection
            collected = gc.collect()
            
            # PyTorch memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            self.last_cleanup = current_time
            
            logging.debug(f"Memory cleanup: collected {collected} objects")
    
    def check_performance_issues(self, metrics: Dict) -> List[str]:
        """Identify performance issues and suggest fixes"""
        issues = []
        
        # Memory issues
        if metrics['memory_usage_pct'] > self.max_memory_usage:
            issues.append(f"High memory usage ({metrics['memory_usage_pct']:.1%}). Consider reducing batch size or number of environments.")
        
        # Speed issues
        if len(self.step_times) > 10:
            avg_speed = np.mean(self.step_times[-10:])
            if avg_speed < self.target_steps_per_second * 0.5:
                issues.append(f"Slow training speed ({avg_speed:.1f} steps/s). Consider optimizing hyperparameters or using fewer environments.")
        
        # GPU underutilization
        if torch.cuda.is_available() and metrics['gpu_usage_pct'] < 0.3:
            issues.append("Low GPU utilization. Consider increasing batch size or using GPU-optimized operations.")
        
        # Memory leaks
        if len(self.memory_usage) > 20:
            recent_memory = self.memory_usage[-20:]
            if np.mean(recent_memory[-5:]) > np.mean(recent_memory[:5]) + 0.1:
                issues.append("Potential memory leak detected. Memory usage is steadily increasing.")
        
        return issues
    
    def suggest_optimizations(self, metrics: Dict) -> List[str]:
        """Suggest performance optimizations"""
        suggestions = []
        
        # Based on current performance
        if metrics['steps_per_second'] < 50:
            suggestions.append("Consider reducing RL_LOOKBACK_WINDOW or using simpler network architecture")
        
        if metrics['memory_usage_pct'] > 0.7:
            suggestions.append("Reduce batch_size or n_steps in PPO hyperparameters")
        
        if torch.cuda.is_available():
            if metrics['gpu_memory_pct'] < 0.5:
                suggestions.append("GPU memory underutilized. Consider increasing batch_size")
            if metrics['gpu_usage_pct'] < 0.5:
                suggestions.append("GPU compute underutilized. Ensure model is on GPU and using optimized operations")
        
        # Environment-specific suggestions
        cpu_count = psutil.cpu_count()
        if metrics.get('num_envs', 1) > cpu_count:
            suggestions.append(f"Too many environments ({metrics.get('num_envs')} > {cpu_count} CPUs). Reduce NUM_CPU_TO_USE")
        
        return suggestions
    
    def auto_optimize_hyperparams(self, current_metrics: Dict) -> Dict:
        """Automatically suggest optimized hyperparameters"""
        optimized = {}
        
        # Memory-based optimizations
        if current_metrics['memory_usage_pct'] > 0.8:
            optimized['batch_size'] = max(32, int(current_metrics.get('batch_size', 128) * 0.7))
            optimized['n_steps'] = max(1024, int(current_metrics.get('n_steps', 2048) * 0.8))
        
        elif current_metrics['memory_usage_pct'] < 0.5:
            optimized['batch_size'] = min(512, int(current_metrics.get('batch_size', 128) * 1.3))
            optimized['n_steps'] = min(8192, int(current_metrics.get('n_steps', 2048) * 1.2))
        
        # Speed-based optimizations
        if current_metrics['steps_per_second'] < 20:
            optimized['lookback_window'] = max(24, int(current_metrics.get('lookback_window', 60) * 0.8))
            optimized['num_envs'] = max(1, int(current_metrics.get('num_envs', 4) * 0.7))
        
        return optimized
    
    def generate_performance_report(self) -> str:
        """Generate a comprehensive performance report"""
        if not self.step_times:
            return "No performance data available yet."
        
        avg_speed = np.mean(self.step_times)
        avg_memory = np.mean(self.memory_usage)
        avg_gpu = np.mean(self.gpu_usage) if self.gpu_usage else 0
        
        elapsed_hours = (time.time() - self.start_time) / 3600
        
        report = f"""
Training Performance Report
==========================
Duration: {elapsed_hours:.2f} hours
Average Speed: {avg_speed:.1f} steps/second
Average Memory Usage: {avg_memory:.1%}
Average GPU Usage: {avg_gpu:.1%}

Performance Trends:
- Speed trend: {'↑' if len(self.step_times) > 10 and self.step_times[-1] > self.step_times[-10] else '↓' if len(self.step_times) > 10 else '-'}
- Memory trend: {'↑' if len(self.memory_usage) > 10 and self.memory_usage[-1] > self.memory_usage[-10] else '↓' if len(self.memory_usage) > 10 else '-'}

System Info:
- CPUs: {psutil.cpu_count()}
- RAM: {psutil.virtual_memory().total / 1024**3:.1f} GB
- GPU: {'Yes' if torch.cuda.is_available() else 'No'}
"""
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory_gb = torch.cuda.max_memory_allocated() / 1024**3
            report += f"- GPU Model: {gpu_name}\n"
            report += f"- GPU Memory Used: {gpu_memory_gb:.1f} GB\n"
        
        return report

def optimize_training_pipeline():
    """Main function to optimize the entire training pipeline"""
    optimizer = TrainingOptimizer()
    
    print("🚀 Training Pipeline Optimizer")
    print("=" * 50)
    
    # Apply system optimizations
    optimizer.optimize_system_settings()
    
    # Generate system recommendations
    system_info = {
        'cpu_count': psutil.cpu_count(),
        'memory_gb': psutil.virtual_memory().total / 1024**3,
        'gpu_available': torch.cuda.is_available(),
        'gpu_memory_gb': torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
    }
    
    print(f"System Configuration:")
    print(f"  CPUs: {system_info['cpu_count']}")
    print(f"  RAM: {system_info['memory_gb']:.1f} GB")
    print(f"  GPU: {'Available' if system_info['gpu_available'] else 'Not Available'}")
    
    if system_info['gpu_available']:
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.max_memory_allocated() / 1024**3:.1f} GB")
    
    # Recommendations based on system specs
    recommendations = []
    
    if system_info['memory_gb'] < 8:
        recommendations.append("⚠️  Low RAM detected. Reduce batch_size and num_environments")
    elif system_info['memory_gb'] > 32:
        recommendations.append("✅ High RAM available. Can use larger batch_size and more environments")
    
    if system_info['cpu_count'] < 4:
        recommendations.append("⚠️  Limited CPUs. Use NUM_CPU_TO_USE = 2")
    elif system_info['cpu_count'] > 16:
        recommendations.append("✅ Many CPUs available. Can use up to 12 environments")
    
    if not system_info['gpu_available']:
        recommendations.append("⚠️  No GPU detected. Training will be slower on CPU")
        recommendations.append("💡 Consider using cloud GPU instances (Google Colab, AWS, etc.)")
    
    print(f"\nSystem Recommendations:")
    for rec in recommendations:
        print(f"  {rec}")
    
    return optimizer

# Performance monitoring context manager
class PerformanceContext:
    """Context manager for performance monitoring during training"""
    
    def __init__(self, optimizer: TrainingOptimizer):
        self.optimizer = optimizer
        self.step_count = 0
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            print("\n" + "="*50)
            print("TRAINING COMPLETED SUCCESSFULLY")
            print("="*50)
            print(self.optimizer.generate_performance_report())
        else:
            print(f"\nTraining interrupted: {exc_type.__name__}: {exc_val}")
            
    def update(self, step_count: int, model=None):
        """Update performance metrics"""
        self.step_count = step_count
        metrics = self.optimizer.monitor_performance(step_count, model)
        
        # Check for issues every 1000 steps
        if step_count % 1000 == 0:
            issues = self.optimizer.check_performance_issues(metrics)
            if issues:
                print("\n⚠️  Performance Issues Detected:")
                for issue in issues:
                    print(f"   - {issue}")
                    
            suggestions = self.optimizer.suggest_optimizations(metrics)
            if suggestions:
                print("\n💡 Optimization Suggestions:")
                for suggestion in suggestions:
                    print(f"   - {suggestion}")
        
        # Auto cleanup
        self.optimizer.cleanup_memory()
        
        return metrics

# Utility functions for training optimization
def estimate_training_time(total_timesteps: int, current_speed: float = None) -> str:
    """Estimate total training time"""
    if current_speed is None:
        # Conservative estimate based on typical performance
        estimated_speed = 50  # steps per second
    else:
        estimated_speed = max(current_speed, 1)  # Avoid division by zero
    
    total_seconds = total_timesteps / estimated_speed
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    
    return f"{hours}h {minutes}m"

def suggest_optimal_config(system_specs: Dict) -> Dict:
    """Suggest optimal configuration based on system specifications"""
    config = {}
    
    # Environment count based on CPU and memory
    cpu_count = system_specs.get('cpu_count', 4)
    memory_gb = system_specs.get('memory_gb', 8)
    gpu_available = system_specs.get('gpu_available', False)
    
    # Conservative environment count
    max_envs_by_cpu = max(1, int(cpu_count * 0.75))
    max_envs_by_memory = max(1, int(memory_gb // 1.5))  # 1.5GB per environment
    
    config['num_environments'] = min(max_envs_by_cpu, max_envs_by_memory, 8)
    
    # Batch size based on GPU/CPU and memory
    if gpu_available:
        if memory_gb >= 16:
            config['batch_size'] = 256
        elif memory_gb >= 8:
            config['batch_size'] = 128
        else:
            config['batch_size'] = 64
    else:
        # CPU training - smaller batches
        config['batch_size'] = min(128, max(32, int(memory_gb * 8)))
    
    # Network architecture
    if gpu_available and memory_gb >= 8:
        config['net_arch'] = {"pi": [128, 128, 64], "vf": [128, 128, 64]}
    else:
        config['net_arch'] = {"pi": [64, 64], "vf": [64, 64]}
    
    # Lookback window based on memory constraints
    if memory_gb >= 16:
        config['lookback_window'] = 60
    elif memory_gb >= 8:
        config['lookback_window'] = 48
    else:
        config['lookback_window'] = 32
    
    # Training steps
    base_steps = 20_000_000
    if gpu_available:
        config['initial_training_timesteps'] = base_steps
    else:
        config['initial_training_timesteps'] = base_steps // 2  # Reduce for CPU
    
    return config

# Example usage function
def run_optimized_training():
    """Example of how to use the performance optimizer"""
    
    # Initialize optimizer
    optimizer = optimize_training_pipeline()
    
    # Get system specs
    system_specs = {
        'cpu_count': psutil.cpu_count(),
        'memory_gb': psutil.virtual_memory().total / 1024**3,
        'gpu_available': torch.cuda.is_available()
    }
    
    # Get optimized configuration
    optimal_config = suggest_optimal_config(system_specs)
    
    print(f"\nOptimal Configuration:")
    for key, value in optimal_config.items():
        print(f"  {key}: {value}")
    
    # Estimate training time
    estimated_time = estimate_training_time(optimal_config['initial_training_timesteps'])
    print(f"\nEstimated Training Time: {estimated_time}")
    
    return optimizer, optimal_config