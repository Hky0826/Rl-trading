# File: trainer_streamlined.py
# Description: Streamlined high-performance trainer using preprocessed data
# =============================================================================
import logging
import os
import pandas as pd
import numpy as np
import time
import psutil
import torch
import gc
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure as configure_logger
from stable_baselines3.common.utils import ConstantSchedule

import config
from rl_environment import TradingEnv

class StreamlinedTrainer:
    """Streamlined trainer focused on speed and efficiency"""
    
    def __init__(self, ticker: str):
        self.ticker = ticker
        self.start_time = time.time()
        self.setup_logging()
        self.optimize_system()
        
    def setup_logging(self):
        """Setup basic logging"""
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def optimize_system(self):
        """Apply essential system optimizations"""
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.cuda.empty_cache()
            self.logger.info(f"GPU optimizations enabled: {torch.cuda.get_device_name(0)}")
        
        torch.set_num_threads(min(psutil.cpu_count(), 8))
        torch.set_num_interop_threads(2)
        
    def load_data(self):
        """Load preprocessed data"""
        data_path = Path("processed_data") / f"{self.ticker}_processed.parquet"
        
        if not data_path.exists():
            raise FileNotFoundError(
                f"Preprocessed data not found at {data_path}. "
                "Run preprocess_data.py first."
            )
        
        self.logger.info(f"Loading data from {data_path}")
        df = pd.read_parquet(data_path)
        
        # Basic validation and cleaning
        df = df.dropna().sort_index()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].astype(np.float32)
        
        self.logger.info(f"Data loaded: {len(df):,} samples, {df.shape[1]} features")
        return df
        
    def get_system_config(self):
        """Get optimal configuration for current system"""
        # Get the number of CPUs from config
        num_envs = config.NUM_CPU_TO_USE
        
        # Get device info
        gpu_available = torch.cuda.is_available()
        device = "cuda" if gpu_available else "cpu"
        
        config_dict = {
            'num_envs': num_envs,
            'device': device
        }
        
        self.logger.info(f"System config: {config_dict}")
        return config_dict
        
    def create_environment(self, df, num_envs):
        """Create vectorized environment"""
        vec_env_cls = DummyVecEnv if num_envs <= 4 else SubprocVecEnv
        
        env = make_vec_env(
            TradingEnv,
            n_envs=num_envs,
            seed=42,
            vec_env_cls=vec_env_cls,
            env_kwargs={'df': df}
        )
        
        self.logger.info(f"Created {num_envs} environments using {vec_env_cls.__name__}")
        return env
        
    def create_model(self, env, sys_config):
        """Create or load PPO model using config hyperparameters"""
        # Get ALL hyperparameters from config
        hyperparams = config.PPO_HYPERPARAMS.copy()
        
        model_path = Path(config.RL_MODEL_DIR) / f"rl_agent_{self.ticker}_final.zip"
        os.makedirs(config.RL_MODEL_DIR, exist_ok=True)
        
        if model_path.exists():
            self.logger.info(f"Loading existing model from {model_path}")
            
            # Load model first
            model = PPO.load(
                str(model_path), 
                env=env, 
                device=sys_config['device']
            )
            
            # Update ALL hyperparameters for continued training
            updatable_params = [
                'learning_rate', 'clip_range', 'ent_coef', 'vf_coef', 
                'max_grad_norm', 'gamma', 'gae_lambda'
            ]
            
            for param in updatable_params:
                if param in hyperparams and hasattr(model, param):
                    old_value = getattr(model, param)
                    new_value = hyperparams[param]

                    # ✅ wrap floats into callable schedules
                    if param in ["learning_rate", "clip_range"] and isinstance(new_value, (float, int)):
                        from stable_baselines3.common.utils import ConstantSchedule
                        new_value = ConstantSchedule(new_value)

                    setattr(model, param, new_value)
                    self.logger.info(f"Updated {param}: {old_value} -> {new_value}")
            
            # ✅ Extra: force-update optimizer LR if changed
            if "learning_rate" in hyperparams:
                # Replace lr schedule entirely
                lr_value = float(hyperparams["learning_rate"])
                model.lr_schedule = ConstantSchedule(lr_value)

                # Apply to optimizer param groups
                for param_group in model.policy.optimizer.param_groups:
                    param_group['lr'] = lr_value

                self.logger.info(f"Learning rate schedule replaced with constant {lr_value}")

            total_timesteps = config.CONTINUOUS_TRAINING_TIMESTEPS
            
        else:
            self.logger.info("Creating new PPO model with config hyperparameters")
            
            # Log all hyperparameters being used
            self.logger.info("Hyperparameters from config:")
            for key, value in hyperparams.items():
                self.logger.info(f"  {key}: {value}")
            
            model = PPO(
                "MultiInputPolicy",
                env,
                verbose=1,
                device=sys_config['device'],
                tensorboard_log="./ppo_trading_tensorboard/",
                **hyperparams  # Use ALL hyperparams from config
            )
            total_timesteps = config.INITIAL_TRAINING_TIMESTEPS
            
        self.logger.info(f"Model ready on device: {model.device}")
        
        # Log final model configuration
        self.logger.info("Final model configuration:")
        self.logger.info(f"  Learning rate: {model.learning_rate}")
        self.logger.info(f"  Batch size: {model.batch_size}")
        self.logger.info(f"  N steps: {model.n_steps}")
        self.logger.info(f"  N epochs: {model.n_epochs}")
        self.logger.info(f"  Gamma: {model.gamma}")
        self.logger.info(f"  GAE lambda: {model.gae_lambda}")
        self.logger.info(f"  Clip range: {model.clip_range}")
        self.logger.info(f"  Entropy coef: {model.ent_coef}")
        self.logger.info(f"  Value function coef: {model.vf_coef}")
        
        return model, total_timesteps
        
    def create_callback(self, sys_config):
        """Create training callback with saving and monitoring"""
        save_freq = max(2048 * 80 // sys_config['num_envs'], 1000)
        
        class TrainingCallback(BaseCallback):
            def __init__(self, save_freq, save_dir, ticker, training_logger):
                super().__init__()
                self.save_freq = save_freq
                self.save_dir = save_dir
                self.ticker = ticker
                self.training_logger = training_logger
                self.last_save = 0
                self.last_memory_check = 0
                
            def _on_step(self) -> bool:
                # Memory cleanup every 1000 steps
                if self.n_calls - self.last_memory_check > 1000:
                    memory_pct = psutil.virtual_memory().percent
                    if memory_pct > config.MAX_MEMORY_USAGE_PCT:  # Use config value
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        self.training_logger.info(f"Memory cleanup at {memory_pct:.1f}% usage")
                    self.last_memory_check = self.n_calls
                
                # Model saving
                if (self.n_calls % self.save_freq == 0 and 
                    self.n_calls > self.last_save + 5000):
                    
                    model_path = Path(self.save_dir) / f"candidate_{self.ticker}_{self.n_calls}.zip"
                    
                    # Background save
                    import threading
                    def save_model():
                        try:
                            self.model.save(str(model_path))
                            print(f"Saved: {model_path.name}")
                        except Exception as e:
                            self.training_logger.error(f"Save failed: {e}")
                    
                    threading.Thread(target=save_model, daemon=True).start()
                    self.last_save = self.n_calls
                
                # Progress reporting every 10k steps
                if self.n_calls % 10000 == 0:
                    elapsed = time.time() - self.training_start_time
                    speed = self.n_calls / elapsed if elapsed > 0 else 0
                    memory_pct = psutil.virtual_memory().percent
                    
                    print(f"Step {self.n_calls:,} | Speed: {speed:.1f} steps/s | Memory: {memory_pct:.1f}%")
                
                return True
                
            def _on_training_start(self):
                self.training_start_time = time.time()
                
        return TrainingCallback(save_freq, config.RL_MODEL_DIR, self.ticker, self.logger)
        
    def train(self):
        """Run the complete training pipeline"""
        print("=" * 70)
        print("STREAMLINED HIGH-PERFORMANCE TRAINER")
        print("=" * 70)
        
        try:
            # Load data
            df = self.load_data()
            
            # Get system configuration
            sys_config = self.get_system_config()
            
            # Create environment
            env = self.create_environment(df, sys_config['num_envs'])
            
            # Create model with config hyperparameters
            model, total_timesteps = self.create_model(env, sys_config)
            
            # Create callback
            callback = self.create_callback(sys_config)
            
            # Get hyperparams for display
            hyperparams = config.PPO_HYPERPARAMS
            
            # Training info
            estimated_hours = total_timesteps / (50 * 3600)  # Conservative speed estimate
            
            print(f"\nTraining Configuration:")
            print(f"  Ticker: {self.ticker}")
            print(f"  Data samples: {len(df):,}")
            print(f"  Total timesteps: {total_timesteps:,}")
            print(f"  Environments: {sys_config['num_envs']}")
            print(f"  Batch size: {hyperparams.get('batch_size', 'N/A')}")
            print(f"  Learning rate: {hyperparams.get('learning_rate', 'N/A')}")
            print(f"  N steps: {hyperparams.get('n_steps', 'N/A')}")
            print(f"  N epochs: {hyperparams.get('n_epochs', 'N/A')}")
            print(f"  Device: {sys_config['device']}")
            print(f"  Estimated time: {estimated_hours:.1f} hours")
            
            # Start training
            print(f"\nStarting training...")
            start_time = time.time()
            
            model.learn(
                total_timesteps=total_timesteps,
                progress_bar=True,
                callback=callback
            )
            
            # Save final model
            final_path = Path(config.RL_MODEL_DIR) / f"rl_agent_{self.ticker}_final.zip"
            model.save(str(final_path))
            
            training_time = time.time() - start_time
            print(f"\nTraining completed successfully!")
            print(f"Total time: {training_time/3600:.2f} hours")
            print(f"Average speed: {total_timesteps/training_time:.1f} steps/second")
            print(f"Final model: {final_path}")
            print("Run backtester.py to evaluate models")
            
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
            if 'model' in locals():
                emergency_path = Path(config.RL_MODEL_DIR) / f"interrupted_{self.ticker}_{int(time.time())}.zip"
                model.save(str(emergency_path))
                print(f"Model saved to: {emergency_path}")
                
        except Exception as e:
            self.logger.exception(f"Training failed: {e}")
            if 'model' in locals():
                emergency_path = Path(config.RL_MODEL_DIR) / f"error_{self.ticker}_{int(time.time())}.zip"
                model.save(str(emergency_path))
                print(f"Emergency save: {emergency_path}")
            raise
            
        finally:
            if 'env' in locals():
                env.close()

def main():
    """Main training function"""
    ticker = config.TICKERS[0]
    trainer = StreamlinedTrainer(ticker)
    trainer.train()

if __name__ == '__main__':
    main()