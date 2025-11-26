# File: tune_hyperparams.py
# Description: Hyperparameter Tuner for RecurrentPPO (LSTM)
# =============================================================================
import logging
import os
import time
import pandas as pd
import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import torch
import json
from pathlib import Path
from datetime import datetime

# CHANGED: Import RecurrentPPO for LSTM support
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback

import config
from rl_environment import TradingEnv
import reporting

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HyperparameterTuner:
    """Tunes hyperparameters for RecurrentPPO using Optuna"""
    
    def __init__(self, ticker: str, phase: int = 1, n_trials: int = 50, 
                 n_timesteps: int = 100_000, n_eval_episodes: int = 10,
                 use_multiprocessing: bool = False, n_jobs: int = 1,
                 fast_eval: bool = True):
        """
        Initialize hyperparameter tuner.
        """
        self.ticker = ticker
        self.phase = phase
        self.n_trials = n_trials
        self.n_timesteps = n_timesteps
        self.n_eval_episodes = n_eval_episodes
        self.use_multiprocessing = use_multiprocessing
        self.n_jobs = n_jobs if use_multiprocessing else 1
        self.fast_eval = fast_eval
        
        # Load data
        self.train_df, self.eval_df = self.load_and_split_data()
        
        # Results directory
        self.results_dir = Path("hyperparameter_tuning")
        self.results_dir.mkdir(exist_ok=True)
        
        logger.info(f"Initialized tuner for Phase {phase}")
        logger.info(f"Training samples: {len(self.train_df):,}")
        logger.info(f"Evaluation samples: {len(self.eval_df):,}")
    
    def load_and_split_data(self):
        """Load and split data into train/eval sets"""
        data_path = Path("processed_data") / f"{self.ticker}_processed.parquet"
        
        if not data_path.exists():
            raise FileNotFoundError(
                f"Preprocessed data not found at {data_path}. "
                "Run preprocess_data.py first."
            )
        
        df = pd.read_parquet(data_path)
        df = df.dropna().sort_index()
        
        # 80/20 split for train/eval
        split_idx = int(len(df) * 0.8)
        train_df = df.iloc[:split_idx].copy()
        eval_df = df.iloc[split_idx:].copy()
        
        return train_df, eval_df
    
    def create_env(self, df, n_envs=1):
        """Create training environment"""
        env = make_vec_env(
            TradingEnv,
            n_envs=n_envs,
            seed=42,
            vec_env_cls=DummyVecEnv,
            env_kwargs={'df': df, 'phase': self.phase}
        )
        return env
    
    def objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for Optuna optimization.
        Uses RecurrentPPO (LSTM) logic.
        """
        trial_start_time = time.time()
        trial_num = trial.number
        
        # Sample hyperparameters
        hyperparams = self.sample_hyperparameters(trial)
        
        # Log trial start
        logger.info(f"=" * 60)
        logger.info(f"🔬 TRIAL #{trial_num} - Starting (LSTM)")
        logger.info(f"=" * 60)
        
        try:
            # Create environments
            train_env = self.create_env(self.train_df, n_envs=4) # Use 4 envs for LSTM training speed
            eval_env = self.create_env(self.eval_df, n_envs=1)
            
            # Determine device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Create RecurrentPPO model
            logger.info(f"📦 Creating RecurrentPPO model on {device}...")
            model = RecurrentPPO(
                "MultiInputLstmPolicy", # Policy for LSTM
                train_env,
                verbose=0,
                device=device,
                **hyperparams
            )
            
            # Train
            model.learn(total_timesteps=self.n_timesteps)
            
            # Evaluate using stable_baselines3 evaluation helper
            # Note: RecurrentPPO handles states automatically during evaluation
            from stable_baselines3.common.evaluation import evaluate_policy
            mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=self.n_eval_episodes)
            
            training_time = time.time() - trial_start_time

            logger.info(f"✅ Trial #{trial_num} Complete | Reward: {mean_reward:.4f} | Time: {training_time:.1f}s")
            
            # Cleanup
            train_env.close()
            eval_env.close()
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Optuna minimizes, so return negative reward
            return -mean_reward
            
        except Exception as e:
            logger.error(f"❌ Trial #{trial_num} failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return float('inf')  # Worst possible score
    
    def sample_hyperparameters(self, trial: optuna.Trial) -> dict:
        """
        Sample hyperparameters including LSTM specific ones.
        """
        # Learning rate
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        
        # LSTM specific sequence settings
        batch_size = trial.suggest_categorical("batch_size", [128, 256, 512]) # Sequence count
        n_steps = trial.suggest_categorical("n_steps", [1024, 2048])
        
        # LSTM Architecture
        lstm_hidden_size = trial.suggest_categorical("lstm_hidden_size", [64, 128, 256])
        n_lstm_layers = trial.suggest_int("n_lstm_layers", 1, 2)
        shared_lstm = trial.suggest_categorical("shared_lstm", [True, False])
        enable_critic_lstm = trial.suggest_categorical("enable_critic_lstm", [True, False])
        
        # Feature Extractor Architecture
        net_arch_type = trial.suggest_categorical("net_arch", ["small", "medium"])
        if net_arch_type == "small":
            net_arch = dict(pi=[64, 64], vf=[64, 64])
        else:
            net_arch = dict(pi=[128, 128], vf=[128, 128])
        
        return {
            "learning_rate": learning_rate,
            "n_steps": n_steps,
            "batch_size": batch_size,
            "policy_kwargs": {
                "net_arch": net_arch,
                "lstm_hidden_size": lstm_hidden_size,
                "n_lstm_layers": n_lstm_layers,
                "shared_lstm": shared_lstm,
                "enable_critic_lstm": enable_critic_lstm,
                "ortho_init": True,
            }
        }
    
    def optimize(self):
        """Run hyperparameter optimization"""
        print("="*80)
        print(f"🔍 HYPERPARAMETER TUNING - PHASE {self.phase} (LSTM MODE)")
        print("="*80)
        
        study_name = f"phase{self.phase}_{self.ticker}_lstm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        study = optuna.create_study(
            study_name=study_name,
            direction="minimize",
            sampler=TPESampler(seed=42),
            pruner=MedianPruner()
        )
        
        study.optimize(
            self.objective,
            n_trials=self.n_trials,
            n_jobs=self.n_jobs
        )
        
        print("\n" + "="*80)
        print("✅ OPTIMIZATION COMPLETE")
        print("="*80)
        print("Best Params:", study.best_params)
        
        self.save_results(study, study_name)
        return study.best_params

    def save_results(self, study, study_name):
        """Save optimization results"""
        best_params_file = self.results_dir / f"{study_name}_best_params.json"
        with open(best_params_file, 'w') as f:
            json.dump(study.best_params, f, indent=4)
        logger.info(f"✅ Best hyperparameters saved: {best_params_file}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, default=1)
    parser.add_argument("--trials", type=int, default=30)
    args = parser.parse_args()
    
    tuner = HyperparameterTuner(
        ticker=config.TICKERS[0],
        phase=args.phase,
        n_trials=args.trials
    )
    tuner.optimize()

if __name__ == '__main__':
    main()