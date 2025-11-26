"""
Hyperparameter Tuning System for Trading RL Agent
Uses Optuna for efficient hyperparameter optimization
"""
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

from stable_baselines3 import PPO
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
    """Tunes hyperparameters using Optuna"""
    
    def __init__(self, ticker: str, phase: int = 1, n_trials: int = 50, 
                 n_timesteps: int = 100_000, n_eval_episodes: int = 10,
                 use_multiprocessing: bool = False, n_jobs: int = 1):
        """
        Initialize hyperparameter tuner.
        
        Args:
            ticker: Trading instrument ticker
            phase: Curriculum phase (1, 2, or 3)
            n_trials: Number of optimization trials
            n_timesteps: Timesteps per trial (keep low for speed)
            n_eval_episodes: Episodes for evaluation
            use_multiprocessing: Use parallel processing (experimental)
            n_jobs: Number of parallel jobs if multiprocessing enabled
        """
        self.ticker = ticker
        self.phase = phase
        self.n_trials = n_trials
        self.n_timesteps = n_timesteps
        self.n_eval_episodes = n_eval_episodes
        self.use_multiprocessing = use_multiprocessing
        self.n_jobs = n_jobs if use_multiprocessing else 1
        
        # Load data
        self.train_df, self.eval_df = self.load_and_split_data()
        
        # Results directory
        self.results_dir = Path("hyperparameter_tuning")
        self.results_dir.mkdir(exist_ok=True)
        
        logger.info(f"Initialized tuner for Phase {phase}")
        logger.info(f"Training samples: {len(self.train_df):,}")
        logger.info(f"Evaluation samples: {len(self.eval_df):,}")
        if use_multiprocessing:
            logger.info(f"⚡ Multiprocessing enabled: {n_jobs} parallel jobs")
    
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
        
        Args:
            trial: Optuna trial object
            
        Returns:
            float: Negative mean reward (Optuna minimizes)
        """
        trial_start_time = time.time()
        trial_num = trial.number
        
        # Sample hyperparameters
        hyperparams = self.sample_hyperparameters(trial)
        
        # Log trial start
        logger.info(f"=" * 60)
        logger.info(f"🔬 TRIAL #{trial_num} - Starting")
        logger.info(f"=" * 60)
        logger.info("Testing hyperparameters:")
        logger.info(f"  Learning Rate: {hyperparams['learning_rate']:.6f}")
        logger.info(f"  Batch Size: {hyperparams['batch_size']}")
        logger.info(f"  N Steps: {hyperparams['n_steps']}")
        logger.info(f"  N Epochs: {hyperparams['n_epochs']}")
        logger.info(f"  Network: {trial.params['net_arch']}")
        logger.info(f"  Activation: {trial.params['activation_fn']}")
        logger.info("-" * 60)
        
        try:
            # Create environments
            train_env = self.create_env(self.train_df, n_envs=1)
            eval_env = self.create_env(self.eval_df, n_envs=1)
            
            # Determine device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Create model with sampled hyperparameters
            logger.info(f"📦 Creating PPO model on {device}...")
            model = PPO(
                "MultiInputPolicy",
                train_env,
                verbose=0,
                device=device,
                **hyperparams
            )
            
            # Create custom callback for progress tracking
            class TrialProgressCallback(BaseCallback):
                """Custom callback for logging training progress"""
                
                def __init__(self, trial_num, total_timesteps, verbose=0):
                    super(TrialProgressCallback, self).__init__(verbose)
                    self.trial_num = trial_num
                    self.total_timesteps = total_timesteps
                    self.start_time = None
                    self.last_log_time = None
                    self.last_log_step = 0
                
                def _on_training_start(self) -> None:
                    """Called at the beginning of training"""
                    self.start_time = time.time()
                    self.last_log_time = time.time()
                    self.last_log_step = 0
                
                def _on_step(self) -> bool:
                    """Called after each step"""
                    if self.start_time is None:
                        return True
                    
                    current_step = self.num_timesteps
                    current_time = time.time()
                    
                    # Log every 10k steps or 30 seconds, whichever comes first
                    time_since_log = current_time - self.last_log_time
                    steps_since_log = current_step - self.last_log_step
                    
                    if steps_since_log >= 10000 or (time_since_log >= 30 and steps_since_log > 0):
                        elapsed = current_time - self.start_time
                        progress_pct = (current_step / self.total_timesteps) * 100
                        
                        # Calculate speed
                        if elapsed > 0:
                            steps_per_sec = current_step / elapsed
                            eta_seconds = (self.total_timesteps - current_step) / steps_per_sec if steps_per_sec > 0 else 0
                            eta_mins = eta_seconds / 60
                        else:
                            steps_per_sec = 0
                            eta_mins = 0
                        
                        logger.info(
                            f"🏃 Trial #{self.trial_num} | "
                            f"Step {current_step:,}/{self.total_timesteps:,} ({progress_pct:.1f}%) | "
                            f"Speed: {steps_per_sec:.1f} steps/s | "
                            f"ETA: {eta_mins:.1f}m"
                        )
                        
                        self.last_log_time = current_time
                        self.last_log_step = current_step
                    
                    return True
            
            progress_callback = TrialProgressCallback(trial_num, self.n_timesteps)
            
            # Evaluation callback with early stopping
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=None,
                log_path=None,
                eval_freq=max(self.n_timesteps // 10, 1000),
                n_eval_episodes=self.n_eval_episodes,
                deterministic=True,
                render=False,
                verbose=0
            )
            
            # Combine callbacks
            callback_list = [eval_callback, progress_callback]
            
            # Train
            logger.info(f"🚀 Training for {self.n_timesteps:,} timesteps...")
            training_start = time.time()
            
            model.learn(
                total_timesteps=self.n_timesteps,
                callback=callback_list,
                progress_bar=False
            )
            
            training_time = time.time() - training_start
            
            # Evaluate on eval set
            logger.info(f"📊 Evaluating on {self.n_eval_episodes} episodes...")
            eval_start = time.time()
            mean_reward = self.evaluate_model(model, eval_env)
            eval_time = time.time() - eval_start
            
            total_time = time.time() - trial_start_time
            
            # Log results
            logger.info("-" * 60)
            logger.info(f"✅ Trial #{trial_num} Complete")
            logger.info(f"  Mean Reward: {mean_reward:.4f}")
            logger.info(f"  Training Time: {training_time:.1f}s")
            logger.info(f"  Evaluation Time: {eval_time:.1f}s")
            logger.info(f"  Total Time: {total_time:.1f}s")
            logger.info(f"  Speed: {self.n_timesteps/training_time:.1f} steps/s")
            logger.info("=" * 60 + "\n")
            
            # Cleanup
            train_env.close()
            eval_env.close()
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # Report intermediate value for pruning
            trial.report(mean_reward, self.n_timesteps)
            
            # Optuna minimizes, so return negative reward
            return -mean_reward
            
        except Exception as e:
            logger.error(f"❌ Trial #{trial_num} failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return float('inf')  # Worst possible score
    
    def sample_hyperparameters(self, trial: optuna.Trial) -> dict:
        """
        Sample hyperparameters for the trial.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            dict: Sampled hyperparameters
        """
        # Learning rate
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        
        # PPO-specific parameters
        n_steps = trial.suggest_categorical("n_steps", [1024, 2048, 4096])
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512, 1024])
        n_epochs = trial.suggest_int("n_epochs", 3, 20)
        gamma = trial.suggest_float("gamma", 0.9, 0.9999)
        gae_lambda = trial.suggest_float("gae_lambda", 0.9, 0.99)
        clip_range = trial.suggest_float("clip_range", 0.1, 0.4)
        
        # Entropy and value function coefficients
        ent_coef = trial.suggest_float("ent_coef", 0.0, 0.1)
        vf_coef = trial.suggest_float("vf_coef", 0.1, 1.0)
        max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 1.0)
        
        # Network architecture
        net_arch_type = trial.suggest_categorical("net_arch", ["small", "medium", "large"])
        
        if net_arch_type == "small":
            net_arch = {"pi": [128, 128], "vf": [128, 128]}
        elif net_arch_type == "medium":
            net_arch = {"pi": [256, 256], "vf": [256, 256]}
        else:  # large
            net_arch = {"pi": [256, 256, 128], "vf": [256, 256, 128]}
        
        # Activation function
        activation_fn_name = trial.suggest_categorical("activation_fn", ["relu", "tanh"])
        activation_fn = torch.nn.ReLU if activation_fn_name == "relu" else torch.nn.Tanh
        
        hyperparams = {
            "learning_rate": learning_rate,
            "n_steps": n_steps,
            "batch_size": batch_size,
            "n_epochs": n_epochs,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "clip_range": clip_range,
            "ent_coef": ent_coef,
            "vf_coef": vf_coef,
            "max_grad_norm": max_grad_norm,
            "policy_kwargs": {
                "net_arch": net_arch,
                "activation_fn": activation_fn,
                "ortho_init": True,
            }
        }
        
        return hyperparams
    
    def evaluate_model(self, model, eval_env, n_episodes=None):
        """
        Evaluate model performance.
        
        Args:
            model: Trained model
            eval_env: Evaluation environment
            n_episodes: Number of episodes (defaults to self.n_eval_episodes)
            
        Returns:
            float: Mean episode reward
        """
        if n_episodes is None:
            n_episodes = self.n_eval_episodes
        
        episode_rewards = []
        
        for _ in range(n_episodes):
            obs = eval_env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _ = eval_env.step(action)
                episode_reward += reward[0] if isinstance(reward, np.ndarray) else reward
                
                if done:
                    break
            
            episode_rewards.append(episode_reward)
        
        return np.mean(episode_rewards)
    
    def optimize(self):
        """Run hyperparameter optimization"""
        print("="*80)
        print(f"🔍 HYPERPARAMETER TUNING - PHASE {self.phase}")
        print("="*80)
        print(f"Trials: {self.n_trials}")
        print(f"Timesteps per trial: {self.n_timesteps:,}")
        print(f"Evaluation episodes: {self.n_eval_episodes}")
        if self.use_multiprocessing:
            print(f"⚡ Parallel jobs: {self.n_jobs}")
        print(f"Estimated time: {(self.n_trials * self.n_timesteps / 1000):.0f}-{(self.n_trials * self.n_timesteps / 500):.0f} minutes")
        print("="*80 + "\n")
        
        # Create Optuna study
        study_name = f"phase{self.phase}_{self.ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        study = optuna.create_study(
            study_name=study_name,
            direction="minimize",  # Minimize negative reward = maximize reward
            sampler=TPESampler(n_startup_trials=min(10, self.n_trials // 5), seed=42),
            pruner=MedianPruner(n_startup_trials=min(5, self.n_trials // 10), n_warmup_steps=self.n_timesteps // 3)
        )
        
        # Optimize
        logger.info("🚀 Starting optimization...")
        if not self.use_multiprocessing:
            logger.info(f"📊 Progress will be logged every 10k steps or 30 seconds\n")
        
        study.optimize(
            self.objective,
            n_trials=self.n_trials,
            show_progress_bar=False,
            n_jobs=self.n_jobs,  # Use parallel jobs if enabled
            timeout=None
        )
        
        # Results
        print("\n" + "="*80)
        print("✅ OPTIMIZATION COMPLETE")
        print("="*80)
        
        best_trial = study.best_trial
        print(f"\n🏆 Best Trial: #{best_trial.number}")
        print(f"📊 Best Value (Mean Reward): {-best_trial.value:.4f}")
        print(f"\n📋 Best Hyperparameters:")
        
        for key, value in best_trial.params.items():
            print(f"  {key}: {value}")
        
        # Show top 5 trials
        print(f"\n🎯 Top 5 Trials:")
        print("-" * 80)
        sorted_trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else float('inf'))
        for i, trial in enumerate(sorted_trials[:5], 1):
            if trial.value is not None:
                print(f"  #{i}. Trial {trial.number}: Reward = {-trial.value:.4f}")
        print("-" * 80)
        
        # Save results
        self.save_results(study, study_name)
        
        # Generate report
        self.generate_report(study)
        
        return study.best_params
    
    def save_results(self, study, study_name):
        """Save optimization results"""
        # Save best hyperparameters
        best_params_file = self.results_dir / f"{study_name}_best_params.json"
        
        with open(best_params_file, 'w') as f:
            json.dump(study.best_params, f, indent=4)
        
        logger.info(f"✅ Best hyperparameters saved: {best_params_file}")
        
        # Save all trials
        trials_df = study.trials_dataframe()
        trials_file = self.results_dir / f"{study_name}_all_trials.csv"
        trials_df.to_csv(trials_file, index=False)
        
        logger.info(f"✅ All trials saved: {trials_file}")
        
        # Save study using joblib (more compatible)
        try:
            import joblib
            study_file = self.results_dir / f"{study_name}_study.pkl"
            joblib.dump(study, str(study_file))
            logger.info(f"✅ Study object saved: {study_file}")
        except Exception as e:
            logger.warning(f"Could not save study object: {e}")
    
    def generate_report(self, study):
        """Generate optimization report with visualizations"""
        try:
            import matplotlib.pyplot as plt
            from optuna.visualization import (
                plot_optimization_history,
                plot_param_importances,
                plot_slice
            )
            
            study_name = study.study_name
            
            # Optimization history
            fig1 = plot_optimization_history(study)
            fig1.write_html(str(self.results_dir / f"{study_name}_history.html"))
            
            # Parameter importances
            try:
                fig2 = plot_param_importances(study)
                fig2.write_html(str(self.results_dir / f"{study_name}_importances.html"))
            except:
                logger.warning("Could not generate parameter importances plot")
            
            # Slice plot
            try:
                fig3 = plot_slice(study)
                fig3.write_html(str(self.results_dir / f"{study_name}_slice.html"))
            except:
                logger.warning("Could not generate slice plot")
            
            logger.info(f"✅ Visualization reports saved in {self.results_dir}")
            
        except Exception as e:
            logger.warning(f"Could not generate visualizations: {e}")
    
    def apply_best_params_to_config(self, best_params):
        """
        Generate config code with best hyperparameters.
        
        Args:
            best_params: Dictionary of best hyperparameters
        """
        print("\n" + "="*80)
        print("📝 APPLY TO CONFIG.PY")
        print("="*80)
        print("\nAdd this to your config.py:\n")
        
        print("# Optimized PPO Hyperparameters (from tuning)")
        print("def get_device_optimized_hyperparams():")
        print("    device = 'cuda' if torch.cuda.is_available() else 'cpu'")
        print("    ")
        
        # Network architecture
        net_arch_type = best_params['net_arch']
        if net_arch_type == "small":
            net_arch_str = '{"pi": [128, 128], "vf": [128, 128]}'
        elif net_arch_type == "medium":
            net_arch_str = '{"pi": [256, 256], "vf": [256, 256]}'
        else:
            net_arch_str = '{"pi": [256, 256, 128], "vf": [256, 256, 128]}'
        
        # Activation function
        activation_fn = "torch.nn.ReLU" if best_params['activation_fn'] == 'relu' else "torch.nn.Tanh"
        
        print("    return {")
        print(f"        'n_steps': {best_params['n_steps']},")
        print(f"        'batch_size': {best_params['batch_size']},")
        print(f"        'n_epochs': {best_params['n_epochs']},")
        print(f"        'gamma': {best_params['gamma']},")
        print(f"        'gae_lambda': {best_params['gae_lambda']},")
        print(f"        'clip_range': {best_params['clip_range']},")
        print(f"        'ent_coef': {best_params['ent_coef']},")
        print(f"        'vf_coef': {best_params['vf_coef']},")
        print(f"        'max_grad_norm': {best_params['max_grad_norm']},")
        print(f"        'learning_rate': {best_params['learning_rate']},")
        print("        'policy_kwargs': {")
        print(f"            'net_arch': {net_arch_str},")
        print(f"            'activation_fn': {activation_fn},")
        print("            'ortho_init': True,")
        print("        }")
        print("    }")
        print("\nPPO_HYPERPARAMS = get_device_optimized_hyperparams()")
        print("="*80)


def main():
    """Main tuning function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for trading RL agent")
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2, 3],
                       help="Curriculum phase to tune (1, 2, or 3)")
    parser.add_argument("--trials", type=int, default=50,
                       help="Number of optimization trials")
    parser.add_argument("--timesteps", type=int, default=100_000,
                       help="Timesteps per trial (keep low for speed)")
    parser.add_argument("--eval-episodes", type=int, default=10,
                       help="Episodes for evaluation")
    parser.add_argument("--parallel", action="store_true",
                       help="Enable parallel processing (faster but uses more resources)")
    parser.add_argument("--n-jobs", type=int, default=2,
                       help="Number of parallel jobs if --parallel is enabled")
    
    args = parser.parse_args()
    
    # Get ticker from config
    ticker = config.TICKERS[0]
    
    # Speed recommendations
    print("\n💡 SPEED OPTIMIZATION TIPS:")
    print("=" * 80)
    print("1. ⚡ Use --parallel flag for 2-3x speedup (if you have enough RAM)")
    print("   Example: python tune_hyperparams.py --phase 1 --parallel --n-jobs 2")
    print("\n2. 🔥 Use GPU for 5-10x speedup")
    print("   Make sure CUDA is available and configured")
    print("\n3. 📉 Reduce --timesteps for faster trials (trade accuracy for speed)")
    print("   Try: --timesteps 25000 or --timesteps 10000")
    print("\n4. 🎯 Reduce --trials for quicker results")
    print("   Try: --trials 20 or --trials 10")
    print("\n5. 📊 Use fewer --eval-episodes")
    print("   Try: --eval-episodes 5")
    print("=" * 80 + "\n")
    
    # Create tuner
    tuner = HyperparameterTuner(
        ticker=ticker,
        phase=args.phase,
        n_trials=args.trials,
        n_timesteps=args.timesteps,
        n_eval_episodes=args.eval_episodes,
        use_multiprocessing=args.parallel,
        n_jobs=args.n_jobs
    )
    
    # Run optimization
    best_params = tuner.optimize()
    
    # Show how to apply to config
    tuner.apply_best_params_to_config(best_params)
    
    print("\n💡 TIP: After applying best params, train your model:")
    print(f"   Set CURRICULUM_PHASE = {args.phase} in config.py")
    print("   Run: python manual_phase_trainer.py")


if __name__ == '__main__':
    main()